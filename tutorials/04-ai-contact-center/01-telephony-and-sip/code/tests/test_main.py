import unittest
import warnings
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

try:  # removed from the standard library in Python 3.13
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop
except ImportError:
    audioop = None


class SipSdpTests(unittest.TestCase):
    def setUp(self):
        self.invite = main.parse_sip_message(main.SAMPLE_INVITE)
        self.offer = main.parse_sdp(self.invite["body"])

    def test_start_line_and_case_insensitive_headers(self):
        self.assertEqual(self.invite["method"], "INVITE")
        self.assertEqual(self.invite["headers"]["call-id"], "a84b4c76e66710@carrier.example.net")
        self.assertEqual(self.invite["headers"]["cseq"], "314159 INVITE")

    def test_sdp_media_line(self):
        self.assertEqual(self.offer["connection"], "203.0.113.10")
        self.assertEqual(self.offer["port"], 40000)
        self.assertEqual(self.offer["payload_types"], [8, 0, 101])
        self.assertEqual(self.offer["rtpmap"][101], ("telephone-event", 8000))

    def test_negotiation_follows_offer_order(self):
        self.assertEqual(main.negotiate_codec(self.offer), (8, "PCMA", 8000))
        self.assertEqual(main.negotiate_codec(self.offer, ("PCMU",)), (0, "PCMU", 8000))

    def test_static_payload_type_without_rtpmap(self):
        offer = main.parse_sdp("c=IN IP4 10.0.0.1\nm=audio 5004 RTP/AVP 0\n")
        self.assertEqual(offer["rtpmap"][0], ("PCMU", 8000))

    def test_no_common_codec_raises(self):
        with self.assertRaises(ValueError):
            main.negotiate_codec(self.offer, ("OPUS",))


class RtpTests(unittest.TestCase):
    def test_header_roundtrip(self):
        header = main.pack_rtp_header(65535, 2 ** 32 - 1, 42, 0, marker=True)
        self.assertEqual(len(header), 12)
        fields = main.unpack_rtp_header(header)
        self.assertEqual((fields["version"], fields["marker"], fields["seq"]), (2, True, 65535))
        self.assertEqual((fields["timestamp"], fields["ssrc"]), (2 ** 32 - 1, 42))

    def test_twenty_ms_packets_at_8khz(self):
        packets = main.packetize(bytes(8000), payload_type=0)
        self.assertEqual(len(packets), 50)
        second = main.unpack_rtp_header(packets[1])
        self.assertEqual((second["seq"], second["timestamp"], len(second["payload"])), (1, 160, 160))
        self.assertFalse(second["marker"])

    def test_wire_bitrate(self):
        self.assertAlmostEqual(main.rtp_bitrate_kbps(), 80.0)


class MulawTests(unittest.TestCase):
    def test_known_codes(self):
        codes = main.mulaw_encode(np.array([0, 32767, -32768], dtype=np.int16))
        np.testing.assert_array_equal(codes, [0xFF, 0x80, 0x00])
        np.testing.assert_array_equal(main.mulaw_decode(np.array([0xFF, 0x80])), [0, 32124])

    def test_relative_error_is_bounded(self):
        x = np.arange(-32768, 32768, 13, dtype=np.int16)
        y = main.mulaw_decode(main.mulaw_encode(x)).astype(np.float64)
        large = np.abs(x) > 1000
        relative = np.abs(y[large] - x[large]) / np.abs(x[large].astype(np.float64))
        self.assertLess(float(relative.max()), 0.07)  # 16 steps per segment -> at most ~1/32 + clipping

    def test_mulaw_beats_8bit_linear_on_quiet_speech(self):
        quiet = 0.01 * np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000)
        self.assertGreater(main.snr_db(quiet, main.mulaw_roundtrip(quiet)),
                           main.snr_db(quiet, main.quantize_linear(quiet, 8)) + 15)

    @unittest.skipIf(audioop is None, "audioop is not available")
    def test_matches_standard_library(self):
        x = np.arange(-32768, 32768, dtype=np.int16)
        reference = np.frombuffer(audioop.lin2ulaw(x.tobytes(), 2), dtype=np.uint8)
        np.testing.assert_array_equal(main.mulaw_encode(x), reference)
        codes = np.arange(256, dtype=np.uint8)
        reference = np.frombuffer(audioop.ulaw2lin(codes.tobytes(), 2), dtype=np.int16)
        np.testing.assert_array_equal(main.mulaw_decode(codes), reference)


class TelephoneBandTests(unittest.TestCase):
    def test_bandpass_passes_voice_and_blocks_outside(self):
        taps = main.bandpass_fir(300, 3400, 16_000)
        response = np.abs(np.fft.rfft(taps, 16_000))  # 1 Hz bins
        self.assertGreater(response[1000], 0.95)
        self.assertLess(response[100], 0.1)
        self.assertLess(response[6000], 0.01)

    def test_phone_channel_halves_rate_and_removes_fricative_band(self):
        utterance = main.synthetic_utterance()
        phone = main.phone_channel(utterance)
        self.assertEqual(len(phone), len(utterance) // 2)
        self.assertGreater(main.band_energy_fraction(utterance, 16_000, 3400, 8000), 0.1)

    def test_missing_fundamental_keeps_pitch(self):
        phone = main.phone_channel(main.synthetic_utterance(f0=120.0))
        self.assertAlmostEqual(main.autocorrelation_pitch(phone[:3200], 8000), 120.0, delta=3.0)


class DtmfTests(unittest.TestCase):
    def test_every_key_survives_mulaw(self):
        for key in "0123456789*#ABCD":
            self.assertEqual(main.detect_dtmf(main.mulaw_roundtrip(main.dtmf_tone(key))), key)

    def test_speech_like_signal_is_not_a_key(self):
        self.assertIsNone(main.detect_dtmf(main.synthetic_utterance(sample_rate=8000)[:640]))

    def test_goertzel_matches_dft_bin(self):
        x = main.dtmf_tone("5")
        n = len(x)
        k = 1336 * n / 8000
        reference = np.abs(np.sum(x * np.exp(-2j * np.pi * k * np.arange(n) / n))) ** 2
        self.assertAlmostEqual(main.goertzel_power(x, 1336, 8000), reference, delta=reference * 1e-6)


if __name__ == "__main__":
    unittest.main()
