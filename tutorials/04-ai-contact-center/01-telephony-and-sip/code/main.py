"""Lesson 01: what a phone call looks like before it reaches a voice agent.

Builds the pieces of a SIP call that matter to a speech engineer, with NumPy and the
standard library only: SIP/SDP parsing and codec negotiation, RTP packetization,
bit-exact G.711 mu-law, the 300-3400 Hz telephone band, and Goertzel DTMF detection.

Run: python3 code/main.py
"""

from __future__ import annotations

import struct

import numpy as np

# --------------------------------------------------------------------------- SIP / SDP

SAMPLE_INVITE = (
    "INVITE sip:+842873001234@sip.example-agent.io SIP/2.0\r\n"
    "Via: SIP/2.0/UDP 203.0.113.10:5060;branch=z9hG4bK776asdhds\r\n"
    "Max-Forwards: 70\r\n"
    "From: <sip:+84901234567@carrier.example.net>;tag=1928301774\r\n"
    "To: <sip:+842873001234@sip.example-agent.io>\r\n"
    "Call-ID: a84b4c76e66710@carrier.example.net\r\n"
    "CSeq: 314159 INVITE\r\n"
    "Contact: <sip:+84901234567@203.0.113.10:5060>\r\n"
    "Content-Type: application/sdp\r\n"
    "\r\n"
    "v=0\r\n"
    "o=- 4858 4858 IN IP4 203.0.113.10\r\n"
    "s=call\r\n"
    "c=IN IP4 203.0.113.10\r\n"
    "t=0 0\r\n"
    "m=audio 40000 RTP/AVP 8 0 101\r\n"
    "a=rtpmap:8 PCMA/8000\r\n"
    "a=rtpmap:0 PCMU/8000\r\n"
    "a=rtpmap:101 telephone-event/8000\r\n"
    "a=fmtp:101 0-16\r\n"
    "a=ptime:20\r\n"
    "a=sendrecv\r\n"
)

# Static RTP payload types from RFC 3551; dynamic types (96-127) need an a=rtpmap line.
STATIC_PAYLOAD_TYPES = {0: ("PCMU", 8000), 8: ("PCMA", 8000), 9: ("G722", 8000), 18: ("G729", 8000)}


def parse_sip_message(raw: str) -> dict:
    """Split a SIP request into its start line, headers, and body.

    SIP is a text protocol modelled on HTTP: a start line, ``Name: value`` headers, an empty
    line, then an optional body (usually SDP). Header names are case-insensitive.
    """
    head, _, body = raw.replace("\r\n", "\n").partition("\n\n")
    lines = head.split("\n")
    method, uri, version = lines[0].split(" ", 2)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return {"method": method, "uri": uri, "version": version, "headers": headers, "body": body}


def parse_sdp(body: str) -> dict:
    """Read the parts of an SDP offer that decide how audio will flow."""
    sdp: dict = {"connection": None, "port": None, "payload_types": [], "rtpmap": {}, "ptime_ms": 20}
    for line in body.strip().split("\n"):
        key, _, value = line.strip().partition("=")
        if key == "c":
            sdp["connection"] = value.split()[-1]
        elif key == "m" and value.startswith("audio"):
            fields = value.split()
            sdp["port"] = int(fields[1])
            sdp["payload_types"] = [int(pt) for pt in fields[3:]]
        elif key == "a" and value.startswith("rtpmap:"):
            pt, _, encoding = value[len("rtpmap:"):].partition(" ")
            name, rate = encoding.split("/")[:2]
            sdp["rtpmap"][int(pt)] = (name, int(rate))
        elif key == "a" and value.startswith("ptime:"):
            sdp["ptime_ms"] = int(value[len("ptime:"):])
    for pt in sdp["payload_types"]:
        if pt not in sdp["rtpmap"] and pt in STATIC_PAYLOAD_TYPES:
            sdp["rtpmap"][pt] = STATIC_PAYLOAD_TYPES[pt]
    return sdp


def negotiate_codec(offer: dict, supported: tuple[str, ...] = ("PCMU", "PCMA")) -> tuple[int, str, int]:
    """Pick the first audio codec in the *offer's* order that the answerer supports.

    Returns ``(payload_type, codec_name, clock_rate)``. ``telephone-event`` is not an audio codec;
    it carries DTMF alongside the chosen codec.
    """
    wanted = {name.upper() for name in supported}
    for pt in offer["payload_types"]:
        name, rate = offer["rtpmap"].get(pt, ("", 0))
        if name.upper() in wanted:
            return pt, name, rate
    raise ValueError("no common audio codec: the answerer must reply 488 Not Acceptable Here")


# --------------------------------------------------------------------------- RTP

RTP_HEADER = struct.Struct("!BBHII")  # 12 bytes, network byte order


def pack_rtp_header(seq: int, timestamp: int, ssrc: int, payload_type: int, marker: bool = False) -> bytes:
    """Build the fixed 12-byte RTP header (RFC 3550): version 2, no padding/extension/CSRC."""
    first = 2 << 6
    second = (int(marker) << 7) | (payload_type & 0x7F)
    return RTP_HEADER.pack(first, second, seq & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc & 0xFFFFFFFF)


def unpack_rtp_header(packet: bytes) -> dict:
    first, second, seq, timestamp, ssrc = RTP_HEADER.unpack(packet[: RTP_HEADER.size])
    return {
        "version": first >> 6,
        "marker": bool(second >> 7),
        "payload_type": second & 0x7F,
        "seq": seq,
        "timestamp": timestamp,
        "ssrc": ssrc,
        "payload": packet[RTP_HEADER.size:],
    }


def packetize(payload: bytes, payload_type: int, sample_rate: int = 8000, ptime_ms: int = 20,
              ssrc: int = 0x1234ABCD, first_seq: int = 0, first_timestamp: int = 0) -> list[bytes]:
    """Cut an encoded G.711 stream (1 byte per sample) into RTP packets of ``ptime_ms``."""
    samples_per_packet = sample_rate * ptime_ms // 1000
    packets = []
    for index, start in enumerate(range(0, len(payload), samples_per_packet)):
        header = pack_rtp_header(first_seq + index, first_timestamp + index * samples_per_packet,
                                 ssrc, payload_type, marker=index == 0)
        packets.append(header + payload[start:start + samples_per_packet])
    return packets


def rtp_bitrate_kbps(sample_rate: int = 8000, bits_per_sample: int = 8, ptime_ms: int = 20,
                     overhead_bytes: int = 12 + 8 + 20) -> float:
    """Bitrate on the wire for one direction: payload plus RTP/UDP/IPv4 headers."""
    packets_per_second = 1000 / ptime_ms
    payload_bytes = sample_rate * ptime_ms / 1000 * bits_per_sample / 8
    return (payload_bytes + overhead_bytes) * 8 * packets_per_second / 1000


# --------------------------------------------------------------------------- G.711 mu-law

MU_BIAS = 0x84
MU_SEGMENT_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def mulaw_encode(pcm16: np.ndarray) -> np.ndarray:
    """G.711 mu-law encoder: int16 PCM -> uint8 codes (same output as the classic Sun g711.c).

    The 14-bit magnitude is biased by 33, then stored as a 3-bit segment (exponent) and a
    4-bit step inside that segment. Each segment doubles the step size: logarithmic resolution.
    The code is inverted (all bits flipped) so that silence is 0xFF, not a run of zero bits.
    """
    x = np.asarray(pcm16, dtype=np.int32) >> 2  # 14-bit, arithmetic shift as in C
    mask = np.where(x < 0, 0x7F, 0xFF)
    magnitude = np.minimum(np.abs(x), 8159) + (MU_BIAS >> 2)
    segment = np.searchsorted(MU_SEGMENT_ENDS, magnitude, side="left")
    step = (magnitude >> (np.minimum(segment, 7) + 1)) & 0x0F
    code = np.where(segment >= 8, 0x7F, (segment << 4) | step)
    return (code ^ mask).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """Bit-exact G.711 mu-law decoder: uint8 codes -> int16 PCM."""
    u = ~np.asarray(codes, dtype=np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = (((mantissa << 3) + MU_BIAS) << exponent) - MU_BIAS
    return np.where(u & 0x80, -magnitude, magnitude).astype(np.int16)


def mulaw_compress(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """The continuous mu-law curve that G.711 approximates with 8 straight segments."""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def quantize_linear(x: np.ndarray, bits: int) -> np.ndarray:
    """Uniform quantizer with the same bit budget, for comparison with mu-law."""
    levels = 2 ** (bits - 1) - 1
    return np.rint(np.clip(x, -1.0, 1.0) * levels) / levels


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    noise = reference - np.asarray(test, dtype=np.float64)
    return float(10 * np.log10(np.sum(reference ** 2) / max(np.sum(noise ** 2), 1e-20)))


def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """float [-1, 1] -> int16 -> mu-law byte -> int16 -> float."""
    return mulaw_decode(mulaw_encode(float_to_pcm16(x))).astype(np.float64) / 32767


# --------------------------------------------------------------------------- telephone band

def bandpass_fir(low_hz: float, high_hz: float, sample_rate: int, taps: int = 255) -> np.ndarray:
    """Windowed-sinc band-pass FIR: difference of two low-pass filters, Hamming window."""
    if taps % 2 == 0:
        raise ValueError("taps must be odd for a linear-phase filter")
    n = np.arange(taps) - (taps - 1) / 2

    def lowpass(cutoff: float) -> np.ndarray:
        return 2 * cutoff / sample_rate * np.sinc(2 * cutoff / sample_rate * n)

    return (lowpass(high_hz) - lowpass(low_hz)) * np.hamming(taps)


def phone_channel(x: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Approximate the PSTN path: 300-3400 Hz band-pass, 8 kHz sampling, G.711 mu-law."""
    if sample_rate % 8000:
        raise ValueError("sample_rate must be a multiple of 8000")
    filtered = np.convolve(x, bandpass_fir(300.0, 3400.0, sample_rate), mode="same")
    narrow = filtered[:: sample_rate // 8000]  # safe: the filter already removed energy above 4 kHz
    return mulaw_roundtrip(narrow)


def band_energy_fraction(x: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    """Share of a signal's energy between ``low_hz`` and ``high_hz``."""
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    band = (freqs >= low_hz) & (freqs < high_hz)
    return float(spectrum[band].sum() / spectrum.sum())


def synthetic_utterance(sample_rate: int = 16_000, f0: float = 120.0, seed: int = 0) -> np.ndarray:
    """A vowel (harmonics of ``f0`` shaped by three formants) followed by an /s/-like fricative."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.4 * sample_rate)) / sample_rate
    vowel = np.zeros_like(t)
    for k in range(1, int(sample_rate / 2 / f0)):
        f = k * f0
        gain = sum(np.exp(-0.5 * ((f - fc) / bw) ** 2) for fc, bw in ((700, 130), (1200, 150), (2600, 250)))
        vowel += (0.05 + gain) / k ** 0.5 * np.sin(2 * np.pi * f * t)
    vowel *= 0.5 / np.max(np.abs(vowel))
    noise = rng.standard_normal(int(0.25 * sample_rate))
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), d=1.0 / sample_rate)
    spectrum *= np.exp(-0.5 * ((freqs - 6000) / 1200) ** 2)  # /s/ concentrates around 4-8 kHz
    fricative = np.fft.irfft(spectrum, n=len(noise))
    fricative *= 0.25 / np.max(np.abs(fricative))
    return np.concatenate([vowel, fricative])


def autocorrelation_pitch(x: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 400.0) -> float:
    """Period of the strongest autocorrelation peak in the voice range (see module 01, lesson 08)."""
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo, hi = int(sample_rate / fmax), int(sample_rate / fmin)
    return sample_rate / (lo + int(np.argmax(corr[lo:hi])))


# --------------------------------------------------------------------------- DTMF

DTMF_LOW = (697.0, 770.0, 852.0, 941.0)
DTMF_HIGH = (1209.0, 1336.0, 1477.0, 1633.0)
DTMF_KEYS = ("123A", "456B", "789C", "*0#D")


def dtmf_tone(key: str, sample_rate: int = 8000, seconds: float = 0.08) -> np.ndarray:
    """One keypress: the sum of a row tone and a column tone."""
    for row, keys in enumerate(DTMF_KEYS):
        if key in keys:
            col = keys.index(key)
            t = np.arange(int(sample_rate * seconds)) / sample_rate
            return 0.4 * (np.sin(2 * np.pi * DTMF_LOW[row] * t) + np.sin(2 * np.pi * DTMF_HIGH[col] * t))
    raise ValueError(f"not a DTMF key: {key!r}")


def goertzel_power(x: np.ndarray, frequency: float, sample_rate: int) -> float:
    """Energy at one frequency with a 2nd-order recursion: one DFT bin for O(N) work."""
    coeff = 2 * np.cos(2 * np.pi * frequency / sample_rate)
    s_prev = s_prev2 = 0.0
    for sample in np.asarray(x, dtype=np.float64):
        s = sample + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return float(s_prev ** 2 + s_prev2 ** 2 - coeff * s_prev * s_prev2)


def detect_dtmf(x: np.ndarray, sample_rate: int = 8000, min_ratio: float = 10.0) -> str | None:
    """Return the key whose row and column tones both dominate their groups, else ``None``."""
    low = np.array([goertzel_power(x, f, sample_rate) for f in DTMF_LOW])
    high = np.array([goertzel_power(x, f, sample_rate) for f in DTMF_HIGH])
    row, col = int(np.argmax(low)), int(np.argmax(high))
    low_rest = np.delete(low, row).max()
    high_rest = np.delete(high, col).max()
    if low[row] < min_ratio * max(low_rest, 1e-12) or high[col] < min_ratio * max(high_rest, 1e-12):
        return None
    return DTMF_KEYS[row][col]


# --------------------------------------------------------------------------- demo

def main() -> None:
    invite = parse_sip_message(SAMPLE_INVITE)
    offer = parse_sdp(invite["body"])
    pt, codec, rate = negotiate_codec(offer)
    print(f"{invite['method']} from {invite['headers']['from']}")
    print(f"SDP offer: audio to {offer['connection']}:{offer['port']}, payload types {offer['payload_types']}")
    print(f"Negotiated codec: PT {pt} = {codec}/{rate}, ptime {offer['ptime_ms']} ms")

    one_second = mulaw_encode(float_to_pcm16(0.3 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)))
    packets = packetize(one_second.tobytes(), payload_type=0)
    header = unpack_rtp_header(packets[1])
    print(f"1 s of G.711 -> {len(packets)} RTP packets of {len(packets[1])} bytes; "
          f"packet 1 has seq={header['seq']} timestamp={header['timestamp']}")
    print(f"Wire bitrate per direction (IPv4/UDP/RTP): {rtp_bitrate_kbps():.1f} kbps")

    tone = 0.01 * np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000)  # a quiet -40 dBFS tone
    print(f"Quiet tone SNR: mu-law {snr_db(tone, mulaw_roundtrip(tone)):.1f} dB, "
          f"8-bit linear {snr_db(tone, quantize_linear(tone, 8)):.1f} dB")

    utterance = synthetic_utterance()
    phone = phone_channel(utterance)
    print(f"Energy above 3.4 kHz in the wideband utterance: "
          f"{100 * band_energy_fraction(utterance, 16_000, 3400, 8000):.0f}% (gone on a phone line)")
    vowel_phone = phone[: int(0.4 * 8000)]
    print(f"Pitch of the 120 Hz vowel after the 300 Hz high-pass: "
          f"{autocorrelation_pitch(vowel_phone, 8000):.0f} Hz (the missing fundamental)")

    keys = "".join(detect_dtmf(dtmf_tone(k)) or "?" for k in "0909")
    print(f"Goertzel decodes DTMF keypresses: {keys}")


if __name__ == "__main__":
    main()
