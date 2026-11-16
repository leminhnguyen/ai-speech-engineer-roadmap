import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

try:
    import librosa
except ImportError:  # optional comparison
    librosa = None


class ResamplerTests(unittest.TestCase):
    def test_rational_factors(self):
        self.assertEqual(main.design_resampler(8_000, 16_000)[:2], (2, 1))
        self.assertEqual(main.design_resampler(48_000, 16_000)[:2], (1, 3))
        self.assertEqual(main.design_resampler(44_100, 16_000)[:2], (160, 441))

    def test_output_length(self):
        x = np.zeros(8_000)
        self.assertEqual(len(main.resample(x, 8_000, 48_000)), 48_000)
        self.assertEqual(len(main.resample(x, 8_000, 16_000)), 16_000)

    def test_in_band_tone_keeps_frequency_and_level(self):
        for src, dst in ((8_000, 16_000), (16_000, 48_000), (48_000, 16_000), (24_000, 48_000)):
            y = main.resample(main.tone(1_000, src), src, dst)
            self.assertAlmostEqual(main.tone_level_db(y, dst, 1_000), main.tone_level_db(main.tone(1_000, dst), dst, 1_000),
                                   delta=0.1)

    def test_downsampling_rejects_aliases(self):
        x = main.tone(6_000, 48_000)  # above the 4 kHz Nyquist of the target
        filtered = main.resample(x, 48_000, 8_000)
        naive = main.naive_resample(x, 48_000, 8_000)
        self.assertLess(main.tone_level_db(filtered, 8_000, 2_000), -80)
        self.assertGreater(main.tone_level_db(naive, 8_000, 2_000), -10)

    def test_upsampling_creates_no_images(self):
        y = main.resample(main.tone(3_000, 8_000), 8_000, 16_000)
        self.assertLess(main.tone_level_db(y, 16_000, 5_000), -80)  # image of 3 kHz around 4 kHz

    def test_round_trip(self):
        x = main.tone(700, 16_000) + main.tone(2_300, 16_000, amplitude=0.2)
        y = main.resample(main.resample(x, 16_000, 48_000), 48_000, 16_000)
        middle = slice(400, len(x) - 400)
        self.assertLess(float(np.max(np.abs(x[middle] - y[middle]))), 1e-3)

    @unittest.skipIf(librosa is None, "librosa is not installed")
    def test_matches_librosa_on_band_limited_signal(self):
        x = main.tone(440, 16_000) + main.tone(3_000, 16_000, amplitude=0.3)
        ours = main.resample(x, 16_000, 48_000)
        reference = librosa.resample(x, orig_sr=16_000, target_sr=48_000)
        middle = slice(600, len(ours) - 600)
        self.assertLess(float(np.max(np.abs(ours[middle] - reference[middle]))), 1e-3)


class FrameStreamTests(unittest.TestCase):
    def test_frame_sizes(self):
        self.assertEqual(main.frame_samples(8_000, 20), 160)
        self.assertEqual(main.frame_samples(48_000, 10), 480)
        with self.assertRaises(ValueError):
            main.frame_samples(44_100, 15)  # 661.5 samples

    def test_rechunking_preserves_every_sample(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(5_000)
        stream = main.FrameStream(sample_rate=16_000, frame_ms=10)
        frames, start = [], 0
        for size in rng.integers(1, 700, size=40):
            frames += stream.push(signal[start:start + size])
            start += size
            if start >= len(signal):
                break
        frames += stream.push(signal[start:])
        frames += stream.flush()
        self.assertTrue(all(len(f) == 160 for f in frames))
        joined = np.concatenate(frames)
        np.testing.assert_array_equal(joined[: len(signal)], signal)
        self.assertTrue(np.all(joined[len(signal):] == 0))

    def test_flush_on_empty_stream(self):
        self.assertEqual(main.FrameStream(8_000).flush(), [])


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.rules = [
            main.DispatchRule(kind="direct", room_name="vip-line", pin="2468"),
            main.DispatchRule(kind="callee", room_prefix="line-", randomize=False, trunk_ids=("ST_sales",)),
            main.DispatchRule(kind="individual", room_prefix="call-", agent_name="inbound-agent"),
        ]

    def test_individual_rule_gives_each_call_its_own_room(self):
        a = main.route_call(self.rules, "ST_hotline", "+8490111", "+8428300", "c1", seed=1)
        b = main.route_call(self.rules, "ST_hotline", "+8490111", "+8428300", "c2", seed=2)
        self.assertTrue(a["room"].startswith("call-+8490111"))
        self.assertNotEqual(a["room"], b["room"])
        self.assertEqual(a["agent_name"], "inbound-agent")

    def test_pin_selects_direct_room(self):
        routed = main.route_call(self.rules, "ST_hotline", "+8490111", "+8428300", "c3", pin="2468")
        self.assertEqual(routed["room"], "vip-line")

    def test_trunk_specific_callee_rule(self):
        routed = main.route_call(self.rules, "ST_sales", "+8490111", "+8428399", "c4")
        self.assertEqual(routed["room"], "line-+8428399")

    def test_hidden_phone_number(self):
        rules = [main.DispatchRule(kind="individual", hide_phone_number=True)]
        attributes = main.route_call(rules, "ST_x", "+8490111", "+8428300", "c5")["attributes"]
        self.assertNotIn("sip.phoneNumber", attributes)
        self.assertEqual(attributes["sip.callID"], "c5")

    def test_no_match_rejects(self):
        rules = [main.DispatchRule(kind="direct", room_name="r", trunk_ids=("ST_other",))]
        with self.assertRaises(LookupError):
            main.route_call(rules, "ST_x", "+1", "+2", "c6")


if __name__ == "__main__":
    unittest.main()
