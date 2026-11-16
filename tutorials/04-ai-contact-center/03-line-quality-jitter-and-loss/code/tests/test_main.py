import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class LossModelTests(unittest.TestCase):
    def test_gilbert_matches_theory(self):
        lost = main.gilbert_elliott(200_000, p=0.02, r=0.25, seed=3)
        self.assertAlmostEqual(lost.mean(), main.gilbert_loss_rate(0.02, 0.25), delta=0.005)
        self.assertAlmostEqual(main.burst_lengths(lost).mean(), 1 / 0.25, delta=0.2)

    def test_same_average_loss_different_burstiness(self):
        random_loss = main.gilbert_elliott(100_000, p=0.03, r=0.97, seed=4)  # memoryless at 3%
        bursty = main.gilbert_elliott(100_000, p=0.0062, r=0.2, seed=4)     # 3% in bursts of ~5
        self.assertAlmostEqual(random_loss.mean(), bursty.mean(), delta=0.006)
        self.assertGreater(main.burst_lengths(bursty).mean(), 3 * main.burst_lengths(random_loss).mean())

    def test_burst_lengths(self):
        np.testing.assert_array_equal(main.burst_lengths([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]), [2, 1, 3])


class JitterBufferTests(unittest.TestCase):
    def setUp(self):
        self.lost = main.gilbert_elliott(3000, 0.01, 0.4, seed=1)
        self.delays = main.network_delays(3000, seed=2, step_at=1800, step_ms=60.0)

    def test_reordering_detected(self):
        send = np.arange(5) * 20.0
        delays = np.array([50, 80, 30, 40, 40.0])  # packet 1 overtaken by 2 and 3
        order = main.arrival_order(send, delays)
        np.testing.assert_array_equal(order, [0, 2, 1, 3, 4])
        self.assertEqual(main.reordered_count(order), 1)

    def test_bigger_fixed_buffer_fewer_late_more_delay(self):
        small = main.fixed_jitter_buffer(self.delays, 40, self.lost)
        large = main.fixed_jitter_buffer(self.delays, 120, self.lost)
        self.assertGreater(small["late_rate"], large["late_rate"])
        self.assertAlmostEqual(large["mouth_to_ear_ms"] - small["mouth_to_ear_ms"], 80.0)

    def test_lost_packets_are_never_late(self):
        jb = main.fixed_jitter_buffer(self.delays, 0, self.lost)
        self.assertFalse(np.any(jb["late"] & jb["lost"]))

    def test_adaptive_beats_fixed_at_equal_delay(self):
        adaptive = main.adaptive_jitter_buffer(self.delays, self.lost)
        fixed = main.fixed_jitter_buffer(self.delays, adaptive["mouth_to_ear_ms"] - self.delays[0], self.lost)
        self.assertAlmostEqual(fixed["mouth_to_ear_ms"], adaptive["mouth_to_ear_ms"])
        self.assertLess(adaptive["late_rate"], fixed["late_rate"])

    def test_adaptive_follows_route_change(self):
        offsets = main.adaptive_jitter_buffer(self.delays, self.lost)["offsets"]
        self.assertGreater(offsets[2500] - offsets[1500], 40.0)


class ConcealmentTests(unittest.TestCase):
    def setUp(self):
        self.x = main.voiced_signal()
        self.missing = np.zeros(50, dtype=bool)
        self.missing[[10, 25, 26]] = True
        self.received = self.x * ~main.frames_to_samples(self.missing, 160)

    def test_good_frames_untouched_except_crossfade(self):
        y = main.conceal(self.received, self.missing, 160, "pitch")
        np.testing.assert_array_equal(y[:1600], self.x[:1600])
        np.testing.assert_array_equal(y[1800:4000], self.x[1800:4000])  # after the 5 ms crossfade

    def test_pitch_period_estimate(self):
        t = np.arange(480) / 8000
        self.assertEqual(main.estimate_period(np.sin(2 * np.pi * 125 * t), 8000), 64)

    def test_ranking_by_spectral_distance(self):
        scores = {m: main.log_spectral_distance(self.x, main.conceal(self.received, self.missing, 160, m), self.missing)
                  for m in ("zero", "repeat", "pitch")}
        self.assertLess(scores["pitch"], scores["repeat"])
        self.assertLess(scores["repeat"], scores["zero"])

    def test_long_burst_fades_to_silence(self):
        missing = np.zeros(50, dtype=bool)
        missing[10:20] = True
        y = main.conceal(self.x * ~main.frames_to_samples(missing, 160), missing, 160, "pitch")
        self.assertEqual(float(np.max(np.abs(y[160 * 14:160 * 20]))), 0.0)  # silent after 60 ms


class LevelTests(unittest.TestCase):
    def test_clipping_detector(self):
        x = main.voiced_signal()
        self.assertEqual(main.clipping_ratio(x), 0.0)
        self.assertGreater(main.clipping_ratio(np.clip(3 * x, -1, 1)), 0.05)

    def test_agc_raises_quiet_talker_but_not_silence(self):
        x = np.concatenate([np.zeros(4000), 0.03 * main.voiced_signal()])
        y = main.agc(x)
        self.assertEqual(float(np.max(np.abs(y[:4000]))), 0.0)
        self.assertAlmostEqual(main.dbfs(y[8000:]), -20.0, delta=1.5)

    def test_agc_respects_max_gain(self):
        quiet = 0.005 * main.voiced_signal()
        gain = main.dbfs(main.agc(quiet)[4000:]) - main.dbfs(quiet[4000:])
        self.assertLessEqual(gain, 24.0 + 0.1)


class EModelTests(unittest.TestCase):
    def test_no_impairments(self):
        self.assertAlmostEqual(main.r_factor(0, 0), 93.2)
        self.assertAlmostEqual(main.mos_from_r(93.2), 4.41, delta=0.01)

    def test_delay_knee_at_177_ms(self):
        slope_before = main.delay_impairment(170) - main.delay_impairment(160)
        slope_after = main.delay_impairment(260) - main.delay_impairment(250)
        self.assertAlmostEqual(slope_before, 0.24)
        self.assertAlmostEqual(slope_after, 1.34)

    def test_bursty_loss_hurts_more_and_plc_helps(self):
        self.assertLess(main.r_factor(100, 0.03, burst_ratio=3), main.r_factor(100, 0.03, burst_ratio=1))
        self.assertGreater(main.r_factor(100, 0.03, bpl=25.1), main.r_factor(100, 0.03, bpl=4.3))

    def test_mos_bounds(self):
        self.assertEqual(main.mos_from_r(-5), 1.0)
        self.assertEqual(main.mos_from_r(120), 4.5)


if __name__ == "__main__":
    unittest.main()
