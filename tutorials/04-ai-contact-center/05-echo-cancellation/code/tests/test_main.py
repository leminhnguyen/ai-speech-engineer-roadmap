import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class EchoPathTests(unittest.TestCase):
    def test_echo_return_loss_matches_request(self):
        h = main.echo_path(erl_db=12.0)
        self.assertAlmostEqual(10 * np.log10(np.sum(h ** 2)), -12.0, places=6)

    def test_bulk_delay_is_silent(self):
        h = main.echo_path(delay=24)
        self.assertTrue(np.all(h[:24] == 0))
        self.assertNotEqual(h[24], 0)

    def test_speech_like_is_deterministic_and_bounded(self):
        a = main.speech_like(1.0, seed=3)
        np.testing.assert_array_equal(a, main.speech_like(1.0, seed=3))
        self.assertAlmostEqual(float(np.max(np.abs(a))), 0.3, places=6)


class NlmsTests(unittest.TestCase):
    def test_learns_the_path_without_noise(self):
        s = main.scenario(seconds=2.0, noise_db=-200)
        residual, w = main.nlms_echo_canceller(s["far"], s["mic"], mu=1.0, eps=1e-6)
        self.assertLess(main.misalignment_db(s["path"], w), -30)
        self.assertGreater(np.nanmedian(main.erle_db(s["echo"], residual)[-5:]), 30)

    def test_erle_grows_as_filter_converges(self):
        s = main.scenario(seconds=3.0)
        residual, _ = main.nlms_echo_canceller(s["far"], s["mic"])
        erle = main.erle_db(s["echo"], residual - s["noise"], frame=800)
        early, late = np.nanmean(erle[:3]), np.nanmean(erle[-6:])
        self.assertGreater(late, early + 10)
        self.assertGreater(late, 20)

    def test_canceller_removes_false_barge_in(self):
        s = main.scenario(seconds=3.0)
        residual, _ = main.nlms_echo_canceller(s["far"], s["mic"])
        self.assertGreater(main.false_barge_in_frames(s["mic"]), 20)
        self.assertLessEqual(main.false_barge_in_frames(residual[8000:]), 2)

    def test_path_change_drops_then_recovers(self):
        s = main.scenario(seconds=6.0, path_change_at=3.0)
        residual, _ = main.nlms_echo_canceller(s["far"], s["mic"])
        erle = main.erle_db(s["echo"], residual - s["noise"], frame=800)  # 0.1 s frames
        before = np.nanmean(erle[24:30])
        right_after = np.nanmean(erle[30:33])
        recovered = np.nanmean(erle[-6:])
        self.assertLess(right_after, before - 15)
        self.assertGreater(recovered, 20)


class DoubleTalkTests(unittest.TestCase):
    def setUp(self):
        self.s = main.scenario(seconds=4.0, double_talk=(2.0, 3.0))
        self.flag = main.geigel_double_talk(self.s["far"], self.s["mic"])

    def test_geigel_covers_near_end_energy_without_false_alarms(self):
        near = self.s["near"]
        covered = np.sum(near ** 2 * self.flag) / np.sum(near ** 2)
        outside = np.concatenate([self.flag[4000:16000], self.flag[26000:]]).mean()
        self.assertGreater(covered, 0.75)
        self.assertLess(outside, 0.01)

    def test_freezing_protects_the_filter(self):
        free, w_free = main.nlms_echo_canceller(self.s["far"], self.s["mic"])
        held, w_held = main.nlms_echo_canceller(self.s["far"], self.s["mic"], freeze=self.flag)
        self.assertLess(main.misalignment_db(self.s["path"], w_held),
                        main.misalignment_db(self.s["path"], w_free) - 6)

    def test_suppressor_trades_residual_for_near_end_loss(self):
        residual, _ = main.nlms_echo_canceller(self.s["far"], self.s["mic"], freeze=self.flag)
        never = np.zeros_like(self.flag)
        nlp_ok = main.residual_echo_suppressor(residual, self.s["far"], self.flag)
        nlp_blind = main.residual_echo_suppressor(residual, self.s["far"], never)
        echo_only = slice(8000, 16000)
        talk = slice(16000, 24000)
        self.assertLess(np.sum(nlp_ok[echo_only] ** 2), 0.25 * np.sum(residual[echo_only] ** 2))
        kept_ok = np.sum(nlp_ok[talk] ** 2) / np.sum(residual[talk] ** 2)
        kept_blind = np.sum(nlp_blind[talk] ** 2) / np.sum(residual[talk] ** 2)
        self.assertGreater(kept_ok, 0.9)
        self.assertLess(kept_blind, 0.4)


if __name__ == "__main__":
    unittest.main()
