import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

AGENT = 6.0


def event(kind, start, duration, words=(), voiced=True):
    return main.OverlapEvent(kind, start, duration, tuple(words), voiced)


class ControllerTests(unittest.TestCase):
    def test_long_interruption_pauses_after_min_duration(self):
        out = main.simulate_event(event("interruption", 2.0, 1.5, "khoan đã tôi muốn".split()), AGENT,
                                  main.BargeInConfig())
        self.assertEqual((out.action, out.trigger), ("stopped", "vad"))
        self.assertAlmostEqual(out.silenced_at, 2.5)

    def test_short_word_still_stops_via_final_transcript(self):
        out = main.simulate_event(event("interruption", 2.0, 0.3, ["không"]), AGENT, main.BargeInConfig())
        self.assertEqual((out.action, out.trigger), ("stopped", "transcript"))
        self.assertAlmostEqual(out.silenced_at, 2.6)

    def test_cough_is_a_false_interruption_that_resumes(self):
        out = main.simulate_event(event("cough", 2.0, 0.6), AGENT, main.BargeInConfig())
        self.assertEqual(out.action, "resumed")
        self.assertAlmostEqual(out.resumed_at, 2.6 + 2.0)  # timeout counts from the end of the cough

    def test_cough_without_resume_silences_the_agent(self):
        cfg = main.BargeInConfig(resume_false_interruption=False)
        self.assertEqual(main.simulate_event(event("cough", 2.0, 0.6), AGENT, cfg).action, "stopped")

    def test_unvoiced_noise_is_ignored(self):
        out = main.simulate_event(event("noise", 2.0, 1.0, voiced=False), AGENT, main.BargeInConfig())
        self.assertEqual(out.action, "ignored")

    def test_vad_mode_stops_on_backchannel_but_adaptive_does_not(self):
        dạ = event("backchannel", 2.0, 0.3, ["dạ"])
        self.assertEqual(main.simulate_event(dạ, AGENT, main.BargeInConfig()).action, "stopped")
        self.assertEqual(main.simulate_event(dạ, AGENT, main.BargeInConfig(mode="adaptive")).action, "ignored")

    def test_min_words_blocks_one_word_interruptions(self):
        cfg = main.BargeInConfig(min_words=2)
        self.assertEqual(main.simulate_event(event("interruption", 2.0, 0.3, ["không"]), AGENT, cfg).action,
                         "ignored")

    def test_guard_window_ignores_early_echo(self):
        echo = event("echo", 0.4, 1.5, main.AGENT_REPLY[:4])
        self.assertEqual(main.simulate_event(echo, AGENT, main.BargeInConfig()).action, "stopped")
        self.assertEqual(main.simulate_event(echo, AGENT, main.BargeInConfig(guard_s=3.0)).action, "ignored")

    def test_event_after_agent_finished(self):
        self.assertEqual(main.simulate_event(event("interruption", 6.5, 1.0, ["x"]), AGENT,
                                             main.BargeInConfig()).action, "after_end")


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.events = main.make_events(300, seed=0)

    def test_dataset_is_reproducible(self):
        self.assertEqual(self.events, main.make_events(300, seed=0))

    def test_adaptive_cuts_false_stops_without_missing_interruptions(self):
        vad = main.evaluate(self.events, main.BargeInConfig())
        adaptive = main.evaluate(self.events, main.BargeInConfig(mode="adaptive"))
        self.assertLess(adaptive["false_stop_rate"], vad["false_stop_rate"] / 2)
        self.assertLessEqual(adaptive["missed_rate"], vad["missed_rate"] + 0.01)

    def test_higher_min_duration_raises_stop_latency(self):
        sweep = main.sweep_min_duration(self.events, [0.2, 0.5, 1.0])
        self.assertTrue(np.all(np.diff(sweep["median_stop_latency"]) > 0))
        self.assertTrue(np.all(np.diff(sweep["false_pause_rate"]) <= 0))


class TruncationTests(unittest.TestCase):
    def test_timings_cover_the_reply(self):
        timings = main.word_timings(main.AGENT_REPLY, 6.0)
        self.assertAlmostEqual(timings[0, 0], 0.0)
        self.assertAlmostEqual(timings[-1, 1], 6.0)
        np.testing.assert_allclose(timings[1:, 0], timings[:-1, 1])

    def test_truncation_splits_at_half_played_word(self):
        words = ["a", "b", "c"]
        timings = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        self.assertEqual(main.truncate_to_heard(words, timings, 1.6), ("a b", "c"))
        self.assertEqual(main.truncate_to_heard(words, timings, 1.4), ("a", "b c"))
        self.assertEqual(main.truncate_to_heard(words, timings, 10.0), ("a b c", ""))


if __name__ == "__main__":
    unittest.main()
