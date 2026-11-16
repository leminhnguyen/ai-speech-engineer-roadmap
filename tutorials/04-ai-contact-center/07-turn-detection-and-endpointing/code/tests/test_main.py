import math
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class SimulationTests(unittest.TestCase):
    def test_turns_end_with_an_infinite_final_pause(self):
        call = main.simulate_call(n_turns=20, seed=1)
        for turn in call:
            self.assertTrue(turn[-1].final)
            self.assertEqual(turn[-1].length_s, math.inf)
            self.assertTrue(all(not p.final and p.length_s > 0 for p in turn[:-1]))

    def test_transcript_grows_chunk_by_chunk(self):
        call = main.simulate_call(n_turns=30, seed=2)
        multi = next(t for t in call if len(t) > 1)
        for earlier, later in zip(multi, multi[1:]):
            self.assertTrue(later.transcript.startswith(earlier.transcript))

    def test_number_pauses_are_longer_than_hesitations(self):
        calls = main.simulate_calls(n_calls=100)
        self.assertGreater(np.median(main.within_turn_pauses(calls, "number")),
                           np.median(main.within_turn_pauses(calls, "hesitation")))

    def test_deterministic(self):
        a = main.simulate_call(seed=5)
        b = main.simulate_call(seed=5)
        self.assertEqual([[p.length_s for p in t] for t in a], [[p.length_s for p in t] for t in b])


class EouTests(unittest.TestCase):
    def test_open_phone_number_is_incomplete(self):
        self.assertLess(main.eou_probability("số điện thoại của em là 0903 123"), 0.2)
        self.assertGreater(main.eou_probability("số điện thoại của em là 0903 123 456"), 0.5)

    def test_connector_and_particle(self):
        self.assertLess(main.eou_probability("em muốn hỏi về"), 0.2)
        self.assertGreater(main.eou_probability("hóa đơn tháng này ạ"), 0.85)

    def test_features(self):
        f = main.eou_features("mã khách hàng của em là 8 4 7")
        self.assertTrue(f["open_number"])
        self.assertFalse(f["final_particle"])


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = main.simulate_calls(n_calls=150)

    def test_longer_fixed_delay_cuts_less_and_waits_more(self):
        short = main.run_policy(main.FixedEndpointing(0.5), self.calls)
        long = main.run_policy(main.FixedEndpointing(1.5), self.calls)
        self.assertGreater(short["cutoff_rate"], long["cutoff_rate"] + 0.2)
        self.assertAlmostEqual(short["mean_wait_s"], 0.5)
        self.assertAlmostEqual(long["mean_wait_s"], 1.5)

    def test_cutoff_rate_matches_pause_distribution_for_single_pause_turns(self):
        call = [[main.Pause("a", 0.7, False), main.Pause("a b", math.inf, True)],
                [main.Pause("c", 0.3, False), main.Pause("c d", math.inf, True)]]
        result = main.run_policy(main.FixedEndpointing(0.5), [call])
        self.assertEqual(result["cutoff_rate"], 0.5)

    def test_dynamic_stays_within_bounds_and_adapts(self):
        policy = main.DynamicEndpointing(0.5, 3.0, alpha=0.5, margin=2.0)
        slow = main.Pause("x", 1.2, False)
        for _ in range(10):
            policy.observe(slow)
        self.assertAlmostEqual(policy.delay_for(slow), 2.4, places=2)
        for _ in range(20):
            policy.observe(main.Pause("x", 0.01, False))
        self.assertEqual(policy.delay_for(slow), 0.5)
        policy.reset()
        self.assertAlmostEqual(policy.average_pause, 0.4)

    def test_semantic_beats_fixed_on_both_axes(self):
        fixed = main.run_policy(main.FixedEndpointing(0.8), self.calls)
        semantic = main.run_policy(main.SemanticEndpointing(0.5, 3.0), self.calls)
        self.assertLess(semantic["cutoff_rate"], fixed["cutoff_rate"])
        self.assertLess(semantic["mean_wait_s"], fixed["mean_wait_s"])

    def test_dynamic_beats_fixed_at_equal_wait(self):
        dynamic = main.run_policy(main.DynamicEndpointing(0.5, 3.0), self.calls)
        fixed = main.run_policy(main.FixedEndpointing(dynamic["mean_wait_s"]), self.calls)
        self.assertLess(dynamic["cutoff_rate"], fixed["cutoff_rate"])


if __name__ == "__main__":
    unittest.main()
