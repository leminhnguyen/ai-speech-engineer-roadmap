import math
import unittest
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class StageTests(unittest.TestCase):
    def test_lognormal_matches_median_and_p95(self):
        stage = main.Stage("x", 400, 900)
        samples = stage.sample(np.random.default_rng(0), 200_000)
        self.assertAlmostEqual(np.median(samples), 400, delta=6)
        self.assertAlmostEqual(np.percentile(samples, 95), 900, delta=20)

    def test_degenerate_stage_is_constant(self):
        samples = main.Stage("x", 50, 50).sample(np.random.default_rng(0), 10)
        np.testing.assert_allclose(samples, 50)

    def test_invalid_stage(self):
        with self.assertRaises(ValueError):
            main.Stage("x", 100, 50).lognormal()


class SimulationTests(unittest.TestCase):
    def test_contributions_add_up_to_total(self):
        result = main.simulate_turns(main.PipelineConfig(), n=2000)
        parts = sum(result[name] for name in main.STAGE_ORDER)
        np.testing.assert_allclose(parts, result["total"])

    def test_constant_pipeline_is_exact(self):
        const = {f: main.Stage(f, v, v) for f, v in
                 [("uplink", 50), ("jitter", 40), ("end_of_turn", 500), ("stt_final", 700),
                  ("llm_ttft", 300), ("tts_ttfb", 200), ("downlink", 60)]}
        cfg = main.PipelineConfig(provider_rtt_ms=20, **const)
        total = main.simulate_turns(cfg, n=5)["total"]
        # STT (700) is slower than the silence wait (500), so it sets the commit time.
        np.testing.assert_allclose(total, 50 + 40 + 700 + 20 + 300 + 20 + 200 + 60)

    def test_preemptive_never_slower_and_wastes_expected_share(self):
        base = main.PipelineConfig()
        pre = replace(base, preemptive=True, p_transcript_change=0.3)
        a, b = main.simulate_turns(base, n=20_000), main.simulate_turns(pre, n=20_000)
        self.assertTrue(np.all(b["total"] <= a["total"] + 1e-9))
        self.assertAlmostEqual(b["wasted_llm_calls"].mean(), 0.3, delta=0.02)

    def test_preemptive_cannot_answer_before_commit(self):
        cfg = replace(main.PipelineConfig(), preemptive=True, p_transcript_change=0.0,
                      llm_ttft=main.Stage("llm", 1, 1), provider_rtt_ms=0.0)
        result = main.simulate_turns(cfg, n=100)
        np.testing.assert_allclose(result["llm ttft"], 0.0)

    def test_each_rung_of_the_ladder_helps(self):
        medians = [main.summarize(main.simulate_turns(cfg, n=5000)["total"])["p50"]
                   for _, cfg in main.optimization_ladder()]
        self.assertTrue(all(later < earlier for earlier, later in zip(medians, medians[1:])))
        self.assertLess(medians[-1], 1000)

    def test_summary_fields(self):
        s = main.summarize(np.array([100.0, 900.0, 1100.0, 2000.0]))
        self.assertEqual(s["under_1s"], 0.5)
        self.assertLessEqual(s["p50"], s["p95"])


class StreamingTests(unittest.TestCase):
    REPLY = "Dạ, đơn hàng sẽ giao vào thứ năm. Phí giao hàng là ba mươi nghìn đồng, anh muốn đổi không?"

    def test_split_keeps_every_word(self):
        chunks = main.split_for_tts(self.REPLY)
        self.assertEqual(" ".join(chunks), self.REPLY)
        self.assertTrue(chunks[0].endswith("."))

    def test_short_leading_clause_is_not_split(self):
        self.assertFalse(main.split_for_tts(self.REPLY)[0] == "Dạ,")

    def test_full_streaming_is_fastest(self):
        times = {(l, t): main.first_audio_ms(self.REPLY, l, t) for l in (False, True) for t in (False, True)}
        self.assertEqual(min(times, key=times.get), (True, True))
        self.assertEqual(max(times, key=times.get), (False, False))
        first = main.split_for_tts(self.REPLY)[0].split()
        expected = 400 + len(first) * main.TOKENS_PER_WORD / 60 * 1000 + 200
        self.assertTrue(math.isclose(times[(True, True)], expected))

    def test_filler_changes_perceived_not_answer_latency(self):
        answer = np.array([500.0, 2600.0])
        np.testing.assert_allclose(main.perceived_latency_ms(answer, 700.0), [500.0, 700.0])
        np.testing.assert_allclose(main.perceived_latency_ms(answer, None), answer)


if __name__ == "__main__":
    unittest.main()
