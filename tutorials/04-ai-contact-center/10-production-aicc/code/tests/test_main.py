import math
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class TransferTests(unittest.TestCase):
    def test_accepted_transfer_and_hold_time(self):
        trace = main.run_transfer([(5, "transfer_requested"), (6, "dial_started"), (14, "human_answered"),
                                   (35, "human_accepted"), (37, "ai_left")])
        self.assertEqual(trace.final, "completed")
        self.assertAlmostEqual(trace.hold_seconds(), 30.0)

    def test_ringing_timeout_offers_callback(self):
        trace = main.run_transfer([(5, "transfer_requested"), (6, "dial_started"), (50, "human_answered")],
                                  ringing_timeout=30.0)
        self.assertEqual(trace.final, "callback_offered")
        self.assertEqual(trace.times[-1], 36.0)

    def test_declined_transfer_returns_to_ai(self):
        trace = main.run_transfer([(5, "transfer_requested"), (6, "dial_started"), (12, "human_answered"),
                                   (25, "human_declined")])
        self.assertEqual(trace.final, "ai_conversation")

    def test_caller_hangs_up_on_hold(self):
        trace = main.run_transfer([(5, "transfer_requested"), (6, "dial_started"), (20, "caller_hung_up")])
        self.assertEqual(trace.final, "abandoned")

    def test_illegal_event_raises(self):
        with self.assertRaises(ValueError):
            main.next_state("ai_conversation", "human_accepted")
        with self.assertRaises(ValueError):
            main.next_state("completed", "transfer_requested")


class DtmfTests(unittest.TestCase):
    def test_complete(self):
        presses = [(1.0 + 0.5 * i, d) for i, d in enumerate("0901234567")]
        self.assertEqual(main.collect_digits(presses, 10), ("0901234567", "complete"))

    def test_terminated_by_pound(self):
        self.assertEqual(main.collect_digits([(1, "1"), (2, "2"), (3, "#")], 6), ("12", "terminated"))

    def test_timeout_between_digits(self):
        self.assertEqual(main.collect_digits([(1, "1"), (2, "2"), (7, "3")], 6), ("12", "timeout"))
        self.assertEqual(main.collect_digits([(5, "1")], 4), ("", "timeout"))  # no first digit within 4 s


class PiiTests(unittest.TestCase):
    def test_masks_phone_and_id_numbers(self):
        masked = main.mask_pii("gọi 0901 234 567 hoặc 0901-234-567, CCCD 079123456789")
        self.assertNotIn("0901", masked)
        self.assertIn("*******567", masked)
        self.assertIn("*********789", masked)

    def test_keeps_short_numbers(self):
        self.assertEqual(main.mask_pii("đơn hàng 4721 giao lúc 9 giờ"), "đơn hàng 4721 giao lúc 9 giờ")


class KpiTests(unittest.TestCase):
    def test_rates_partition_all_calls(self):
        kpis = main.compute_kpis(main.generate_call_log(1500, seed=3))
        total = (kpis["containment_rate"] + kpis["transfer_rate"] + kpis["abandonment_rate"]
                 + kpis["drop_rate"])
        self.assertAlmostEqual(total, 1.0)
        self.assertEqual(sum(kpis["transfer_reasons"].values()), round(kpis["transfer_rate"] * 1500))

    def test_hand_built_log(self):
        log = [main.CallRecord("a", 9, "resolved", 100.0, (800, 900)),
               main.CallRecord("b", 9, "transferred", 200.0, (1000,), "complaint"),
               main.CallRecord("c", 9, "abandoned", 10.0, ())]
        kpis = main.compute_kpis(log)
        self.assertAlmostEqual(kpis["containment_rate"], 1 / 3)
        self.assertAlmostEqual(kpis["aht_s"], 150.0)
        self.assertEqual(kpis["p50_turn_latency_ms"], 900.0)


class ErlangTests(unittest.TestCase):
    def test_erlang_b_known_values(self):
        self.assertAlmostEqual(main.erlang_b(1.0, 1), 0.5)
        self.assertAlmostEqual(main.erlang_b(2.0, 2), 0.4)  # (4/2) / (1 + 2 + 4/2)

    def test_erlang_c_known_value(self):
        # 2 Erlangs on 3 agents: C = 4/9 ~ 0.444
        self.assertAlmostEqual(main.erlang_c(2.0, 3), 4 / 9, places=6)
        self.assertEqual(main.erlang_c(3.0, 3), 1.0)

    def test_service_level_and_asa(self):
        traffic = main.offered_load(100, 180)  # 5 Erlangs
        self.assertAlmostEqual(traffic, 5.0)
        levels = [main.service_level(traffic, m, 180, 20) for m in range(6, 12)]
        self.assertTrue(all(b > a for a, b in zip(levels, levels[1:])))
        self.assertTrue(math.isinf(main.average_speed_of_answer(traffic, 5, 180)))

    def test_staffing_meets_target_with_fewest_agents(self):
        m = main.agents_for_service_level(300, 300, 0.8, 20)
        traffic = main.offered_load(300, 300)
        self.assertGreaterEqual(main.service_level(traffic, m, 300, 20), 0.8)
        self.assertLess(main.service_level(traffic, m - 1, 300, 20), 0.8)

    def test_channels_and_workers(self):
        channels = main.channels_for_blocking(10.0, 0.01)
        self.assertLessEqual(main.erlang_b(10.0, channels), 0.01)
        self.assertGreater(main.erlang_b(10.0, channels - 1), 0.01)
        self.assertEqual(main.workers_needed(67, 25, 0.7), 4)

    def test_hourly_profile_peaks_mid_morning(self):
        profile = main.hourly_profile()
        self.assertEqual(int(np.argmax(profile)), 10)
        self.assertAlmostEqual(profile.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
