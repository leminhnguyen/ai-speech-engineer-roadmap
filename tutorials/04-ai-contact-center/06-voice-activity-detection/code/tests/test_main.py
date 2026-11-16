import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class FeatureTests(unittest.TestCase):
    def test_flatness_separates_noise_from_harmonics(self):
        rng = np.random.default_rng(0)
        white = main.frame_signal(rng.standard_normal(main.FRAME * 20))
        voiced = main.frame_signal(main.speech_like(1.0, seed=2))
        self.assertGreater(np.median(main.spectral_flatness(white)), 0.4)
        self.assertLess(np.median(main.spectral_flatness(voiced[np.std(voiced, axis=1) > 0.05])), 0.2)

    def test_zero_crossing_rate_of_a_tone(self):
        t = np.arange(main.FRAME) / main.SAMPLE_RATE
        frames = np.sin(2 * np.pi * 1000 * t + 0.1)[None, :]
        self.assertAlmostEqual(float(main.zero_crossing_rate(frames)[0]), 2000 / main.SAMPLE_RATE, delta=0.01)

    def test_noise_floor_follows_minimum_and_rises_slowly(self):
        energy = np.array([-50.0] * 10 + [-20.0] * 40 + [-50.0] * 5)
        floor = main.track_noise_floor(energy, rise_db=0.1)
        self.assertAlmostEqual(float(floor[49]), -46.0, places=6)  # 40 frames x 0.1 dB
        self.assertEqual(float(floor[-1]), -50.0)

    def test_mix_at_snr(self):
        dialog = main.synthetic_dialog(seconds=8.0)
        n = main.noise("white", len(dialog["signal"]))
        noisy = main.mix_at_snr(dialog["signal"], n, 10.0, dialog["speech"])
        added = noisy - dialog["signal"]
        snr = 10 * np.log10(np.mean(dialog["signal"][dialog["speech"]] ** 2) / np.mean(added ** 2))
        self.assertAlmostEqual(snr, 10.0, places=6)


class StateMachineTests(unittest.TestCase):
    def test_defaults_match_silero(self):
        p = main.VADParams()
        self.assertEqual((p.min_speech_duration, p.min_silence_duration, p.prefix_padding_duration), (0.05, 0.55, 0.5))
        self.assertAlmostEqual(p.deactivation(), 0.35)
        self.assertAlmostEqual(main.VADParams(activation_threshold=0.1).deactivation(), 0.01)
        self.assertAlmostEqual(p.frame_duration, 0.032)

    def test_hysteresis_keeps_speech_between_thresholds(self):
        probs = np.array([0.0] * 5 + [0.9] * 10 + [0.4] * 30 + [0.0] * 30)
        result = main.run_vad(probs)
        self.assertEqual([kind for _, kind in result.events], ["START_OF_SPEECH", "END_OF_SPEECH"])
        start, end = result.segments[0]
        self.assertAlmostEqual(end, 45 * 0.032)  # 0.4 stays above the 0.35 deactivation threshold

    def test_short_blip_is_ignored(self):
        probs = np.array([0.0] * 5 + [0.9] + [0.0] * 20)
        self.assertEqual(main.run_vad(probs, main.VADParams(min_speech_duration=0.1)).events, [])

    def test_short_pause_is_bridged_long_pause_splits(self):
        speech, short, long = [0.9] * 20, [0.0] * 10, [0.0] * 25  # 0.32 s and 0.8 s pauses
        probs = np.array([0.0] * 5 + speech + short + speech + long + speech + [0.0] * 30)
        self.assertEqual(len(main.run_vad(probs).segments), 2)
        self.assertEqual(len(main.run_vad(probs, main.VADParams(min_silence_duration=0.2)).segments), 3)

    def test_event_times_include_the_timers(self):
        probs = np.array([0.0] * 10 + [0.9] * 40 + [0.0] * 40)
        result = main.run_vad(probs)
        (t_start, _), (t_end, _) = result.events
        self.assertAlmostEqual(t_start, 12 * 0.032)  # 2 frames to confirm 50 ms of speech
        self.assertAlmostEqual(t_end, (50 + 18) * 0.032)  # 18 frames to confirm 550 ms of silence
        self.assertAlmostEqual(result.segments[0][0], 0.0)  # 0.32 s onset - 0.5 s padding, clipped at 0


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialog = main.synthetic_dialog()

    def noisy(self, kind, snr):
        return main.mix_at_snr(self.dialog["signal"], main.noise(kind, len(self.dialog["signal"])), snr,
                               self.dialog["speech"])

    def test_clean_conditions_work(self):
        m = main.evaluate(self.noisy("white", 20), self.dialog)
        self.assertGreater(m["f1"], 0.95)
        self.assertLess(m["start_delay_s"], 0.15)
        self.assertAlmostEqual(m["end_delay_s"], 0.55, delta=0.1)

    def test_lower_snr_degrades(self):
        self.assertLess(main.evaluate(self.noisy("white", 0), self.dialog)["f1"],
                        main.evaluate(self.noisy("white", 20), self.dialog)["f1"] - 0.3)

    def test_babble_causes_more_false_alarms_than_hiss(self):
        self.assertGreater(main.evaluate(self.noisy("babble", 10), self.dialog)["false_alarm"],
                           main.evaluate(self.noisy("white", 10), self.dialog)["false_alarm"] + 0.02)

    def test_prefix_padding_prevents_onset_clipping(self):
        x = self.noisy("white", 20)
        slow = dict(min_speech_duration=0.15)
        clipped = main.evaluate(x, self.dialog, main.VADParams(prefix_padding_duration=0.0, **slow))
        padded = main.evaluate(x, self.dialog, main.VADParams(prefix_padding_duration=0.5, **slow))
        self.assertGreater(clipped["onset_clipping_ms"], 40)
        self.assertEqual(padded["onset_clipping_ms"], 0.0)

    def test_min_silence_trades_fragmentation_for_end_delay(self):
        x = self.noisy("white", 20)
        short = main.evaluate(x, self.dialog, main.VADParams(min_silence_duration=0.1))
        default = main.evaluate(x, self.dialog)
        self.assertGreater(short["segments"], default["segments"] + 5)
        self.assertLess(short["end_delay_s"], default["end_delay_s"] - 0.3)


if __name__ == "__main__":
    unittest.main()
