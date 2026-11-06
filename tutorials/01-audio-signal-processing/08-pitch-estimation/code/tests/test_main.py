import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

SR = 16_000


def steady(f0, seconds=0.4, **kwargs):
    return main.harmonic_signal(main.constant_pitch(f0, SR, seconds), SR, **kwargs)


class PitchEstimationTests(unittest.TestCase):
    def test_autocorrelation_matches_direct_sum(self):
        frame = np.random.default_rng(1).standard_normal((2, 64))
        r = main.autocorrelation(frame, 20)
        direct = np.array([[np.dot(f[: 64 - tau], f[tau:]) for tau in range(21)] for f in frame])
        np.testing.assert_allclose(r, direct, atol=1e-10)

    def test_difference_function_matches_definition(self):
        frames = np.random.default_rng(2).standard_normal((3, 100))
        max_lag, window = 30, 70
        d = main.difference_function(frames, max_lag)
        direct = np.array([[np.sum((f[:window] - f[tau : tau + window]) ** 2) for tau in range(max_lag + 1)] for f in frames])
        np.testing.assert_allclose(d, direct, atol=1e-9)
        np.testing.assert_allclose(d[:, 0], 0.0, atol=1e-9)

    def test_normalized_difference_starts_at_one_and_hits_zero_at_the_period(self):
        tone = np.sin(2 * np.pi * np.arange(400) / 50.0)[None, :]  # period of exactly 50 samples
        curve = main.cumulative_mean_normalized_difference(main.difference_function(tone, 120))[0]
        self.assertEqual(curve[0], 1.0)
        self.assertLess(curve[50], 1e-10)
        self.assertLess(curve[100], 1e-10)
        self.assertGreater(curve[25], 1.0)  # half a period: the shifted copy is inverted

    def test_parabolic_offset_recovers_vertex(self):
        x = np.arange(10.0)
        values = (x - 4.3) ** 2
        self.assertAlmostEqual(main.parabolic_offset(values, 4), 0.3)

    def test_yin_is_accurate_on_steady_pitches(self):
        for f0 in (65.0, 110.0, 147.3, 220.0, 390.0):
            estimate, voiced, _ = main.yin(steady(f0), SR)
            self.assertTrue(voiced[5:-5].all(), f0)
            self.assertLess(np.max(np.abs(main.cents(estimate[5:-5], f0))), 2.0, f0)

    def test_loudest_harmonic_is_not_the_pitch(self):
        vowel = steady(150.0)
        frame = main.frame_signal(vowel, 640, 160)[15]
        self.assertAlmostEqual(main.spectral_peak_pitch(frame, SR), 750.0, delta=5.0)
        self.assertAlmostEqual(main.autocorrelation_pitch(vowel, SR)[15], 150.0, delta=0.5)
        self.assertAlmostEqual(main.yin(vowel, SR)[0][15], 150.0, delta=0.5)

    def test_missing_fundamental_is_still_found(self):
        missing = steady(200.0, harmonics=tuple(range(2, 11)))
        frame = main.frame_signal(missing, 640, 160)[15]
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(640) / 640)
        spectrum = np.abs(np.fft.rfft(frame * hann))
        self.assertLess(spectrum[8], 1e-3 * spectrum.max())  # bin 8 is 200 Hz: only leakage there
        self.assertAlmostEqual(float(np.nanmedian(main.yin(missing, SR)[0])), 200.0, delta=0.5)

    def test_first_dip_avoids_subharmonic_errors(self):
        high = steady(300.0)
        self.assertAlmostEqual(float(np.median(main.deepest_dip_pitch(high, SR)[5:-5])), 100.0, delta=1.0)
        self.assertAlmostEqual(float(np.nanmedian(main.yin(high, SR)[0][5:-5])), 300.0, delta=1.0)

    def test_noise_and_silence_are_unvoiced(self):
        f0, voiced, aperiodicity = main.yin(main.hiss(SR, 0.4), SR)
        self.assertFalse(voiced.any())
        self.assertTrue(np.all(np.isnan(f0)))
        self.assertGreater(np.min(aperiodicity[5:-5]), 0.5)
        self.assertFalse(main.yin(np.zeros(SR // 4), SR)[1].any())

    def test_cents_and_gross_pitch_error(self):
        self.assertAlmostEqual(float(main.cents(200.0, 100.0)), 1_200.0)
        self.assertAlmostEqual(float(main.cents(100.0 * 2 ** (1 / 12), 100.0)), 100.0)
        estimate = np.array([100.0, 210.0, 50.0, np.nan, 130.0])
        reference = np.array([100.0, 200.0, 100.0, 100.0, np.nan])
        self.assertAlmostEqual(main.gross_pitch_error(estimate, reference), 1.0 / 3.0)

    def test_intonation_track_with_short_frames(self):
        utterance, truth = main.intonation_utterance(SR)
        f0, voiced, _ = main.yin(utterance, SR, frame_length=640)
        scores = main.evaluate_track(f0, voiced, truth, 640, 160)
        self.assertGreater(scores["voicing_accuracy"], 0.95)
        self.assertEqual(scores["gross_pitch_error"], 0.0)
        self.assertLess(scores["median_cents"], 15.0)

    def test_long_frames_blur_a_moving_pitch(self):
        utterance, truth = main.intonation_utterance(SR)
        f0, voiced, _ = main.yin(utterance, SR, frame_length=1_024)
        self.assertLess(main.evaluate_track(f0, voiced, truth, 1_024, 160)["voicing_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
