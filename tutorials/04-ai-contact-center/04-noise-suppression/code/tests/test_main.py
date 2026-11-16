import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class StftTests(unittest.TestCase):
    def test_perfect_reconstruction(self):
        x = np.random.default_rng(0).standard_normal(5_001)
        np.testing.assert_allclose(main.istft(main.stft(x), len(x)), x, atol=1e-12)

    def test_unit_gain_is_identity_and_zero_gain_is_silence(self):
        x = main.synthetic_speech(1.0)
        spec = main.stft(x)
        np.testing.assert_allclose(main.istft(np.ones_like(spec) * spec, len(x)), x, atol=1e-12)
        self.assertEqual(float(np.max(np.abs(main.istft(0 * spec, len(x))))), 0.0)


class SignalTests(unittest.TestCase):
    def test_mix_hits_requested_snr(self):
        clean = main.synthetic_speech(1.0)
        for snr in (-5.0, 0.0, 10.0):
            noisy, _ = main.mix_at_snr(clean, main.make_noise("fan", len(clean)), snr)
            self.assertAlmostEqual(main.snr_db(clean, noisy), snr, places=6)

    def test_pink_noise_slope(self):
        x = main.colored_noise(2 ** 16, 1.0)
        power = np.abs(np.fft.rfft(x)) ** 2
        low, high = power[100:200].mean(), power[1000:2000].mean()
        self.assertAlmostEqual(10 * np.log10(low / high), 10.0, delta=1.5)  # 10 dB per decade

    def test_noises_are_unit_variance(self):
        for kind in ("fan", "street", "keyboard", "babble"):
            self.assertAlmostEqual(float(np.std(main.make_noise(kind, 8000))), 1.0, places=6)


class NoiseEstimationTests(unittest.TestCase):
    def test_minimum_statistics_tracks_stationary_noise_through_speech(self):
        clean = main.synthetic_speech()
        noisy, noise = main.mix_at_snr(clean, main.make_noise("fan", len(clean)), 5.0)
        est = main.minimum_statistics(np.abs(main.stft(noisy)) ** 2)
        truth = np.abs(main.stft(noise)) ** 2
        error_db = 10 * np.log10(est[100:].mean() / truth[100:].mean())
        self.assertLess(abs(error_db), 2.0)


class SuppressionTests(unittest.TestCase):
    def setUp(self):
        self.clean = main.synthetic_speech()
        self.noisy, self.noise = main.mix_at_snr(self.clean, main.make_noise("fan", len(self.clean)), 5.0)
        self.pauses = main.speech_absent_frames(self.clean)

    def test_gains_are_bounded(self):
        power = np.abs(main.stft(self.noisy)) ** 2
        psd = main.minimum_statistics(power)
        for gain in (main.spectral_subtraction_gain(power, psd, floor=0.05), main.wiener_dd_gain(power, psd, floor=0.05)):
            self.assertGreaterEqual(float(gain.min()), 0.05 - 1e-12)
            self.assertLessEqual(float(gain.max()), 1.0)

    def test_wiener_improves_snr_on_stationary_noise(self):
        out = main.enhance(self.noisy)
        self.assertGreater(main.snr_db(self.clean, out) - main.snr_db(self.clean, self.noisy), 5.0)

    def test_tradeoff_more_reduction_costs_more_distortion(self):
        power = np.abs(main.stft(self.noisy)) ** 2
        psd = main.minimum_statistics(power)
        gentle = main.distortion_and_reduction(self.clean, self.noise, main.wiener_dd_gain(power, psd, floor=0.5))
        aggressive = main.distortion_and_reduction(self.clean, self.noise, main.wiener_dd_gain(power, psd, floor=0.02))
        self.assertGreater(aggressive[1], gentle[1])  # more noise reduction
        self.assertGreater(aggressive[0], gentle[0])  # and more speech distortion

    def test_decision_directed_has_less_musical_noise(self):
        _, g_sub = main.enhance(self.noisy, "subtraction", return_gain=True, over=1.0, floor=0.0)
        _, g_wiener = main.enhance(self.noisy, "wiener", return_gain=True)
        raw = main.isolated_peak_fraction(np.abs(main.stft(self.noise)) ** 2, self.pauses)
        self.assertGreater(main.musical_noise(self.noise, g_sub, self.pauses), 10 * raw)
        self.assertLess(main.musical_noise(self.noise, g_wiener, self.pauses),
                        0.6 * main.musical_noise(self.noise, g_sub, self.pauses))

    def test_babble_is_hard_for_noise_suppression(self):
        gains = {}
        for kind in ("fan", "babble"):
            noisy, _ = main.mix_at_snr(self.clean, main.make_noise(kind, len(self.clean)), 5.0)
            gains[kind] = main.snr_db(self.clean, main.enhance(noisy)) - main.snr_db(self.clean, noisy)
        self.assertGreater(gains["fan"], gains["babble"] + 4.0)

    def test_isolated_peak_detector(self):
        power = np.ones((20, 20))
        power[10, 10] = 100.0
        self.assertAlmostEqual(main.isolated_peak_fraction(power, np.ones(20, dtype=bool)), 1 / 400)


if __name__ == "__main__":
    unittest.main()
