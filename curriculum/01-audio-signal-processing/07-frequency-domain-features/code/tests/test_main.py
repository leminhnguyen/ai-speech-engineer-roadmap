import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

SR, N_FFT, HOP = 16_000, 512, 160


def interior(samples):
    """Magnitude spectrogram without the zero-padded edge frames."""
    return main.magnitude_spectrogram(samples, N_FFT, HOP)[:, 5:-5]


class FrequencyDomainFeatureTests(unittest.TestCase):
    def test_on_bin_tone_has_exact_centroid_and_hann_bandwidth(self):
        magnitude = interior(main.sine_wave(1_000.0, SR, 0.5))
        np.testing.assert_allclose(main.spectral_centroid(magnitude, SR, N_FFT), 1_000.0, atol=1e-6)
        # A periodic Hann window spreads an on-bin tone over three bins weighted 1/4, 1/2, 1/4.
        bin_spacing = SR / N_FFT
        np.testing.assert_allclose(main.spectral_bandwidth(magnitude, SR, N_FFT), bin_spacing / np.sqrt(2), rtol=1e-6)

    def test_two_tone_centroid_is_the_magnitude_weighted_mean(self):
        samples = main.sine_wave(1_000.0, SR, 0.5, amplitude=0.6) + main.sine_wave(3_000.0, SR, 0.5, amplitude=0.2)
        centroid = main.spectral_centroid(interior(samples), SR, N_FFT)
        np.testing.assert_allclose(centroid, (0.6 * 1_000.0 + 0.2 * 3_000.0) / 0.8, rtol=1e-6)

    def test_features_ignore_overall_gain(self):
        vowel = main.synthetic_vowel((730.0, 1_090.0, 2_440.0), 120.0, SR, 0.3)
        loud, quiet = interior(vowel), interior(0.05 * vowel)
        for feature in (main.spectral_centroid, main.spectral_bandwidth, main.spectral_rolloff):
            np.testing.assert_allclose(feature(loud, SR, N_FFT), feature(quiet, SR, N_FFT), rtol=1e-9)
        # Flatness is gain-invariant only while no bin hits the amin floor. The gaps between a vowel's
        # harmonics do, so check it on white noise, whose bins all stay well above the floor.
        noise = main.highpass_noise(0.0, SR, 0.3)
        np.testing.assert_allclose(main.spectral_flatness(interior(noise)), main.spectral_flatness(interior(0.05 * noise)), rtol=1e-9)

    def test_white_noise_band_energy_ratio_matches_bandwidths(self):
        magnitude = interior(main.highpass_noise(0.0, SR, 2.0))
        ratio = main.band_energy_ratio(magnitude ** 2, 2_000.0, SR, N_FFT)
        # Equal power per hertz: 0-2 kHz against 2-8 kHz is 1:3.
        self.assertAlmostEqual(float(np.median(ratio)), 1.0 / 3.0, delta=0.04)

    def test_white_noise_flatness_is_exp_minus_euler_gamma(self):
        magnitude = interior(main.highpass_noise(0.0, SR, 2.0))
        # Periodogram bins of white noise are exponentially distributed, so GM / AM -> exp(-0.5772).
        self.assertAlmostEqual(float(np.mean(main.spectral_flatness(magnitude))), np.exp(-np.euler_gamma), delta=0.02)

    def test_tone_is_far_less_flat_than_noise(self):
        tone = np.median(main.spectral_flatness(interior(main.sine_wave(1_000.0, SR, 0.5))))
        noise = np.median(main.spectral_flatness(interior(main.highpass_noise(0.0, SR, 0.5))))
        self.assertLess(tone, 1e-3)
        self.assertGreater(noise, 0.4)

    def test_rolloff_bounds(self):
        flat = np.ones((N_FFT // 2 + 1, 1))
        np.testing.assert_allclose(main.spectral_rolloff(flat, SR, N_FFT, 0.5), 4_000.0)
        magnitude = interior(main.sine_wave(1_000.0, SR, 0.5))
        rolloff = main.spectral_rolloff(magnitude, SR, N_FFT, 0.85)
        self.assertTrue(np.all((rolloff >= 1_000.0) & (rolloff <= 1_000.0 + SR / N_FFT)))

    def test_rms_from_magnitude_matches_windowed_frame_rms(self):
        samples = main.synthetic_vowel((270.0, 2_290.0, 3_010.0), 150.0, SR, 0.2)
        frame = samples[800:800 + N_FFT] * main.hann_window(N_FFT)
        magnitude = np.abs(np.fft.rfft(frame))[:, None]
        np.testing.assert_allclose(main.rms_from_magnitude(magnitude, N_FFT), np.sqrt(np.mean(frame ** 2)), rtol=1e-12)

    def test_vowel_and_hiss_land_on_opposite_sides(self):
        vowel = interior(main.synthetic_vowel((730.0, 1_090.0, 2_440.0), 120.0, SR, 0.3))
        hiss = interior(main.highpass_noise(4_000.0, SR, 0.3))
        self.assertGreater(np.min(main.band_energy_ratio(vowel ** 2, 2_000.0, SR, N_FFT)), 10.0)
        self.assertLess(np.max(main.band_energy_ratio(hiss ** 2, 2_000.0, SR, N_FFT)), 0.1)
        self.assertLess(np.max(main.spectral_centroid(vowel, SR, N_FFT)), 2_000.0)
        self.assertGreater(np.min(main.spectral_centroid(hiss, SR, N_FFT)), 4_000.0)

    def test_frame_labels_recover_the_asa_segments(self):
        utterance, segments = main.asa_utterance(SR)
        magnitude = main.magnitude_spectrogram(utterance, N_FFT, HOP)
        labels = main.label_frames(magnitude, SR, N_FFT)
        times = np.arange(magnitude.shape[1]) * HOP / SR
        for label, start, end in segments:
            middle = np.argmin(np.abs(times - (start + end) / 2))
            self.assertEqual(labels[middle], label)


if __name__ == "__main__":
    unittest.main()
