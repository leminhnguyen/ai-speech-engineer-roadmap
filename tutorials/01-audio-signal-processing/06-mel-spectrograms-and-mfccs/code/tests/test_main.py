import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class MelAndMfccTests(unittest.TestCase):
    def test_mel_scale_anchor_points_and_round_trip(self):
        self.assertAlmostEqual(float(main.hz_to_mel(0.0)), 0.0)
        self.assertAlmostEqual(float(main.hz_to_mel(1_000.0)), 1_000.0, delta=0.1)
        frequencies = np.array([50.0, 440.0, 3_000.0, 7_900.0])
        np.testing.assert_allclose(main.mel_to_hz(main.hz_to_mel(frequencies)), frequencies)

    def test_mel_frequencies_widen_in_hertz(self):
        edges = main.mel_frequencies(42, 0.0, 8_000.0)
        self.assertAlmostEqual(edges[0], 0.0)
        self.assertAlmostEqual(edges[-1], 8_000.0)
        np.testing.assert_allclose(np.diff(main.hz_to_mel(edges)), np.diff(main.hz_to_mel(edges))[0])
        self.assertTrue(np.all(np.diff(np.diff(edges)) > 0))

    def test_filterbank_shape_and_triangles(self):
        filters = main.mel_filterbank(16_000, 512, 40, normalize=False)
        self.assertEqual(filters.shape, (40, 257))
        self.assertTrue(np.all(filters >= 0.0))
        self.assertTrue(np.all(filters.max(axis=1) <= 1.0 + 1e-12))
        self.assertTrue(np.all(filters.max(axis=1) > 0.5))

    def test_unnormalized_filters_sum_to_one_between_first_and_last_centres(self):
        filters = main.mel_filterbank(16_000, 1_024, 40, normalize=False)
        edges = main.mel_frequencies(42, 0.0, 8_000.0)
        bins_hz = np.arange(513) * 16_000 / 1_024
        inside = (bins_hz >= edges[1]) & (bins_hz <= edges[-2])
        np.testing.assert_allclose(filters.sum(axis=0)[inside], 1.0, atol=1e-12)

    def test_normalized_filters_have_equal_area(self):
        filters = main.mel_filterbank(16_000, 8_192, 20)
        areas = filters.sum(axis=1) * 16_000 / 8_192
        np.testing.assert_allclose(areas, 1.0, rtol=0.02)

    def test_tone_lands_in_the_band_centred_near_its_frequency(self):
        t = np.arange(16_000) / 16_000
        tone = np.sin(2 * np.pi * 1_000.0 * t)
        mel = main.mel_spectrogram(tone, 16_000, n_fft=512, hop_length=160, n_mels=40)
        centres = main.mel_frequencies(42, 0.0, 8_000.0)[1:-1]
        loudest = int(np.argmax(mel[:, 50]))
        self.assertLess(abs(centres[loudest] - 1_000.0), 60.0)

    def test_dct_matrix_is_orthonormal(self):
        basis = main.dct_matrix(40, 40)
        np.testing.assert_allclose(np.einsum("in,jn->ij", basis, basis), np.eye(40), atol=1e-12)
        np.testing.assert_allclose(basis[0], 1.0 / np.sqrt(40))

    def test_flat_log_mel_has_only_c0(self):
        log_mel = np.full((40, 3), -20.0)
        coefficients = main.mfcc_from_log_mel(log_mel, n_mfcc=13)
        np.testing.assert_allclose(coefficients[0], -20.0 * np.sqrt(40))
        np.testing.assert_allclose(coefficients[1:], 0.0, atol=1e-10)

    def test_gain_change_only_moves_c0(self):
        vowel = main.synthetic_vowel((730.0, 1_090.0, 2_440.0), 120.0, 16_000, 0.3)
        loud = main.mfcc(vowel, 16_000)
        quiet = main.mfcc(0.1 * vowel, 16_000)
        np.testing.assert_allclose(quiet[0] - loud[0], -20.0 * np.sqrt(40), atol=1e-8)
        np.testing.assert_allclose(quiet[1:], loud[1:], atol=1e-8)

    def test_mfcc_separates_vowels_more_than_pitches(self):
        def profile(formants, pitch):
            vowel = main.synthetic_vowel(formants, pitch, 16_000, 0.3)
            return main.mfcc(vowel, 16_000)[1:].mean(axis=1)

        a_low = profile((730.0, 1_090.0, 2_440.0), 120.0)
        a_high = profile((730.0, 1_090.0, 2_440.0), 200.0)
        i_low = profile((270.0, 2_290.0, 3_010.0), 120.0)
        self.assertLess(np.linalg.norm(a_low - a_high), 0.5 * np.linalg.norm(a_low - i_low))

    def test_cepstral_mean_normalization_removes_per_coefficient_mean(self):
        features = np.arange(12.0).reshape(3, 4)
        normalized = main.cepstral_mean_normalization(features)
        np.testing.assert_allclose(normalized.mean(axis=1), 0.0)
        np.testing.assert_allclose(normalized[0], [-1.5, -0.5, 0.5, 1.5])


if __name__ == "__main__":
    unittest.main()
