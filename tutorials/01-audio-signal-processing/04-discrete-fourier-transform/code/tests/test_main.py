import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class DiscreteFourierTransformTests(unittest.TestCase):
    def test_dft_matches_numpy_fft(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=64)
        np.testing.assert_allclose(main.dft(samples), np.fft.fft(samples), atol=1e-9)

    def test_idft_inverts_dft(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(size=50)
        np.testing.assert_allclose(main.idft(main.dft(samples)).real, samples, atol=1e-9)

    def test_dc_bin_is_sum_of_samples(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(main.dft(samples)[0].real, 10.0)

    def test_real_signal_spectrum_is_conjugate_symmetric(self):
        samples = main.sine_wave(300.0, 4_000, 0.01)
        spectrum = main.dft(samples)
        n = len(samples)
        np.testing.assert_allclose(spectrum[1:], np.conj(spectrum[1:][::-1]), atol=1e-9)
        self.assertEqual(n, 40)

    def test_bin_frequencies_and_resolution(self):
        np.testing.assert_allclose(main.bin_frequencies(8, 8_000)[:4], [0, 1_000, 2_000, 3_000])
        self.assertEqual(main.frequency_resolution(2_000, 8_000), 4.0)

    def test_magnitude_spectrum_recovers_sine_amplitude(self):
        samples = main.sine_wave(500.0, 8_000, 0.25, amplitude=0.6)
        magnitude = main.magnitude_spectrum(samples)
        k = int(round(500.0 / main.frequency_resolution(len(samples), 8_000)))
        self.assertAlmostEqual(float(magnitude[k]), 0.6, places=6)
        magnitude[k] = 0.0
        self.assertLess(float(magnitude.max()), 1e-6)

    def test_dominant_frequency_of_on_grid_tone(self):
        samples = main.sine_wave(500.0, 8_000, 0.25)
        self.assertEqual(main.dominant_frequency(samples, 8_000), 500.0)

    def test_off_grid_tone_leaks_into_neighbours(self):
        on_grid = main.magnitude_spectrum(main.sine_wave(440.0, 8_000, 0.25, amplitude=0.5))
        off_grid = main.magnitude_spectrum(main.sine_wave(442.0, 8_000, 0.25, amplitude=0.5))
        self.assertLess(float(off_grid.max()), float(on_grid.max()))
        self.assertGreater(int(np.sum(off_grid > 0.01)), int(np.sum(on_grid > 0.01)))

    def test_power_spectrum_db_floor(self):
        db = main.power_spectrum_db(np.zeros(16))
        self.assertTrue(np.all(db == -100.0))


if __name__ == "__main__":
    unittest.main()
