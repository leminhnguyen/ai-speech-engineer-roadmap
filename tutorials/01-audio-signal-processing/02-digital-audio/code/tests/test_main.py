import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class DigitalAudioTests(unittest.TestCase):
    def test_nyquist_frequency(self):
        self.assertEqual(main.nyquist_frequency(16_000), 8_000.0)

    def test_alias_frequency_folds_above_nyquist(self):
        self.assertEqual(main.alias_frequency(7_000, 10_000), 3_000.0)
        self.assertEqual(main.alias_frequency(7_000, 8_000), 1_000.0)

    def test_fft_observes_alias(self):
        samples = main.sine_wave(7_000, 10_000, 0.1)
        self.assertEqual(main.dominant_frequency(samples, 10_000), 3_000.0)

    def test_quantize_dequantize_error_is_bounded(self):
        samples = np.array([-1.0, -0.25, 0.0, 0.25, 1.0])
        restored = main.dequantize_pcm(main.quantize_pcm(samples, 8), 8)
        self.assertLessEqual(float(np.max(np.abs(samples - restored))), 1 / 254)

    def test_naive_downsample_keeps_every_factor_sample(self):
        np.testing.assert_array_equal(main.naive_downsample(np.arange(10), 3), [0, 3, 6, 9])


if __name__ == "__main__":
    unittest.main()
