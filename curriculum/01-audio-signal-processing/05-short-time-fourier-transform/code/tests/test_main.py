import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class ShortTimeFourierTransformTests(unittest.TestCase):
    def test_hann_window_shape_and_symmetry(self):
        periodic = main.hann_window(8)
        np.testing.assert_allclose(periodic, [0.0, 0.1464466, 0.5, 0.8535534, 1.0, 0.8535534, 0.5, 0.1464466], atol=1e-7)
        np.testing.assert_allclose(main.hann_window(9, periodic=False), np.hanning(9))

    def test_hann_satisfies_constant_overlap_add(self):
        window = main.hann_window(256)
        half = main.overlap_add_window(window, 128, 10)[256:-256]
        quarter = main.overlap_add_window(window, 64, 20)[256:-256]
        np.testing.assert_allclose(half, 1.0, atol=1e-12)
        np.testing.assert_allclose(quarter, 2.0, atol=1e-12)

    def test_stft_shape_with_center_padding(self):
        spectrum = main.stft(np.zeros(16_000), n_fft=512, hop_length=128)
        self.assertEqual(spectrum.shape, (257, 1 + 16_000 // 128))

    def test_stft_column_is_fft_of_windowed_frame(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=2_048)
        spectrum = main.stft(samples, n_fft=256, hop_length=64, center=False)
        frame = samples[3 * 64 : 3 * 64 + 256] * main.hann_window(256)
        np.testing.assert_allclose(spectrum[:, 3], np.fft.rfft(frame), atol=1e-10)

    def test_istft_inverts_stft(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(size=4_000)
        spectrum = main.stft(samples, n_fft=512, hop_length=128)
        np.testing.assert_allclose(main.istft(spectrum, 128, length=len(samples)), samples, atol=1e-10)

    def test_hann_window_reduces_leakage(self):
        tone = main.sine_wave(442.0, 8_000, 0.25)
        rectangular = main.energy_outside_peak(np.abs(np.fft.rfft(tone)))
        hann = main.energy_outside_peak(np.abs(np.fft.rfft(tone * main.hann_window(len(tone)))))
        self.assertGreater(rectangular, 0.03)
        self.assertLess(hann, 0.001)

    def test_stft_tracks_a_chirp(self):
        chirp = main.linear_chirp(500.0, 3_500.0, 16_000, 1.0)
        spectrum = np.abs(main.stft(chirp, n_fft=512, hop_length=256))
        peaks = main.stft_frequencies(16_000, 512)[np.argmax(spectrum[:, 2:-2], axis=0)]
        self.assertTrue(np.all(np.diff(peaks) >= 0))
        self.assertLess(peaks[0], 700.0)
        self.assertGreater(peaks[-1], 3_300.0)

    def test_resolution_numbers(self):
        config = main.resolution(16_000, 400, 160)
        self.assertEqual(config, {"window_ms": 25.0, "hop_ms": 10.0, "bin_hz": 40.0})

    def test_frames_to_time(self):
        np.testing.assert_allclose(main.frames_to_time(4, 160, 16_000), [0.0, 0.01, 0.02, 0.03])

    def test_power_to_db_reference_and_top_db(self):
        db = main.power_to_db(np.array([1.0, 0.1, 1e-12]), top_db=80.0)
        np.testing.assert_allclose(db, [0.0, -10.0, -80.0])


if __name__ == "__main__":
    unittest.main()
