import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class WaveformTests(unittest.TestCase):
    def test_time_axis_has_expected_length(self):
        self.assertEqual(len(main.time_axis(8_000, 0.25)), 2_000)

    def test_sine_wave_stays_in_amplitude_range(self):
        samples = main.sine_wave(440.0, 8_000, 0.1, amplitude=0.25)
        self.assertLessEqual(float(np.max(np.abs(samples))), 0.25 + 1e-12)

    def test_dominant_frequency_matches_a_bin_center(self):
        samples = main.sine_wave(500.0, 8_000, 0.25)
        self.assertEqual(main.dominant_frequency(samples, 8_000), 500.0)

    def test_mix_averages_equal_length_signals(self):
        first = np.array([0.0, 1.0])
        second = np.array([1.0, -1.0])
        np.testing.assert_allclose(main.mix([first, second]), [0.5, 0.0])

    def test_pcm16_wav_round_trip_is_small(self):
        samples = main.sine_wave(440.0, 8_000, 0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            main.write_wav(path, samples, 8_000)
            restored, rate = main.read_wav(path)
        self.assertEqual(rate, 8_000)
        self.assertLess(float(np.max(np.abs(samples - restored))), 4e-5)


if __name__ == "__main__":
    unittest.main()
