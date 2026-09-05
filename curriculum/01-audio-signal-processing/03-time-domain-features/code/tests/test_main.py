import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class TimeDomainFeatureTests(unittest.TestCase):
    def test_frame_signal_shape_and_hop(self):
        frames = main.frame_signal(np.arange(10, dtype=float), frame_length=4, hop_length=2)
        self.assertEqual(frames.shape, (4, 4))
        np.testing.assert_array_equal(frames[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(frames[1], [2, 3, 4, 5])
        np.testing.assert_array_equal(frames[-1], [6, 7, 8, 9])

    def test_frame_signal_zero_pads_last_frame(self):
        frames = main.frame_signal(np.arange(7, dtype=float), frame_length=4, hop_length=2)
        self.assertEqual(frames.shape, (3, 4))
        np.testing.assert_array_equal(frames[-1], [4, 5, 6, 0])

    def test_amplitude_envelope_tracks_peak(self):
        signal = np.concatenate([np.full(8, 0.25), np.full(8, -0.9)])
        envelope = main.amplitude_envelope(signal, frame_length=8, hop_length=8)
        np.testing.assert_allclose(envelope, [0.25, 0.9])

    def test_rms_of_sine_is_amplitude_over_sqrt2(self):
        tone = main.sine_wave(100.0, 8_000, 1.0, amplitude=0.8)
        rms = main.rms_energy(tone, frame_length=8_000, hop_length=8_000)
        self.assertAlmostEqual(float(rms[0]), 0.8 / np.sqrt(2), places=4)

    def test_rms_of_silence_is_zero(self):
        rms = main.rms_energy(np.zeros(2_048), frame_length=1_024, hop_length=512)
        np.testing.assert_array_equal(rms, np.zeros_like(rms))

    def test_zero_crossing_rate_matches_two_f_over_sr(self):
        sample_rate = 8_000
        tone = main.sine_wave(400.0, sample_rate, 1.0)
        zcr = main.zero_crossing_rate(tone, frame_length=sample_rate, hop_length=sample_rate)
        self.assertAlmostEqual(float(zcr[0]), 2 * 400 / sample_rate, delta=0.001)

    def test_zero_crossing_rate_bounds(self):
        alternating = np.tile([1.0, -1.0], 512)
        zcr = main.zero_crossing_rate(alternating, frame_length=1_024, hop_length=1_024)
        self.assertAlmostEqual(float(zcr[0]), 1.0)
        silence = main.zero_crossing_rate(np.zeros(1_024), frame_length=1_024, hop_length=1_024)
        self.assertEqual(float(silence[0]), 0.0)

    def test_frames_to_time(self):
        np.testing.assert_allclose(main.frames_to_time(3, 512, 16_000), [0.0, 0.032, 0.064])

    def test_amplitude_to_db(self):
        np.testing.assert_allclose(main.amplitude_to_db(np.array([1.0, 0.1])), [0.0, -20.0])
        self.assertEqual(float(main.amplitude_to_db(np.array([0.0]))[0]), -80.0)


if __name__ == "__main__":
    unittest.main()
