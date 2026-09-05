"""Lesson 03: framing and time-domain features with NumPy.

Run: python3 code/main.py
"""

from __future__ import annotations

import numpy as np


def sine_wave(frequency_hz: float, sample_rate: int, seconds: float, amplitude: float = 0.8) -> np.ndarray:
    """Create a normalized sine signal sampled at a fixed rate."""
    if sample_rate <= 0 or seconds <= 0:
        raise ValueError("sample_rate and seconds must be positive")
    t = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice a 1-D signal into overlapping frames of shape (num_frames, frame_length).

    The last partial frame is zero-padded so no samples are dropped. This mirrors
    how feature extractors turn a long signal into a sequence of short windows.
    """
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-D array")
    if len(samples) == 0:
        return np.zeros((0, frame_length))
    num_frames = 1 + int(np.ceil(max(len(samples) - frame_length, 0) / hop_length))
    padded_length = (num_frames - 1) * hop_length + frame_length
    padded = np.pad(samples, (0, padded_length - len(samples)))
    starts = np.arange(num_frames) * hop_length
    indices = starts[:, None] + np.arange(frame_length)[None, :]
    return padded[indices]


def amplitude_envelope(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Maximum absolute amplitude in each frame."""
    frames = frame_signal(samples, frame_length, hop_length)
    return np.max(np.abs(frames), axis=1)


def rms_energy(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Root-mean-square energy of each frame."""
    frames = frame_signal(samples, frame_length, hop_length)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def zero_crossing_rate(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Fraction of adjacent sample pairs in each frame whose sign changes.

    A frame of length L has L-1 adjacent pairs, so the result lies in [0, 1].
    """
    frames = frame_signal(samples, frame_length, hop_length)
    signs = np.sign(frames)
    signs[signs == 0] = 1.0  # treat exact zeros as positive so silence is not full of crossings
    crossings = np.abs(np.diff(signs, axis=1)) > 0
    return np.mean(crossings, axis=1)


def frames_to_time(num_frames: int, hop_length: int, sample_rate: int) -> np.ndarray:
    """Time in seconds at which each frame starts."""
    return np.arange(num_frames) * hop_length / sample_rate


def amplitude_to_db(values: np.ndarray, reference: float = 1.0, floor_db: float = -80.0) -> np.ndarray:
    """Convert a linear amplitude (such as RMS) to decibels relative to a reference."""
    values = np.asarray(values, dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(values, 1e-12) / reference)
    return np.maximum(db, floor_db)


def main() -> None:
    sample_rate, frame_length, hop_length = 16_000, 1024, 512

    # Two 0.5-second segments: a soft low tone, then a loud high tone.
    soft_low = sine_wave(200.0, sample_rate, 0.5, amplitude=0.2)
    loud_high = sine_wave(2_000.0, sample_rate, 0.5, amplitude=0.8)
    signal = np.concatenate([soft_low, loud_high])

    frames = frame_signal(signal, frame_length, hop_length)
    print(f"{len(signal)} samples -> {frames.shape[0]} frames of {frames.shape[1]} samples")

    envelope = amplitude_envelope(signal, frame_length, hop_length)
    rms = rms_energy(signal, frame_length, hop_length)
    zcr = zero_crossing_rate(signal, frame_length, hop_length)

    half = frames.shape[0] // 2
    print(f"Envelope  first half ~{envelope[:half - 1].mean():.3f}, second half ~{envelope[half + 1:].mean():.3f}")
    print(f"RMS       first half ~{rms[:half - 1].mean():.3f}, second half ~{rms[half + 1:].mean():.3f}")
    print(f"RMS in dB first half ~{amplitude_to_db(rms[:half - 1]).mean():.1f} dB, "
          f"second half ~{amplitude_to_db(rms[half + 1:]).mean():.1f} dB")
    print(f"ZCR       first half ~{zcr[:half - 1].mean():.3f}, second half ~{zcr[half + 1:].mean():.3f}")
    print("A sine at f Hz crosses zero about 2f times per second, so ZCR ~ 2f / sample_rate.")
    print(f"Predicted ZCR: 200 Hz -> {2 * 200 / sample_rate:.3f}, 2000 Hz -> {2 * 2_000 / sample_rate:.3f}")


if __name__ == "__main__":
    main()
