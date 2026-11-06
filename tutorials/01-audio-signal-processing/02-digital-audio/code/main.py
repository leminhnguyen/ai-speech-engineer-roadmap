"""Lesson 02: sampling, aliasing, and PCM quantization with NumPy.

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


def nyquist_frequency(sample_rate: int) -> float:
    """Return the highest unambiguous frequency for a sample rate."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return sample_rate / 2.0


def alias_frequency(frequency_hz: float, sample_rate: int) -> float:
    """Fold a continuous frequency into the observable [0, Nyquist] range."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    wrapped = (abs(frequency_hz) + sample_rate / 2.0) % sample_rate - sample_rate / 2.0
    return abs(float(wrapped))


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    """Find the strongest non-DC real-FFT bin."""
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    magnitudes = np.abs(np.fft.rfft(samples))
    magnitudes[0] = 0.0
    return float(frequencies[np.argmax(magnitudes)])


def quantize_pcm(samples: np.ndarray, bits: int = 16) -> np.ndarray:
    """Map normalized floats to a signed integer PCM grid."""
    if bits < 2 or bits > 32:
        raise ValueError("bits must be between 2 and 32")
    max_integer = 2 ** (bits - 1) - 1
    return np.rint(np.clip(samples, -1.0, 1.0) * max_integer).astype(np.int64)


def dequantize_pcm(values: np.ndarray, bits: int = 16) -> np.ndarray:
    """Map signed PCM values back to normalized floating-point samples."""
    max_integer = 2 ** (bits - 1) - 1
    return np.asarray(values, dtype=np.float64) / max_integer


def naive_downsample(samples: np.ndarray, factor: int) -> np.ndarray:
    """Keep every nth sample; intentionally no anti-alias low-pass filter."""
    if factor < 1:
        raise ValueError("factor must be positive")
    return np.asarray(samples)[::factor]


def main() -> None:
    sample_rate = 10_000
    true_frequency = 7_000.0
    tone = sine_wave(true_frequency, sample_rate, 0.1)
    print(f"Nyquist at {sample_rate} Hz: {nyquist_frequency(sample_rate):.0f} Hz")
    print(f"{true_frequency:.0f} Hz folds to {alias_frequency(true_frequency, sample_rate):.0f} Hz")
    print(f"FFT observes {dominant_frequency(tone, sample_rate):.0f} Hz")

    original = sine_wave(7_000.0, 24_000, 0.1)
    decimated = naive_downsample(original, 3)
    print(f"Naively decimating 24 kHz to 8 kHz observes {dominant_frequency(decimated, 8_000):.0f} Hz")

    values = quantize_pcm(sine_wave(440.0, 16_000, 0.02), bits=8)
    error = np.max(np.abs(sine_wave(440.0, 16_000, 0.02) - dequantize_pcm(values, bits=8)))
    print(f"8-bit quantization has at most {error:.4f} absolute error in this tone")


if __name__ == "__main__":
    main()
