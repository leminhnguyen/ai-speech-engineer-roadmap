"""Lesson 01: synthesize and inspect waveforms with NumPy.

Run: python3 code/main.py
"""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def time_axis(sample_rate: int, seconds: float) -> np.ndarray:
    """Return equally spaced sample times in seconds."""
    if sample_rate <= 0 or seconds <= 0:
        raise ValueError("sample_rate and seconds must be positive")
    return np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate


def sine_wave(frequency_hz: float, sample_rate: int, seconds: float, amplitude: float = 0.5,
              phase_radians: float = 0.0) -> np.ndarray:
    """Create a mono sine waveform in the normalized range [-1, 1]."""
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude must be in [0, 1]")
    t = time_axis(sample_rate, seconds)
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_radians)


def mix(signals: Iterable[np.ndarray]) -> np.ndarray:
    """Average equal-length signals, preserving a valid normalized waveform."""
    items = [np.asarray(signal, dtype=np.float64) for signal in signals]
    if not items:
        raise ValueError("at least one signal is required")
    length = len(items[0])
    if any(len(item) != length for item in items):
        raise ValueError("all signals must have the same length")
    return np.mean(np.stack(items), axis=0)


def pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert normalized floating-point samples to signed 16-bit PCM."""
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a mono PCM16 WAV using only the standard library."""
    values = pcm16(samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack("<" + "h" * len(values), *values))


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a mono PCM16 WAV and return normalized float samples plus sample rate."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("only mono PCM16 WAV files are supported in this lesson")
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    values = np.array(struct.unpack("<" + "h" * (len(raw) // 2), raw), dtype=np.int16)
    return values.astype(np.float64) / 32767.0, sample_rate


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    """Estimate the strongest non-DC frequency using the real FFT."""
    signal = np.asarray(samples, dtype=np.float64)
    if len(signal) < 2:
        raise ValueError("at least two samples are required")
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    magnitudes = np.abs(np.fft.rfft(signal))
    magnitudes[0] = 0.0
    return float(frequencies[np.argmax(magnitudes)])


def main() -> None:
    sample_rate = 16_000
    seconds = 1.0
    a4 = sine_wave(440.0, sample_rate, seconds)
    chord = mix([a4, sine_wave(660.0, sample_rate, seconds), sine_wave(880.0, sample_rate, seconds)])
    with tempfile.TemporaryDirectory(prefix="sound_waveforms_") as directory:
        output = Path(directory) / "a4.wav"
        write_wav(output, a4, sample_rate)
        restored, restored_rate = read_wav(output)
        error = float(np.max(np.abs(a4 - restored)))
    print(f"A4: {len(a4)} samples at {sample_rate} Hz")
    print(f"Dominant frequency: {dominant_frequency(a4, sample_rate):.1f} Hz")
    print(f"Chord dominant frequency: {dominant_frequency(chord, sample_rate):.1f} Hz")
    print(f"PCM16 round-trip at {restored_rate} Hz; max error={error:.6f}")


if __name__ == "__main__":
    main()
