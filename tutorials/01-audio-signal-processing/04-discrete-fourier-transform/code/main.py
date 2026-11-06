"""Lesson 04: the discrete Fourier transform from its definition, verified against NumPy's FFT.

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


def dft(samples: np.ndarray) -> np.ndarray:
    """Compute X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N) directly from the definition.

    This is O(N^2): a full N x N matrix of complex exponentials. It exists to be read,
    not to be fast. Compare with np.fft.fft, which computes the same numbers in O(N log N).
    """
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-D array")
    n = len(samples)
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    index = np.arange(n)
    twiddle = np.exp(-2j * np.pi * np.outer(index, index) / n)
    # einsum avoids a BLAS complex-matmul path that emits spurious warnings on some platforms.
    return np.einsum("kn,n->k", twiddle, samples)


def idft(spectrum: np.ndarray) -> np.ndarray:
    """Invert the DFT: x[n] = (1/N) sum_k X[k] * exp(+2*pi*i*k*n/N)."""
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    n = len(spectrum)
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    index = np.arange(n)
    twiddle = np.exp(2j * np.pi * np.outer(index, index) / n)
    return np.einsum("nk,k->n", twiddle, spectrum) / n


def bin_frequencies(num_samples: int, sample_rate: int) -> np.ndarray:
    """Frequency in hertz of each DFT bin k = 0..N-1, i.e. k * sample_rate / N."""
    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError("num_samples and sample_rate must be positive")
    return np.arange(num_samples) * sample_rate / num_samples


def frequency_resolution(num_samples: int, sample_rate: int) -> float:
    """Spacing between adjacent DFT bins in hertz."""
    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError("num_samples and sample_rate must be positive")
    return sample_rate / num_samples


def magnitude_spectrum(samples: np.ndarray) -> np.ndarray:
    """One-sided magnitude |X[k]| for k = 0..N//2, scaled so a sine of amplitude A peaks at A.

    Real signals have a mirror-image spectrum, so only the first N//2 + 1 bins carry new
    information. Doubling the interior bins accounts for the energy in the discarded mirror.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n == 0:
        return np.zeros(0)
    full = np.abs(np.fft.fft(samples)) / n
    half = full[: n // 2 + 1].copy()
    half[1 : (n + 1) // 2] *= 2.0
    return half


def power_spectrum_db(samples: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
    """One-sided power spectrum in decibels, 20*log10(|X|), clamped at a floor."""
    magnitude = magnitude_spectrum(samples)
    return np.maximum(20.0 * np.log10(np.maximum(magnitude, 1e-12)), floor_db)


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    """Frequency of the strongest non-DC bin in the one-sided spectrum."""
    magnitude = magnitude_spectrum(samples)
    if len(magnitude) < 2:
        return 0.0
    magnitude[0] = 0.0
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return float(frequencies[np.argmax(magnitude)])


def main() -> None:
    sample_rate, seconds = 8_000, 0.25
    n = round(sample_rate * seconds)
    print(f"{n} samples at {sample_rate} Hz -> bin spacing {frequency_resolution(n, sample_rate):.1f} Hz")

    signal = (
        sine_wave(440.0, sample_rate, seconds, amplitude=0.5)
        + sine_wave(1_000.0, sample_rate, seconds, amplitude=0.3)
        + sine_wave(2_400.0, sample_rate, seconds, amplitude=0.1)
    )

    slow = dft(signal)
    fast = np.fft.fft(signal)
    print(f"Definition vs np.fft.fft: max |difference| = {np.max(np.abs(slow - fast)):.2e}")

    restored = idft(slow).real
    print(f"Round trip DFT -> IDFT: max |difference| = {np.max(np.abs(restored - signal)):.2e}")

    magnitude = magnitude_spectrum(signal)
    frequencies = bin_frequencies(n, sample_rate)[: n // 2 + 1]
    top = np.argsort(magnitude)[-3:][::-1]
    for k in top:
        print(f"bin {k:4d} -> {frequencies[k]:7.1f} Hz, amplitude {magnitude[k]:.3f}")

    # A frequency that falls between bins leaks into its neighbours.
    off_grid = sine_wave(442.0, sample_rate, seconds, amplitude=0.5)
    leaked = magnitude_spectrum(off_grid)
    print(f"442 Hz sits between bins; peak amplitude drops to {leaked.max():.3f} and spreads to neighbours")
    print(f"Dominant frequency is still reported as {dominant_frequency(off_grid, sample_rate):.0f} Hz")


if __name__ == "__main__":
    main()
