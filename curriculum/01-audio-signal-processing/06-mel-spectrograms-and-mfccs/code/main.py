"""Lesson 06: mel filterbanks, log-mel spectrograms, and MFCCs with NumPy.

Run: python3 code/main.py
"""

from __future__ import annotations

import numpy as np


def synthetic_vowel(
    formants_hz: tuple[float, ...],
    pitch_hz: float,
    sample_rate: int,
    seconds: float,
    bandwidth_hz: float = 90.0,
) -> np.ndarray:
    """A vowel-like sound: harmonics of pitch_hz shaped by resonant peaks at formants_hz.

    The pitch sets where the harmonics sit; the formants set the smooth spectral envelope
    that decides which vowel we hear. Keeping them separate is the point of this lesson.
    """
    if sample_rate <= 0 or seconds <= 0 or pitch_hz <= 0:
        raise ValueError("sample_rate, seconds, and pitch_hz must be positive")
    t = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    harmonics = np.arange(pitch_hz, sample_rate / 2, pitch_hz)
    envelope = np.zeros_like(harmonics)
    for rank, formant in enumerate(formants_hz):
        envelope += (0.5 ** rank) / (1.0 + ((harmonics - formant) / bandwidth_hz) ** 2)
    # einsum avoids a BLAS matmul path that emits spurious warnings on some platforms (see Lesson 04).
    signal = np.einsum("h,ht->t", envelope, np.sin(2.0 * np.pi * harmonics[:, None] * t[None, :]))
    return 0.8 * signal / np.max(np.abs(signal))


def hz_to_mel(frequency_hz):
    """HTK mel scale: m = 2595 * log10(1 + f / 700). Linear-ish below 1 kHz, logarithmic above."""
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    """Inverse of hz_to_mel."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_frequencies(num: int, fmin: float, fmax: float) -> np.ndarray:
    """num frequencies in hertz, evenly spaced on the mel scale from fmin to fmax."""
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), num))


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Triangular mel filters with shape (n_mels, n_fft // 2 + 1).

    Filter m rises linearly from edge m to edge m+1 and falls back to zero at edge m+2,
    where the n_mels + 2 edges are evenly spaced in mels. With normalize=True each
    triangle is scaled to unit area (Slaney style), so wide high-frequency filters do not
    collect more energy just because they are wide.
    """
    if n_mels <= 0 or n_fft <= 0 or sample_rate <= 0:
        raise ValueError("sample_rate, n_fft, and n_mels must be positive")
    fmax = sample_rate / 2.0 if fmax is None else fmax
    if not 0.0 <= fmin < fmax <= sample_rate / 2.0:
        raise ValueError("need 0 <= fmin < fmax <= sample_rate / 2")
    bins_hz = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    edges = mel_frequencies(n_mels + 2, fmin, fmax)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bins_hz[None, :] - lower) / (center - lower)
    falling = (upper - bins_hz[None, :]) / (upper - center)
    weights = np.maximum(0.0, np.minimum(rising, falling))
    if normalize:
        weights *= 2.0 / (upper - lower)
    return weights


def hann_window(length: int) -> np.ndarray:
    """Periodic Hann window, as in Lesson 05."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def power_spectrogram(samples: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Centred, Hann-windowed |STFT|^2 with shape (n_fft // 2 + 1, num_frames), as in Lesson 05."""
    samples = np.pad(np.asarray(samples, dtype=np.float64), n_fft // 2)
    num_frames = 1 + (len(samples) - n_fft) // hop_length
    indices = (np.arange(num_frames) * hop_length)[:, None] + np.arange(n_fft)[None, :]
    return (np.abs(np.fft.rfft(samples[indices] * hann_window(n_fft), axis=1)) ** 2).T


def power_to_db(power: np.ndarray, reference: float = 1.0, amin: float = 1e-10, top_db: float | None = 80.0) -> np.ndarray:
    """10*log10(power / reference), floored at amin and optionally clipped to top_db below the peak."""
    power = np.asarray(power, dtype=np.float64)
    db = 10.0 * np.log10(np.maximum(power, amin)) - 10.0 * np.log10(max(amin, reference))
    if top_db is not None:
        db = np.maximum(db, db.max() - top_db)
    return db


def apply_filterbank(filters: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Weighted sum of STFT bins per mel band: (n_mels, bins) x (bins, frames) -> (n_mels, frames)."""
    return np.einsum("mk,kt->mt", filters, power)


def mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Mel power spectrogram: each column of |STFT|^2 multiplied by the filterbank."""
    filters = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    return apply_filterbank(filters, power_spectrogram(samples, n_fft, hop_length))


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II basis with shape (n_out, n_in).

    Row k is cos(pi * k * (2n + 1) / (2 * n_in)), scaled so the full square matrix is orthogonal.
    Row 0 is a constant (the average level); higher rows are progressively faster ripples.
    """
    if n_out <= 0 or n_in <= 0 or n_out > n_in:
        raise ValueError("need 0 < n_out <= n_in")
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    basis = np.sqrt(2.0 / n_in) * np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    basis[0] /= np.sqrt(2.0)
    return basis


def mfcc_from_log_mel(log_mel: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """MFCCs: the DCT of each log-mel column, keeping the first n_mfcc coefficients."""
    log_mel = np.asarray(log_mel, dtype=np.float64)
    return np.einsum("km,mt->kt", dct_matrix(n_mfcc, log_mel.shape[0]), log_mel)


def mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 40,
) -> np.ndarray:
    """Waveform -> power spectrogram -> mel filterbank -> dB -> DCT -> first n_mfcc rows."""
    log_mel = power_to_db(mel_spectrogram(samples, sample_rate, n_fft, hop_length, n_mels))
    return mfcc_from_log_mel(log_mel, n_mfcc)


def cepstral_mean_normalization(features: np.ndarray) -> np.ndarray:
    """Subtract each coefficient's mean over time, removing a fixed channel or microphone colouring."""
    features = np.asarray(features, dtype=np.float64)
    return features - features.mean(axis=1, keepdims=True)


def main() -> None:
    sample_rate, n_fft, hop_length, n_mels, n_mfcc = 16_000, 512, 160, 40, 13
    print(f"Mel scale: 1000 Hz -> {float(hz_to_mel(1_000.0)):.1f} mel, 4000 Hz -> {float(hz_to_mel(4_000.0)):.1f} mel, "
          f"8000 Hz -> {float(hz_to_mel(8_000.0)):.1f} mel")

    filters = mel_filterbank(sample_rate, n_fft, n_mels)
    edges = mel_frequencies(n_mels + 2, 0.0, sample_rate / 2)
    widths = edges[2:] - edges[:-2]
    print(f"Filterbank {filters.shape}: first filter spans {widths[0]:.0f} Hz, last spans {widths[-1]:.0f} Hz")

    a = synthetic_vowel((730.0, 1_090.0, 2_440.0), 120.0, sample_rate, 0.5)
    i = synthetic_vowel((270.0, 2_290.0, 3_010.0), 120.0, sample_rate, 0.5)
    a_high = synthetic_vowel((730.0, 1_090.0, 2_440.0), 200.0, sample_rate, 0.5)

    power = power_spectrogram(a, n_fft, hop_length)
    log_mel = power_to_db(apply_filterbank(filters, power))
    coefficients = mfcc_from_log_mel(log_mel, n_mfcc)
    print(f"{len(a)} samples -> power {power.shape} -> log-mel {log_mel.shape} -> MFCC {coefficients.shape}")

    def profile(signal):
        return mfcc(signal, sample_rate, n_mfcc, n_fft, hop_length, n_mels)[1:].mean(axis=1)

    print(f"MFCC distance /a/ at 120 Hz vs /a/ at 200 Hz: {np.linalg.norm(profile(a) - profile(a_high)):6.1f}")
    print(f"MFCC distance /a/ at 120 Hz vs /i/ at 120 Hz: {np.linalg.norm(profile(a) - profile(i)):6.1f}")
    print("Same vowel, different pitch: close. Different vowel, same pitch: far.")

    # A fixed channel gain adds a constant in dB, which lands entirely in c0 and vanishes after CMN.
    quieter = mfcc(0.1 * a, sample_rate, n_mfcc, n_fft, hop_length, n_mels)
    shift = quieter - coefficients
    print(f"Scaling the signal by 0.1 shifts c0 by {shift[0].mean():.1f}, other coefficients by at most {np.abs(shift[1:]).max():.1e}")


if __name__ == "__main__":
    main()
