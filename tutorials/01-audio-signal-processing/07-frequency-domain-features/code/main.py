"""Lesson 07: frequency-domain features (band energy ratio, spectral centroid, bandwidth,
rolloff, and flatness) with NumPy.

Run: python3 code/main.py
"""

from __future__ import annotations

import numpy as np


def sine_wave(frequency_hz: float, sample_rate: int, seconds: float, amplitude: float = 0.8) -> np.ndarray:
    """A pure tone, as in earlier lessons."""
    t = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def synthetic_vowel(
    formants_hz: tuple[float, ...],
    pitch_hz: float,
    sample_rate: int,
    seconds: float,
    bandwidth_hz: float = 90.0,
) -> np.ndarray:
    """Harmonics of pitch_hz shaped by resonant peaks at formants_hz (see Lesson 06)."""
    if sample_rate <= 0 or seconds <= 0 or pitch_hz <= 0:
        raise ValueError("sample_rate, seconds, and pitch_hz must be positive")
    t = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    harmonics = np.arange(pitch_hz, sample_rate / 2, pitch_hz)
    envelope = np.zeros_like(harmonics)
    for rank, formant in enumerate(formants_hz):
        envelope += (0.5 ** rank) / (1.0 + ((harmonics - formant) / bandwidth_hz) ** 2)
    signal = np.einsum("h,ht->t", envelope, np.sin(2.0 * np.pi * harmonics[:, None] * t[None, :]))
    return 0.8 * signal / np.max(np.abs(signal))


def highpass_noise(
    cutoff_hz: float,
    sample_rate: int,
    seconds: float,
    order: int = 2,
    amplitude: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Seeded Gaussian noise through a Butterworth-shaped high-pass, |H(f)| = 1 / sqrt(1 + (fc / f)^(2 * order)).

    With cutoff_hz around 4 kHz this resembles the hiss of /s/: no harmonics, most energy high up,
    but a gentle slope rather than an empty low band. cutoff_hz=0 gives white noise.
    """
    if not 0.0 <= cutoff_hz < sample_rate / 2 or order <= 0:
        raise ValueError("need 0 <= cutoff_hz < sample_rate / 2 and a positive order")
    num_samples = round(sample_rate * seconds)
    noise = np.random.default_rng(seed).standard_normal(num_samples)
    frequencies = np.fft.rfftfreq(num_samples, 1.0 / sample_rate)
    gain = np.zeros_like(frequencies)
    positive = frequencies > 0
    gain[positive] = 1.0 / np.sqrt(1.0 + (cutoff_hz / frequencies[positive]) ** (2 * order))
    shaped = np.fft.irfft(np.fft.rfft(noise) * gain, n=num_samples)
    return amplitude * shaped / np.max(np.abs(shaped))


def hann_window(length: int) -> np.ndarray:
    """Periodic Hann window, as in Lesson 05."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def magnitude_spectrogram(samples: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Centred, Hann-windowed |STFT| with shape (n_fft // 2 + 1, num_frames), as in Lesson 05."""
    samples = np.pad(np.asarray(samples, dtype=np.float64), n_fft // 2)
    num_frames = 1 + (len(samples) - n_fft) // hop_length
    indices = (np.arange(num_frames) * hop_length)[:, None] + np.arange(n_fft)[None, :]
    return np.abs(np.fft.rfft(samples[indices] * hann_window(n_fft), axis=1)).T


def bin_frequencies(sample_rate: int, n_fft: int) -> np.ndarray:
    """Centre frequency in hertz of each STFT bin."""
    return np.arange(n_fft // 2 + 1) * sample_rate / n_fft


def band_energy_ratio(power: np.ndarray, split_hz: float, sample_rate: int, n_fft: int, amin: float = 1e-10) -> np.ndarray:
    """Per frame: energy below split_hz divided by energy at or above it.

    Takes a power spectrogram (|STFT|^2). Voiced speech puts most energy low, so the ratio is large;
    hiss puts it high, so the ratio is small. Often reported in dB.
    """
    power = np.asarray(power, dtype=np.float64)
    low = bin_frequencies(sample_rate, n_fft) < split_hz
    return power[low].sum(axis=0) / np.maximum(power[~low].sum(axis=0), amin)


def _weights(magnitude: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    """Each column normalised to sum to 1, so it can be read as a distribution over frequency."""
    magnitude = np.asarray(magnitude, dtype=np.float64)
    return magnitude / np.maximum(magnitude.sum(axis=0, keepdims=True), amin)


def spectral_centroid(magnitude: np.ndarray, sample_rate: int, n_fft: int) -> np.ndarray:
    """Magnitude-weighted mean frequency per frame: sum_k f_k |X_k| / sum_k |X_k|."""
    return np.einsum("k,kt->t", bin_frequencies(sample_rate, n_fft), _weights(magnitude))


def spectral_bandwidth(magnitude: np.ndarray, sample_rate: int, n_fft: int, p: float = 2.0) -> np.ndarray:
    """Magnitude-weighted spread around the centroid: (sum_k w_k |f_k - centroid|^p)^(1/p).

    With p=2 this is the standard deviation of the frequency distribution.
    """
    frequencies = bin_frequencies(sample_rate, n_fft)
    deviation = np.abs(frequencies[:, None] - spectral_centroid(magnitude, sample_rate, n_fft)[None, :])
    return np.sum(_weights(magnitude) * deviation ** p, axis=0) ** (1.0 / p)


def spectral_rolloff(magnitude: np.ndarray, sample_rate: int, n_fft: int, roll_percent: float = 0.85) -> np.ndarray:
    """Lowest bin frequency below which roll_percent of the frame's magnitude lies."""
    if not 0.0 < roll_percent < 1.0:
        raise ValueError("roll_percent must be between 0 and 1")
    cumulative = np.cumsum(np.asarray(magnitude, dtype=np.float64), axis=0)
    first = np.argmax(cumulative >= roll_percent * cumulative[-1:], axis=0)
    return bin_frequencies(sample_rate, n_fft)[first]


def spectral_flatness(magnitude: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    """Geometric mean over arithmetic mean of the power spectrum, per frame.

    Close to 1 when every bin holds similar power (noise), close to 0 when a few bins hold it all (tones).
    """
    power = np.maximum(np.asarray(magnitude, dtype=np.float64) ** 2, amin)
    return np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)


def rms_from_magnitude(magnitude: np.ndarray, n_fft: int) -> np.ndarray:
    """Frame RMS of the windowed frame via Parseval, so loudness comes from the same spectrogram."""
    power = np.asarray(magnitude, dtype=np.float64) ** 2
    total = 2.0 * power.sum(axis=0) - power[0] - (power[-1] if n_fft % 2 == 0 else 0.0)
    return np.sqrt(total / n_fft ** 2)


def label_frames(
    magnitude: np.ndarray,
    sample_rate: int,
    n_fft: int,
    silence_rms: float = 0.01,
    centroid_split_hz: float = 2_500.0,
) -> np.ndarray:
    """A two-feature rule: quiet frames are silence, bright frames are fricatives, the rest are voiced."""
    rms = rms_from_magnitude(magnitude, n_fft)
    centroid = spectral_centroid(magnitude, sample_rate, n_fft)
    return np.where(rms < silence_rms, "silence", np.where(centroid > centroid_split_hz, "fricative", "voiced"))


def asa_utterance(sample_rate: int) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """Silence, /a/, an /s/-like hiss, /a/, silence: a synthetic 'asa' with known segment times."""
    pieces = [
        ("silence", np.zeros(round(0.15 * sample_rate))),
        ("voiced", synthetic_vowel((730.0, 1_090.0, 2_440.0), 120.0, sample_rate, 0.35)),
        ("fricative", highpass_noise(4_000.0, sample_rate, 0.25)),
        ("voiced", synthetic_vowel((730.0, 1_090.0, 2_440.0), 110.0, sample_rate, 0.35)),
        ("silence", np.zeros(round(0.15 * sample_rate))),
    ]
    segments, start = [], 0.0
    for label, piece in pieces:
        end = start + len(piece) / sample_rate
        segments.append((label, start, end))
        start = end
    return np.concatenate([piece for _, piece in pieces]), segments


def main() -> None:
    sample_rate, n_fft, hop_length = 16_000, 512, 160
    hop_label = f"n_fft={n_fft} ({1_000 * n_fft / sample_rate:.0f} ms), hop={hop_length}"
    print(f"STFT setup: {hop_label}, bin spacing {sample_rate / n_fft:.2f} Hz")

    sounds = {
        "1 kHz tone": sine_wave(1_000.0, sample_rate, 0.5),
        "vowel /a/": synthetic_vowel((730.0, 1_090.0, 2_440.0), 120.0, sample_rate, 0.5),
        "vowel /i/": synthetic_vowel((270.0, 2_290.0, 3_010.0), 120.0, sample_rate, 0.5),
        "/s/-like hiss": highpass_noise(4_000.0, sample_rate, 0.5),
        "white noise": highpass_noise(0.0, sample_rate, 0.5),
    }
    print(f"\n{'sound':<14} {'BER@2k (dB)':>11} {'centroid':>9} {'bandwidth':>10} {'rolloff85':>10} {'flatness':>9}")
    for name, samples in sounds.items():
        magnitude = magnitude_spectrogram(samples, n_fft, hop_length)[:, 5:-5]  # skip padded edge frames
        ber = 10.0 * np.log10(band_energy_ratio(magnitude ** 2, 2_000.0, sample_rate, n_fft))
        print(f"{name:<14} {np.median(ber):11.1f} "
              f"{np.median(spectral_centroid(magnitude, sample_rate, n_fft)):7.0f}Hz "
              f"{np.median(spectral_bandwidth(magnitude, sample_rate, n_fft)):8.0f}Hz "
              f"{np.median(spectral_rolloff(magnitude, sample_rate, n_fft)):8.0f}Hz "
              f"{np.median(spectral_flatness(magnitude)):9.3f}")

    utterance, segments = asa_utterance(sample_rate)
    magnitude = magnitude_spectrogram(utterance, n_fft, hop_length)
    predicted = label_frames(magnitude, sample_rate, n_fft)
    times = np.arange(magnitude.shape[1]) * hop_length / sample_rate
    truth = np.empty(len(times), dtype=object)
    for label, start, end in segments:
        truth[(times >= start) & (times < end)] = label
    # Frames within half a window of a boundary mix two segments, so score only the interiors.
    boundaries = np.array([end for _, _, end in segments[:-1]])
    interior = np.min(np.abs(times[:, None] - boundaries[None, :]), axis=1) > n_fft / sample_rate / 2
    accuracy = np.mean(predicted[interior] == truth[interior])
    print(f"\nSynthetic 'asa': {len(times)} frames, rule on RMS + centroid labels "
          f"{accuracy:.0%} of {interior.sum()} interior frames correctly")


if __name__ == "__main__":
    main()
