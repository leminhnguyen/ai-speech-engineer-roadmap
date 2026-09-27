"""Lesson 05: windowing, the short-time Fourier transform, and spectrograms with NumPy.

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


def linear_chirp(start_hz: float, end_hz: float, sample_rate: int, seconds: float, amplitude: float = 0.8) -> np.ndarray:
    """A sine whose frequency rises linearly from start_hz to end_hz over the signal."""
    if sample_rate <= 0 or seconds <= 0:
        raise ValueError("sample_rate and seconds must be positive")
    t = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    phase = 2.0 * np.pi * (start_hz * t + (end_hz - start_hz) * t ** 2 / (2.0 * seconds))
    return amplitude * np.sin(phase)


def hann_window(length: int, periodic: bool = True) -> np.ndarray:
    """Hann window w[n] = 0.5 - 0.5 cos(2*pi*n / D).

    The periodic form (D = length) is the one used for spectral analysis; the symmetric
    form (D = length - 1) is the textbook filter-design version returned by np.hanning.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return np.ones(1)
    denominator = length if periodic else length - 1
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / denominator)


def frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice a 1-D signal into frames of shape (num_frames, frame_length), dropping any partial tail."""
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-D array")
    if len(samples) < frame_length:
        raise ValueError("signal is shorter than one frame")
    num_frames = 1 + (len(samples) - frame_length) // hop_length
    starts = np.arange(num_frames) * hop_length
    return samples[starts[:, None] + np.arange(frame_length)[None, :]]


def stft(
    samples: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray | None = None,
    center: bool = True,
) -> np.ndarray:
    """Short-time Fourier transform with shape (n_fft // 2 + 1, num_frames).

    Frame, multiply each frame by a window, and take the one-sided FFT of every frame.
    With center=True the signal is zero-padded by n_fft // 2 on both sides, so frame t
    is centred on sample t * hop_length, matching librosa's default.
    """
    if window is None:
        window = hann_window(n_fft)
    window = np.asarray(window, dtype=np.float64)
    if len(window) != n_fft:
        raise ValueError("window length must equal n_fft")
    samples = np.asarray(samples, dtype=np.float64)
    if center:
        samples = np.pad(samples, n_fft // 2)
    frames = frame_signal(samples, n_fft, hop_length) * window
    return np.fft.rfft(frames, axis=1).T


def istft(
    spectrum: np.ndarray,
    hop_length: int,
    window: np.ndarray | None = None,
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """Invert an STFT by weighted overlap-add.

    Each frame is inverse-transformed, multiplied by the window again, and added at its
    position. Dividing by the running sum of squared windows undoes both multiplications.
    """
    spectrum = np.asarray(spectrum)
    n_fft = 2 * (spectrum.shape[0] - 1)
    if window is None:
        window = hann_window(n_fft)
    window = np.asarray(window, dtype=np.float64)
    frames = np.fft.irfft(spectrum.T, n=n_fft, axis=1) * window
    total = n_fft + hop_length * (len(frames) - 1)
    output = np.zeros(total)
    window_sum = np.zeros(total)
    for t, frame in enumerate(frames):
        start = t * hop_length
        output[start : start + n_fft] += frame
        window_sum[start : start + n_fft] += window ** 2
    covered = window_sum > 1e-10
    output[covered] /= window_sum[covered]
    if center:
        output = output[n_fft // 2 :]
    if length is None:
        return output[: total - 2 * (n_fft // 2)] if center else output
    return np.pad(output, (0, max(length - len(output), 0)))[:length]


def power_spectrogram(samples: np.ndarray, n_fft: int, hop_length: int, center: bool = True) -> np.ndarray:
    """|STFT|^2 with a Hann window."""
    return np.abs(stft(samples, n_fft, hop_length, center=center)) ** 2


def power_to_db(power: np.ndarray, reference: float = 1.0, amin: float = 1e-10, top_db: float | None = 80.0) -> np.ndarray:
    """10*log10(power / reference), floored at amin and optionally clipped to top_db below the peak."""
    power = np.asarray(power, dtype=np.float64)
    db = 10.0 * np.log10(np.maximum(power, amin)) - 10.0 * np.log10(max(amin, reference))
    if top_db is not None:
        db = np.maximum(db, db.max() - top_db)
    return db


def stft_frequencies(sample_rate: int, n_fft: int) -> np.ndarray:
    """Centre frequency in hertz of each STFT row."""
    return np.arange(n_fft // 2 + 1) * sample_rate / n_fft


def frames_to_time(num_frames: int, hop_length: int, sample_rate: int) -> np.ndarray:
    """Time in seconds of each STFT column (the frame centre when center=True)."""
    return np.arange(num_frames) * hop_length / sample_rate


def resolution(sample_rate: int, n_fft: int, hop_length: int) -> dict[str, float]:
    """The three numbers that describe an STFT configuration."""
    return {
        "window_ms": 1_000.0 * n_fft / sample_rate,
        "hop_ms": 1_000.0 * hop_length / sample_rate,
        "bin_hz": sample_rate / n_fft,
    }


def overlap_add_window(window: np.ndarray, hop_length: int, num_frames: int) -> np.ndarray:
    """Sum of num_frames copies of window shifted by hop_length.

    A flat interior means the window/hop pair satisfies constant overlap-add (COLA):
    every sample is weighted equally once all frames are summed.
    """
    window = np.asarray(window, dtype=np.float64)
    total = len(window) + hop_length * (num_frames - 1)
    output = np.zeros(total)
    for t in range(num_frames):
        output[t * hop_length : t * hop_length + len(window)] += window
    return output


def energy_outside_peak(magnitude: np.ndarray, half_width: int = 3) -> float:
    """Fraction of spectral energy farther than half_width bins from the strongest bin."""
    energy = np.asarray(magnitude, dtype=np.float64) ** 2
    peak = int(np.argmax(energy))
    near = energy[max(peak - half_width, 0) : peak + half_width + 1].sum()
    return float(1.0 - near / energy.sum())


def main() -> None:
    sample_rate, n_fft, hop_length = 16_000, 512, 128
    config = resolution(sample_rate, n_fft, hop_length)
    print(f"n_fft={n_fft}, hop={hop_length} at {sample_rate} Hz -> "
          f"{config['window_ms']:.0f} ms window, {config['hop_ms']:.0f} ms hop, {config['bin_hz']:.2f} Hz bins")

    # Windowing: an off-bin tone with and without a Hann window.
    tone = sine_wave(442.0, 8_000, 0.25, amplitude=0.5)
    rectangular = np.abs(np.fft.rfft(tone))
    windowed = np.abs(np.fft.rfft(tone * hann_window(len(tone))))
    print(f"442 Hz energy outside +/-3 bins: rectangular {energy_outside_peak(rectangular):.1%}, "
          f"Hann {energy_outside_peak(windowed):.2%}")

    # A 1-second chirp from 300 Hz to 3 kHz: one FFT hides the sweep, the STFT follows it.
    chirp = linear_chirp(300.0, 3_000.0, sample_rate, 1.0)
    spectrum = stft(chirp, n_fft, hop_length)
    frequencies = stft_frequencies(sample_rate, n_fft)
    times = frames_to_time(spectrum.shape[1], hop_length, sample_rate)
    print(f"{len(chirp)} samples -> STFT of shape {spectrum.shape} (frequency bins, frames)")
    peaks = frequencies[np.argmax(np.abs(spectrum), axis=0)]
    for t in (10, 62, 115):
        expected = 300.0 + 2_700.0 * times[t]
        print(f"frame {t:3d} at {times[t]:.2f} s -> peak {peaks[t]:7.1f} Hz (instantaneous {expected:7.1f} Hz)")

    restored = istft(spectrum, hop_length, length=len(chirp))
    print(f"Round trip STFT -> ISTFT: max |difference| = {np.max(np.abs(restored - chirp)):.2e}")

    ola = overlap_add_window(hann_window(n_fft), n_fft // 4, 12)
    interior = ola[n_fft : -n_fft]
    print(f"Hann at 75% overlap sums to {interior.min():.3f}..{interior.max():.3f} (constant overlap-add)")


if __name__ == "__main__":
    main()
