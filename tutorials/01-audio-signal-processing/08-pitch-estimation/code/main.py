"""Lesson 08: pitch (F0) estimation with autocorrelation and YIN, using NumPy.

Run: python3 code/main.py
"""

from __future__ import annotations

import numpy as np


def harmonic_signal(
    f0_hz,
    sample_rate: int,
    formants_hz: tuple[float, ...] = (730.0, 1_090.0, 2_440.0),
    bandwidth_hz: float = 90.0,
    harmonics: tuple[int, ...] | None = None,
    amplitude: float = 0.8,
) -> np.ndarray:
    """A voiced sound whose pitch may change over time.

    f0_hz holds one frequency per sample (a pitch contour); see constant_pitch and glide.
    Harmonic h has instantaneous frequency h * f0(t); its phase is the running sum of that frequency,
    so a glide stays smooth. Amplitudes follow a formant envelope evaluated at the harmonic's frequency.
    Pass harmonics=(2, 3, ...) to leave out the fundamental.
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    if f0.ndim != 1 or len(f0) == 0 or np.any(f0 <= 0):
        raise ValueError("f0_hz must be a non-empty 1-D array of positive frequencies")
    ranks = np.arange(1, int(sample_rate / 2 / f0.min()) + 1) if harmonics is None else np.asarray(harmonics)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    signal = np.zeros_like(f0)
    for rank in ranks:
        frequency = rank * f0
        gain = sum((0.5 ** i) / (1.0 + ((frequency - formant) / bandwidth_hz) ** 2) for i, formant in enumerate(formants_hz))
        signal += np.where(frequency < sample_rate / 2, gain, 0.0) * np.sin(rank * phase)
    return amplitude * signal / np.max(np.abs(signal))


def constant_pitch(f0_hz: float, sample_rate: int, seconds: float) -> np.ndarray:
    """A constant pitch contour, ready for harmonic_signal."""
    return np.full(round(sample_rate * seconds), float(f0_hz))


def glide(start_hz: float, end_hz: float, sample_rate: int, seconds: float) -> np.ndarray:
    """A pitch contour that moves geometrically (evenly in semitones) from start_hz to end_hz."""
    return start_hz * (end_hz / start_hz) ** np.linspace(0.0, 1.0, round(sample_rate * seconds))


def hiss(sample_rate: int, seconds: float, amplitude: float = 0.2, seed: int = 0) -> np.ndarray:
    """Seeded white noise standing in for an unvoiced consonant."""
    noise = np.random.default_rng(seed).standard_normal(round(sample_rate * seconds))
    return amplitude * noise / np.max(np.abs(noise))


def frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Centred frames with shape (num_frames, frame_length); frame t is centred on sample t * hop_length."""
    samples = np.pad(np.asarray(samples, dtype=np.float64), frame_length // 2)
    num_frames = 1 + (len(samples) - frame_length) // hop_length
    indices = (np.arange(num_frames) * hop_length)[:, None] + np.arange(frame_length)[None, :]
    return samples[indices]


def autocorrelation(frames: np.ndarray, max_lag: int) -> np.ndarray:
    """r[t, tau] = sum_n x_t[n] x_t[n + tau] for tau = 0..max_lag, computed with the FFT.

    Zero-padding to at least twice the frame length turns the FFT's circular correlation into a linear one.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    n_fft = 1 << int(np.ceil(np.log2(2 * frames.shape[1])))
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    return np.fft.irfft(np.abs(spectrum) ** 2, n=n_fft, axis=1)[:, : max_lag + 1]


def difference_function(frames: np.ndarray, max_lag: int) -> np.ndarray:
    """YIN step 2: d[t, tau] = sum_{j < W} (x_t[j] - x_t[j + tau])^2 with W = frame_length - max_lag.

    Expanded as energy(start) + energy(shifted) - 2 * cross term, so every lag costs one FFT pass.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    window = frames.shape[1] - max_lag
    if window <= 0:
        raise ValueError("frame_length must be longer than max_lag")
    n_fft = 1 << int(np.ceil(np.log2(frames.shape[1] + window)))
    cross = np.fft.irfft(
        np.conj(np.fft.rfft(frames[:, :window], n=n_fft, axis=1)) * np.fft.rfft(frames, n=n_fft, axis=1),
        n=n_fft,
        axis=1,
    )[:, : max_lag + 1]
    cumulative = np.concatenate([np.zeros((frames.shape[0], 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    lags = np.arange(max_lag + 1)
    shifted_energy = cumulative[:, lags + window] - cumulative[:, lags]
    return np.maximum(cumulative[:, window : window + 1] + shifted_energy - 2.0 * cross, 0.0)


def cumulative_mean_normalized_difference(difference: np.ndarray) -> np.ndarray:
    """YIN step 3: d'[0] = 1 and d'[tau] = d[tau] / mean(d[1..tau]).

    Dividing by the running mean removes the dip at tau = 0 and puts every frame on the same scale:
    a perfectly periodic frame reaches 0 at its period, while noise hovers around 1.
    """
    difference = np.atleast_2d(np.asarray(difference, dtype=np.float64))
    lags = np.arange(1, difference.shape[1])
    running_mean = np.cumsum(difference[:, 1:], axis=1) / lags
    normalized = np.ones_like(difference)
    np.divide(difference[:, 1:], running_mean, out=normalized[:, 1:], where=running_mean > 0)
    return normalized


def parabolic_offset(values: np.ndarray, index: int) -> float:
    """Sub-sample offset of an extremum from a parabola through values[index - 1 : index + 2]."""
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * centre + right
    return 0.0 if denominator == 0 else float(0.5 * (left - right) / denominator)


def yin(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_length: int = 640,
    hop_length: int = 160,
    threshold: float = 0.1,
    silence_rms: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-wise F0 with YIN (de Cheveigné and Kawahara, 2002).

    Returns (f0_hz, voiced, aperiodicity). f0 is NaN on unvoiced frames. A frame is voiced when the
    normalised difference dips below threshold somewhere in the allowed lag range; the first such dip
    wins, which is what prevents choosing two or three periods (an octave too low).
    """
    tau_min, tau_max = int(np.floor(sample_rate / fmax)), int(np.ceil(sample_rate / fmin))
    if not 1 <= tau_min < tau_max < frame_length:
        raise ValueError("need sample_rate / fmax >= 1 and sample_rate / fmin < frame_length")
    frames = frame_signal(samples, frame_length, hop_length)
    normalized = cumulative_mean_normalized_difference(difference_function(frames, tau_max))
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    f0 = np.full(len(frames), np.nan)
    voiced = np.zeros(len(frames), dtype=bool)
    aperiodicity = np.ones(len(frames))
    for t, curve in enumerate(normalized):
        search = curve[tau_min : tau_max + 1]
        below = np.flatnonzero(search < threshold)
        if below.size == 0 or rms[t] < silence_rms:
            aperiodicity[t] = float(search.min())
            continue
        tau = tau_min + below[0]
        while tau + 1 <= tau_max and curve[tau + 1] < curve[tau]:  # slide to the bottom of the dip
            tau += 1
        voiced[t] = True
        aperiodicity[t] = float(curve[tau])
        f0[t] = sample_rate / (tau + parabolic_offset(curve, tau))
    return f0, voiced, aperiodicity


def autocorrelation_pitch(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_length: int = 640,
    hop_length: int = 160,
) -> np.ndarray:
    """Baseline: the lag of the largest autocorrelation peak in [sr / fmax, sr / fmin], per frame."""
    tau_min, tau_max = int(np.floor(sample_rate / fmax)), int(np.ceil(sample_rate / fmin))
    r = autocorrelation(frame_signal(samples, frame_length, hop_length), tau_max)
    f0 = np.full(len(r), np.nan)
    for t, curve in enumerate(r):
        if curve[0] <= 0:
            continue
        tau = tau_min + int(np.argmax(curve[tau_min : tau_max + 1]))
        f0[t] = sample_rate / (tau + parabolic_offset(curve, tau))
    return f0


def deepest_dip_pitch(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_length: int = 640,
    hop_length: int = 160,
) -> np.ndarray:
    """YIN without its step 4: take the global minimum of d' instead of the first dip below a threshold.

    A periodic frame dips at T, 2T, 3T, ... and the later dips can be just as deep, so this often
    reports half or a third of the true pitch. It exists to show why YIN picks the first dip.
    """
    tau_min, tau_max = int(np.floor(sample_rate / fmax)), int(np.ceil(sample_rate / fmin))
    normalized = cumulative_mean_normalized_difference(difference_function(frame_signal(samples, frame_length, hop_length), tau_max))
    f0 = np.empty(len(normalized))
    for t, curve in enumerate(normalized):
        tau = tau_min + int(np.argmin(curve[tau_min : tau_max + 1]))
        f0[t] = sample_rate / (tau + parabolic_offset(curve, tau))
    return f0


def spectral_peak_pitch(frame: np.ndarray, sample_rate: int) -> float:
    """A naive guess: the frequency of the loudest bin. It finds the strongest harmonic, not the pitch."""
    frame = np.asarray(frame, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(len(frame)) / len(frame))
    magnitude = np.abs(np.fft.rfft(frame * window))
    peak = int(np.argmax(magnitude[1:])) + 1
    return (peak + parabolic_offset(magnitude, peak)) * sample_rate / len(frame)


def cents(estimate_hz, reference_hz) -> np.ndarray:
    """Pitch difference in cents: 1200 * log2(estimate / reference). 100 cents is one semitone."""
    return 1_200.0 * np.log2(np.asarray(estimate_hz, dtype=np.float64) / np.asarray(reference_hz, dtype=np.float64))


def gross_pitch_error(estimate_hz: np.ndarray, reference_hz: np.ndarray) -> float:
    """Fraction of frames voiced in both that are off by more than 20 percent (the usual GPE definition)."""
    both = np.isfinite(estimate_hz) & np.isfinite(reference_hz)
    if not np.any(both):
        return float("nan")
    return float(np.mean(np.abs(estimate_hz[both] / reference_hz[both] - 1.0) > 0.2))


def intonation_utterance(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """A rising voiced glide, a voiceless hiss, a falling glide, then silence.

    Returns the waveform and the true F0 for every sample (NaN where unvoiced).
    """
    contours = [glide(100.0, 180.0, sample_rate, 0.6), None, glide(170.0, 110.0, sample_rate, 0.5), None]
    lengths = [0.6, 0.15, 0.5, 0.15]
    pieces, truth = [], []
    for contour, seconds in zip(contours, lengths):
        if contour is None:
            n = round(sample_rate * seconds)
            pieces.append(hiss(sample_rate, seconds) if len(pieces) == 1 else np.zeros(n))
            truth.append(np.full(n, np.nan))
        else:
            pieces.append(harmonic_signal(contour, sample_rate))
            truth.append(contour)
    return np.concatenate(pieces), np.concatenate(truth)


def evaluate_track(
    f0: np.ndarray,
    voiced: np.ndarray,
    truth_per_sample: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> dict[str, float]:
    """Score a frame-wise F0 track against a per-sample reference.

    Frames whose window straddles a voiced/unvoiced boundary are skipped, since neither answer is wrong there.
    """
    centres = np.minimum(np.arange(len(f0)) * hop_length, len(truth_per_sample) - 1)
    truth = truth_per_sample[centres]
    is_voiced = np.isfinite(truth_per_sample)
    clean = np.array([
        is_voiced[max(0, c - frame_length // 2): c + frame_length // 2].all()
        or not is_voiced[max(0, c - frame_length // 2): c + frame_length // 2].any()
        for c in centres
    ])
    truth_voiced = np.isfinite(truth)
    both = clean & truth_voiced & voiced
    return {
        "frames": float(len(f0)),
        "scored": float(clean.sum()),
        "voicing_accuracy": float(np.mean(voiced[clean] == truth_voiced[clean])),
        "gross_pitch_error": gross_pitch_error(f0[clean], truth[clean]),
        "median_cents": float(np.median(np.abs(cents(f0[both], truth[both])))) if both.any() else float("nan"),
    }


def main() -> None:
    sample_rate, frame_length, hop_length = 16_000, 640, 160
    print(f"Frame {frame_length} samples ({1_000 * frame_length / sample_rate:.0f} ms), "
          f"lag search {sample_rate // 400}-{int(np.ceil(sample_rate / 60))} samples for 60-400 Hz")

    vowel = harmonic_signal(constant_pitch(150.0, sample_rate, 0.3), sample_rate)
    frame = frame_signal(vowel, frame_length, hop_length)[15]
    print("\nOne frame of /a/ at 150 Hz:")
    print(f"  loudest spectral peak: {spectral_peak_pitch(frame, sample_rate):7.1f} Hz  <- a harmonic near the first formant")
    print(f"  autocorrelation:       {autocorrelation_pitch(vowel, sample_rate)[15]:7.1f} Hz")
    print(f"  YIN:                   {yin(vowel, sample_rate)[0][15]:7.1f} Hz")

    high = harmonic_signal(constant_pitch(300.0, sample_rate, 0.3), sample_rate)
    print("\nA steady 300 Hz voice, where d' dips at T, 2T, and 3T:")
    print(f"  deepest dip of d':     {np.median(deepest_dip_pitch(high, sample_rate)[5:-5]):7.1f} Hz  <- picked 3T")
    print(f"  YIN (first dip < 0.1): {np.nanmedian(yin(high, sample_rate)[0][5:-5]):7.1f} Hz")

    missing = harmonic_signal(constant_pitch(200.0, sample_rate, 0.3), sample_rate, harmonics=tuple(range(2, 11)))
    print(f"\nHarmonics 2-10 of 200 Hz, no energy at 200 Hz: YIN says {np.nanmedian(yin(missing, sample_rate)[0]):.1f} Hz")

    utterance, truth = intonation_utterance(sample_rate)
    print("\nIntonation utterance (rising glide, hiss, falling glide, silence):")
    print(f"  {'frame':>7} {'threshold':>9} {'voicing ok':>10} {'GPE':>6} {'median error':>13}")
    for length, threshold in ((1_024, 0.1), (640, 0.1), (640, 0.3)):
        f0, voiced, _ = yin(utterance, sample_rate, frame_length=length, hop_length=hop_length, threshold=threshold)
        scores = evaluate_track(f0, voiced, truth, length, hop_length)
        print(f"  {1_000 * length / sample_rate:5.0f}ms {threshold:9.1f} {scores['voicing_accuracy']:10.1%} "
              f"{scores['gross_pitch_error']:6.1%} {scores['median_cents']:8.1f} cents")


if __name__ == "__main__":
    main()
