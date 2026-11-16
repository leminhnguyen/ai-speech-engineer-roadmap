"""Lesson 04: single-channel noise suppression for call audio, built on the STFT.

Implements, with NumPy only: synthetic call-center noises, a perfect-reconstruction STFT,
minimum-statistics noise tracking, power spectral subtraction, a Wiener filter with the
decision-directed a priori SNR, and the metrics that expose the trade-off between removing noise
and damaging speech (SNR, segmental SNR, speech distortion, noise reduction, musical noise).

Run: python3 code/main.py
"""

from __future__ import annotations

import numpy as np

SR = 8000
N_FFT = 256  # 32 ms at 8 kHz
HOP = 128

# --------------------------------------------------------------------------- signals


def synthetic_speech(seconds: float = 3.0, sample_rate: int = SR, seed: int = 0) -> np.ndarray:
    """Alternating vowels (gliding F0, moving formants), fricatives and pauses."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    out = np.zeros(n)
    t0 = int(0.15 * sample_rate)
    vowels = ((700, 1200, 2500), (300, 2200, 2900), (500, 900, 2400), (400, 1900, 2600))
    while t0 < n - int(0.3 * sample_rate):
        kind = rng.choice(["vowel", "vowel", "fricative"])
        length = int(rng.uniform(0.12, 0.3) * sample_rate)
        t = np.arange(length) / sample_rate
        if kind == "vowel":
            f0 = rng.uniform(110, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(1, 3) * t))
            phase = 2 * np.pi * np.cumsum(f0) / sample_rate
            formants = vowels[rng.integers(len(vowels))]
            seg = np.zeros(length)
            for h in range(1, 30):
                f = h * f0.mean()
                if f > sample_rate / 2 - 200:
                    break
                gain = sum(np.exp(-0.5 * ((f - fc) / 120) ** 2) for fc in formants) + 0.03
                seg += gain / h ** 0.5 * np.sin(h * phase)
        else:
            noise = rng.standard_normal(length)
            spec = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(length, 1 / sample_rate)
            spec *= np.exp(-0.5 * ((freqs - 3300) / 500) ** 2)
            seg = np.fft.irfft(spec, n=length) * 0.4
        seg *= np.hanning(length) ** 0.5
        out[t0:t0 + length] += seg / (np.max(np.abs(seg)) + 1e-12) * rng.uniform(0.3, 0.6)
        t0 += length + int(rng.choice([0.02, 0.05, 0.25]) * sample_rate)  # some real pauses
    return out


def colored_noise(n: int, exponent: float, seed: int = 0) -> np.ndarray:
    """Noise with power spectrum ~ 1 / f^exponent (0 white, 1 pink, 2 brown)."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(spec / f ** (exponent / 2), n=n)
    return x / np.std(x)


def make_noise(kind: str, n: int, sample_rate: int = SR, seed: int = 1) -> np.ndarray:
    """Four noises a contact-center caller brings along, all unit variance."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    if kind == "fan":  # stationary: hum harmonics + pink noise
        x = colored_noise(n, 1.0, seed) + sum(0.6 / k * np.sin(2 * np.pi * 50 * k * t) for k in range(2, 8))
    elif kind == "street":  # slowly varying brown/pink noise with a passing vehicle swell
        swell = 1 + 1.5 * np.exp(-0.5 * ((t - t[-1] * 0.6) / 0.5) ** 2)
        x = (0.7 * colored_noise(n, 1.6, seed) + 0.3 * colored_noise(n, 0.5, seed + 1)) * swell
    elif kind == "keyboard":  # impulsive clicks on a quiet floor
        x = 0.05 * rng.standard_normal(n)
        for start in rng.choice(n - 200, size=int(n / sample_rate * 7), replace=False):
            x[start:start + 120] += rng.standard_normal(120) * np.exp(-np.arange(120) / 15) * 4
    elif kind == "babble":  # several distant talkers: non-stationary and speech-shaped
        x = sum(synthetic_speech(n / sample_rate, sample_rate, seed=seed + 10 + k) for k in range(5))
    else:
        raise ValueError(f"unknown noise {kind!r}")
    return x / np.std(x)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale ``noise`` so the mixture has the requested SNR over the whole signal."""
    noise = noise[: len(clean)]
    scale = np.sqrt(np.sum(clean ** 2) / (np.sum(noise ** 2) * 10 ** (snr_db / 10)))
    return clean + scale * noise, scale * noise


# --------------------------------------------------------------------------- STFT


def _window(n_fft: int) -> np.ndarray:
    return np.sqrt(np.hanning(n_fft + 1)[:-1])  # periodic sqrt-Hann: analysis x synthesis = Hann


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Frames x bins complex spectrogram, with the signal padded so every sample is covered twice."""
    x = np.concatenate([np.zeros(n_fft - hop), np.asarray(x, dtype=float), np.zeros(n_fft)])
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(x[idx] * _window(n_fft), axis=1)


def istft(spec: np.ndarray, length: int, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Weighted overlap-add inverse; exact for an unmodified ``stft`` at 50% overlap."""
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * _window(n_fft)
    out = np.zeros(hop * (len(frames) - 1) + n_fft)
    for i, frame in enumerate(frames):
        out[i * hop:i * hop + n_fft] += frame
    return out[n_fft - hop:n_fft - hop + length]


# --------------------------------------------------------------------------- noise estimation


def minimum_statistics(power: np.ndarray, window_frames: int = 94, alpha: float = 0.85,
                       bias: float = 1.6) -> np.ndarray:
    """Simplified minimum-statistics noise PSD tracker (after Martin, 2001).

    Smooth each bin's periodogram over time, then take the minimum over the last ``window_frames``
    (1.5 s at 16 ms hop). Speech rarely occupies a bin for that long, so the minimum follows the
    noise floor. The minimum of a noisy estimate is biased low, hence ``bias``. Needs no VAD.
    """
    smoothed = np.empty_like(power)
    acc = power[0]
    for i, frame in enumerate(power):
        acc = alpha * acc + (1 - alpha) * frame
        smoothed[i] = acc
    noise = np.empty_like(power)
    for i in range(len(power)):
        noise[i] = smoothed[max(0, i - window_frames + 1):i + 1].min(axis=0)
    return bias * noise


# --------------------------------------------------------------------------- suppression rules


def spectral_subtraction_gain(power: np.ndarray, noise_psd: np.ndarray, over: float = 1.0,
                              floor: float = 0.0) -> np.ndarray:
    """Power subtraction: |S|^2 = |X|^2 - over * N, as a gain on |X|, with a spectral floor."""
    gain_sq = 1.0 - over * noise_psd / np.maximum(power, 1e-12)
    return np.sqrt(np.maximum(gain_sq, floor ** 2))


def wiener_dd_gain(power: np.ndarray, noise_psd: np.ndarray, alpha: float = 0.98,
                   floor: float = 0.1, over: float = 1.0) -> np.ndarray:
    """Wiener gain xi / (1 + xi) with Ephraim-Malah's decision-directed a priori SNR ``xi``.

    xi mixes the previous frame's *cleaned* power (smooth, stable) with this frame's instantaneous
    estimate max(gamma - 1, 0) (fast, noisy). alpha near 1 suppresses musical noise. ``over`` > 1
    over-estimates the noise: more suppression, more speech damage.
    """
    noise_psd = over * noise_psd
    gains = np.empty_like(power)
    prev_clean = np.zeros(power.shape[1])
    for i in range(len(power)):
        gamma = power[i] / np.maximum(noise_psd[i], 1e-12)  # a posteriori SNR
        xi = alpha * prev_clean / np.maximum(noise_psd[i], 1e-12) + (1 - alpha) * np.maximum(gamma - 1, 0)
        g = np.maximum(xi / (1 + xi), floor)
        gains[i] = g
        prev_clean = (g ** 2) * power[i]
    return gains


def enhance(noisy: np.ndarray, method: str = "wiener", noise_psd: np.ndarray | None = None,
            return_gain: bool = False, **kwargs):
    """Run one suppressor end to end. Returns the waveform (and the T x F gain if asked)."""
    spec = stft(noisy)
    power = np.abs(spec) ** 2
    noise_psd = minimum_statistics(power) if noise_psd is None else noise_psd
    if method == "subtraction":
        gain = spectral_subtraction_gain(power, noise_psd, **kwargs)
    elif method == "wiener":
        gain = wiener_dd_gain(power, noise_psd, **kwargs)
    else:
        raise ValueError(f"unknown method {method!r}")
    out = istft(gain * spec, len(noisy))
    return (out, gain) if return_gain else out


# --------------------------------------------------------------------------- metrics


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    err = reference - test
    return float(10 * np.log10(np.sum(reference ** 2) / max(np.sum(err ** 2), 1e-20)))


def segmental_snr_db(reference: np.ndarray, test: np.ndarray, frame: int = 256,
                     floor: float = -10.0, ceil: float = 35.0) -> float:
    values = []
    for start in range(0, len(reference) - frame + 1, frame):
        ref = reference[start:start + frame]
        if np.sum(ref ** 2) < 1e-6 * frame:  # skip silent frames
            continue
        err = ref - test[start:start + frame]
        values.append(np.clip(10 * np.log10(np.sum(ref ** 2) / max(np.sum(err ** 2), 1e-20)), floor, ceil))
    return float(np.mean(values))


def distortion_and_reduction(clean: np.ndarray, noise: np.ndarray, gain: np.ndarray) -> tuple[float, float]:
    """Apply the *same* T x F gain to speech and noise separately (the gain is linear per bin).

    Returns (speech distortion in dB, noise reduction in dB). Distortion = energy of what the
    gain removed or changed in the speech, relative to the speech; lower is better.
    """
    s_hat = istft(gain * stft(clean), len(clean))
    n_hat = istft(gain * stft(noise), len(noise))
    distortion = 10 * np.log10(np.sum((clean - s_hat) ** 2) / np.sum(clean ** 2))
    reduction = 10 * np.log10(np.sum(noise ** 2) / max(np.sum(n_hat ** 2), 1e-20))
    return float(distortion), float(reduction)


def speech_absent_frames(clean: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
    """STFT frames whose clean-speech energy is ``threshold_db`` below the loudest frame."""
    energy = np.sum(np.abs(stft(clean)) ** 2, axis=1)
    return 10 * np.log10(energy / energy.max() + 1e-30) < threshold_db


def isolated_peak_fraction(power: np.ndarray, frames: np.ndarray, margin_db: float = 10.0) -> float:
    """Share of time-frequency bins, within ``frames``, that stand ``margin_db`` above their neighbours.

    Musical noise is exactly this: lone spectral peaks that survive suppression and twinkle on and
    off like tones. Each bin is compared with the median of its 5 x 5 time-frequency neighbourhood.
    """
    level = 10 * np.log10(np.asarray(power) + 1e-12)
    padded = np.pad(level, 2, mode="edge")
    neighbourhood = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    local_median = np.median(neighbourhood, axis=(-2, -1))
    return float(((level - local_median) > margin_db)[frames].mean())


def musical_noise(noise: np.ndarray, gain: np.ndarray, frames: np.ndarray) -> float:
    """Isolated-peak fraction of the residual noise ``gain * noise`` during speech pauses."""
    return isolated_peak_fraction(np.abs(gain * stft(noise)) ** 2, frames)


# --------------------------------------------------------------------------- demo


def main() -> None:
    clean = synthetic_speech()
    print(f"Synthetic speech: {len(clean) / SR:.1f} s at {SR} Hz")
    noise = make_noise("fan", len(clean))
    noisy, scaled = mix_at_snr(clean, noise, 5.0)
    print(f"Input: SNR {snr_db(clean, noisy):.1f} dB, segmental SNR {segmental_snr_db(clean, noisy):.1f} dB")

    spec = stft(clean)
    print(f"STFT round trip error: {np.max(np.abs(istft(spec, len(clean)) - clean)):.1e}")
    true_psd = np.abs(stft(scaled)) ** 2
    est = minimum_statistics(np.abs(stft(noisy)) ** 2)
    ratio_db = 10 * np.log10(est[100:].mean() / true_psd[100:].mean())
    print(f"Minimum statistics noise level vs truth: {ratio_db:+.1f} dB")

    pauses = speech_absent_frames(clean)
    print(f"{pauses.sum()} of {len(pauses)} STFT frames are speech pauses; isolated peaks in the raw noise: "
          f"{100 * isolated_peak_fraction(np.abs(stft(scaled)) ** 2, pauses):.1f}%")
    for name, method, kwargs in (("subtraction, no floor", "subtraction", {"over": 1.0, "floor": 0.0}),
                                 ("subtraction, floor 0.1", "subtraction", {"over": 2.0, "floor": 0.1}),
                                 ("Wiener + decision-directed", "wiener", {})):
        out, gain = enhance(noisy, method, return_gain=True, **kwargs)
        dist, red = distortion_and_reduction(clean, scaled, gain)
        print(f"{name:<27} SNR {snr_db(clean, out):5.1f} dB  segSNR {segmental_snr_db(clean, out):5.1f} dB  "
              f"noise -{red:4.1f} dB  speech distortion {dist:6.1f} dB  isolated peaks {100 * musical_noise(scaled, gain, pauses):4.1f}%")

    print("SNR improvement of the Wiener filter by noise type (input 5 dB):")
    for kind in ("fan", "street", "keyboard", "babble"):
        noisy_k, _ = mix_at_snr(clean, make_noise(kind, len(clean)), 5.0)
        print(f"  {kind:<9} {snr_db(clean, enhance(noisy_k)) - snr_db(clean, noisy_k):+5.1f} dB")


if __name__ == "__main__":
    main()
