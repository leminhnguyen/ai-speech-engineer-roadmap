"""Lesson 05: when the agent hears itself. An NLMS acoustic/line echo canceller from scratch.

Builds an echo scenario (the agent's TTS played into a phone, coming back through an echo path),
an NLMS adaptive filter that learns that path, ERLE measurement, Geigel double-talk detection,
echo-path change, a residual echo suppressor, and a check of how many frames an energy VAD would
wrongly call a barge-in.

Run: python3 code/main.py
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 8000


# --------------------------------------------------------------------------- signals

def speech_like(seconds: float, sample_rate: int = SAMPLE_RATE, f0: float = 140.0, seed: int = 0,
                level: float = 0.3, breathiness: float = 0.5) -> np.ndarray:
    """Syllable-like bursts of harmonic sound with a drifting pitch and short gaps.

    Stands in for TTS or caller speech: non-stationary, coloured, with pauses, which is exactly
    what makes echo cancellation harder than with white noise.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    out = np.zeros(n)
    start = 0
    while start < n:
        length = int(rng.uniform(0.12, 0.28) * sample_rate)
        gap = int(rng.uniform(0.03, 0.12) * sample_rate)
        end = min(start + length, n)
        t = np.arange(end - start) / sample_rate
        pitch = f0 * (1 + 0.15 * rng.uniform(-1, 1)) * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
        syllable = np.zeros(end - start)
        formant = rng.uniform(500, 900)
        for k in range(1, int(3400 / (f0 * 1.2))):
            gain = np.exp(-0.5 * ((k * f0 - formant) / 400) ** 2) + 0.3 / k
            syllable += gain * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        syllable += breathiness * np.std(syllable) * rng.standard_normal(end - start)
        syllable *= np.hanning(end - start)
        out[start:end] = syllable
        start = end + gap
    return level * out / (np.max(np.abs(out)) + 1e-12)


def echo_path(taps: int = 256, delay: int = 24, decay_ms: float = 6.0, sample_rate: int = SAMPLE_RATE,
              erl_db: float = 10.0, seed: int = 1) -> np.ndarray:
    """An echo impulse response: a bulk delay, then an exponentially decaying random tail.

    ``erl_db`` is the echo return loss: how much quieter the echo is than the far-end signal.
    Line echo from a 2-to-4-wire hybrid typically has 6-20 dB ERL; a speakerphone can be near 0 dB.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(taps)
    tail = np.arange(taps - delay)
    h[delay:] = rng.standard_normal(taps - delay) * np.exp(-tail / (decay_ms * sample_rate / 1000))
    return h / np.sqrt(np.sum(h ** 2)) * 10 ** (-erl_db / 20)


def make_echo(far_end: np.ndarray, path: np.ndarray) -> np.ndarray:
    return np.convolve(far_end, path)[: len(far_end)]


# --------------------------------------------------------------------------- canceller

def geigel_double_talk(far_end: np.ndarray, mic: np.ndarray, window: int = 256, threshold: float = 0.5,
                       hangover: int = 240, far_floor: float = 0.01) -> np.ndarray:
    """Geigel detector: near-end talk if |mic[n]| > threshold * max |far_end| over the last ``window``.

    Assumes the echo is at least ``-20*log10(threshold)`` dB below the far end (6 dB for 0.5).
    ``hangover`` keeps the flag on for a while so adaptation is not resumed mid-word. When the far
    end is silent (``max < far_floor``) there is no echo to protect, so the test is skipped.
    """
    padded = np.concatenate([np.zeros(window - 1), np.abs(far_end)])
    far_max = np.lib.stride_tricks.sliding_window_view(padded, window).max(axis=1)
    raw = (np.abs(mic) > threshold * far_max) & (far_max > far_floor)
    flag = np.zeros(len(mic), dtype=bool)
    remaining = 0
    for n, hit in enumerate(raw):
        remaining = hangover if hit else max(remaining - 1, 0)
        flag[n] = remaining > 0
    return flag


def nlms_echo_canceller(far_end: np.ndarray, mic: np.ndarray, taps: int = 256, mu: float = 0.5,
                        eps: float = 0.1, freeze: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Normalized LMS: estimate the echo from the far-end reference and subtract it.

    For each sample: y_hat = w . x, e = mic - y_hat, w += mu * e * x / (x . x + eps).
    ``freeze[n] = True`` stops adaptation (used during double talk). Returns (residual, weights).
    ``eps`` regularizes the normalization: at syllable onsets ``x . x`` is tiny, and without a floor
    comparable to the noise energy over ``taps`` samples, noise alone would kick the filter around.
    """
    w = np.zeros(taps)
    padded = np.concatenate([np.zeros(taps - 1), far_end])
    residual = np.zeros(len(mic))
    for n in range(len(mic)):
        x = padded[n:n + taps][::-1]
        e = mic[n] - w @ x
        residual[n] = e
        if freeze is None or not freeze[n]:
            w += mu * e * x / (x @ x + eps)
    return residual, w


def residual_echo_suppressor(residual: np.ndarray, far_end: np.ndarray, double_talk: np.ndarray,
                             frame: int = 160, far_threshold: float = 0.003,
                             attenuation_db: float = -30.0) -> np.ndarray:
    """Non-linear processor (NLP): attenuate frames where only the far end is active.

    Removes the residual the linear filter cannot, at the price of chopping the caller whenever the
    double-talk detector misses: the classic half-duplex feel of a speakerphone.
    """
    out = residual.copy()
    gain = 10 ** (attenuation_db / 20)
    for start in range(0, len(residual), frame):
        sl = slice(start, start + frame)
        far_active = np.sqrt(np.mean(far_end[sl] ** 2)) > far_threshold
        if far_active and not double_talk[sl].any():
            out[sl] *= gain
    return out


# --------------------------------------------------------------------------- metrics

def frame_power(x: np.ndarray, frame: int) -> np.ndarray:
    usable = len(x) // frame * frame
    return np.mean(x[:usable].reshape(-1, frame) ** 2, axis=1)


def erle_db(echo: np.ndarray, residual_echo: np.ndarray, frame: int = 400,
            min_echo_db: float = -50.0) -> np.ndarray:
    """Echo return loss enhancement per frame: how much the canceller removed, in dB.

    Frames where the echo itself is weaker than ``min_echo_db`` (far end silent) are NaN: there is
    nothing to cancel there, so the ratio would only measure noise.
    """
    echo_power = frame_power(echo, frame)
    ratio = 10 * np.log10((echo_power + 1e-12) / (frame_power(residual_echo, frame) + 1e-12))
    return np.where(10 * np.log10(echo_power + 1e-12) > min_echo_db, ratio, np.nan)


def misalignment_db(true_path: np.ndarray, estimate: np.ndarray) -> float:
    """Normalized distance between the true echo path and the learned filter (lower is better)."""
    n = min(len(true_path), len(estimate))
    return float(10 * np.log10(np.sum((true_path[:n] - estimate[:n]) ** 2) / np.sum(true_path[:n] ** 2)))


def false_barge_in_frames(signal: np.ndarray, frame: int = 160, threshold_db: float = -40.0) -> int:
    """Frames an energy VAD would call 'user speaking' while the caller is actually silent."""
    level = 10 * np.log10(frame_power(signal, frame) + 1e-12)
    return int(np.sum(level > threshold_db))


# --------------------------------------------------------------------------- scenarios

def scenario(seconds: float = 4.0, erl_db: float = 10.0, double_talk: tuple[float, float] | None = None,
             path_change_at: float | None = None, noise_db: float = -60.0, seed: int = 0) -> dict:
    """Far-end TTS, its echo, optional caller speech (double talk) and an optional echo-path change."""
    rng = np.random.default_rng(seed + 100)
    n = int(seconds * SAMPLE_RATE)
    far = speech_like(seconds, f0=150.0, seed=seed)
    path_a = echo_path(erl_db=erl_db, seed=seed + 1)
    echo = make_echo(far, path_a)
    if path_change_at is not None:
        cut = int(path_change_at * SAMPLE_RATE)
        path_b = echo_path(delay=40, erl_db=erl_db, seed=seed + 2)
        echo[cut:] = make_echo(far, path_b)[cut:]
    near = np.zeros(n)
    if double_talk is not None:
        a, b = (int(t * SAMPLE_RATE) for t in double_talk)
        near[a:b] = speech_like((b - a) / SAMPLE_RATE, f0=110.0, seed=seed + 7, level=0.25)
    noise = rng.standard_normal(n) * 10 ** (noise_db / 20)
    return {"far": far, "echo": echo, "near": near, "noise": noise, "mic": echo + near + noise,
            "path": path_a}


def main() -> None:
    s = scenario(seconds=3.0)
    residual, w = nlms_echo_canceller(s["far"], s["mic"])
    erle = erle_db(s["echo"], residual - s["noise"])
    print(f"Echo return loss of the path: 10 dB; after 0.5 s the NLMS filter adds "
          f"{np.nanmedian(erle[10:]):.0f} dB of ERLE (median)")
    print(f"Filter misalignment after 3 s: {misalignment_db(s['path'], w):.1f} dB")
    print(f"Frames an energy VAD calls speech while the caller is silent: "
          f"{false_barge_in_frames(s['mic'])} before AEC, {false_barge_in_frames(residual[4000:])} after")

    dt = scenario(seconds=4.0, double_talk=(2.0, 3.0))
    flag = geigel_double_talk(dt["far"], dt["mic"])
    free, _ = nlms_echo_canceller(dt["far"], dt["mic"])
    held, _ = nlms_echo_canceller(dt["far"], dt["mic"], freeze=flag)
    after = slice(int(3.2 * SAMPLE_RATE), int(4.0 * SAMPLE_RATE))
    for name, res in (("no double-talk detector", free), ("Geigel freeze", held)):
        leftover = res[after] - dt["noise"][after]
        print(f"ERLE just after double talk, {name}: {np.nanmedian(erle_db(dt['echo'][after], leftover)):.1f} dB")


if __name__ == "__main__":
    main()
