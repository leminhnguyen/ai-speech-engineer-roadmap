"""Lesson 06: is anyone speaking? A frame-level VAD with Silero-style hysteresis and timers.

Builds frame features (log energy over an adaptive noise floor, spectral flatness, zero-crossing
rate), turns them into a speech probability, and runs the same state machine and parameters that
LiveKit's Silero VAD plugin exposes: activation/deactivation thresholds, minimum speech and silence
durations, and prefix padding. Then measures it like a product: frame F1, false alarms per noise
type and SNR, onset clipping, and how quickly speech start and end are reported.

Run: python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

SAMPLE_RATE = 8000
FRAME = 256  # 32 ms at 8 kHz, the same window Silero uses


# --------------------------------------------------------------------------- signals

def speech_like(seconds: float, sample_rate: int = SAMPLE_RATE, f0: float = 140.0, seed: int = 0,
                level: float = 0.3) -> np.ndarray:
    """Syllable-like voiced bursts (harmonics + breath noise) with short inter-syllable gaps."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    out = np.zeros(n)
    start = 0
    while start < n:
        length = int(rng.uniform(0.12, 0.28) * sample_rate)
        gap = int(rng.uniform(0.02, 0.08) * sample_rate)
        end = min(start + length, n)
        t = np.arange(end - start) / sample_rate
        pitch = f0 * (1 + 0.15 * rng.uniform(-1, 1)) * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
        syllable = np.zeros(end - start)
        formant = rng.uniform(500, 900)
        for k in range(1, int(3400 / (f0 * 1.2))):
            gain = np.exp(-0.5 * ((k * f0 - formant) / 400) ** 2) + 0.3 / k
            syllable += gain * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        syllable += 0.3 * np.std(syllable) * rng.standard_normal(end - start)
        out[start:end] = syllable * np.hanning(end - start) ** 0.5
        start = end + gap
    return level * out / (np.max(np.abs(out)) + 1e-12)


def synthetic_dialog(seconds: float = 30.0, seed: int = 0, sample_rate: int = SAMPLE_RATE) -> dict:
    """Caller utterances made of phrases with short within-utterance pauses, separated by silence.

    Returns the clean signal, a per-sample ``speech`` mask (inside a phrase), and the utterance
    boundaries in seconds (an utterance spans its phrases and the pauses between them).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    signal = np.zeros(n)
    speech = np.zeros(n, dtype=bool)
    utterances = []
    t = rng.uniform(0.6, 1.0)
    index = 0
    while True:
        phrases = int(rng.integers(1, 4))
        start = t
        for p in range(phrases):
            length = rng.uniform(0.5, 1.4)
            a, b = int(t * sample_rate), int((t + length) * sample_rate)
            if b >= n:
                break
            signal[a:b] = speech_like((b - a) / sample_rate + 0.01, sample_rate, f0=rng.uniform(100, 220), seed=seed * 100 + index,
                                      level=rng.uniform(0.2, 0.4))[: b - a]
            speech[a:b] = True
            index += 1
            t += length + (rng.uniform(0.15, 0.4) if p < phrases - 1 else 0.0)
        else:
            utterances.append((start, t))
            t += rng.uniform(0.9, 1.8)
            continue
        break
    return {"signal": signal, "speech": speech, "utterances": utterances}


def noise(kind: str, n: int, seed: int = 1, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Call-center noises: ``white`` (line hiss), ``hum`` (fan rumble plus a tone), ``babble`` (other talkers)."""
    rng = np.random.default_rng(seed)
    if kind == "white":
        out = rng.standard_normal(n)
    elif kind == "hum":
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
        out = np.fft.irfft(spectrum / np.sqrt(np.maximum(freqs, 100.0)), n=n)  # pink: fan, air conditioning
        t = np.arange(n) / sample_rate
        out = out / np.std(out) + 0.7 * np.sin(2 * np.pi * 300 * t)  # plus a tonal mains/engine component
    elif kind == "babble":
        out = np.zeros(n)
        for talker in range(6):
            voice = speech_like(n / sample_rate + 1, sample_rate, f0=100 + 25 * talker, seed=seed * 50 + talker)
            shift = int(rng.integers(0, sample_rate))
            out += voice[shift:shift + n]
    else:
        raise ValueError(f"unknown noise kind: {kind}")
    return out / np.std(out)


def mix_at_snr(clean: np.ndarray, noise_signal: np.ndarray, snr_db: float, speech: np.ndarray) -> np.ndarray:
    """Scale noise so that speech-active samples sit ``snr_db`` above it."""
    speech_power = np.mean(clean[speech] ** 2)
    noise_power = np.mean(noise_signal ** 2)
    return clean + noise_signal * np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))


# --------------------------------------------------------------------------- features

def frame_signal(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    usable = len(x) // frame * frame
    return x[:usable].reshape(-1, frame)


def log_energy_db(frames: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)


def spectral_flatness(frames: np.ndarray) -> np.ndarray:
    """Geometric over arithmetic mean of the power spectrum: ~1 for white noise, small for voiced speech."""
    power = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1)) ** 2 + 1e-12
    return np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=1)), axis=1)


def track_noise_floor(energy_db: np.ndarray, rise_db: float = 0.05, init_frames: int = 10) -> np.ndarray:
    """Minimum tracking: follow the energy down immediately, creep up slowly (``rise_db`` per frame).

    Speech is loud and intermittent, noise is the level the signal keeps returning to.
    """
    floor = np.empty_like(energy_db)
    level = float(np.min(energy_db[:init_frames]))
    for i, e in enumerate(energy_db):
        level = min(e, level + rise_db)
        floor[i] = level
    return floor


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def track_noise_flatness(log_flatness: np.ndarray, near_floor: np.ndarray, smoothing: float = 0.95,
                         init_frames: int = 10) -> np.ndarray:
    """Running average of log spectral flatness over frames that sit near the noise floor.

    Hiss is flat and a fan is not, so "how much flatter than the background" is the useful cue,
    not flatness on its own.
    """
    baseline = np.empty_like(log_flatness)
    level = float(np.mean(log_flatness[:init_frames]))
    for i, value in enumerate(log_flatness):
        if near_floor[i]:
            level = smoothing * level + (1 - smoothing) * value
        baseline[i] = level
    return baseline


def speech_probability(x: np.ndarray, frame: int = FRAME, snr_mid_db: float = 8.0, snr_scale_db: float = 3.0,
                       contrast_mid: float = 0.5, contrast_scale: float = 0.5) -> np.ndarray:
    """A hand-built stand-in for Silero's network output: one probability per 32 ms frame.

    Two cues relative to the background are added in the logit domain, so either can compensate
    for the other: loudness above the tracked noise floor, and spectral structure (lower flatness)
    compared with the tracked flatness of that background.
    """
    frames = frame_signal(x, frame)
    energy = log_energy_db(frames)
    snr = energy - track_noise_floor(energy)
    log_flat = np.log(spectral_flatness(frames))
    contrast = track_noise_flatness(log_flat, near_floor=snr < 3.0) - log_flat
    loudness = (snr - snr_mid_db) / snr_scale_db
    structure = np.clip((contrast - contrast_mid) / contrast_scale, -3.0, 3.0)
    return sigmoid(loudness + structure)


# --------------------------------------------------------------------------- the state machine

@dataclass
class VADParams:
    """The knobs of ``silero.VAD.load(...)`` in LiveKit, with the same names and defaults."""

    min_speech_duration: float = 0.05
    min_silence_duration: float = 0.55
    prefix_padding_duration: float = 0.5
    activation_threshold: float = 0.5
    deactivation_threshold: float | None = None
    frame_duration: float = FRAME / SAMPLE_RATE

    def deactivation(self) -> float:
        if self.deactivation_threshold is not None:
            return self.deactivation_threshold
        return max(self.activation_threshold - 0.15, 0.01)


@dataclass
class VADResult:
    segments: list = field(default_factory=list)  # (start_s, end_s) of speech chunks, start includes padding
    events: list = field(default_factory=list)    # (time_s, "START_OF_SPEECH" | "END_OF_SPEECH")
    states: list = field(default_factory=list)    # state name per frame


def run_vad(probs: np.ndarray, params: VADParams | None = None) -> VADResult:
    """SILENCE -> MAYBE_SPEECH -> SPEECH -> MAYBE_SILENCE -> SILENCE, with hysteresis and timers.

    START_OF_SPEECH fires once ``min_speech_duration`` of frames stay above the activation
    threshold; END_OF_SPEECH fires after ``min_silence_duration`` below the deactivation threshold.
    Event times are when the decision is *made*; segment boundaries are where speech *was*.
    """
    p = params or VADParams()
    dur = p.frame_duration
    need_speech = max(1, math.ceil(p.min_speech_duration / dur - 1e-9))
    need_silence = max(1, math.ceil(p.min_silence_duration / dur - 1e-9))
    pad = p.prefix_padding_duration
    result = VADResult()
    state, count, onset, offset = "SILENCE", 0, 0, 0
    for i, prob in enumerate(probs):
        if state == "SILENCE" and prob >= p.activation_threshold:
            state, count, onset = "MAYBE_SPEECH", 0, i
        if state == "MAYBE_SPEECH":
            if prob >= p.activation_threshold:
                count += 1
                if count >= need_speech:
                    state = "SPEECH"
                    result.events.append(((i + 1) * dur, "START_OF_SPEECH"))
            else:
                state = "SILENCE"
        elif state == "SPEECH" and prob < p.deactivation():
            state, count, offset = "MAYBE_SILENCE", 0, i
        if state == "MAYBE_SILENCE":
            if prob < p.deactivation():
                count += 1
                if count >= need_silence:
                    state = "SILENCE"
                    result.events.append(((i + 1) * dur, "END_OF_SPEECH"))
                    result.segments.append((max(onset * dur - pad, 0.0), offset * dur))
            else:
                state = "SPEECH"
        result.states.append(state)
    if state in ("SPEECH", "MAYBE_SILENCE"):
        end = (offset if state == "MAYBE_SILENCE" else len(probs)) * dur
        result.segments.append((max(onset * dur - pad, 0.0), end))
    return result


# --------------------------------------------------------------------------- evaluation

def frames_from_samples(mask: np.ndarray, frame: int = FRAME) -> np.ndarray:
    """A frame is speech if most of its samples are."""
    return frame_signal(mask.astype(np.float64), frame).mean(axis=1) > 0.5


def frames_from_segments(segments: list, n_frames: int, frame_duration: float = FRAME / SAMPLE_RATE,
                         include_padding: bool = False, pad: float = 0.0) -> np.ndarray:
    centers = (np.arange(n_frames) + 0.5) * frame_duration
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in segments:
        mask |= (centers >= (start if include_padding else start + pad)) & (centers < end)
    return mask


def frame_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    false_alarm = fp / max(int(np.sum(~truth)), 1)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": false_alarm}


def onset_clipping_ms(segments: list, utterances: list) -> float:
    """Mean speech lost at the start of each utterance (0 if the padded segment starts early enough)."""
    lost = []
    for start, end in utterances:
        overlapping = [s for s, e in segments if s < end and e > start]
        if overlapping:
            lost.append(max(0.0, min(overlapping) - start) * 1000)
    return float(np.mean(lost)) if lost else float("nan")


def event_delays(events: list, utterances: list, wait: float = 3.0) -> dict:
    """How long after each true utterance start/end the VAD *reported* START/END_OF_SPEECH.

    Utterances the VAD never announced (or announced before they began, because it was already
    triggered by noise) are skipped, so check ``segments`` and false alarms alongside.
    """
    starts = [t for t, kind in events if kind == "START_OF_SPEECH"]
    ends = [t for t, kind in events if kind == "END_OF_SPEECH"]
    start_delay = [min((t - a for t in starts if a <= t <= b), default=np.nan) for a, b in utterances]
    end_delay = [min((t - b for t in ends if b <= t <= b + wait), default=np.nan) for _, b in utterances]
    nan_mean = lambda v: float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan")
    return {"start": nan_mean(start_delay), "end": nan_mean(end_delay)}


def evaluate(noisy: np.ndarray, dialog: dict, params: VADParams | None = None) -> dict:
    params = params or VADParams()
    probs = speech_probability(noisy)
    result = run_vad(probs, params)
    # An utterance (phrases plus the short pauses inside it) is the unit an agent should see as one turn.
    truth = frames_from_segments(dialog["utterances"], len(probs), params.frame_duration)
    predicted = frames_from_segments(result.segments, len(probs), params.frame_duration,
                                     pad=params.prefix_padding_duration)
    metrics = frame_metrics(predicted, truth)
    metrics["segments"] = len(result.segments)
    metrics["utterances"] = len(dialog["utterances"])
    metrics["onset_clipping_ms"] = onset_clipping_ms(result.segments, dialog["utterances"])
    metrics.update({f"{k}_delay_s": v for k, v in event_delays(result.events, dialog["utterances"]).items()})
    return metrics


def main() -> None:
    dialog = synthetic_dialog()
    print(f"Synthetic call: {len(dialog['utterances'])} caller utterances in {len(dialog['signal']) / SAMPLE_RATE:.0f} s")
    params = VADParams()
    print(f"Silero-style defaults: activation {params.activation_threshold}, deactivation {params.deactivation():.2f}, "
          f"min speech {params.min_speech_duration} s, min silence {params.min_silence_duration} s, "
          f"prefix padding {params.prefix_padding_duration} s")
    for kind in ("white", "hum", "babble"):
        for snr in (20, 5):
            noisy = mix_at_snr(dialog["signal"], noise(kind, len(dialog["signal"])), snr, dialog["speech"])
            m = evaluate(noisy, dialog)
            print(f"{kind:>6} @ {snr:>2} dB SNR: F1 {m['f1']:.2f}, false alarms {100 * m['false_alarm']:4.1f}%, "
                  f"{m['segments']} segments for {m['utterances']} utterances, "
                  f"START after {1000 * m['start_delay_s']:.0f} ms, END after {1000 * m['end_delay_s']:.0f} ms")
    noisy = mix_at_snr(dialog["signal"], noise("white", len(dialog["signal"])), 20, dialog["speech"])
    for pad in (0.0, 0.5):
        m = evaluate(noisy, dialog, VADParams(prefix_padding_duration=pad))
        print(f"prefix_padding_duration={pad}: {m['onset_clipping_ms']:.0f} ms of each utterance onset lost")


if __name__ == "__main__":
    main()
