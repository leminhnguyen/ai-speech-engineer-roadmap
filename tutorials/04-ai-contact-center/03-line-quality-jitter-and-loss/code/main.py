"""Lesson 03: what the network does to a call, and how the receiver hides it.

Simulates the impairments between a caller's handset and the agent with NumPy:
Gilbert-Elliott burst loss, delay jitter, fixed and adaptive jitter buffers, three packet loss
concealment (PLC) strategies, clipping and level problems, and the ITU-T G.107 E-model that turns
delay and loss into an estimated MOS.

Run: python3 code/main.py
"""

from __future__ import annotations

import numpy as np

PTIME_MS = 20

# --------------------------------------------------------------------------- packet loss


def gilbert_elliott(n: int, p: float, r: float, loss_good: float = 0.0, loss_bad: float = 1.0,
                    seed: int = 0) -> np.ndarray:
    """Two-state Markov loss model. Returns a boolean array, True = packet lost.

    ``p`` = P(good -> bad), ``r`` = P(bad -> good). In the classic "simple Gilbert" setting
    (loss_good=0, loss_bad=1) the mean loss rate is p / (p + r) and the mean burst length 1 / r.
    """
    rng = np.random.default_rng(seed)
    u_state, u_loss = rng.random(n), rng.random(n)
    lost = np.zeros(n, dtype=bool)
    bad = False
    for i in range(n):
        bad = (u_state[i] >= r) if bad else (u_state[i] < p)
        lost[i] = u_loss[i] < (loss_bad if bad else loss_good)
    return lost


def gilbert_loss_rate(p: float, r: float) -> float:
    return p / (p + r)


def burst_lengths(lost: np.ndarray) -> np.ndarray:
    """Lengths of consecutive runs of lost packets."""
    padded = np.concatenate([[False], np.asarray(lost, dtype=bool), [False]]).astype(int)
    edges = np.diff(padded)
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


# --------------------------------------------------------------------------- jitter


def network_delays(n: int, base_ms: float = 40.0, jitter_ms: float = 8.0, spike_prob: float = 0.004,
                   spike_ms: float = 120.0, step_at: int | None = None, step_ms: float = 0.0,
                   seed: int = 0) -> np.ndarray:
    """One-way delay per packet: path delay + gamma-distributed queueing + rare spikes.

    Spikes decay over the next packets, as a congested queue drains. ``step_at``/``step_ms`` model a
    route change (a mobile handover, a carrier reroute) that shifts the path delay mid-call.
    """
    rng = np.random.default_rng(seed)
    queueing = rng.gamma(shape=2.0, scale=jitter_ms / 2.0, size=n)
    spikes = np.zeros(n)
    for start in np.flatnonzero(rng.random(n) < spike_prob):
        length = min(int(spike_ms / PTIME_MS) + 1, n - start)
        spikes[start:start + length] = np.maximum(spikes[start:start + length],
                                                  spike_ms - PTIME_MS * np.arange(length))
    path = np.full(n, base_ms)
    if step_at is not None:
        path[step_at:] += step_ms
    return path + queueing + spikes


def arrival_order(send_ms: np.ndarray, delay_ms: np.ndarray) -> np.ndarray:
    """Sequence numbers in the order the packets arrive."""
    return np.argsort(send_ms + delay_ms, kind="stable")


def reordered_count(order: np.ndarray) -> int:
    """Packets that arrive after a packet with a higher sequence number."""
    running_max = np.maximum.accumulate(order)
    return int(np.sum(order < running_max))


def fixed_jitter_buffer(delay_ms: np.ndarray, buffer_ms: float, lost: np.ndarray | None = None) -> dict:
    """Play packet i at send_i + d_0 + buffer_ms. Anything arriving later than that is discarded.

    ``d_0`` is the first packet's delay, the only one the receiver knows when it starts playing.
    """
    lost = np.zeros(len(delay_ms), dtype=bool) if lost is None else lost
    playout_offset = delay_ms[0] + buffer_ms
    late = (delay_ms > playout_offset) & ~lost
    return {"late": late, "lost": lost, "missing": late | lost,
            "mouth_to_ear_ms": float(playout_offset), "late_rate": float(late.mean())}


def adaptive_jitter_buffer(delay_ms: np.ndarray, lost: np.ndarray | None = None, quantile: float = 0.97,
                           window: int = 500, margin_ms: float = 5.0, adapt_every: int = 25,
                           step_down_ms: float = 2.0) -> dict:
    """Adaptive playout: target = a high quantile of recently observed delays, plus a margin.

    The deadline rises at once when the target rises (late packets are audible) but falls by at most
    ``step_down_ms`` per update (a single quiet stretch should not shrink the buffer). It changes only
    every ``adapt_every`` packets, standing in for the silence gaps between talkspurts where a real
    buffer can stretch or shrink unnoticed. Production buffers such as WebRTC NetEq refine the same
    idea with delay histograms and time-stretching.
    """
    lost = np.zeros(len(delay_ms), dtype=bool) if lost is None else lost
    offset = float(delay_ms[0]) + 40.0  # conservative start, before any statistics exist
    offsets = np.empty(len(delay_ms))
    for i in range(len(delay_ms)):
        if i and i % adapt_every == 0:
            recent = delay_ms[max(0, i - window):i][~lost[max(0, i - window):i]]
            if len(recent):
                target = float(np.quantile(recent, quantile)) + margin_ms
                offset = target if target > offset else max(target, offset - step_down_ms)
        offsets[i] = offset
    late = (delay_ms > offsets) & ~lost
    return {"late": late, "lost": lost, "missing": late | lost, "offsets": offsets,
            "mouth_to_ear_ms": float(offsets.mean()), "late_rate": float(late.mean())}


# --------------------------------------------------------------------------- concealment


def voiced_signal(seconds: float = 1.0, sample_rate: int = 8000, f0: float = 140.0, seed: int = 0) -> np.ndarray:
    """A vowel-like signal with slowly gliding pitch (Vietnamese tones glide too)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    pitch = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
    x = sum((0.6 / h) * np.sin(h * phase + rng.uniform(0, np.pi)) for h in range(1, 18) if h * f0 < 3400)
    return 0.6 * x / np.max(np.abs(x))


def frames_to_samples(missing: np.ndarray, frame_len: int) -> np.ndarray:
    return np.repeat(np.asarray(missing, dtype=bool), frame_len)


def estimate_period(history: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 400.0) -> int:
    """Pitch period (samples) of the most recent good audio, by normalized autocorrelation."""
    lo, hi = int(sample_rate / fmax), int(sample_rate / fmin)
    x = history - history.mean()
    best, best_score = lo, -np.inf
    for lag in range(lo, min(hi, len(x) // 2)):
        a, b = x[lag:], x[:-lag]
        score = float(np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12))
        if score > best_score:
            best, best_score = lag, score
    return best


def conceal(received: np.ndarray, missing: np.ndarray, frame_len: int, method: str,
            sample_rate: int = 8000, hold_ms: float = 10.0, fade_out_ms: float = 50.0,
            crossfade_ms: float = 5.0) -> np.ndarray:
    """Fill missing frames. ``received`` may hold anything in missing frames.

    - ``zero``: silence (what a naive receiver does)
    - ``repeat``: replay the last good frame over and over
    - ``pitch``: repeat the last pitch period cyclically, the idea behind G.711 Appendix I PLC

    Synthetic audio keeps full level for ``hold_ms``, then fades to silence over ``fade_out_ms``
    (a long burst must not turn into a robotic drone), and is crossfaded into the next good frame.
    """
    y = np.asarray(received, dtype=np.float64).copy()
    missing = np.asarray(missing, dtype=bool)
    n_cross = int(sample_rate * crossfade_ms / 1000)
    pos, source, tail = 0, None, None
    for f, gone in enumerate(missing):
        start, stop = f * frame_len, (f + 1) * frame_len
        if not gone:
            if tail is not None:  # first good frame after a burst
                ramp = np.linspace(0.0, 1.0, n_cross)
                y[start:start + n_cross] = ramp * y[start:start + n_cross] + (1 - ramp) * tail
            pos, source, tail = 0, None, None
            continue
        if source is None:  # burst begins: freeze the material we will repeat
            history = y[max(0, start - 3 * frame_len):start]
            if method == "zero" or len(history) < frame_len:
                source = np.zeros(1)
            elif method == "repeat":
                source = history[-frame_len:]
            elif method == "pitch":
                source = history[-estimate_period(history, sample_rate):]
            else:
                raise ValueError(f"unknown method {method!r}")
        k = np.arange(pos, pos + frame_len + n_cross)
        t_ms = k / sample_rate * 1000
        gain = np.clip(1.0 - (t_ms - hold_ms) / fade_out_ms, 0.0, 1.0)
        synthetic = source[k % len(source)] * gain
        y[start:stop] = synthetic[:frame_len]
        tail = synthetic[frame_len:]
        pos += frame_len
    return y


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    noise = np.asarray(reference) - np.asarray(test)
    return float(10 * np.log10(np.sum(reference ** 2) / max(np.sum(noise ** 2), 1e-20)))


def segmental_snr_db(reference: np.ndarray, test: np.ndarray, frame_len: int = 160,
                     floor: float = -10.0, ceil: float = 35.0) -> float:
    """Mean of per-frame SNRs clamped to [floor, ceil]: closer to perception than global SNR."""
    values = []
    for start in range(0, len(reference) - frame_len + 1, frame_len):
        ref = reference[start:start + frame_len]
        err = ref - test[start:start + frame_len]
        if np.sum(ref ** 2) < 1e-8:
            continue
        values.append(np.clip(10 * np.log10(np.sum(ref ** 2) / max(np.sum(err ** 2), 1e-20)), floor, ceil))
    return float(np.mean(values))


def log_spectral_distance(reference: np.ndarray, test: np.ndarray, missing: np.ndarray, frame_len: int = 160,
                          hop: int = 40, floor_db: float = -60.0) -> float:
    """Mean log-spectral distance (dB) over sliding windows that touch a missing frame. Ignores phase.

    Windows slide by ``hop`` so they straddle frame boundaries, where repetition clicks live. Power
    below ``floor_db`` relative to the reference window's peak is clamped, so silence scores a large
    but finite distance.
    """
    window = np.hanning(frame_len)
    touched = frames_to_samples(missing, frame_len)
    distances = []
    for start in range(0, len(reference) - frame_len + 1, hop):
        if not touched[start:start + frame_len].any():
            continue
        p_ref = 10 * np.log10(np.abs(np.fft.rfft(reference[start:start + frame_len] * window)) ** 2 + 1e-12)
        p_tst = 10 * np.log10(np.abs(np.fft.rfft(test[start:start + frame_len] * window)) ** 2 + 1e-12)
        top = p_ref.max()
        p_ref, p_tst = np.maximum(p_ref, top + floor_db), np.maximum(p_tst, top + floor_db)
        distances.append(np.sqrt(np.mean((p_ref - p_tst) ** 2)))
    return float(np.mean(distances))


# --------------------------------------------------------------------------- level


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


def clipping_ratio(x: np.ndarray, threshold: float = 0.99, min_run: int = 3) -> float:
    """Share of samples in runs of >= ``min_run`` samples at or above full scale.

    A single sample at full scale can be legitimate; flat runs are the signature of clipping.
    """
    hot = np.abs(np.asarray(x)) >= threshold
    runs = burst_lengths(hot)
    return float(runs[runs >= min_run].sum() / len(x)) if len(x) else 0.0


def agc(x: np.ndarray, sample_rate: int = 8000, target_dbfs: float = -20.0, max_gain_db: float = 24.0,
        gate_dbfs: float = -50.0, attack: float = 0.5, release: float = 0.05, frame_ms: int = 10) -> np.ndarray:
    """Frame-based automatic gain control with a noise gate and asymmetric smoothing.

    Gain moves quickly down (attack) when the talker gets loud and slowly up (release) when quiet,
    and freezes in frames below the gate so background noise is not pumped up between words.
    """
    frame = int(sample_rate * frame_ms / 1000)
    y = np.asarray(x, dtype=np.float64).copy()
    gain_db = 0.0
    for start in range(0, len(y), frame):
        seg = y[start:start + frame]
        level = dbfs(seg)
        if level > gate_dbfs:
            wanted = np.clip(target_dbfs - level, -max_gain_db, max_gain_db)
            rate = attack if wanted < gain_db else release
            gain_db += rate * (wanted - gain_db)
        y[start:start + frame] = seg * 10 ** (gain_db / 20)
    return np.clip(y, -1.0, 1.0)


# --------------------------------------------------------------------------- E-model


def delay_impairment(one_way_ms: float) -> float:
    """Id from mouth-to-ear delay, the common simplification of G.107 (Cole & Rosenbluth, 2001)."""
    d = float(one_way_ms)
    return 0.024 * d + 0.11 * (d - 177.3) * (d > 177.3)


def effective_equipment_impairment(loss: float, burst_ratio: float = 1.0, ie: float = 0.0, bpl: float = 25.1) -> float:
    """Ie,eff from G.107: codec impairment ``ie`` plus packet loss ``loss`` (0..1).

    G.711 has ie = 0; bpl = 25.1 with PLC (4.3 without), per ITU-T G.113 Appendix I.
    ``burst_ratio`` = 1 for random loss, > 1 for bursty loss (1 / (p + r) for a Gilbert model).
    """
    ppl = 100.0 * loss
    return ie + (95.0 - ie) * ppl / (ppl / burst_ratio + bpl)


def r_factor(one_way_ms: float, loss: float = 0.0, burst_ratio: float = 1.0, bpl: float = 25.1) -> float:
    return 93.2 - delay_impairment(one_way_ms) - effective_equipment_impairment(loss, burst_ratio, bpl=bpl)


def mos_from_r(r: float) -> float:
    if r <= 0:
        return 1.0
    if r >= 100:
        return 4.5
    return 1 + 0.035 * r + r * (r - 60) * (100 - r) * 7e-6


# --------------------------------------------------------------------------- demo


def main() -> None:
    n = 3000  # one minute of 20 ms packets
    lost = gilbert_elliott(n, p=0.01, r=0.4, seed=1)
    bursts = burst_lengths(lost)
    print(f"Gilbert-Elliott p=0.01 r=0.4: {100 * lost.mean():.1f}% lost (theory "
          f"{100 * gilbert_loss_rate(0.01, 0.4):.1f}%), mean burst {bursts.mean():.2f} packets")

    delays = network_delays(n, seed=2, step_at=1800, step_ms=60.0)  # route change after 36 s
    send = np.arange(n) * PTIME_MS
    print(f"Delay p50 {np.percentile(delays, 50):.0f} ms, p99 {np.percentile(delays, 99):.0f} ms, "
          f"{reordered_count(arrival_order(send, delays))} packets reordered")
    for buffer_ms in (20, 60, 100, 140):
        jb = fixed_jitter_buffer(delays, buffer_ms, lost)
        print(f"  fixed buffer {buffer_ms:>3} ms: late {100 * jb['late_rate']:5.2f}%, "
              f"network + buffer delay {jb['mouth_to_ear_ms']:.0f} ms")
    ajb = adaptive_jitter_buffer(delays, lost)
    print(f"  adaptive buffer   : late {100 * ajb['late_rate']:5.2f}%, mean delay {ajb['mouth_to_ear_ms']:.0f} ms")

    x = voiced_signal()
    missing = np.zeros(50, dtype=bool)
    missing[[10, 25, 26, 40, 41, 42]] = True  # one single loss, a pair, a triple
    received = x * ~frames_to_samples(missing, 160)
    for method in ("zero", "repeat", "pitch"):
        y = conceal(received, missing, 160, method)
        print(f"PLC {method:<6}: segmental SNR {segmental_snr_db(x, y):5.1f} dB, "
              f"log-spectral distance on lost frames {log_spectral_distance(x, y, missing):5.1f} dB")

    loud = np.clip(3.0 * x, -1, 1)
    print(f"Clipped share of a 3x-too-hot signal: {100 * clipping_ratio(loud):.1f}%")
    quiet = 0.02 * x
    print(f"AGC: {dbfs(quiet):.1f} dBFS in -> {dbfs(agc(quiet)[4000:]):.1f} dBFS out")

    for d, loss, br in ((100, 0.0, 1.0), (100, 0.01, 1.0), (250, 0.03, 2.5)):
        r = r_factor(d, loss, br)
        print(f"E-model: {d} ms, {100 * loss:.0f}% loss, burst ratio {br}: R = {r:.0f}, MOS = {mos_from_r(r):.2f}")


if __name__ == "__main__":
    main()
