"""Lesson 02: the plumbing between a SIP participant and an AgentSession.

Three pieces a voice-agent engineer touches indirectly on every call, rebuilt with NumPy:

1. A polyphase windowed-sinc resampler for the rate changes along the call path
   (8 kHz SIP -> 16 kHz STT/VAD, 24 kHz TTS -> 48 kHz room audio, and back).
2. A frame stream that turns arbitrary network chunks into fixed 10/20 ms frames.
3. A dispatch-rule simulator that reproduces how LiveKit's three inbound rule types
   choose a room for a call, and which SIP attributes the caller carries into it.

Run: python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np

# --------------------------------------------------------------------------- resampling


def kaiser_lowpass(cutoff: float, num_taps: int, beta: float = 8.6) -> np.ndarray:
    """Windowed-sinc low-pass FIR. ``cutoff`` is in cycles per sample (Nyquist = 0.5)."""
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for a symmetric, linear-phase filter")
    n = np.arange(num_taps) - (num_taps - 1) / 2
    return 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(num_taps, beta)


def design_resampler(orig_sr: int, target_sr: int, zeros: int = 16, rolloff: float = 0.94,
                     beta: float = 8.6) -> tuple[int, int, np.ndarray]:
    """Return ``(up, down, h)`` for a rational resampler ``target/orig = up/down``.

    The anti-alias/anti-image filter runs at the virtual rate ``orig_sr * up``. Its cutoff is
    the lower of the two Nyquist frequencies, scaled by ``rolloff`` to leave a transition band.
    ``zeros`` sets the filter length in zero crossings on each side: longer = sharper.
    """
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    stretch = max(up, down)
    h = kaiser_lowpass(0.5 * rolloff / stretch, 2 * zeros * stretch + 1, beta) * up
    return up, down, h


def resample(x: np.ndarray, orig_sr: int, target_sr: int, zeros: int = 16) -> np.ndarray:
    """Polyphase rational resampling, without ever building the zero-stuffed signal.

    Conceptually: insert ``up - 1`` zeros between samples, low-pass, keep every ``down``-th
    sample. Output sample ``n`` sits at virtual index ``u = n * down + D`` (``D`` = filter delay),
    so it only needs filter taps ``h[u - k * up]``: one *phase* of the filter, ``len(h) / up`` taps.
    """
    x = np.asarray(x, dtype=np.float64)
    if orig_sr == target_sr:
        return x.copy()
    up, down, h = design_resampler(orig_sr, target_sr, zeros)
    num_taps = len(h)
    delay = (num_taps - 1) // 2
    per_phase = -(-num_taps // up)  # ceil
    phases = np.zeros((up, per_phase))
    for p in range(up):
        taps = h[p::up]
        phases[p, : len(taps)] = taps
    n_out = -(-len(x) * up // down)
    u = np.arange(n_out) * down + delay
    phase, base = u % up, u // up
    idx = base[:, None] - np.arange(per_phase)[None, :]
    valid = (idx >= 0) & (idx < len(x))
    gathered = np.where(valid, x[np.clip(idx, 0, len(x) - 1)], 0.0)
    return np.sum(gathered * phases[phase], axis=1)


def naive_resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear interpolation, the shortcut that skips the anti-alias filter."""
    t_out = np.arange(int(len(x) * target_sr / orig_sr)) / target_sr
    return np.interp(t_out, np.arange(len(x)) / orig_sr, x)


def tone(frequency: float, sample_rate: int, seconds: float = 0.5, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


def tone_level_db(x: np.ndarray, sample_rate: int, frequency: float) -> float:
    """Level of one frequency in dB re full scale sine, via a Hann-windowed DFT bin."""
    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * window)) * 2 / window.sum()
    freqs = np.fft.rfftfreq(len(x), d=1 / sample_rate)
    k = int(np.argmin(np.abs(freqs - frequency)))
    return float(20 * np.log10(spectrum[max(k - 2, 0): k + 3].max() + 1e-12))


# --------------------------------------------------------------------------- frames


def frame_samples(sample_rate: int, frame_ms: int) -> int:
    samples = sample_rate * frame_ms / 1000
    if samples != int(samples):
        raise ValueError(f"{frame_ms} ms is not a whole number of samples at {sample_rate} Hz")
    return int(samples)


@dataclass
class FrameStream:
    """Re-chunk arbitrary audio pieces into fixed frames, like an audio track's 10/20 ms frames.

    Network and model APIs deliver audio in whatever chunk size suits them (a 160-byte RTP
    payload, a 4096-sample TTS chunk). Downstream components such as VAD expect fixed frames.
    """

    sample_rate: int
    frame_ms: int = 20
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def size(self) -> int:
        return frame_samples(self.sample_rate, self.frame_ms)

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        self._pending = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float64)])
        whole = len(self._pending) // self.size
        frames = [self._pending[i * self.size:(i + 1) * self.size] for i in range(whole)]
        self._pending = self._pending[whole * self.size:]
        return frames

    def flush(self) -> list[np.ndarray]:
        """Zero-pad and emit the last partial frame, if any."""
        if len(self._pending) == 0:
            return []
        last = np.zeros(self.size)
        last[: len(self._pending)] = self._pending
        self._pending = np.zeros(0)
        return [last]


# Audio formats along a LiveKit telephony call. The SIP leg is G.711 (lesson 01); the room carries
# audio between participants; models consume and produce their own rates.
CALL_PATH = (
    ("Phone line (G.711)", 8_000, 20),
    ("Room track", 48_000, 10),
    ("VAD / STT input", 16_000, 10),
    ("TTS output", 24_000, 10),
)


def bytes_per_frame(sample_rate: int, frame_ms: int, bytes_per_sample: int = 2) -> int:
    return frame_samples(sample_rate, frame_ms) * bytes_per_sample


# --------------------------------------------------------------------------- dispatch rules


@dataclass
class DispatchRule:
    """A simplified LiveKit inbound dispatch rule (see the docs for the full API).

    kind: ``"individual"`` (new room per caller), ``"direct"`` (one fixed room, optional PIN),
    or ``"callee"`` (room named after the dialed number).
    """

    kind: str
    room_prefix: str = ""
    room_name: str = ""
    pin: str = ""
    randomize: bool = True
    trunk_ids: tuple[str, ...] = ()
    agent_name: str = ""
    hide_phone_number: bool = False
    rule_id: str = "SDR_demo"

    def matches(self, trunk_id: str, pin: str = "") -> bool:
        trunk_ok = not self.trunk_ids or trunk_id in self.trunk_ids  # no trunk list = wildcard
        return trunk_ok and (not self.pin or pin == self.pin)


def dispatch_room(rule: DispatchRule, caller: str, callee: str, rng: np.random.Generator) -> str:
    """Room a call lands in. The random suffix format is illustrative, not LiveKit's exact one."""
    suffix = "_" + "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), 8))
    if rule.kind == "individual":
        return f"{rule.room_prefix}{caller}{suffix}"
    if rule.kind == "direct":
        return rule.room_name
    if rule.kind == "callee":
        return f"{rule.room_prefix}{callee}{suffix if rule.randomize else ''}"
    raise ValueError(f"unknown rule kind: {rule.kind!r}")


def route_call(rules: list[DispatchRule], trunk_id: str, caller: str, callee: str, call_id: str,
               pin: str = "", seed: int = 0) -> dict:
    """Pick the first matching rule, then return the room, agent and SIP participant attributes."""
    rng = np.random.default_rng(seed)
    for rule in rules:
        if rule.matches(trunk_id, pin):
            attributes = {
                "sip.callID": call_id,
                "sip.callStatus": "ringing",
                "sip.trunkID": trunk_id,
                "sip.trunkPhoneNumber": callee,
                "sip.ruleID": rule.rule_id,
            }
            if not rule.hide_phone_number:
                attributes["sip.phoneNumber"] = caller
            return {"room": dispatch_room(rule, caller, callee, rng), "agent_name": rule.agent_name,
                    "attributes": attributes}
    raise LookupError("no dispatch rule matches: the call is rejected")


# --------------------------------------------------------------------------- demo


def main() -> None:
    print("Audio formats along a telephony call:")
    for name, rate, ms in CALL_PATH:
        print(f"  {name:<20} {rate:>6} Hz, {ms} ms frames = {frame_samples(rate, ms):>4} samples, "
              f"{bytes_per_frame(rate, ms):>4} bytes as int16")

    for src, dst in ((8_000, 16_000), (24_000, 48_000), (48_000, 16_000), (48_000, 8_000)):
        up, down, h = design_resampler(src, dst)
        print(f"Resample {src} -> {dst} Hz: up {up}, down {down}, {len(h)} taps, "
              f"{-(-len(h) // up)} multiplies per output sample")

    wide = tone(1_000, 48_000) + tone(6_000, 48_000)
    good = resample(wide, 48_000, 8_000)
    bad = naive_resample(wide, 48_000, 8_000)
    print(f"6 kHz tone after 48k -> 8k: aliased to 2 kHz at {tone_level_db(good, 8_000, 2_000):.0f} dB "
          f"(filtered) vs {tone_level_db(bad, 8_000, 2_000):.0f} dB (naive)")

    stream = FrameStream(sample_rate=8_000, frame_ms=20)
    rng = np.random.default_rng(1)
    frames = []
    for size in rng.integers(30, 400, size=20):
        frames += stream.push(np.zeros(size))
    frames += stream.flush()
    print(f"20 uneven chunks -> {len(frames)} frames of {stream.size} samples")

    rules = [
        DispatchRule(kind="direct", room_name="vip-line", pin="2468", rule_id="SDR_vip"),
        DispatchRule(kind="individual", room_prefix="call-", agent_name="inbound-agent", rule_id="SDR_main"),
    ]
    routed = route_call(rules, "ST_hotline", "+84901234567", "+842873001234", "a84b4c76e66710")
    print(f"Inbound call -> room {routed['room']!r}, agent {routed['agent_name']!r}")
    print(f"  attributes: {routed['attributes']}")


if __name__ == "__main__":
    main()
