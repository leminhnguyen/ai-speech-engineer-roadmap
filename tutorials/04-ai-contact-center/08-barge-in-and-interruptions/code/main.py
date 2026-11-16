"""Lesson 08: barge-in, backchannels and false interruptions.

A deterministic event-timeline simulator of the decision a voice agent makes every time
audio arrives while it is speaking: stop, pause and resume, or ignore. The controller is
parameterised like LiveKit's ``InterruptionOptions`` (``min_duration``, ``min_words``,
``false_interruption_timeout``, ``resume_false_interruption``) so you can sweep the same
knobs you will tune in production. It also truncates the agent's reply to what the caller
actually heard.

Run: python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- events

BACKCHANNELS = ("dạ", "vâng", "ừ", "ờ", "uh-huh", "okay", "yes", "mm")
TWO_WORD_BACKCHANNELS = (("dạ", "vâng"), ("vâng", "ạ"), ("ừ", "ừ"), ("okay", "okay"))
SHORT_INTERRUPTIONS = (["không"], ["khoan"], ["wait"], ["no"], ["sai", "rồi"])
LONG_INTERRUPTIONS = (
    "khoan đã tôi muốn đổi địa chỉ giao hàng",
    "không phải đơn đó đơn hôm qua cơ",
    "sorry I meant the other account",
    "chị ơi cho em gặp nhân viên",
    "wait that is not my order number",
)
AGENT_REPLY = ("your order four seven two one will arrive on thursday between nine and eleven "
               "and the delivery fee is thirty thousand dong").split()


@dataclass(frozen=True)
class OverlapEvent:
    """Audio from the caller's side that starts while the agent is speaking.

    ``kind`` is the ground truth: only ``"interruption"`` should stop the agent.
    ``voiced`` says whether a VAD would fire on it (speech, a cough, or the agent's own echo).
    ``words`` is what STT would transcribe from it.
    """

    kind: str
    start: float
    duration: float
    words: tuple = ()
    voiced: bool = True

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class BargeInConfig:
    """The knobs of a barge-in controller. Defaults mirror LiveKit ``InterruptionOptions``."""

    mode: str = "vad"                    # "vad" or "adaptive"
    min_duration: float = 0.5            # s of speech before VAD counts it as an interruption
    min_words: int = 0                   # words required (0 = VAD alone is enough)
    false_interruption_timeout: float | None = 2.0  # s of no transcript before "false"
    resume_false_interruption: bool = True
    guard_s: float = 0.0                 # ignore overlaps that start this early in the agent turn
    stt_final_delay: float = 0.3         # final transcript arrives this long after speech ends
    interim_delay: float = 0.2           # interim words lag the audio by this much
    classifier_delay: float = 0.25       # adaptive mode decides this long after onset


@dataclass
class Outcome:
    """What the controller did with one overlap event."""

    action: str                    # "ignored", "resumed", "stopped", "after_end"
    silenced_at: float | None = None  # when the agent's audio went quiet
    resumed_at: float | None = None
    trigger: str = ""              # "vad", "transcript", "classifier" or ""
    notes: list = field(default_factory=list)


def is_backchannel(words: tuple, duration: float) -> bool:
    """A toy stand-in for an acoustic backchannel classifier: one short listener token.

    Deliberately simple, so it makes the mistakes real classifiers make on edge cases:
    a two-token backchannel such as "dạ vâng" looks like an interruption to it.
    """
    return len(words) == 1 and words[0].lower() in BACKCHANNELS and duration < 0.8


def adaptive_says_interrupt(event: OverlapEvent) -> bool:
    """Toy adaptive detector. Ignores non-speech and backchannels; it cannot recognise echo."""
    return event.voiced and len(event.words) > 0 and not is_backchannel(event.words, event.duration)


def word_arrival_times(event: OverlapEvent, interim_delay: float) -> np.ndarray:
    """When each interim word becomes available to the controller."""
    n = len(event.words)
    return event.start + (np.arange(1, n + 1) / max(n, 1)) * event.duration + interim_delay


def simulate_event(event: OverlapEvent, agent_duration: float, cfg: BargeInConfig) -> Outcome:
    """Decide what the agent does about one overlapping event.

    Two paths can interrupt, as in LiveKit: the VAD path (enough speech, optionally enough
    interim words) *pauses* the agent; a final transcript arriving while the agent still
    speaks *stops* it. A pause with no transcript within ``false_interruption_timeout`` is a
    false interruption and, if ``resume_false_interruption``, the agent continues. The timeout
    counts silence after the caller stops speaking.
    """
    if event.start >= agent_duration:
        return Outcome("after_end")
    if event.start < cfg.guard_s:
        return Outcome("ignored", notes=["inside guard window"])
    if cfg.mode == "adaptive" and not adaptive_says_interrupt(event):
        return Outcome("ignored", notes=["classified as non-interruption"])

    has_transcript = len(event.words) >= max(1, cfg.min_words)
    t_final = event.end + cfg.stt_final_delay if has_transcript else None

    t_pause = None
    if event.voiced and event.duration >= cfg.min_duration:
        t_pause = event.start + cfg.min_duration
        if cfg.mode == "adaptive":
            t_pause = event.start + max(cfg.min_duration, cfg.classifier_delay)
        if cfg.min_words > 0:
            arrivals = word_arrival_times(event, cfg.interim_delay)
            t_pause = max(t_pause, float(arrivals[cfg.min_words - 1])) if len(arrivals) >= cfg.min_words else None
    if t_pause is not None and t_pause >= agent_duration:
        t_pause = None

    if t_pause is not None:
        timeout = cfg.false_interruption_timeout
        silence_from = max(t_pause, event.end)  # the timeout counts silence after the caller stops
        if t_final is not None and (timeout is None or t_final <= silence_from + timeout):
            return Outcome("stopped", silenced_at=t_pause, trigger="vad")
        if timeout is None:
            return Outcome("stopped", silenced_at=t_pause, trigger="vad", notes=["no false-interruption check"])
        if cfg.resume_false_interruption:
            return Outcome("resumed", silenced_at=t_pause, resumed_at=silence_from + timeout, trigger="vad")
        return Outcome("stopped", silenced_at=t_pause, trigger="vad", notes=["false interruption, not resumed"])

    if t_final is not None and t_final < agent_duration:
        return Outcome("stopped", silenced_at=t_final, trigger="transcript")
    return Outcome("ignored")


# --------------------------------------------------------------------------- datasets and metrics

def make_events(n: int = 400, seed: int = 0, agent_duration: float = 6.0,
                include_echo: bool = False) -> list:
    """A reproducible mix of overlaps a contact-center agent hears while speaking."""
    rng = np.random.default_rng(seed)
    kinds = ["interruption", "backchannel", "cough", "noise"] + (["echo"] if include_echo else [])
    probs = np.array([0.4, 0.3, 0.15, 0.15] + ([0.25] if include_echo else []))
    probs = probs / probs.sum()
    events = []
    for _ in range(n):
        kind = str(rng.choice(kinds, p=probs))
        start = float(rng.uniform(0.3, agent_duration - 1.0))
        if kind == "interruption":
            if rng.random() < 0.3:
                words = tuple(SHORT_INTERRUPTIONS[rng.integers(len(SHORT_INTERRUPTIONS))])
                duration = float(rng.uniform(0.25, 0.55))
            else:
                words = tuple(LONG_INTERRUPTIONS[rng.integers(len(LONG_INTERRUPTIONS))].split())
                duration = float(np.clip(0.28 * len(words) * rng.uniform(0.8, 1.2), 0.8, 3.0))
            events.append(OverlapEvent(kind, start, duration, words, True))
        elif kind == "backchannel":
            if rng.random() < 0.25:  # "dạ vâng": still a backchannel, but two tokens long
                words = TWO_WORD_BACKCHANNELS[rng.integers(len(TWO_WORD_BACKCHANNELS))]
                duration = float(rng.uniform(0.4, 0.8))
            else:
                words = (BACKCHANNELS[rng.integers(len(BACKCHANNELS))],)
                duration = float(rng.uniform(0.15, 0.6))
            events.append(OverlapEvent(kind, start, duration, words, True))
        elif kind == "cough":
            events.append(OverlapEvent(kind, start, float(rng.uniform(0.1, 0.7)), (), True))
        elif kind == "noise":
            events.append(OverlapEvent(kind, start, float(rng.uniform(0.2, 1.5)), (), False))
        else:  # the agent's own voice leaking back through a speakerphone with no echo canceller
            start = float(rng.uniform(0.0, 1.2))
            n_words = int(rng.integers(2, 6))
            first = int(start / agent_duration * len(AGENT_REPLY))
            words = tuple(AGENT_REPLY[first:first + n_words])
            events.append(OverlapEvent(kind, start, float(rng.uniform(0.6, 2.0)), words, True))
    return events


def evaluate(events: list, cfg: BargeInConfig, agent_duration: float = 6.0) -> dict:
    """Rates that matter to a caller, computed against the ground-truth ``kind``."""
    real = [e for e in events if e.kind == "interruption"]
    other = [e for e in events if e.kind != "interruption"]
    real_out = [simulate_event(e, agent_duration, cfg) for e in real]
    other_out = [simulate_event(e, agent_duration, cfg) for e in other]
    latencies = [o.silenced_at - e.start for e, o in zip(real, real_out) if o.action == "stopped"]
    return {
        "missed_rate": float(np.mean([o.action != "stopped" for o in real_out])) if real else 0.0,
        "false_stop_rate": float(np.mean([o.action == "stopped" for o in other_out])) if other else 0.0,
        "false_pause_rate": float(np.mean([o.action == "resumed" for o in other_out])) if other else 0.0,
        "median_stop_latency": float(np.median(latencies)) if latencies else float("nan"),
        "p90_stop_latency": float(np.percentile(latencies, 90)) if latencies else float("nan"),
    }


def sweep_min_duration(events: list, values, **config) -> dict:
    """Evaluate the controller for each ``min_duration`` value."""
    rows = [evaluate(events, BargeInConfig(min_duration=float(v), **config)) for v in values]
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


# --------------------------------------------------------------------------- what the caller heard

def word_timings(words, total_duration: float) -> np.ndarray:
    """Estimate (start, end) per word when TTS gives no alignment: time proportional to length."""
    weights = np.array([len(w) + 1.0 for w in words])
    edges = np.concatenate([[0.0], np.cumsum(weights) / weights.sum() * total_duration])
    return np.stack([edges[:-1], edges[1:]], axis=1)


def truncate_to_heard(words, timings: np.ndarray, playout_position: float,
                      min_fraction: float = 0.5) -> tuple:
    """Split a reply into the part the caller heard and the part they never heard.

    A word counts as heard when at least ``min_fraction`` of it was played. The heard part is
    what belongs in the LLM's chat history; keeping the whole reply makes the model believe it
    said things the caller never heard.
    """
    starts, ends = timings[:, 0], timings[:, 1]
    played = np.clip((playout_position - starts) / np.maximum(ends - starts, 1e-9), 0.0, 1.0)
    cut = int(np.sum(played >= min_fraction))
    return " ".join(words[:cut]), " ".join(words[cut:])


def main() -> None:
    agent_duration = 6.0
    default = BargeInConfig()
    examples = [
        OverlapEvent("interruption", 2.0, 1.6, tuple("khoan đã tôi muốn đổi địa chỉ".split())),
        OverlapEvent("interruption", 2.0, 0.35, ("không",)),
        OverlapEvent("backchannel", 2.0, 0.3, ("dạ",)),
        OverlapEvent("cough", 2.0, 0.6, ()),
        OverlapEvent("echo", 0.4, 1.5, tuple(AGENT_REPLY[:4])),
    ]
    print("Default VAD-mode controller (min_duration=0.5, min_words=0, timeout=2.0, resume=True):")
    for event in examples:
        out = simulate_event(event, agent_duration, default)
        label = " ".join(event.words) or "-"
        when = f", quiet from {out.silenced_at:.2f} s" if out.silenced_at is not None else ""
        print(f"  {event.kind:<12} '{label}' -> {out.action}{when} ({out.trigger or 'no trigger'})")

    events = make_events(400, seed=0)
    for name, cfg in (("vad, min_duration 0.5", BargeInConfig()),
                      ("vad, min_words 2", BargeInConfig(min_words=2)),
                      ("adaptive", BargeInConfig(mode="adaptive"))):
        m = evaluate(events, cfg)
        print(f"{name:<22} missed {m['missed_rate']:.0%}  false stops {m['false_stop_rate']:.0%}  "
              f"false pauses {m['false_pause_rate']:.0%}  median stop {m['median_stop_latency']:.2f} s")

    echo = make_events(400, seed=1, include_echo=True)
    for guard in (0.0, 3.0):
        m = evaluate(echo, BargeInConfig(mode="adaptive", guard_s=guard))
        print(f"with echo leakage, guard {guard:.0f} s: false stops {m['false_stop_rate']:.0%}")

    timings = word_timings(AGENT_REPLY, 6.0)
    heard, unheard = truncate_to_heard(AGENT_REPLY, timings, playout_position=2.4)
    print(f"Interrupted at 2.4 s. Heard: '{heard}'")
    print(f"Never heard (drop from chat history): '{unheard}'")


if __name__ == "__main__":
    main()
