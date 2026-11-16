"""Lesson 07: knowing when the caller has finished. Endpointing policies on simulated calls.

Simulates Vietnamese call-center turns in which callers pause *inside* a turn (reading a phone
number in groups, hesitating mid-request), then compares three ways of deciding that a turn is
over: a fixed silence timer, a timer that adapts to the caller's own pauses (dynamic endpointing),
and a timer chosen by an end-of-utterance probability computed from the transcript so far
(semantic endpointing). Every policy trades cut-offs against waiting.

Run: python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

# Each template is one caller turn split where callers typically pause. The pause *kind* sets
# its length distribution: reading digits produces longer, more regular pauses than hesitation.
TURN_TEMPLATES = (
    (("số điện thoại của em là", "0903", "123", "456"), "number"),
    (("em ở số 12", "đường Lê Lợi", "phường Bến Nghé", "quận 1 ạ"), "hesitation"),
    (("em muốn hỏi về", "hóa đơn tháng này ạ"), "hesitation"),
    (("em muốn đổi gói cước", "sang gói tháng nhé"), "hesitation"),
    (("không", "em không đặt đơn đó"), "hesitation"),
    (("mã khách hàng của em là", "8 4 7", "2 9 1"), "number"),
    (("dạ vâng",), None),
    (("đúng rồi ạ",), None),
    (("cho em gặp nhân viên",), None),
    (("chi nhánh gần nhất ở đâu ạ",), None),
)

PAUSE_MEDIAN_S = {"hesitation": 0.35, "number": 0.6}
PAUSE_SIGMA = {"hesitation": 0.5, "number": 0.35}


@dataclass
class Pause:
    """A silence after a chunk of caller speech, and what the agent knows at that moment."""

    transcript: str  # everything the caller has said in this turn so far
    length_s: float  # how long the caller stays silent (inf: the caller is done and waits)
    final: bool      # True if this silence is the real end of the turn


def simulate_call(n_turns: int = 12, seed: int = 0, tempo: float | None = None) -> list[list[Pause]]:
    """One caller, several turns. ``tempo`` scales every pause (slow talkers pause longer).

    Tempo varies a lot between callers (log-normal, roughly 0.45x to 2.2x for the middle 95%),
    which is what gives a per-caller adaptive timer something to learn.
    """
    rng = np.random.default_rng(seed)
    tempo = float(np.exp(0.4 * rng.standard_normal())) if tempo is None else tempo
    turns = []
    for _ in range(n_turns):
        chunks, kind = TURN_TEMPLATES[int(rng.integers(len(TURN_TEMPLATES)))]
        pauses = []
        for i in range(len(chunks)):
            text = " ".join(chunks[: i + 1])
            if i == len(chunks) - 1:
                pauses.append(Pause(text, math.inf, True))
            else:
                length = tempo * PAUSE_MEDIAN_S[kind] * float(np.exp(PAUSE_SIGMA[kind] * rng.standard_normal()))
                pauses.append(Pause(text, length, False))
        turns.append(pauses)
    return turns


def simulate_calls(n_calls: int = 200, turns_per_call: int = 12, seed: int = 0) -> list[list[list[Pause]]]:
    return [simulate_call(turns_per_call, seed=seed * 10_000 + i) for i in range(n_calls)]


# --------------------------------------------------------------------------- end-of-utterance cues

CONNECTORS = {"là", "và", "với", "về", "của", "thì", "mà", "nhưng", "hoặc", "số", "ở", "cho"}
FINAL_PARTICLES = {"ạ", "nhé", "nha", "không", "vâng", "rồi", "ơi"}


def eou_features(transcript: str) -> dict:
    """Cheap transcript cues that a turn is (in)complete.

    - ``trailing_connector``: the turn ends on a word that needs a continuation ("là", "về", ...).
    - ``open_number``: a number is being read ("số điện thoại", "mã ... là") and has too few digits.
    - ``final_particle``: Vietnamese sentence-final particles ("ạ", "nhé", ...) usually close a turn.
    """
    words = transcript.lower().split()
    digits = len(re.sub(r"\D", "", transcript))
    reading_number = any(cue in transcript.lower() for cue in ("số điện thoại", "mã "))
    expected = 10 if "điện thoại" in transcript.lower() else 6
    return {
        "trailing_connector": words[-1] in CONNECTORS,
        "open_number": reading_number and digits < expected,
        "final_particle": words[-1] in FINAL_PARTICLES,
    }


def eou_probability(transcript: str) -> float:
    """A toy end-of-utterance model: a logistic score over the three cues.

    LiveKit's turn detectors learn this from data (the audio model also hears intonation); the
    weights here are hand-set to show the mechanism.
    """
    f = eou_features(transcript)
    score = 0.4 + 2.2 * f["final_particle"] - 3.0 * f["trailing_connector"] - 3.0 * f["open_number"]
    return 1 / (1 + math.exp(-score))


# --------------------------------------------------------------------------- endpointing policies

class FixedEndpointing:
    """Reply after ``delay`` seconds of silence, whatever was said."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay

    def reset(self) -> None:
        pass

    def delay_for(self, pause: Pause) -> float:
        return self.delay

    def observe(self, pause: Pause) -> None:
        pass


class DynamicEndpointing:
    """Adapt the silence timer to this caller's own pauses, within ``[min_delay, max_delay]``.

    Keeps an exponential moving average of the pauses after which the caller *continued*
    talking, and waits ``margin`` times that. A fast talker gets a short timer, a slow one a long
    timer. This illustrates the idea behind LiveKit's ``endpointing={"mode": "dynamic", ...}``;
    it is not LiveKit's exact algorithm.
    """

    def __init__(self, min_delay: float = 0.5, max_delay: float = 3.0, alpha: float = 0.9, margin: float = 2.0,
                 initial_pause: float = 0.4):
        self.min_delay, self.max_delay, self.alpha, self.margin = min_delay, max_delay, alpha, margin
        self.initial_pause = initial_pause
        self.reset()

    def reset(self) -> None:
        self.average_pause = self.initial_pause

    def delay_for(self, pause: Pause) -> float:
        return float(np.clip(self.margin * self.average_pause, self.min_delay, self.max_delay))

    def observe(self, pause: Pause) -> None:
        if not pause.final:
            self.average_pause = self.alpha * self.average_pause + (1 - self.alpha) * pause.length_s


class SemanticEndpointing:
    """Wait ``min_delay`` when the transcript looks finished, up to ``max_delay`` when it does not."""

    def __init__(self, min_delay: float = 0.5, max_delay: float = 3.0, threshold: float = 0.5):
        self.min_delay, self.max_delay, self.threshold = min_delay, max_delay, threshold

    def reset(self) -> None:
        pass

    def delay_for(self, pause: Pause) -> float:
        return self.min_delay if eou_probability(pause.transcript) >= self.threshold else self.max_delay

    def observe(self, pause: Pause) -> None:
        pass


def run_policy(policy, calls: list) -> dict:
    """Replay calls through a policy.

    A *cut-off* happens when the timer expires during a pause after which the caller would have
    continued: the agent replies to half a sentence. The *wait* is the silence the caller sits
    through after really finishing. After a cut-off the caller talks again anyway, so adaptive
    policies still observe that pause; the rest of the turn is not scored.
    """
    turns = cut_turns = 0
    waits = []
    for call in calls:
        policy.reset()
        for turn in call:
            turns += 1
            for pause in turn:
                delay = policy.delay_for(pause)
                if pause.final:
                    waits.append(delay)
                elif pause.length_s >= delay:
                    cut_turns += 1
                    policy.observe(pause)
                    break
                policy.observe(pause)
    waits = np.array(waits)
    return {
        "cutoff_rate": cut_turns / turns,
        "mean_wait_s": float(np.mean(waits)) if len(waits) else float("nan"),
        "p95_wait_s": float(np.percentile(waits, 95)) if len(waits) else float("nan"),
        "turns": turns,
    }


def within_turn_pauses(calls: list, kind: str | None = None) -> np.ndarray:
    """All non-final pause lengths, optionally only those of one template kind."""
    lengths = []
    for call in calls:
        for turn in call:
            transcript = turn[-1].transcript
            turn_kind = next((k for chunks, k in TURN_TEMPLATES if " ".join(chunks) == transcript), None)
            if kind is None or turn_kind == kind:
                lengths.extend(p.length_s for p in turn if not p.final)
    return np.array(lengths)


def main() -> None:
    calls = simulate_calls()
    pauses = within_turn_pauses(calls)
    print(f"{len(calls)} simulated calls, {sum(len(c) for c in calls)} turns, {len(pauses)} pauses inside turns")
    print(f"Pauses inside turns longer than 0.5 s: {100 * np.mean(pauses > 0.5):.0f}%, "
          f"longer than 1.0 s: {100 * np.mean(pauses > 1.0):.0f}%")
    for text in ("số điện thoại của em là", "số điện thoại của em là 0903", "số điện thoại của em là 0903 123 456",
                 "em muốn hỏi về", "hóa đơn tháng này ạ", "không"):
        print(f"  p(end of turn | '{text}') = {eou_probability(text):.2f}")
    policies = {
        "fixed 0.5 s": FixedEndpointing(0.5),
        "fixed 1.2 s": FixedEndpointing(1.2),
        "dynamic 0.5-3.0 s": DynamicEndpointing(0.5, 3.0),
        "semantic 0.5-3.0 s": SemanticEndpointing(0.5, 3.0),
    }
    for name, policy in policies.items():
        r = run_policy(policy, calls)
        print(f"{name:>20}: {100 * r['cutoff_rate']:4.1f}% of turns cut off, caller waits "
              f"{r['mean_wait_s']:.2f} s on average (p95 {r['p95_wait_s']:.2f} s)")


if __name__ == "__main__":
    main()
