"""Lesson 10: running an AI contact center: hand-offs, keypads, KPIs and capacity.

Builds the operational pieces around a voice agent with NumPy and the standard library:
a warm-transfer state machine with timeouts and fallbacks, keypad (DTMF) digit collection,
PII masking for logs, KPIs over a call log, and Erlang B / Erlang C capacity planning for the
agent workers and the human team behind the AI.

Run: python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

import numpy as np

# --------------------------------------------------------------------------- warm transfer

TRANSITIONS = {
    ("ai_conversation", "transfer_requested"): "caller_on_hold",
    ("ai_conversation", "caller_hung_up"): "abandoned",
    ("caller_on_hold", "dial_started"): "dialing_human",
    ("caller_on_hold", "caller_hung_up"): "abandoned",
    ("dialing_human", "human_answered"): "briefing",
    ("dialing_human", "ringing_timeout"): "callback_offered",
    ("dialing_human", "caller_hung_up"): "abandoned",
    ("briefing", "human_accepted"): "connected",
    ("briefing", "human_declined"): "ai_conversation",
    ("briefing", "voicemail_detected"): "callback_offered",
    ("briefing", "caller_hung_up"): "abandoned",
    ("connected", "ai_left"): "completed",
    ("callback_offered", "callback_booked"): "completed",
    ("callback_offered", "caller_hung_up"): "abandoned",
}
TERMINAL = {"completed", "abandoned"}
ON_HOLD = {"caller_on_hold", "dialing_human", "briefing"}


def next_state(state: str, event: str) -> str:
    """One legal transition, or ValueError. Illegal events are bugs, not edge cases."""
    if state in TERMINAL:
        raise ValueError(f"{state!r} is terminal; got {event!r}")
    try:
        return TRANSITIONS[(state, event)]
    except KeyError:
        raise ValueError(f"illegal event {event!r} in state {state!r}") from None


@dataclass
class TransferTrace:
    states: list = field(default_factory=list)
    times: list = field(default_factory=list)

    @property
    def final(self) -> str:
        return self.states[-1]

    def hold_seconds(self) -> float:
        """Total time the caller spent hearing hold music instead of a person or the AI."""
        total = 0.0
        for (state, start), end in zip(zip(self.states, self.times), self.times[1:]):
            if state in ON_HOLD:
                total += end - start
        return total


def run_transfer(events, ringing_timeout: float = 30.0, start_state: str = "ai_conversation") -> TransferTrace:
    """Replay timestamped events ``[(t, event), ...]``. Inserts a ``ringing_timeout`` event
    when the human has not answered ``ringing_timeout`` seconds after dialing started."""
    trace = TransferTrace([start_state], [0.0])
    dial_started_at = None
    for t, event in sorted(events, key=lambda item: item[0]):
        if trace.final == "dialing_human" and dial_started_at is not None and t - dial_started_at > ringing_timeout:
            trace.states.append(next_state(trace.final, "ringing_timeout"))
            trace.times.append(dial_started_at + ringing_timeout)
        if trace.final in TERMINAL:
            break
        try:
            state = next_state(trace.final, event)
        except ValueError:
            if trace.final == "callback_offered" and event == "human_answered":
                continue  # the human answered too late; the caller has already moved on
            raise
        if event == "dial_started":
            dial_started_at = t
        trace.states.append(state)
        trace.times.append(t)
    return trace


# --------------------------------------------------------------------------- keypad input

def collect_digits(presses, num_digits: int, stop_digit: str = "#", inter_digit_timeout: float = 4.0,
                   start_time: float = 0.0) -> tuple:
    """Collect DTMF digits ``[(t, digit), ...]`` the way an IVR prompt does.

    Stops when ``num_digits`` are collected (``"complete"``), when ``stop_digit`` is pressed
    (``"terminated"``), or when the gap since the prompt or the previous digit exceeds
    ``inter_digit_timeout`` (``"timeout"``). Defaults mirror LiveKit's ``GetDtmfTask``.
    """
    digits, last = "", start_time
    for t, digit in sorted(presses, key=lambda item: item[0]):
        if t - last > inter_digit_timeout:
            return digits, "timeout"
        if digit == stop_digit:
            return digits, "terminated"
        digits += digit
        last = t
        if len(digits) == num_digits:
            return digits, "complete"
    return digits, "timeout"


DIGIT_RUN = re.compile(r"(?<!\d)(?:\d[\s.\-]?){6,}\d(?!\d)")


def mask_pii(text: str, keep_last: int = 3) -> str:
    """Mask long digit runs (phone, card, ID numbers) in transcripts and logs, keeping the last digits."""
    def _mask(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return "*" * (len(digits) - keep_last) + digits[-keep_last:]
    return DIGIT_RUN.sub(_mask, text)


# --------------------------------------------------------------------------- KPIs

OUTCOMES = ("resolved", "transferred", "abandoned", "dropped")
TRANSFER_REASONS = ("asked for a human", "two failed attempts", "complaint", "out of scope")


@dataclass(frozen=True)
class CallRecord:
    call_id: str
    hour: int
    outcome: str
    duration_s: float
    turn_latencies_ms: tuple
    transfer_reason: str | None = None


def generate_call_log(n: int = 2000, seed: int = 0) -> list:
    """A synthetic but plausible day of inbound calls to an AI hotline."""
    rng = np.random.default_rng(seed)
    hours = rng.choice(np.arange(24), size=n, p=hourly_profile() / hourly_profile().sum())
    records = []
    for i in range(n):
        outcome = str(rng.choice(OUTCOMES, p=[0.64, 0.24, 0.08, 0.04]))
        turns = int(rng.integers(2, 14)) if outcome != "abandoned" else int(rng.integers(0, 3))
        latencies = tuple(np.round(rng.lognormal(math.log(950), 0.25, turns)).astype(int).tolist())
        duration = float(rng.lognormal(math.log(150 if outcome == "resolved" else 110), 0.5))
        reason = str(rng.choice(TRANSFER_REASONS, p=[0.45, 0.25, 0.2, 0.1])) if outcome == "transferred" else None
        records.append(CallRecord(f"call-{i:05d}", int(hours[i]), outcome, duration, latencies, reason))
    return records


def compute_kpis(log: list) -> dict:
    """The numbers an AICC is judged on. Every rate uses all offered calls as its denominator."""
    n = len(log)
    counts = {o: sum(r.outcome == o for r in log) for o in OUTCOMES}
    handled = [r.duration_s for r in log if r.outcome in ("resolved", "transferred")]
    latencies = np.array([x for r in log for x in r.turn_latencies_ms], dtype=float)
    reasons = {reason: sum(r.transfer_reason == reason for r in log) for reason in TRANSFER_REASONS}
    return {
        "calls": n,
        "containment_rate": counts["resolved"] / n,
        "transfer_rate": counts["transferred"] / n,
        "abandonment_rate": counts["abandoned"] / n,
        "drop_rate": counts["dropped"] / n,
        "aht_s": float(np.mean(handled)) if handled else 0.0,
        "p50_turn_latency_ms": float(np.percentile(latencies, 50)) if latencies.size else 0.0,
        "p95_turn_latency_ms": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
        "transfer_reasons": reasons,
    }


# --------------------------------------------------------------------------- capacity

def hourly_profile() -> np.ndarray:
    """Relative inbound volume by hour of day: a morning and an afternoon peak."""
    hours = np.arange(24)
    profile = 0.03 + np.exp(-0.5 * ((hours - 10) / 1.6) ** 2) + 0.8 * np.exp(-0.5 * ((hours - 15) / 1.8) ** 2)
    return profile / profile.max()


def offered_load(calls_per_hour: float, aht_s: float) -> float:
    """Traffic in Erlangs: the average number of calls in progress (Little's law)."""
    return calls_per_hour * aht_s / 3600.0


def erlang_b(traffic: float, servers: int) -> float:
    """Probability that a new call finds all ``servers`` busy and is blocked (no queue)."""
    b = 1.0
    for k in range(1, servers + 1):
        b = traffic * b / (k + traffic * b)
    return b


def erlang_c(traffic: float, agents: int) -> float:
    """Probability that a call has to wait in queue (infinite patience, M/M/c)."""
    if agents <= traffic:
        return 1.0
    b = erlang_b(traffic, agents)
    return agents * b / (agents - traffic * (1 - b))


def service_level(traffic: float, agents: int, aht_s: float, target_s: float) -> float:
    """Share of calls answered within ``target_s`` seconds."""
    if agents <= traffic:
        return 0.0
    return 1.0 - erlang_c(traffic, agents) * math.exp(-(agents - traffic) * target_s / aht_s)


def average_speed_of_answer(traffic: float, agents: int, aht_s: float) -> float:
    if agents <= traffic:
        return math.inf
    return erlang_c(traffic, agents) * aht_s / (agents - traffic)


def agents_for_service_level(calls_per_hour: float, aht_s: float, target_sl: float = 0.8,
                             target_s: float = 20.0) -> int:
    """Fewest humans that answer ``target_sl`` of calls within ``target_s`` (the 80/20 rule)."""
    traffic = offered_load(calls_per_hour, aht_s)
    agents = max(1, math.floor(traffic) + 1)
    while service_level(traffic, agents, aht_s, target_s) < target_sl:
        agents += 1
    return agents


def channels_for_blocking(traffic: float, max_blocking: float = 0.01) -> int:
    """SIP channels or concurrent agent sessions needed so at most ``max_blocking`` of calls get a busy signal."""
    servers = 1
    while erlang_b(traffic, servers) > max_blocking:
        servers += 1
    return servers


def workers_needed(concurrent_sessions: int, sessions_per_worker: int, load_threshold: float = 0.7) -> int:
    """Agent worker hosts, keeping each below the load at which it stops accepting jobs."""
    return math.ceil(concurrent_sessions / max(1, math.floor(sessions_per_worker * load_threshold)))


def main() -> None:
    happy = [(5, "transfer_requested"), (6, "dial_started"), (14, "human_answered"),
             (35, "human_accepted"), (37, "ai_left")]
    no_answer = [(5, "transfer_requested"), (6, "dial_started"), (50, "human_answered")]
    declined = [(5, "transfer_requested"), (6, "dial_started"), (12, "human_answered"), (25, "human_declined")]
    for name, events in (("accepted", happy), ("no answer", no_answer), ("declined", declined)):
        trace = run_transfer(events)
        print(f"Warm transfer, {name:<9}: {' -> '.join(trace.states)} (hold {trace.hold_seconds():.0f} s)")

    presses = [(1.2, "0"), (1.8, "9"), (2.3, "0"), (2.9, "1"), (3.4, "2"), (4.0, "3"), (4.4, "#")]
    print(f"Keypad entry: {collect_digits(presses, num_digits=10)}")
    print(mask_pii("Số điện thoại của tôi là 0901 234 567, CCCD 079123456789."))

    kpis = compute_kpis(generate_call_log())
    print(f"Containment {kpis['containment_rate']:.0%}, transfers {kpis['transfer_rate']:.0%}, "
          f"abandoned {kpis['abandonment_rate']:.0%}, AHT {kpis['aht_s']:.0f} s, "
          f"p95 turn latency {kpis['p95_turn_latency_ms']:.0f} ms")

    peak_calls, ai_aht, human_aht = 1200, 160, 300
    ai_traffic = offered_load(peak_calls, ai_aht)
    channels = channels_for_blocking(ai_traffic)
    transfers = peak_calls * kpis["transfer_rate"]
    humans = agents_for_service_level(transfers, human_aht)
    print(f"Peak hour: {peak_calls} calls -> {ai_traffic:.0f} Erlangs of AI traffic, "
          f"{channels} concurrent sessions for 1% blocking, "
          f"{workers_needed(channels, sessions_per_worker=25)} worker hosts at 25 sessions each")
    print(f"{transfers:.0f} transfers/hour need {humans} human agents for 80% answered in 20 s "
          f"(ASA {average_speed_of_answer(offered_load(transfers, human_aht), humans, human_aht):.0f} s)")


if __name__ == "__main__":
    main()
