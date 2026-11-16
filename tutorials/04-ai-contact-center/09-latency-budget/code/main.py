"""Lesson 09: the latency budget of one conversational turn on a phone line.

Monte-Carlo simulation of the time from the end of the caller's speech to the first agent
audio the caller hears, stage by stage, with the optimizations that move it: tuned
endpointing, preemptive generation, provider co-location, streaming, sentence chunking and
fillers. NumPy and the standard library only.

Run: python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

import numpy as np

Z95 = 1.6448536269514722  # standard normal 95th percentile


@dataclass(frozen=True)
class Stage:
    """One stage of the critical path, described by its median and 95th percentile (ms)."""

    name: str
    median_ms: float
    p95_ms: float

    def lognormal(self) -> tuple:
        """(mu, sigma) of a lognormal with this median and p95. Latency is right-skewed."""
        if self.p95_ms < self.median_ms or self.median_ms <= 0:
            raise ValueError("need 0 < median <= p95")
        return math.log(self.median_ms), (math.log(self.p95_ms) - math.log(self.median_ms)) / Z95

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        mu, sigma = self.lognormal()
        return rng.lognormal(mu, sigma, n) if sigma > 0 else np.full(n, self.median_ms)


@dataclass(frozen=True)
class PipelineConfig:
    """Everything that shapes one turn's latency. Defaults: an unoptimized phone agent."""

    uplink: Stage = Stage("uplink", 60, 140)            # phone -> carrier -> LiveKit SIP
    jitter: Stage = Stage("jitter buffer", 40, 80)
    end_of_turn: Stage = Stage("end of turn", 600, 1100)  # silence wait + endpointing delay
    stt_final: Stage = Stage("stt final", 120, 300)       # runs in parallel with the silence wait
    llm_ttft: Stage = Stage("llm ttft", 450, 1100)
    tts_ttfb: Stage = Stage("tts ttfb", 220, 500)
    downlink: Stage = Stage("downlink", 80, 160)          # LiveKit -> carrier -> phone
    provider_rtt_ms: float = 180.0      # agent <-> LLM/TTS provider round trip (other region)
    preemptive: bool = False            # start the LLM before the turn is committed
    interim_lag_ms: float = 150.0       # when a usable interim transcript exists after speech ends
    p_transcript_change: float = 0.2   # chance the committed transcript differs -> redo the LLM call


STAGE_ORDER = ("uplink", "jitter buffer", "end of turn", "llm ttft", "tts ttfb", "downlink")


def simulate_turns(cfg: PipelineConfig, n: int = 20_000, seed: int = 0) -> dict:
    """Sample ``n`` turns. Returns per-stage critical-path contributions and ``total`` (ms).

    Time 0 is the end of the caller's speech at the phone. The turn is committed when both
    the end-of-turn wait and the final transcript are done (they overlap). With preemptive
    generation the LLM starts on the interim transcript; its first token cannot be used
    before the commit, and is thrown away if the committed transcript changed.
    """
    rng = np.random.default_rng(seed)
    up, jit = cfg.uplink.sample(rng, n), cfg.jitter.sample(rng, n)
    eot, stt = cfg.end_of_turn.sample(rng, n), cfg.stt_final.sample(rng, n)
    ttft, ttfb = cfg.llm_ttft.sample(rng, n), cfg.tts_ttfb.sample(rng, n)
    down = cfg.downlink.sample(rng, n)
    rtt = cfg.provider_rtt_ms

    arrive = up + jit
    committed = arrive + np.maximum(eot, stt)
    first_token = committed + rtt + ttft
    wasted = np.zeros(n, dtype=bool)
    if cfg.preemptive:
        changed = rng.random(n) < cfg.p_transcript_change
        early = np.maximum(committed, arrive + cfg.interim_lag_ms + rtt + ttft)
        first_token = np.where(changed, first_token, early)
        wasted = changed
    first_audio = first_token + rtt + ttfb
    total = first_audio + down
    return {
        "uplink": up, "jitter buffer": jit, "end of turn": committed - arrive,
        "llm ttft": first_token - committed, "tts ttfb": first_audio - first_token, "downlink": down,
        "total": total, "wasted_llm_calls": wasted,
    }


def summarize(values_ms: np.ndarray) -> dict:
    p50, p95, p99 = np.percentile(values_ms, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99),
            "under_1s": float(np.mean(values_ms < 1000))}


def optimization_ladder() -> list:
    """Cumulative optimizations, each on top of the previous one."""
    base = PipelineConfig()
    tuned = replace(base, end_of_turn=Stage("end of turn", 420, 800))
    preemptive = replace(tuned, preemptive=True)
    colocated = replace(preemptive, provider_rtt_ms=10.0)
    lean = replace(colocated, llm_ttft=Stage("llm ttft", 300, 650), tts_ttfb=Stage("tts ttfb", 150, 300))
    return [("baseline", base), ("+ tuned endpointing", tuned), ("+ preemptive generation", preemptive),
            ("+ co-located providers", colocated), ("+ smaller prompt, faster TTS", lean)]


# --------------------------------------------------------------------------- streaming

SPEAKING_RATE_WPS = 2.6   # words per second of synthesized speech
TOKENS_PER_WORD = 1.4


def split_for_tts(text: str, min_words: int = 4) -> list:
    """Cut a reply into chunks TTS can start on: sentence ends always, commas once a chunk is long enough."""
    chunks, current = [], []
    for token in text.split():
        current.append(token)
        ends_sentence = re.search(r"[.!?]$", token) is not None
        ends_clause = token.endswith(",") and len(current) >= min_words
        if ends_sentence or ends_clause:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def first_audio_ms(text: str, streaming_llm: bool, streaming_tts: bool, ttft_ms: float = 400.0,
                   tokens_per_s: float = 60.0, tts_ttfb_ms: float = 200.0, tts_rtf: float = 0.15,
                   min_words: int = 4) -> float:
    """Time from the LLM request to the first synthesized audio for one reply.

    Without LLM streaming, TTS waits for the whole reply. Without TTS streaming, the first
    audio waits for synthesis of the whole text it was given (``tts_rtf`` x its duration).
    """
    words = text.split()
    first_chunk = split_for_tts(text, min_words)[0].split()
    tts_words = first_chunk if streaming_llm else words
    text_ready = ttft_ms + len(tts_words) * TOKENS_PER_WORD / tokens_per_s * 1000
    if streaming_tts:
        return text_ready + tts_ttfb_ms
    audio_s = len(tts_words) / SPEAKING_RATE_WPS
    return text_ready + tts_ttfb_ms + tts_rtf * audio_s * 1000


def perceived_latency_ms(answer_ms: np.ndarray, filler_at_ms: float | None) -> np.ndarray:
    """What the caller experiences: the first sound, useful or not. Report it next to, never instead of, answer latency."""
    if filler_at_ms is None:
        return np.asarray(answer_ms, dtype=float)
    return np.minimum(answer_ms, filler_at_ms)


def main() -> None:
    print("Median / p95 time from end of caller speech to first agent audio (ms):")
    for name, cfg in optimization_ladder():
        result = simulate_turns(cfg)
        s = summarize(result["total"])
        wasted = result["wasted_llm_calls"].mean()
        extra = f"  wasted LLM calls {wasted:.0%}" if cfg.preemptive else ""
        print(f"  {name:<30} p50 {s['p50']:5.0f}  p95 {s['p95']:5.0f}  under 1 s {s['under_1s']:4.0%}{extra}")

    base = simulate_turns(PipelineConfig())
    shares = {k: float(np.median(base[k])) for k in STAGE_ORDER}
    largest = max(shares, key=shares.get)
    print(f"Largest median contribution in the baseline: {largest} ({shares[largest]:.0f} ms)")

    reply = ("Dạ, đơn hàng của anh sẽ được giao vào thứ năm. Phí giao hàng là ba mươi nghìn đồng, "
             "anh có muốn đổi địa chỉ không?")
    print(f"TTS chunks: {split_for_tts(reply)}")
    for llm in (False, True):
        for tts in (False, True):
            print(f"  streaming LLM={llm!s:<5} TTS={tts!s:<5} -> first audio {first_audio_ms(reply, llm, tts):5.0f} ms")

    tool_answer = np.full(5, 2600.0)
    print(f"Slow tool call: answer at {tool_answer[0]:.0f} ms, "
          f"first sound with a filler at 700 ms: {perceived_latency_ms(tool_answer, 700.0)[0]:.0f} ms")


if __name__ == "__main__":
    main()
