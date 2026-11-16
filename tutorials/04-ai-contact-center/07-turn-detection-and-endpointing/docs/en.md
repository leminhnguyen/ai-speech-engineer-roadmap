# Turn Detection and Endpointing: Knowing When the Caller Has Finished

> Reply too early and you answer half a phone number. Reply too late and the caller says "alô?" into the silence. Endpointing is the decision in between, made dozens of times per call.

**Type:** Learn + Build  
**Language:** Python  
**Prerequisites:** [Voice Activity Detection](../../06-voice-activity-detection/docs/en.md), [LiveKit Voice Agent Architecture](../../02-livekit-voice-agent-architecture/docs/en.md)  
**Time:** ~70 minutes

## Learning Objectives

- Separate the three questions VAD, endpointing and turn detection answer, and where each adds latency.
- Quantify the trade-off between cutting callers off and making them wait, using pause distributions.
- Implement fixed, dynamic (per-caller adaptive) and semantic (transcript-aware) endpointing and compare them on the same calls.
- Configure LiveKit's `TurnHandlingOptions` for turn detection and endpointing, and choose a strategy for Vietnamese, which LiveKit's turn detectors do not cover.

## The Problem

The VAD in lesson 06 reports END_OF_SPEECH when the caller has been silent for `min_silence_duration`. That tells you the caller *stopped*. It does not tell you the caller *finished*.

Phone conversations are full of pauses inside a turn. A caller reading a phone number says "*số điện thoại của em là*", pauses, "*0903*", pauses, "*123 456*". A caller giving an address stops to remember the ward. A caller changing a plan says "*em muốn đổi gói cước*", thinks, and adds "*sang gói tháng nhé*". If the agent treats each pause as the end of the turn, it answers fragments: it confirms a 4-digit phone number, or starts explaining all plans before the caller has said which one.

The opposite error is just as visible. A long fixed wait makes every reply feel slow, and on a phone line, where callers cannot see that the agent is "thinking", two seconds of silence sounds like a dropped call.

![Timeline of a caller reading a phone number in three chunks with pauses of 0.4 and 0.8 seconds. A fixed 0.5 second silence timer expires in the second pause, and the agent talks over the caller. A turn-aware timer sees end-of-turn probabilities of 0.00 and 0.07 while the number is incomplete, keeps waiting, and replies 0.5 seconds after the number is complete at probability 0.60.](../assets/concept.svg)

## Three Questions, Three Components

| Question | Component | Typical signal | Adds latency? |
| --- | --- | --- | --- |
| Is there speech in this frame? | VAD | 32 ms frame probability | Yes: `min_silence_duration` before END_OF_SPEECH |
| How long should silence last before we act? | Endpointing | Silence timer between `min_delay` and `max_delay` | Yes: the timer itself |
| Is the turn semantically complete? | Turn detector | Words so far, and in LiveKit's audio model also intonation | Inference time, but it lets the timer be short when the answer is "yes" |

The turn detector does not replace the timer. It *chooses* the timer: when it is confident the caller has finished, the agent waits `min_delay`; when it thinks the caller will continue, the agent waits up to `max_delay`.

## Pauses Inside a Turn

![Histogram of pauses inside turns. Hesitations inside a request cluster around 0.25 to 0.45 seconds; pauses between groups of digits are longer, spread from 0.3 to beyond 1.5 seconds. A 0.5 second timer is shorter than 43% of all pauses, and a 1.0 second timer is shorter than 9%.](../assets/generated/pause-distributions.png)

The simulated calls in `main.py` mix ten Vietnamese call-center turn types: phone numbers and customer codes read in groups, addresses, requests with a hesitation, and one-chunk answers such as "*dạ vâng*". The mix deliberately over-represents chunked turns, because those are where endpointing fails. Every caller also has a personal tempo: some pause half as long as average, some twice as long.

The histogram makes the problem concrete. Any fixed timer is a threshold on this distribution: every pause longer than the timer becomes a cut-off. A 0.5 s timer is shorter than 43% of pauses; even 1.0 s is shorter than 9% of them.

## Three Policies

**Fixed.** Reply after `delay` seconds of silence. Cut-off rate is the share of turns containing a pause longer than `delay`; every finished turn waits exactly `delay`.

**Dynamic.** Learn this caller's pauses. The policy keeps an exponential moving average of the pauses after which the caller continued, and waits a multiple of it, clipped to `[min_delay, max_delay]`:

$$\bar p \leftarrow \alpha\,\bar p + (1-\alpha)\,p_{\text{observed}}, \qquad d = \operatorname{clip}(2\bar p,\ d_{\min},\ d_{\max})$$

A fast talker gets a short timer and a slow, elderly caller a long one. This is the idea behind LiveKit's `endpointing={"mode": "dynamic"}`, which adapts within `[min_delay, max_delay]` from pause statistics with an EMA coefficient `alpha`. The code here illustrates the idea; it is not LiveKit's exact algorithm.

**Semantic.** Read the words. A toy end-of-utterance model scores three cues in the transcript so far: a trailing connector that needs a continuation ("*là*", "*về*", "*của*"), a number that is still too short (fewer than 10 digits after "*số điện thoại*"), and a Vietnamese sentence-final particle that usually closes a turn ("*ạ*", "*nhé*", "*nha*"):

$$p_{\text{EOU}} = \sigma(0.4 + 2.2\,[\text{particle}] - 3.0\,[\text{connector}] - 3.0\,[\text{open number}]), \qquad d = \begin{cases} d_{\min} & p_{\text{EOU}} \ge 0.5 \\ d_{\max} & \text{otherwise} \end{cases}$$

![Cut-off rate against average waiting time for the three policies. The fixed timer runs from 52% cut-offs at 0.2 seconds to under 1% at 2 seconds, with 0.5 seconds giving 29%. The dynamic policy lies slightly below the fixed curve. The semantic policy lies far below both: 10% cut-offs at 0.5 seconds of waiting and 2% at 1 second.](../assets/generated/endpointing-tradeoff.png)

Every policy is a curve, not a point: you choose where to sit on it. The results on the same 200 simulated calls:

| Policy | Turns cut off | Average wait after finishing | p95 wait |
| --- | ---: | ---: | ---: |
| Fixed 0.5 s | 28.6% | 0.50 s | 0.50 s |
| Fixed 1.2 s | 4.7% | 1.20 s | 1.20 s |
| Dynamic, 0.5–3.0 s | 9.1% | 0.88 s | 1.25 s |
| Semantic, 0.5–3.0 s | 9.8% | 0.50 s | 0.50 s |

Knowing the caller helps a little on average and a lot for outliers: for one caller who pauses twice as long as average, a fixed 0.5 s timer cuts off 47% of turns and the dynamic policy 13%, at the price of a 1.9 s wait. Knowing the words helps a lot, because the most dangerous pauses, inside numbers and after connectors, are exactly the ones the transcript explains. The semantic policy still fails where the words mislead: "*không*" (no) looks complete but is often followed by "*em không đặt đơn đó*", and a plain clause such as "*em muốn đổi gói cước*" gives no hint that more is coming.

![Silences in one slow caller's call, in order. Bars show pauses inside turns (0.4 to 0.9 seconds); ticks mark ends of turns. The dynamic timer rises from 0.8 to about 1.25 seconds as it learns this caller's long pauses. The semantic timer sits at 0.5 seconds for most silences and jumps to 3.0 seconds where a number is still open.](../assets/generated/delay-trace.png)

## Build It

`code/main.py` simulates calls and replays them through each policy:

| Function or class | What it does |
| --- | --- |
| `simulate_call`, `simulate_calls` | Callers with a personal tempo, turns from ten templates, log-normal pauses by kind |
| `eou_features`, `eou_probability` | Transcript cues and a logistic end-of-utterance score |
| `FixedEndpointing`, `DynamicEndpointing`, `SemanticEndpointing` | The three policies, sharing `reset`, `delay_for` and `observe` |
| `run_policy` | Replays calls: a timer that expires inside a turn is a cut-off; at the end of a turn, the delay is the caller's wait |
| `within_turn_pauses` | Pause lengths for the histogram, optionally by kind |

The replay loop is the whole evaluation:

```python
for pause in turn:
    delay = policy.delay_for(pause)
    if pause.final:
        waits.append(delay)               # caller finished: they wait for the timer
    elif pause.length_s >= delay:
        cut_turns += 1                    # timer fired while the caller was mid-turn
        policy.observe(pause)             # ...and the caller kept talking: learn from it
        break
    policy.observe(pause)                 # caller resumed: adaptive policies learn
```

Run it from this lesson directory:

```bash
python3 code/main.py
python3 -m unittest discover -s code/tests -v
python3 code/generate_figures.py --output-dir assets/generated
```

Expected output: 43% of pauses inside turns are longer than 0.5 s; `p(end of turn | 'số điện thoại của em là 0903') = 0.07` and `0.60` once all ten digits are present; and the policy table above.

## Use It in Practice: Turn Handling in LiveKit

In `livekit-agents` 1.8, turn settings live in one nested option, `turn_handling=TurnHandlingOptions(...)`. The older flat arguments such as `min_endpointing_delay` are deprecated.

| Setting | Default | Meaning |
| --- | --- | --- |
| `turn_detection` | `inference.TurnDetector()` | Also `"stt"`, `"vad"`, `"realtime_llm"`, `"manual"` |
| `endpointing["mode"]` | `"fixed"` | `"dynamic"` adapts within the bounds from pause statistics |
| `endpointing["min_delay"]` | 0.5 s (0.3 s with the audio turn detector) | Wait when the turn looks finished; in `"vad"` mode the wait is roughly the larger of this and the VAD's silence duration |
| `endpointing["max_delay"]` | 3.0 s (2.5 s with the audio turn detector) | Upper bound when the turn looks unfinished |
| `endpointing["alpha"]` | 0.9 | EMA coefficient for dynamic mode |
| `user_turn_limit` | none | `max_words` / `max_duration`; when exceeded, `Agent.on_user_turn_exceeded` runs |

**Turn detection modes.** The default audio turn detector (`v1` on LiveKit Inference, `v1-mini` on local CPU) listens to the audio itself, so it hears intonation as well as words. The older text detector (`MultilingualModel`) is deprecated and will be removed in 2.0. `"stt"` uses the STT provider's own end-of-utterance signal. `"vad"` uses silence only. `"manual"` leaves the decision to your code through `session.commit_user_turn()`, `session.clear_user_turn()` and `session.interrupt()`.

**Vietnamese.** LiveKit's turn detectors support 14 languages: English, Arabic, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Dutch, Portuguese, Turkish and Chinese. Vietnamese is not among them. For uncovered languages the docs recommend `turn_detection="vad"`, or `"stt"` with a provider that detects end of utterance, and tuning the delays. A reasonable starting point for a Vietnamese hotline:

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import silero

session = AgentSession(
    vad=silero.VAD.load(min_silence_duration=0.4),
    # stt=..., llm=..., tts=...
    turn_handling=TurnHandlingOptions(
        turn_detection="vad",  # Vietnamese is not covered by LiveKit's turn detectors
        endpointing={"mode": "dynamic", "min_delay": 0.6, "max_delay": 2.5},
        user_turn_limit={"max_duration": 30.0},
    ),
)
```

Then measure, as this lesson does, on your own recorded calls: the cut-off rate per turn type and the wait after finished turns. Three further levers help with Vietnamese:

- **Structure the hard turns.** For phone numbers, customer codes and OTPs, ask for them in a dedicated step and accept DTMF keypresses ("bấm phím # khi xong"). In manual mode, a `#` received through `sip_dtmf_received` can call `session.commit_user_turn()`.
- **Change the delays per step.** `session.update_options(endpointing_opts={...})` changes endpointing at runtime: lengthen `min_delay` while collecting a number or an address, shorten it for yes/no confirmations.
- **Design for cut-offs anyway.** Tell the LLM that the caller may have been interrupted and that it should ask the caller to continue when an entity looks incomplete, instead of confirming it.

Options and defaults follow the LiveKit docs for `livekit-agents` 1.8 ([turn handling options](https://docs.livekit.io/reference/agents/turn-handling-options), [turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector), [tuning turns](https://docs.livekit.io/agents/logic/turns/tuning)).

## Pitfalls

- **Lowering the VAD silence to cut latency.** It shortens every pause the system tolerates, and the cut-off rate explodes. Endpointing and turn detection are the right levers.
- **Averaging away the hard turns.** A 5% overall cut-off rate can hide a 40% rate on phone numbers. Report cut-offs per turn type.
- **Measuring latency only on finished turns.** A policy that replies fast by replying early looks great on latency. Always report cut-offs next to the wait.
- **Assuming English turn-taking.** Vietnamese uses sentence-final particles and different hesitation patterns; English-tuned thresholds and English-only turn models do not transfer.
- **Ignoring backchannels.** "*dạ*", "*vâng*", "*ừ*" are often not turns at all. They are covered in lesson 08 on barge-in.

## Exercises

1. Add a turn template for a 6-digit OTP read digit by digit. Find the fixed delay that keeps cut-offs on that template below 5%, and the average wait it costs on all other turns.
2. Change the semantic policy to interpolate, `d = d_max − p_EOU (d_max − d_min)`, and compare it with the threshold version on the trade-off plot.
3. Combine the two adaptive policies: use the semantic decision, but replace `max_delay` with the dynamic timer. Does the combination dominate both?
4. Make the toy EOU model wrong 10% of the time at random. How quickly does the semantic curve approach the fixed curve?
5. Record ten real calls, mark each pause as inside or at the end of a turn, and fit the log-normal parameters used in `PAUSE_MEDIAN_S` and `PAUSE_SIGMA`.

## Key Terms

| Term | Meaning |
| --- | --- |
| End of speech | The VAD's decision that the caller stopped making sound. |
| End of turn (EOU) | The decision that the caller has finished what they wanted to say. |
| Endpointing | The silence timer that converts end of speech into end of turn. |
| Cut-off | Replying during a pause inside the caller's turn. |
| Dynamic endpointing | A timer adapted to the caller's own pause statistics. |
| Semantic turn detection | Choosing the timer from what was said (and, in audio models, how it was said). |
| Sentence-final particle | A Vietnamese word such as "*ạ*", "*nhé*" or "*nha*" that typically closes an utterance. |

## Further Reading

- M. Heldner, J. Edlund, "Pauses, gaps and overlaps in conversations", *Journal of Phonetics*, 2010
- G. Skantze, "Turn-taking in conversational systems and human-robot interaction: a review", *Computer Speech & Language*, 2021
- [LiveKit: turns overview](https://docs.livekit.io/agents/logic/turns)
- [LiveKit: turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector)
- [LiveKit: turn handling options reference](https://docs.livekit.io/reference/agents/turn-handling-options)
