# ☎️ 04 · AI Contact Center with LiveKit

> A voice agent that works in a browser demo can still fail on a phone line. This module covers the part in between.

[← All modules](../README.md) · [🗺️ Roadmap](../../roadmap/README.md) · [📡 SOTA: real-time dialogue](../../sota/topics/dialogue.md)

An AI contact center (AICC) answers real phone calls. The caller's voice arrives through the phone network as narrowband 8 kHz audio, sometimes with packet loss, background noise, or an echo of the agent's own voice. The agent has to decide when the caller has finished speaking, let the caller interrupt it, reply in under a second, and hand the call to a human when it should.

The module starts with the protocols (SIP, RTP, G.711), works through the audio problems (jitter, loss, noise, echo), then the conversation-timing problems (VAD, end-of-turn detection, barge-in), and ends with latency and production operations. Each lesson builds the underlying algorithm from scratch in NumPy, tests it, and then shows the matching [LiveKit Agents](https://docs.livekit.io/agents) configuration.

![Course map: caller audio crosses the phone network (lessons 01 and 03), LiveKit SIP and the room (02), the audio front-end for noise and echo (04 and 05) and turn-taking with VAD, endpointing and barge-in (06 to 08) before reaching the STT, LLM and TTS agent, whose speech returns to the caller. Lesson 09 budgets the latency of that loop and lesson 10 covers transfers, keypad input, outbound calls and operations.](../../assets/aicc-course-map.svg)

## 📚 Lessons

| | Lesson | Time | What you will build |
| :---: | --- | :---: | --- |
| | **Part 1 · The phone call** | | |
| ✅ | [01 · Telephony and SIP](01-telephony-and-sip/docs/en.md) | ~70 min | Parse SIP and SDP, packetize RTP, implement bit-exact G.711 µ-law, simulate the 300–3400 Hz band, and detect DTMF with Goertzel. |
| ✅ | [02 · LiveKit Voice Agent Architecture](02-livekit-voice-agent-architecture/docs/en.md) | ~70 min | Map trunks, dispatch rules, rooms and `AgentSession` onto the call; build a rational resampler and frame chunker for 8/16/48 kHz audio. |
| | **Part 2 · Audio quality** | | |
| ✅ | [03 · Line Quality: Jitter and Packet Loss](03-line-quality-jitter-and-loss/docs/en.md) | ~75 min | Simulate bursty loss and jitter, build fixed and adaptive jitter buffers, compare concealment methods, and estimate MOS with the E-model. |
| ✅ | [04 · Noise Suppression](04-noise-suppression/docs/en.md) | ~75 min | Track the noise floor, implement spectral subtraction and a Wiener filter, and measure the noise-reduction versus speech-distortion trade-off. |
| ✅ | [05 · Echo Cancellation](05-echo-cancellation/docs/en.md) | ~75 min | Cancel echo with an NLMS adaptive filter, measure ERLE, and protect adaptation with double-talk detection. |
| | **Part 3 · Conversation timing** | | |
| ✅ | [06 · Voice Activity Detection](06-voice-activity-detection/docs/en.md) | ~60 min | Build a VAD with hysteresis and duration rules that mirror Silero's parameters, and evaluate it across SNRs. |
| ✅ | [07 · Turn Detection and Endpointing](07-turn-detection-and-endpointing/docs/en.md) | ~70 min | Trade cut-offs against latency with fixed, dynamic and semantic endpointing on simulated dialogs. |
| ✅ | [08 · Barge-in and Interruptions](08-barge-in-and-interruptions/docs/en.md) | ~70 min | Separate real interruptions from backchannels and echo, and truncate the agent's transcript to what the caller heard. |
| | **Part 4 · Production** | | |
| ✅ | [09 · Latency Budget](09-latency-budget/docs/en.md) | ~70 min | Simulate per-stage latency, read p50/p95, and quantify preemptive generation, streaming and co-location. |
| ✅ | [10 · Production AICC](10-production-aicc/docs/en.md) | ~80 min | Model warm transfers as a state machine, size the human team with Erlang C, and compute contact-center KPIs. |

## 🎯 Before You Start

- **Required:** [01 · Audio Signal Processing](../README.md#-01--audio-signal-processing), especially [Digital Audio](../01-audio-signal-processing/02-digital-audio/docs/en.md), [the DFT](../01-audio-signal-processing/04-discrete-fourier-transform/docs/en.md) and [the STFT](../01-audio-signal-processing/05-short-time-fourier-transform/docs/en.md).
- **Helpful:** a basic idea of what ASR and TTS models do, and some `asyncio` if you want to run the LiveKit examples.
- **Not required:** a LiveKit account or a phone number. Every lesson's code, tests and notebook run offline with NumPy. The LiveKit snippets are references for when you build the real agent.

## 🧭 Design Principles

- **Design for 8 kHz.** Most calls arrive as narrowband G.711. Wideband (G.722 HD voice) is a bonus, not an assumption.
- **Measure from the caller's ear.** Latency is measured from the end of the caller's speech to the first audio the caller hears. Quality is judged on what reaches the STT model.
- **Code owns state, the model owns wording.** Transfers, bookings and payments are committed by code after explicit confirmation, never because the LLM said so.
- **Vietnamese is a first-class case.** Tones survive the phone band, short backchannels (*dạ*, *vâng*, *ừ*) look like interruptions, and LiveKit's turn-detector models do not cover Vietnamese yet. The lessons show what to do instead.

## 🔖 LiveKit Version

The LiveKit snippets target `livekit-agents` **1.8** and were checked against the [LiveKit docs](https://docs.livekit.io) in November 2026. In 1.8, turn-taking settings live under `AgentSession(turn_handling=TurnHandlingOptions(...))`, and audio input options under `session.start(room_options=room_io.RoomOptions(...))`. The older flat arguments such as `min_endpointing_delay` and `allow_interruptions` are deprecated. If you are on a newer version, check the [changelog](https://github.com/livekit/agents/releases) before copying a snippet.

## 🚀 Run the Lessons

From the repository root:

```bash
python3 tutorials/04-ai-contact-center/01-telephony-and-sip/code/main.py
python3 -m unittest discover -s tutorials/04-ai-contact-center/01-telephony-and-sip/code/tests -v
```

`make test`, `make figures` and `make notebooks` include this module automatically.
