# Echo Cancellation: When the Agent Hears Itself

> A human on a phone call ignores their own voice coming back. A voice agent has no idea it is listening to itself unless something subtracts the echo first.

**Type:** Learn + Build  
**Language:** Python  
**Prerequisites:** [Telephony and SIP](../../01-telephony-and-sip/docs/en.md), [Line Quality: Jitter, Loss and Level](../../03-line-quality-jitter-and-loss/docs/en.md)  
**Time:** ~75 minutes

## Learning Objectives

- Distinguish line echo (from a 2-to-4-wire hybrid) from acoustic echo (speaker to microphone), and estimate how loud each is.
- Explain why echo hurts a voice agent far more than a human: false barge-in, self-transcription, and a filter that never settles.
- Implement an NLMS adaptive echo canceller and measure it with ERLE.
- Detect double talk with the Geigel detector and show why adaptation must freeze while the caller talks.
- Show the trade-off of a residual echo suppressor: it removes what the linear filter misses but can chop the caller.
- Map these mechanisms to where echo control actually lives in a LiveKit telephony deployment, and to the agent-side defenses you control.

## The Problem

On a call, the agent's synthesized voice leaves through the phone network, plays from the caller's earpiece or speakerphone, and some of it comes back through the caller's microphone. Old analog lines add a second source: the *hybrid* that converts the four-wire network to the two-wire local loop reflects part of the signal back. Either way, the microphone signal is

$$d(n) = \underbrace{(h * x)(n)}_{\text{echo}} + \underbrace{s(n)}_{\text{caller}} + \underbrace{v(n)}_{\text{noise}},$$

where $x$ is the far-end signal (the agent's voice), $h$ the unknown echo path, and $s$ the caller.

A person tolerates this. A voice agent does not. Its VAD sees energy while the agent is speaking and reports "the user started talking". The agent stops mid-sentence, waits, hears nothing, and perhaps starts again. If the echo is loud enough, STT transcribes the agent's own words, and the LLM answers itself. This is the most confusing failure mode in a phone deployment, because the logs show a user turn that the caller never spoke.

![Diagram of the echo loop: the agent's voice travels through the phone network to the handset speaker, leaks along the echo path into the handset microphone together with the caller's voice, and returns toward VAD and STT. An echo canceller fed with a reference copy of the far-end signal subtracts the echo; without it, VAD reports a false barge-in and the agent stops speaking.](../assets/concept.svg)

## Two Kinds of Echo

| | Line (hybrid) echo | Acoustic echo |
| --- | --- | --- |
| Where | 2-to-4-wire hybrid in the network or a PBX | Loudspeaker to microphone in the caller's device or room |
| Echo return loss (ERL) | Typically 6–20 dB below the far end | 0–30 dB; a speakerphone or car kit can be near 0 dB |
| Path length | Tens of milliseconds | 50–300 ms of room reverberation |
| Changes | Rarely | Every time the caller moves the phone or a door opens |
| Who cancels it | Carrier echo cancellers (ITU-T G.168) | The handset, the car kit, or the browser's WebRTC AEC |

The canceller in this lesson is the same idea in both cases: model the echo path with an adaptive FIR filter, predict the echo from the far-end reference, and subtract it.

## The NLMS Canceller

The canceller keeps a filter $\hat h$ of $L$ taps and the last $L$ far-end samples $\mathbf{x}(n)$. For each sample:

$$e(n) = d(n) - \hat{\mathbf{h}}^\top \mathbf{x}(n), \qquad \hat{\mathbf{h}} \leftarrow \hat{\mathbf{h}} + \frac{\mu\, e(n)\, \mathbf{x}(n)}{\mathbf{x}(n)^\top\mathbf{x}(n) + \epsilon}$$

The residual $e(n)$ is both the output (what goes on to VAD and STT) and the error signal that drives learning. Normalizing by the reference energy makes the step size independent of how loud the agent speaks. The regularization $\epsilon$ matters more than it looks: at the start of every syllable $\mathbf{x}^\top\mathbf{x}$ is tiny, and without a floor comparable to the noise energy over $L$ samples, background noise alone kicks the filter away from the solution. In this lesson's scenario, raising $\epsilon$ from $10^{-4}$ to $0.1$ moves the ERLE after four seconds from about 10 dB to almost 30 dB.

The figure of merit is **ERLE**, the echo return loss enhancement: how much quieter the echo is after the canceller.

$$\mathrm{ERLE} = 10 \log_{10} \frac{\mathbb{E}[\text{echo}^2]}{\mathbb{E}[\text{residual echo}^2]}$$

![Top: the microphone carries the agent's echo, and the residual after the NLMS canceller shrinks within two seconds, grows when the echo path changes at 3 seconds, and shrinks again. Bottom: ERLE rises from about 6 dB to about 30 dB, drops to near 2 dB at the path change, and recovers over the next two to three seconds.](../assets/generated/erle-convergence.png)

Two things to notice. Convergence takes seconds, not milliseconds, because speech is a poorly conditioned training signal: it is coloured, intermittent, and pitched. The first sentence of every call, usually the agent's greeting, is spoken while the canceller is still learning. And when the echo path changes, the canceller must re-learn it; ERLE collapses to near zero for a moment.

## Double Talk: the Caller Talks Over the Agent

When the caller speaks, $s(n)$ enters the error signal. The NLMS update treats the caller's voice as "echo I failed to predict" and tries to model it, which drives the filter away from the true echo path. Afterwards, the echo comes back louder than before.

The fix is to **freeze adaptation during double talk**. The Geigel detector is the classic test: declare double talk when the microphone is louder than a fraction of the recent far-end peak,

$$|d(n)| > T \cdot \max_{0 \le k < L} |x(n-k)|, \qquad T = 0.5 \;(\text{echo assumed} \ge 6\ \text{dB down}),$$

with a hangover so adaptation does not resume in the middle of a word.

![Top: echo of the agent, caller speech between 2 and 3 seconds, and the Geigel double-talk flags under the caller's syllables. Bottom: with adaptation always on, ERLE falls below 0 dB during and after double talk; with adaptation frozen on double talk, ERLE stays positive and returns to about 20 dB afterwards.](../assets/generated/double-talk.png)

Geigel assumes a fixed minimum echo loss. That holds for line echo and fails on a speakerphone, where the echo can be as loud as the caller. Production cancellers use correlation- or coherence-based detectors and keep two filters (a cautious foreground and an aggressive background) so a wrong decision costs less.

## Residual Echo Suppression and Its Price

A linear filter never removes everything: the path changes, the loudspeaker distorts, and the filter is still converging. A **non-linear processor (NLP)** attenuates the residual when only the far end is active. The cost appears whenever the double-talk detector is wrong: the suppressor also attenuates the caller, which produces the clipped, half-duplex sound of a cheap speakerphone. In the tests, a suppressor that trusts a working detector keeps more than 90% of the caller's energy; a suppressor that never detects double talk keeps less than 40%.

For a voice agent the trade-off matters in both directions:

![Frame levels over four seconds. Without a canceller, the agent's echo crosses the -40 dBFS energy-VAD threshold 57 times before the caller speaks. With NLMS it crosses twice, and with NLMS plus the residual suppressor it never does, while the caller's speech between 2.6 and 3.4 seconds still rises to about -22 dBFS.](../assets/generated/false-barge-in.png)

## Build It

`code/main.py` implements every piece with NumPy:

| Function | What it does |
| --- | --- |
| `speech_like`, `echo_path`, `scenario` | Synthetic agent speech, an echo path with a chosen ERL and bulk delay, and full scenarios with double talk or a path change |
| `nlms_echo_canceller` | Sample-by-sample NLMS with optional adaptation freeze |
| `geigel_double_talk` | Geigel detector with far-end gating and hangover |
| `residual_echo_suppressor` | Frame-level NLP gated by far-end activity and the double-talk flag |
| `erle_db`, `misalignment_db` | Per-frame ERLE and the distance between the learned filter and the true path |
| `false_barge_in_frames` | How many frames an energy VAD would wrongly call speech |

The core loop is five lines:

```python
for n in range(len(mic)):
    x = padded[n:n + taps][::-1]          # last `taps` far-end samples, newest first
    e = mic[n] - w @ x                    # residual = mic - predicted echo
    residual[n] = e
    if freeze is None or not freeze[n]:   # stop learning during double talk
        w += mu * e * x / (x @ x + eps)
```

Run it from this lesson directory:

```bash
python3 code/main.py
python3 -m unittest discover -s code/tests -v
python3 code/generate_figures.py --output-dir assets/generated
```

Expected output: a median ERLE of about 26 dB after the first half second, filter misalignment near -28 dB after three seconds, 64 false speech frames before the canceller and 0 after, and an ERLE just after double talk of about 6 dB without a double-talk detector versus about 20 dB with the Geigel freeze.

## Use It in Practice: Where Echo Control Lives with LiveKit

You will not run an NLMS filter inside your agent. The important question is *who* cancels echo on each path, and what the agent must do when that fails.

| Path | Who cancels echo | What reaches the agent |
| --- | --- | --- |
| Browser or mobile app | WebRTC `echoCancellation` in the client, configured with the client's audio capture options | Mostly echo-free audio |
| Phone call over SIP | The handset or car kit (acoustic), carrier equipment (line echo) | Whatever the phone network delivers, sometimes with audible echo |
| Agent side | No echo canceller is documented for the agent; LiveKit's enhanced noise cancellation models are recommended for agents and telephony | Echo the network did not remove |

LiveKit's WebRTC echo cancellation and noise suppression run on the client only ([noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation)). On a phone call there is no browser in the loop, so treat residual echo as something the agent must tolerate. The defenses live in turn handling:

```python
from livekit.agents import AgentSession, TurnHandlingOptions

session = AgentSession(
    # stt=..., llm=..., tts=...
    turn_handling=TurnHandlingOptions(
        interruption={
            "min_duration": 0.8,                # default 0.5 s: ignore short echo bursts
            "min_words": 2,                     # default 0: require words from STT, not just energy
            "false_interruption_timeout": 2.0,  # if no transcript follows, it was not a real interruption
            "resume_false_interruption": True,  # ...so continue the sentence
        },
    ),
)
```

- `interruption.min_duration` and `interruption.min_words` make the agent wait for evidence that is hard for an echo to fake. The cost is a slower reaction to genuine barge-ins (lesson 08 quantifies that trade-off).
- `false_interruption_timeout` and `resume_false_interruption` turn a false barge-in into a short pause instead of an abandoned sentence.
- The **adaptive** interruption mode uses an audio model to separate real interruptions from backchannels. It is on by default on LiveKit Cloud and in dev mode with VAD and STT; self-hosted agents enable it with `interruption={"mode": "adaptive"}`.
- `AgentSession` also has an `aec_warmup_duration` (default 3.0 s) that blocks interruptions right after the agent starts speaking, so a *client-side* canceller can calibrate. The docs describe it for client AEC; whether it applies to SIP participants is not documented, so do not rely on it for phone calls.
- For must-not-be-cut speech, such as a legal disclosure, use `session.say(text, allow_interruptions=False)`. This is a targeted half-duplex mode: the agent finishes the sentence no matter what comes back.

Options and defaults follow the LiveKit docs for `livekit-agents` 1.8 ([turn handling options](https://docs.livekit.io/reference/agents/turn-handling-options), [adaptive interruption handling](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling)).

## Pitfalls

- **Testing only with headsets.** A headset has almost no acoustic echo. Test with a speakerphone, a car kit and a cheap Android phone at full volume; that is where false barge-ins appear.
- **Adapting during double talk.** An always-adapting filter gets worse every time the caller interrupts, which is exactly when good cancellation matters.
- **An unregularized NLMS.** A tiny `eps` looks harmless in noiseless tests and diverges in real noise.
- **Over-suppression.** An aggressive residual suppressor removes echo and also the start of the caller's barge-in, so the agent never hears "wait, no".
- **Reading echo as a model bug.** If the transcript contains the agent's own words attributed to the user, fix the audio path and the interruption settings before you touch the prompt.
- **Running enhancement twice.** If the agent applies noise cancellation, do not also apply noise cancellation models in a web frontend; LiveKit warns when both do it. Standard WebRTC echo cancellation can stay on.

## Exercises

1. Sweep `mu` over 0.1, 0.3, 0.5 and 1.0 with `eps=0.1`. Plot convergence time to 20 dB ERLE against the final ERLE. What is the trade-off?
2. Set `erl_db=0` (a speakerphone). Does the Geigel detector still work with `threshold=0.5`? Find a threshold that keeps false alarms below 5%, and measure what it costs in missed double talk.
3. Lengthen the echo path to 128 ms (1024 taps at 8 kHz) and run the canceller with 256 taps. How much ERLE is left on the table, and what happens to the false barge-in count?
4. Implement a two-filter canceller: a background filter that always adapts, and a foreground filter that copies it only when the background's residual is lower. Compare it with the Geigel freeze in the double-talk scenario.
5. Feed `false_barge_in_frames` the residual of each variant and connect the result to `interruption.min_duration`: how long must an echo burst last to trigger an interruption at 0.5 s and at 0.8 s?

## Key Terms

| Term | Meaning |
| --- | --- |
| Far end / near end | The agent's side (reference signal) and the caller's side (microphone). |
| Echo path | The unknown impulse response from the loudspeaker or hybrid back to the microphone. |
| ERL | Echo return loss: how much quieter the echo is than the far-end signal. |
| ERLE | Echo return loss enhancement: the additional attenuation provided by the canceller. |
| NLMS | Normalized least mean squares: an adaptive filter whose step is normalized by input energy. |
| Double talk | Both sides speaking at once; adaptation must pause. |
| Geigel detector | Double-talk test comparing the microphone peak with the recent far-end peak. |
| NLP | Non-linear processor: residual echo suppression after the linear filter. |
| False barge-in | The agent stops speaking because it detected its own echo as user speech. |

## Further Reading

- [ITU-T G.168: Digital network echo cancellers](https://www.itu.int/rec/T-REC-G.168)
- S. Haykin, *Adaptive Filter Theory*, chapters on LMS and NLMS
- [WebRTC AEC3 overview (source tree)](https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/)
- [LiveKit: noise cancellation and audio processing](https://docs.livekit.io/transport/media/noise-cancellation)
- [LiveKit: turn handling options](https://docs.livekit.io/reference/agents/turn-handling-options)
