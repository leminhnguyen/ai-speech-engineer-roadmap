# Line Quality: Jitter, Packet Loss, and Level

> Packets arrive late, out of order, or not at all. The receiver must decide, every 20 ms, how long to wait and what to play when the packet is not there.

**Type:** Learn + Build  
**Language:** Python  
**Prerequisites:** [Telephony and SIP](../../01-telephony-and-sip/docs/en.md), [LiveKit Voice Agent Architecture](../../02-livekit-voice-agent-architecture/docs/en.md), [Pitch Estimation](../../../01-audio-signal-processing/08-pitch-estimation/docs/en.md)  
**Time:** ~80 minutes

## Learning Objectives

- Model bursty packet loss with a Gilbert–Elliott chain and explain why burstiness matters more than the average loss rate.
- Simulate network jitter, reordering and a mid-call route change.
- Build fixed and adaptive jitter buffers and read the trade-off between added delay and late packets.
- Implement three packet loss concealment (PLC) strategies and measure them with a metric that matches perception.
- Detect clipping and implement a gated automatic gain control (AGC).
- Estimate call quality from delay and loss with a simplified ITU-T G.107 E-model.
- Decide what a LiveKit agent can control, what it cannot, and what it should log.

## The Problem

Customers rarely complain that "the jitter was high". They say the agent "didn't hear me", "kept interrupting", or "sounded robotic". Behind those reports are a handful of network and level impairments that a voice agent inherits from the phone call. RTP runs over UDP, which does not retransmit, so every packet that is lost or arrives too late becomes a hole in the audio. The receiver hides holes with concealment, and a loud or quiet caller adds clipping or noise.

For a human listener the damage shows up as a lower MOS. For an agent it shows up as STT errors, missed or false VAD triggers (lessons 06–08), and wrong turn timing. You usually cannot fix the network, but you can measure it and choose settings that degrade gracefully.

![Pipeline from the caller's handset through the carrier network, a jitter buffer, and a decoder with packet loss concealment to the agent's VAD and STT. Below each stage, a dashed box names what goes wrong there (clipping and noise; bursty loss, jitter and reordering; late packets and added delay; gaps and clicks; higher WER and false VAD triggers), and a solid box names what limits it (AGC and noise cancellation; region pinning and a better trunk; adaptive playout delay; pitch-period concealment; measuring loss, jitter and MOS per call).](../assets/concept.svg)

## Loss Comes in Bursts

The simplest loss model drops each packet independently with probability `p`. Real networks do not behave like that: a congested queue or a radio fade drops several consecutive packets. The **Gilbert–Elliott** model captures this with two states:

| State | Loss probability | Leave with probability |
| --- | --- | --- |
| Good | `loss_good` (≈ 0) | `p` (to Bad) |
| Bad | `loss_bad` (≈ 1) | `r` (to Good) |

With `loss_good = 0` and `loss_bad = 1`, the average loss is `p / (p + r)` and the mean burst length is `1 / r`. Two channels with the same 3% average loss can have mean bursts of one packet or five. The bursty one sounds far worse, because concealment works well for 20 ms and badly for 100 ms. When you read a carrier's quality report, ask for burst statistics, not only the loss percentage.

## Jitter and the Jitter Buffer

Packets leave the sender every 20 ms but arrive with varying delay: path delay plus queueing, with occasional spikes when a queue fills. When the delay difference between consecutive packets exceeds 20 ms, packets arrive out of order. RTP's sequence numbers and timestamps (lesson 01) let the receiver put them back in order.

A **jitter buffer** holds packets until a *playout deadline*. A packet that arrives after its deadline is useless and is discarded: **late is the same as lost**. The deadline is a trade-off:

- A short buffer adds little delay but discards more late packets.
- A long buffer discards fewer packets but adds delay to every turn of the conversation.

A **fixed** buffer picks one offset at the start of the call. An **adaptive** buffer tracks the recent delay distribution and moves the deadline, ideally during silence so the change is inaudible. In `main.py` the target is a high quantile of the last 500 delays plus a small margin. The deadline rises immediately when needed and falls slowly.

![Left: one minute of packet delays with a route change after 36 seconds that adds 60 milliseconds. A fixed 60 ms buffer, sized for the first half, discards most packets after the change, while the adaptive deadline rises with it. Right: late-packet percentage against network plus buffer delay, on a log scale. The adaptive buffer, tuned online, reaches the same or lower late rates as the best fixed size chosen in hindsight, especially between 100 and 115 milliseconds.](../assets/generated/jitter-buffer.png)

The left panel shows why fixed buffers fail in mobile networks: the "right" size changes during the call. The right panel compares the adaptive buffer with every fixed size *chosen in hindsight*. That is the best a fixed buffer could ever do, and the adaptive one matches or beats it without knowing the future.

For a voice agent, every millisecond of buffer is added to response latency (lesson 09). Conversely, every late packet is a hole that STT must transcribe through. Neither extreme is free.

## Concealing a Lost Packet

When the deadline passes with no packet, the decoder must still output 20 ms of audio. Three strategies:

| Strategy | Output | Typical artifact |
| --- | --- | --- |
| Zero fill | Silence | Audible dropout; can split a word for STT and look like end-of-speech to a VAD |
| Repeat last frame | The previous 20 ms again | Buzz at 50 Hz, because 20 ms is not a whole number of pitch periods |
| Pitch-period repetition | The last pitch period, repeated | Smooth for short gaps; the idea behind G.711 Appendix I PLC |

All three should fade out over long bursts. A voiced sound repeated for 200 ms becomes an unnatural drone, so `conceal` keeps full level for 10 ms, then fades to silence over the next 50 ms. It also crossfades into the next good packet to avoid a click.

![Three panels around two lost packets in a voiced signal. Zero fill leaves a flat line for 40 milliseconds, with a log-spectral distance of 21 dB. Repeating the last frame keeps the level but distorts the waveform, 5 dB. Repeating the last pitch period follows the original closely before fading, 3.4 dB.](../assets/generated/plc-comparison.png)

How to score concealment matters. Waveform metrics such as SNR or segmental SNR penalize any phase difference, so a repeated period that is perfectly natural but slightly out of phase can score *worse* than silence. `main.py` prints both: segmental SNR barely separates the three methods, while the phase-insensitive **log-spectral distance**, computed on windows that straddle frame boundaries, ranks them the way a listener would. Choose metrics that reflect what the consumer of the audio, human or STT, actually notices.

## Level: Clipping and AGC

Two level problems are common on calls:

- **Clipping.** A caller shouting into a phone held close, or a gateway applying too much gain, drives samples to full scale. Clipped audio contains harmonics that were never spoken, and it cannot be repaired afterwards. `clipping_ratio` looks for runs of at least three samples at full scale, which is the signature of clipping rather than a legitimate peak.
- **Low level.** A caller on speakerphone across the room arrives 20–30 dB too quiet, close to the noise floor where VAD and STT start failing.

An **AGC** measures the level of each 10 ms frame and moves a gain toward a target. Three details make it safe for speech: it reacts quickly when the signal gets louder (attack) and slowly when it gets quieter (release), it has a maximum gain, and a **noise gate** freezes the gain in frames that are only background noise. Without the gate, an AGC raises the noise in every pause, which a VAD then reads as speech.

## Putting a Number on It: the E-model

ITU-T G.107's **E-model** predicts conversational quality from transmission parameters. It starts from a transmission rating `R` and subtracts impairments. For a G.711 call, a widely used simplification keeps two:

$$R = 93.2 - I_d - I_{e,\mathrm{eff}}, \qquad I_{e,\mathrm{eff}} = I_e + (95 - I_e)\,\frac{P_{pl}}{P_{pl}/\mathrm{BurstR} + B_{pl}}$$

`I_d` grows by about 0.024 per millisecond of mouth-to-ear delay up to 177 ms, then much faster. `P_pl` is the packet loss in percent, `BurstR` is 1 for random loss and larger for bursty loss, and G.711 with PLC has `I_e = 0` and `B_pl = 25.1` (ITU-T G.113 Appendix I). `R` maps to an estimated MOS between 1 and 4.5.

![Estimated MOS against mouth-to-ear delay for G.711 with PLC. With no loss the curve stays above 4.3 until the 177 ms knee and then drops steeply. 1% random loss costs about 0.1 MOS. 3% loss costs about 0.3, and the same 3% in bursts costs a little more again. Dashed lines mark R 80 (satisfied) and R 70 (some users dissatisfied).](../assets/generated/emodel-mos.png)

The E-model was built for human conversations, not for agents. It is still a useful shared language with carriers and network teams, and its delay knee is a reminder: once the network and jitter buffer use 150 ms, every millisecond the agent pipeline adds is expensive.

## Build It

| Function | What it does |
| --- | --- |
| `gilbert_elliott`, `gilbert_loss_rate`, `burst_lengths` | Two-state bursty loss model and its statistics |
| `network_delays`, `arrival_order`, `reordered_count` | Gamma queueing jitter, decaying spikes, an optional route change, and reordering |
| `fixed_jitter_buffer`, `adaptive_jitter_buffer` | Playout deadlines, late packets, and the delay each buffer adds |
| `conceal`, `estimate_period` | Zero fill, frame repetition and pitch-period repetition, with fade-out and crossfade |
| `segmental_snr_db`, `log_spectral_distance` | Waveform and phase-insensitive spectral scores for concealment |
| `clipping_ratio`, `agc`, `dbfs` | Clipping detection and a gated attack/release AGC |
| `delay_impairment`, `effective_equipment_impairment`, `r_factor`, `mos_from_r` | Simplified G.107 E-model |

The adaptive buffer in a few lines:

```python
if i and i % adapt_every == 0:
    recent = delay_ms[max(0, i - window):i][~lost[max(0, i - window):i]]
    target = float(np.quantile(recent, quantile)) + margin_ms
    offset = target if target > offset else max(target, offset - step_down_ms)
```

Run it from this lesson directory:

```bash
python3 code/main.py
python3 -m unittest discover -s code/tests -v
python3 code/generate_figures.py --output-dir assets/generated
```

Expected output includes about 2.2% loss in bursts of 2.4 packets, `fixed buffer 60 ms: late 18.03%` against `adaptive buffer: late 2.93%, mean delay 114 ms`, log-spectral distances of about 20, 8 and 6 dB for zero fill, frame repetition and pitch repetition, and E-model MOS values of 4.36, 4.26 and 3.52.

## Use It in Practice: What a LiveKit Agent Controls

The caller's RTP terminates at LiveKit SIP, so the jitter buffer and decoder on the phone leg belong to that media gateway, not to your agent. Your levers are around it:

| Concern | What you can do in LiveKit | Source |
| --- | --- | --- |
| Long network paths | SIP endpoints are global by default: inbound calls go to the region closest to the carrier's endpoint. Pin a region with `{sip_subdomain}.{region}.sip.livekit.cloud`, and set `destination_country` on outbound trunks | [Region pinning](https://docs.livekit.io/telephony/features/region-pinning) |
| Dead media | If no RTP arrives within 30 s, or within 15 s after audio has started, the call ends with a media timeout. Tune with `media_timeout` in the trunk's `SIPMediaConfig` | [SIP primer](https://docs.livekit.io/reference/telephony/sip-primer) |
| Narrowband audio | Enable G.722 HD voice where your carrier supports it | [HD voice](https://docs.livekit.io/telephony/features/hd-voice) |
| Level | The agent's room input has built-in automatic gain control, on by default. Disable it with `room_io.AudioInputOptions(auto_gain_control=False)` if a stage upstream already normalizes level | [Noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation) |
| Noise | Agent-side noise cancellation for SIP callers (lesson 04) | [Noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation) |
| Evidence | Session recordings in LiveKit Cloud, or Egress to your own storage, to listen to bad calls | [Recording](https://docs.livekit.io/testing/observability/recording) |

What to log for every call, keyed by `sip.callID`:

- The carrier, trunk and region that handled it.
- Whether the call ended by hangup, by media timeout, or by your agent.
- Per-turn STT confidence or WER on sampled calls, and the agent's latency metrics (lesson 09).
- A short audio sample when a turn fails (no transcript, or a very low-confidence one), so you can tell a network problem from a model problem.

Network-level counters such as loss and jitter live in the media gateway and the carrier. When a cluster of complaints appears, correlate your per-call logs with the carrier's reports by time and `Call-ID`.

## Pitfalls

- **Judging a carrier by average loss.** 1% in long bursts can sound worse than 2% spread out. Ask for burst statistics.
- **Equating "high jitter" with "bad calls".** Jitter that the buffer absorbs costs only delay. Late packets cost audio. Measure both.
- **Grading concealment with SNR.** Waveform SNR punishes phase shifts that nobody hears. Use spectral measures, listening tests or STT accuracy.
- **Stacking gain stages.** Carrier AGC, the agent's AGC and a normalization step in your STT client can fight each other and pump noise. Keep one, and gate it.
- **Treating "media timeout" as user hangup.** They have different causes and fixes. Log the disconnect reason.
- **Measuring in the lab only.** Wi-Fi in an office is not a 4G handover on a motorbike. Test on the networks your callers use.

## Exercises

1. Find Gilbert–Elliott parameters that give 3% loss with mean bursts of 1, 2 and 5 packets. Run `conceal(..., "pitch")` on each pattern and compare the log-spectral distance.
2. Remove the route change (`step_at=None`) and rerun the jitter buffer comparison. Does the adaptive buffer still help? Why?
3. Change `adaptive_jitter_buffer` so that it never lowers its deadline. Plot the delay it adds over a long call with occasional spikes.
4. Add a fourth PLC method that reverses the last pitch period on alternate repetitions, and measure whether it reduces the drone on 60 ms bursts.
5. Using the E-model, find the largest mouth-to-ear delay that keeps R ≥ 80 for G.711 with 1% random loss. Subtract a 150 ms network-and-buffer budget, and state the latency left for your agent's response.

## Key Terms

| Term | Meaning |
| --- | --- |
| Jitter | Variation in packet delay. |
| Jitter buffer | Receiver queue that delays playout so late packets can still be used. |
| Playout deadline | The time by which a packet must arrive to be played. |
| Late loss | A packet that arrived after its deadline and was discarded. |
| Gilbert–Elliott model | Two-state Markov model of bursty packet loss. |
| PLC | Packet loss concealment: synthesizing audio for missing packets. |
| Log-spectral distance | RMS difference between two log power spectra; insensitive to phase. |
| AGC | Automatic gain control: adaptive gain toward a target level. |
| E-model | ITU-T G.107 model that predicts call quality (R, MOS) from impairments. |
| Media timeout | LiveKit SIP ending a call because RTP stopped arriving. |

## Further Reading

- [ITU-T G.107: The E-model](https://www.itu.int/rec/T-REC-G.107) and [G.113: Transmission impairments due to speech processing](https://www.itu.int/rec/T-REC-G.113)
- [ITU-T G.711 Appendix I: packet loss concealment for G.711](https://www.itu.int/rec/T-REC-G.711)
- R. Ramjee, J. Kurose, D. Towsley, H. Schulzrinne, "Adaptive Playout Mechanisms for Packetized Audio Applications in Wide-Area Networks", IEEE INFOCOM 1994
- R. G. Cole, J. H. Rosenbluth, "Voice over IP Performance Monitoring", ACM SIGCOMM CCR, 2001
- [LiveKit: region pinning](https://docs.livekit.io/telephony/features/region-pinning) and [SIP primer](https://docs.livekit.io/reference/telephony/sip-primer)
