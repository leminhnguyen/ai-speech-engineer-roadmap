# 📡 SOTA Tracker

> What is new in speech AI, and why it matters. Dated, curated, and kept short.

[← Back to home](../README.md) · [🗺️ Roadmap](../roadmap/README.md) · [🛠️ Tutorials](../tutorials/README.md)

**Last reviewed:** August 2026 · ⭐ = must-read · 🇻🇳 = Vietnamese

Research in this area moves quickly, and many recent works are preprints. Treat “SOTA” claims as benchmark-specific and check the latest paper version, code, and evaluation setup before relying on them.

## 🧭 The Big Picture

The field is moving from isolated speech tasks toward **unified models that understand, reason over, and generate audio in real time**. Four shifts explain most new papers:

| Shift | From | To |
| --- | --- | --- |
| 🔤 **Representation** | Mel spectrograms fed to task-specific models | Discrete semantic and acoustic tokens that a language model can read and write |
| 🎧 **Scope** | One model per task (ASR, TTS, SV) | One audio-language model that listens, reasons, and speaks |
| 🗣️ **Interaction** | Turn-based pipelines (ASR → LLM → TTS) | End-to-end, streaming, full-duplex dialogue with interruptions and overlap |
| 📏 **Evaluation** | WER and MOS | Reasoning, paralinguistics, latency, turn-taking, and multi-speaker benchmarks |

Start with the [surveys](topics/surveys.md), then follow the track closest to your work.

## 🗂️ Tracks

<!-- BEGIN TRACKS -->
| | Track | Entries | Newest |
| :---: | --- | :---: | :---: |
| 🧭 | [Surveys & Taxonomies](topics/surveys.md) | 4 | 2025-04 |
| 🔤 | [Speech Tokenization & Neural Codecs](topics/tokenization.md) | 2 | 2024-09 |
| 🎙️ | [Speech Recognition](topics/asr.md) | 3 | 2026-03 |
| 🎧 | [Audio Understanding & Reasoning](topics/understanding.md) | 3 | 2025-07 |
| 🗣️ | [Real-Time Spoken Dialogue & Omni Models](topics/dialogue.md) | 7 | 2025-07 |
| 🎨 | [Speech Generation & Multilingual TTS](topics/generation.md) | 5 | 2026-04 |
| 👥 | [Speaker & Diarization](topics/speaker.md) | 1 | 2025-07 |
| 📏 | [Evaluation, Benchmarks & Data](topics/evaluation.md) | 6 | 2026-07 |
<!-- END TRACKS -->

## 🆕 Most Recent Work

The ten newest entries by release date.

<!-- BEGIN LATEST -->
| Date | Work | Track | Why it matters |
| :---: | --- | --- | --- |
| 2026-07 | **DuplexChat** | [Evaluation, Benchmarks & Data](topics/evaluation.md) | Large-scale, speaker-separated dialogue speech for training full-duplex models. |
| 2026-04 | ⭐ **OmniVoice** | [Speech Generation & Multilingual TTS](topics/generation.md) | Pushes zero-shot TTS toward omnilingual coverage with a diffusion language model. |
| 2026-04 | ⭐ **HumDial-FDBench** | [Evaluation, Benchmarks & Data](topics/evaluation.md) | Benchmark for full-duplex interaction quality in spoken dialogue systems. |
| 2026-03 | ⭐ **Gipformer** 🇻🇳 | [Speech Recognition](topics/asr.md) | 68M Zipformer RNN-T for Vietnamese call-center speech, MIT-licensed; v1.5 beats PhoWhisper-large on hard call-center test sets. |
| 2026-02 | **WavBench** | [Evaluation, Benchmarks & Data](topics/evaluation.md) | Evaluates spoken models on reasoning, colloquial speech, and paralinguistic cues. |
| 2026-01 | **Qwen3-TTS** | [Speech Generation & Multilingual TTS](topics/generation.md) | Multilingual TTS from the Qwen3 family. |
| 2026-01 | **Qwen3-ASR** | [Speech Recognition](topics/asr.md) | LLM-based multilingual recognizer from the Qwen3 family. |
| 2026-01 | **ART** | [Evaluation, Benchmarks & Data](topics/evaluation.md) | Audio reasoning benchmark for multimodal LLMs. |
| 2025-07 | ⭐ **Streaming Sortformer** | [Speaker & Diarization](topics/speaker.md) | Online diarization with a speaker cache that keeps speakers in arrival-time order. |
| 2025-07 | ⭐ **Step-Audio 2** | [Real-Time Spoken Dialogue & Omni Models](topics/dialogue.md) | End-to-end audio LM for industrial-grade understanding and conversation, aware of paralinguistic cues. |
<!-- END LATEST -->

## 📐 How Entries Are Chosen

The tracker is curated, not exhaustive. An entry belongs here when it meets at least one of these:

- **Moves a benchmark** in a way that holds up against strong baselines.
- **Introduces an idea** others are building on, such as a new tokenizer, architecture, or training recipe.
- **Ships something usable**: open weights, code, or a dataset that changes what practitioners can do.
- **Defines how we measure**: a benchmark or evaluation protocol that is being adopted.

Each entry records its release month, a primary link, and one line on why it matters, so the list reads as a timeline of ideas rather than a pile of links.

### 🎓 Graduation to the Roadmap

When a work stops being news and becomes required background, for example the way Whisper or VITS did, it moves to the matching section of the [roadmap](../roadmap/README.md) and leaves this tracker.

## 🤝 Add an Entry

1. Add an item to [`data/entries.yaml`](data/entries.yaml). The file header documents every field.
2. Regenerate the pages:

   ```bash
   make sota
   ```

3. Commit the YAML change together with the regenerated pages. CI runs `make sota-check` and fails if they drift.

Never edit files in [`topics/`](topics/) by hand; they are generated.
