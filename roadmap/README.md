# 🗺️ Roadmap: AI Speech Engineer

> The foundation every speech engineer needs, in the order that makes it stick. Curated from six years of building speech systems.

[← Back to home](../README.md) · [🛠️ Tutorials](../tutorials/README.md) · [📡 SOTA Tracker](../sota/README.md)

This roadmap covers knowledge that stays useful for years: math and machine learning, audio signal processing, the standard toolkit, and the core speech technologies. Fast-moving research lives in the [SOTA tracker](../sota/README.md) instead, so this page stays stable.

## 🧭 The Path

```mermaid
flowchart TD
    START([🐌 AI Speech Engineer]):::title --> P1

    subgraph P1["🧠 01 · Foundations"]
        direction LR
        A1[🐍 Python] ~~~ A2[📈 ML] ~~~ A3[🧬 Deep Learning] ~~~ A4[🔊 Signal Processing]
    end

    P1 --> P2

    subgraph P2["💼 02 · Tools & Frameworks"]
        direction LR
        B1[🔥 PyTorch] ~~~ B2[🎵 librosa / torchaudio] ~~~ B3[🛠️ ffmpeg / sox] ~~~ B4[🤗 Hugging Face]
    end

    P2 --> TRANS[🤖 Transformers]
    TRANS --> P3

    subgraph P3["🌱 03 · Core Speech Technologies"]
        direction LR
        C1[🎙️ ASR] ~~~ C2[🗣️ TTS] ~~~ C3[🔐 SV] ~~~ C4[👥 SD] ~~~ C5[🎭 VC]
    end

    P3 --> P4[📡 Stay current with the SOTA tracker]

    classDef title fill:#1a5276,color:#fff,stroke:#1a5276,font-size:16px
    style P1 fill:#dae8fc,stroke:#6c8ebf
    style P2 fill:#fff2cc,stroke:#d6b656
    style P3 fill:#f3e5f5,stroke:#9c27b0
    style P4 fill:#fce4ec,stroke:#e91e63
```

| Phase | Duration | Focus | Practice it |
| --- | :---: | --- | --- |
| 🧠 [**01 · Foundations**](01-foundations.md) | 3 months | Python, machine learning, deep learning, audio signal processing | [Tutorial module 01](../tutorials/01-audio-signal-processing/) |
| 💼 [**02 · Tools & Frameworks**](02-tools-and-frameworks.md) | 3 months | PyTorch, librosa, torchaudio, audio utilities, Hugging Face Audio | — |
| 🌱 [**03 · Core Speech Technologies**](03-core-speech-technologies.md) | 12 months | Transformers, ASR, TTS, speaker verification, diarization, voice conversion | Planned |
| 🇻🇳 [**Vietnamese Speech**](vietnamese-speech.md) | Alongside 03 | Phonemizers, VLSP reports, theses, Vietnamese models | — |
| 📡 [**SOTA Tracker**](../sota/README.md) | Continuous | Speech LMs, codecs, full-duplex dialogue, modern TTS and benchmarks | — |

Durations are guidelines, not deadlines. Move on when you can **explain** the core idea, **reproduce** a small implementation, and **describe** its failure modes.

## 🔖 Legend

| Mark | Meaning |
| :---: | --- |
| ⭐ | Essential. If you only have time for a few resources, read these. |
| 📺 | Video or course |
| 📃 | Course or written material |
| 🇻🇳 | Vietnamese-specific resource |
| 📡 | Link to the newest work in the SOTA tracker |
