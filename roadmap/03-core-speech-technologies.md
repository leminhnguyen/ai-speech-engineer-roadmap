# 🌱 03 · Core Speech Technologies

> **Duration:** ~12 months · **Goal:** understand the canonical architecture of each core task, train or fine-tune one model per task, and know how each is evaluated.

[← Tools & Frameworks](02-tools-and-frameworks.md) · [Roadmap](README.md) · [🇻🇳 Vietnamese Speech](vietnamese-speech.md)

Papers are listed oldest first so each section reads as a history of ideas. The newest work for each task is in the [📡 SOTA tracker](../sota/README.md).

**Sections:** [Transformers](#-transformers) · [ASR](#%EF%B8%8F-automatic-speech-recognition-asr) · [TTS](#%EF%B8%8F-text-to-speech-tts) · [Speaker Verification](#-speaker-verification-sv) · [Speaker Diarization](#-speaker-diarization-sd) · [Voice Conversion](#-voice-conversion-vc)

## 🤖 Transformers

Every modern speech model builds on attention. Learn it before the tasks below.

- ⭐ [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- ⭐ [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 📺 [Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)

## 🎙️ Automatic Speech Recognition (ASR)

- ⭐ [Sequence Modeling With CTC (2017)](https://distill.pub/2017/ctc/)
- [SpecAugment (2019)](https://blog.research.google/2019/04/specaugment-new-data-augmentation.html)
- ⭐ [Wav2Vec2 (2020)](https://arxiv.org/abs/2005.08100)
- [Generation of large-scale simulated utterances in virtual rooms to train deep-neural networks for far-field speech recognition in Google Home (2020)](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/509254e34b4c496eb3cfa1c2be1e1b5fc874bee3.pdf)
- [Illustrated Wav2Vec2 (2021)](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)
- ⭐ [Whisper (2022)](https://arxiv.org/abs/2212.04356)
- [Fast Conformer (2023)](https://arxiv.org/abs/2305.05084)
- ⭐ [Zipformer: A Faster and Better Encoder for ASR (2023)](https://arxiv.org/abs/2310.11230) - U-Net-like encoder and ScaledAdam optimizer; faster and more memory-efficient than Conformer (ICLR 2024)
- ⭐ [SpeechBrain ASR Tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-recognition-from-scratch.html)

> 📡 **Latest:** [Speech Recognition](../sota/topics/asr.md) · [Audio Understanding](../sota/topics/understanding.md) &nbsp;·&nbsp; 🇻🇳 [Vietnamese ASR](vietnamese-speech.md#%EF%B8%8F-asr)

## 🗣️ Text-to-Speech (TTS)

- ⭐ [WaveNet: A Generative Model for Raw Audio (2016)](https://arxiv.org/abs/1609.03499)
- ⭐ [Tacotron: Towards End-to-End Speech Synthesis (2017)](https://arxiv.org/abs/1703.10135)
- [WaveGlow: A Flow-based Generative Network for Speech Synthesis (2018)](https://arxiv.org/abs/1811.00002)
- [FastSpeech: Fast, Robust and Controllable Text to Speech (2019)](https://arxiv.org/abs/1905.09263)
- ⭐ [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (2020)](https://arxiv.org/abs/2006.04558)
- ⭐ [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (2020)](https://arxiv.org/abs/2010.05646)
- ⭐ [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (2021)](https://arxiv.org/abs/2106.06103)
- [JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech (2022)](https://arxiv.org/abs/2203.16852)
- [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality (2022)](https://arxiv.org/abs/2205.04421)
- [Kokoro TTS (2024)](https://kokorottsai.com/) - 82M-parameter model built on the StyleTTS 2 architecture

> 📡 **Latest:** [Speech Generation & Multilingual TTS](../sota/topics/generation.md) · [Tokenization & Codecs](../sota/topics/tokenization.md) &nbsp;·&nbsp; 🇻🇳 [Vietnamese TTS](vietnamese-speech.md#%EF%B8%8F-tts)

## 🔐 Speaker Verification (SV)

- ⭐ [Speaker Verification Introduction](https://maelfabien.github.io/machinelearning/Speech1/#)
- ⭐ [X-vectors: Robust DNN Embeddings for Speaker Recognition (2017)](https://danielpovey.com/files/2017_interspeech_embeddings.pdf)
- ⭐ [VoxCeleb: A Large-Scale Speaker Identification Dataset (2017)](https://arxiv.org/abs/1706.08612)
- [I-vector Paper (2018)](https://www.sciencedirect.com/science/article/pii/S1877050918314042/pdf)
- ⭐ [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification (2020)](https://arxiv.org/abs/2005.07143)
- [ResNeXt and Res2Net Structures for Speaker Verification (2020)](https://arxiv.org/abs/2007.02480)
- [CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking (2023)](https://arxiv.org/abs/2303.00332)
- [3D-Speaker: A Large-Scale Multi-Device, Multi-Distance, and Multi-Dialect Corpus for Speech Representation Disentanglement (2023)](https://arxiv.org/abs/2306.15354)
- [Golden Gemini is All You Need: Finding the Sweet Spots for Speaker Verification (2024)](https://arxiv.org/abs/2312.03620)
- [ERes2NetV2: Boosting Short-Duration Speaker Verification Performance with Computational Efficiency (2024)](https://arxiv.org/html/2406.02167v1)
- [ReDimNet: Reshape Dimensions Network for Speaker Recognition (2024)](https://arxiv.org/abs/2407.18223)

> 📡 **Latest:** [Speaker & Diarization](../sota/topics/speaker.md)

## 👥 Speaker Diarization (SD)

- ⭐ [pyannote.audio: Neural Building Blocks for Speaker Diarization (2019)](https://arxiv.org/abs/1911.01255)
- ⭐ [A Review of Speaker Diarization: Recent Advances with Deep Learning (2021)](https://arxiv.org/abs/2101.09624)
- [Multi-scale Speaker Diarization with Dynamic Scale Weighting (2022)](https://arxiv.org/pdf/2203.15974)
- ⭐ [Speaker Diarization: An Introductory Overview (2023)](https://lajavaness.medium.com/speaker-diarization-an-introductory-overview-c070a3bfea70)
- [Comparing State-of-the-Art Speaker Diarization Frameworks: Pyannote vs NeMo (2023)](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)
- [DiarizationLM: Speaker Diarization Post-Processing with Large Language Models (2024)](https://arxiv.org/html/2401.03506v10)
- [Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens (2024)](https://arxiv.org/abs/2409.06656)
- [Speaker Diarization: From Traditional Methods to the Modern Models (2025)](https://leminhnguyen.github.io/post/speech-research/speaker-diarization/)

> 📡 **Latest:** [Speaker & Diarization](../sota/topics/speaker.md)

## 🎭 Voice Conversion (VC)

- ⭐ [AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss (2019)](https://arxiv.org/abs/1905.05879)
- ⭐ [An Overview of Voice Conversion and its Challenges: From Statistical Modeling to Deep Learning (2020)](https://arxiv.org/abs/2008.03648)
- [AGAIN-VC: A One-shot Voice Conversion using Activation Guidance and Adaptive Instance Normalization (2021)](https://arxiv.org/abs/2011.00316)
- [YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion (2022)](https://arxiv.org/abs/2112.02418)
- [kNN-VC: Voice Conversion With Just Nearest Neighbors (2023)](https://arxiv.org/abs/2305.18975)
- ⭐ [Seed-VC: Towards Robust Zero-shot Voice Conversion and Singing Voice Conversion (2024)](https://arxiv.org/abs/2411.09943)

> 📡 **Latest:** [Speech Generation](../sota/topics/generation.md) &nbsp;·&nbsp; 🇻🇳 [Vietnamese VC](vietnamese-speech.md#-voice-conversion)

## ✅ You Are Ready to Move On When You Can

- Train or fine-tune one model per task and report it with the standard metric: WER/CER, MOS or UTMOS, EER, DER, and speaker similarity.
- Explain why CTC, attention, and transducer decoders exist and when each wins.
- Read a new paper in this area and place it relative to the papers above.
