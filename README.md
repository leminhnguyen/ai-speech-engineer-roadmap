
# 🥑 ROADMAP: AI Speech Engineer

![GitHub stars](https://img.shields.io/github/stars/leminhnguyen/ai-speech-engineer-roadmap?style=social)
![GitHub forks](https://img.shields.io/github/forks/leminhnguyen/ai-speech-engineer-roadmap?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/leminhnguyen/ai-speech-engineer-roadmap)

> A curated roadmap based on my 5 years of experience form zero to become a skilled AI Speech Engineer. 🚀👨‍💻  
> This roadmap covers everything from fundamentals to cutting-edge research trends in the speech domain.

---

## 📅 Overview Timeline

| Phase                        | Duration   | Focus Areas                               |
|-----------------------------|------------|-------------------------------------------|
| 🧠 Foundations              | 2 month    | Math, Python, Signal Processing           |
| 💼 Tools & Frameworks       | 3 months   | Libraries, Audio Tools, Hugging Face      |
| 🌱 Core Technologies        | 12 months   | ASR, TTS, Speaker Verification & Diarization |
| 🔬 Research Trends          | Continuous | Audio-Language Models                     |

---

## 🧠 #1 Foundations (1 month)

### 🔹 Python Basic
- 📺 [Python Tutorial for Beginners](https://www.youtube.com/watch?v=YYXdXT2l-Gg&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)

### 🔹 Audio Signal Processing for ML
- 📺 [YouTube Series](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)

---

## 💼 #2 Tools & Frameworks (2 months)

### 🧰 Frameworks & Libraries
- `PyTorch` - Training models framework
- `librosa` - Audio preprocessing (STFT, MFCCs, etc.)
- `torchaudio`- Audio loading, transforms, and model wrappers
- `ffmpeg`, `sox`, `pydub` - Audio conversion, slicing, format handling
- `noisereduce` – Simple noise reduction from raw audio

### 🖥️ Tools
- [Audacity](https://www.audacityteam.org/) - A free & powerful software for editing & visualizing audio
- [Audacity Tutorial](https://www.youtube.com/watch?v=vlzOb4OLj94)

### 🤗 Hugging Face Course
- [Hugging Face Audio](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) - Learn to tackle a range of audio-related tasks and gain experiments with speech datasets.

---

## 🌱 #3 Dive Into Speech Core Technologies (12 months)

### 🎙️ Automatic Speech Recognition (ASR)
- [SpeechBrain ASR](https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-recognition-from-scratch.html)
- [SpecAugment](https://blog.research.google/2019/04/specaugment-new-data-augmentation.html)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Generation of large-scale simulated utterances in virtual rooms...](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/509254e34b4c496eb3cfa1c2be1e1b5fc874bee3.pdf)
- [Illustrated Wav2Vec2](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)
- [CTC](https://distill.pub/2017/ctc/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Wav2Vec2](https://arxiv.org/abs/2005.08100)
- [Whisper](https://arxiv.org/abs/2212.04356)
- [Fast Conformer](https://arxiv.org/abs/2305.05084)

### 🗣️ Text-to-Speech (TTS)
- [My graduation thesis (Vietnamese) (2021)](materials/graduation-thesis.pdf)
- [HMM-based Vietnamese TTS](https://theses.hal.science/tel-01260884/document)
- [Wavenet: A Generative Model for Raw Audio (2016)](https://arxiv.org/abs/1609.03499)
- [Tacotron: Towards End-to-End Speech Synthesis (2017)](https://arxiv.org/abs/1703.10135)
- [WaveGlow: A Flow-based Generative Network for Speech Synthesis (2018)](https://arxiv.org/abs/1811.00002)
- [FastSpeech 1: Fast, Robust and Controllable Text to Speech (2020)](https://arxiv.org/abs/1905.09263)
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (2021)](https://arxiv.org/abs/2006.04558)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (2020)](https://arxiv.org/abs/2010.05646)
- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (2021)](https://arxiv.org/abs/2106.06103)
- [JETS: JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech (2022)](https://arxiv.org/abs/2203.16852)
- [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality (2022)](https://arxiv.org/abs/2205.04421)

#### 🇻🇳 Vietnamese Resources
- [Viphoneme](https://github.com/v-nhandt21/Viphoneme)
- [Text2PhonemeSequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence)

### 🔐 Speaker Verification (SV)
- [Speech Verification Introduction](https://maelfabien.github.io/machinelearning/Speech1/#)
- [X-vector Paper](https://danielpovey.com/files/2017_interspeech_embeddings.pdf)
- [I-vector Paper](https://www.sciencedirect.com/science/article/pii/S1877050918314042/pdf)
- [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation...](https://arxiv.org/abs/2005.07143)
- [VoxCeleb: a large-scale speaker identification dataset](https://arxiv.org/abs/1706.08612)
- [ResNeXt and Res2Net Structures for Speaker Verification](https://arxiv.org/abs/2007.02480)
- [Golden Gemini is All You Need: Finding the Sweet Spots for Speaker Verification](https://arxiv.org/abs/2312.03620)
- [CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking](https://arxiv.org/abs/2303.00332)
- [RedimNet: Reshape Dimensions Network for Speaker Recognition](https://arxiv.org/abs/2407.18223)
- [3D-Speaker: A Large-Scale Multi-Device, Multi-Distance, and Multi-Dialect Corpus...](https://arxiv.org/abs/2306.15354)
- [ERes2NetV2: Boosting Short-Duration Speaker...](https://arxiv.org/html/2406.02167v1)

### 👥 Speaker Diarization (SD)
- [Speaker Diarization: An Introductory Overview](https://lajavaness.medium.com/speaker-diarization-an-introductory-overview-c070a3bfea70)
- [Speaker Diarization: From Traditional Methods to the Modern Models](https://leminhnguyen.github.io/post/speech-research/speaker-diarization/)
- [pyannote.audio: neural building blocks for speaker diarization](https://arxiv.org/abs/1911.01255)
- [A Review of Speaker Diarization: Recent Advances with Deep Learning](https://arxiv.org/abs/2101.09624)
- [Comparing state-of-the-art speaker diarization frameworks : Pyannote vs Nemo](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)
- [Multi-scale Speaker Diarization with Dynamic Scale Weighting](https://arxiv.org/pdf/2203.15974)
- [DiarizationLM: Speaker Diarization Post-Processing with Large Language Models](https://arxiv.org/html/2401.03506v10)
- [Sortformer: Seamless Integration of Speaker Diarization and ASR...](https://arxiv.org/abs/2409.06656)
---

## 🔬 #4 Research Trends

### 🤯 Audio Language Models
- [Recent Advances in Speech Language Models: A Survey](https://arxiv.org/pdf/2410.03751)
- [Audio-Language Models for Audio-Centric Tasks: A survey](https://arxiv.org/pdf/2501.15177)
- [CosyVoice: A Scalable Multilingual Zero-shot Text to Speech...](https://arxiv.org/abs/2407.05407)
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885)
- [FunAudioLLM: Voice Understanding and Generation Foundation Models...](https://arxiv.org/html/2407.04051v1)
- [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale...](https://arxiv.org/abs/2311.07919)
- [MiniCPM: Unveiling the Potential of Small Language Models with Scalable...](https://arxiv.org/abs/2404.06395)
