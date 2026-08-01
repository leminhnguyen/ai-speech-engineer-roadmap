# AI Speech Engineer Roadmap

> A curated roadmap based on six years of experience, designed to help learners become skilled AI Speech Engineers. The repository pairs this reading roadmap with focused build tutorials where they add practical value.

## Timeline

| Phase | Duration | Focus |
| --- | --- | --- |
| Foundations | 3 months | Python, ML, deep learning, signal processing |
| Tools & Frameworks | 3 months | Libraries, audio tools, Hugging Face |
| Core Technologies | 12 months | ASR, TTS, voice conversion, speaker recognition |
| Research Trends | Continuous | Audio-language models and new benchmarks |

## 1. Foundations

### Python Basics

- [Python Tutorial for Beginners](https://www.youtube.com/watch?v=YYXdXT2l-Gg&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)

### Machine Learning Basics

- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

### Deep Learning Basics

- [What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)

### Audio Signal Processing for ML

This topic has a hands-on tutorial track. Each lesson introduces the mental model, implements it with NumPy, and then compares it with the practical library workflow.

**Tutorial progress: 2 of 12 lessons**

| Status | Build alongside this roadmap topic | Core outcome |
| --- | --- | --- |
| ✓ | [01 · Sound and Waveforms](curriculum/01-audio-signal-processing/01-sound-and-waveforms/docs/en.md) | Synthesize, mix, inspect, and save PCM WAV audio. |
| ✓ | [02 · Digital Audio](curriculum/01-audio-signal-processing/02-digital-audio/docs/en.md) | Explain sampling and aliasing; reproduce PCM quantization. |
| Coming next | Time-Domain Features | Build amplitude envelope, RMS energy, and zero-crossing rate. |
| Planned | Fourier to MFCC | Progress from spectra and STFT to mel features and MFCCs. |

Run completed tutorials locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make audit test figures notebooks
```

- [Audio Signal Processing for Machine Learning — video series](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)

## 2. Tools & Frameworks

### Frameworks and Libraries

- `PyTorch` — training models.
- `librosa` — audio preprocessing, STFT, and MFCCs.
- `torchaudio` — loading, transforms, and model wrappers.
- `ffmpeg`, `sox`, and `pydub` — conversion, slicing, and format handling.
- `noisereduce` — simple noise reduction from raw audio.

### Tools and Courses

- [Audacity](https://www.audacityteam.org/) and its [tutorial](https://www.youtube.com/watch?v=vlzOb4OLj94)
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/en/chapter1/audio_data)

## 3. Core Speech Technologies

### Transformers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)

### Automatic Speech Recognition

- [CTC](https://distill.pub/2017/ctc/), [SpecAugment](https://blog.research.google/2019/04/specaugment-new-data-augmentation.html), [Wav2Vec 2.0](https://arxiv.org/abs/2005.08100), and [Illustrated Wav2Vec 2.0](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)
- [Large-scale simulated utterances in virtual rooms](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/509254e34b4c496eb3cfa1c2be1e1b5fc874bee3.pdf)
- [Whisper](https://arxiv.org/abs/2212.04356), [Fast Conformer](https://arxiv.org/abs/2305.05084), and [Zipformer](https://arxiv.org/abs/2310.11230)
- [ChunkFormer](https://arxiv.org/abs/2502.14673) and Vietnamese [Gipformer](https://github.com/ggroup-ai-lab/gipformer)
- [SpeechBrain ASR from scratch](https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-recognition-from-scratch.html)
- [VLSP 2025 ASR — Twinkle Team](materials/asr_vlsp_2025_twinkle_team.pdf)

### Text-to-Speech

- [Vietnamese HMM-based TTS](https://theses.hal.science/tel-01260884/document), [WaveNet](https://arxiv.org/abs/1609.03499), and [Tacotron](https://arxiv.org/abs/1703.10135)
- [WaveGlow](https://arxiv.org/abs/1811.00002), [FastSpeech](https://arxiv.org/abs/1905.09263), and [FastSpeech 2](https://arxiv.org/abs/2006.04558)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646), [VITS](https://arxiv.org/abs/2106.06103), [JETS](https://arxiv.org/abs/2203.16852), and [NaturalSpeech](https://arxiv.org/abs/2205.04421)
- [Kokoro TTS](https://kokorottsai.com/) and [my Vietnamese TTS thesis](materials/graduation-thesis.pdf)
- Vietnamese resources: [Viphoneme](https://github.com/v-nhandt21/Viphoneme), [Text2PhonemeSequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence), and [VLSP 2021 TTS](materials/vlsp_tts_2021_navi_team.pdf)

### Speaker Verification

- [Speaker Verification Introduction](https://maelfabien.github.io/machinelearning/Speech1/#), [x-vectors](https://danielpovey.com/files/2017_interspeech_embeddings.pdf), and [i-vectors](https://www.sciencedirect.com/science/article/pii/S1877050918314042/pdf)
- [VoxCeleb](https://arxiv.org/abs/1706.08612), [ECAPA-TDNN](https://arxiv.org/abs/2005.07143), [ResNeXt/Res2Net](https://arxiv.org/abs/2007.02480), and [CAM++](https://arxiv.org/abs/2303.00332)
- [3D-Speaker](https://arxiv.org/abs/2306.15354), [ERes2NetV2](https://arxiv.org/html/2406.02167v1), [Golden Gemini](https://arxiv.org/abs/2312.03620), and [RedimNet](https://arxiv.org/abs/2407.18223)

### Speaker Diarization

- [Introductory overview](https://lajavaness.medium.com/speaker-diarization-an-introductory-overview-c070a3bfea70), [pyannote.audio](https://arxiv.org/abs/1911.01255), and [a deep-learning survey](https://arxiv.org/abs/2101.09624)
- [Comparing pyannote and NeMo diarization frameworks](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)
- [Multi-scale diarization](https://arxiv.org/pdf/2203.15974), [DiarizationLM](https://arxiv.org/html/2401.03506v10), [Sortformer](https://arxiv.org/abs/2409.06656), and [Streaming Sortformer](https://arxiv.org/abs/2507.18446)
- [Speaker Diarization: From Traditional Methods to Modern Models](https://leminhnguyen.github.io/post/speech-research/speaker-diarization/)

### Voice Conversion

- [AutoVC](https://arxiv.org/abs/1905.05879), [VC overview](https://arxiv.org/abs/2008.03648), [AGAIN-VC](https://arxiv.org/abs/2011.00316), and [YourTTS](https://arxiv.org/abs/2112.02418)
- [kNN-VC](https://arxiv.org/abs/2305.18975), [Seed-VC](https://arxiv.org/abs/2411.09943), and [VLSP 2025 VC](materials/vc_vlsp_2025_twinkle_team.pdf)

## 4. Research Trends

> Updated August 2026. Research moves quickly; treat SOTA claims as benchmark-specific and verify the latest paper version, code, and evaluation setup.

### Surveys and Taxonomies

- [Recent Advances in Speech Language Models](https://arxiv.org/abs/2410.03751)
- [Audio-Language Models for Audio-Centric Tasks](https://arxiv.org/abs/2501.15177)
- [Landscape of Spoken Language Models](https://arxiv.org/abs/2504.08528)
- [Discrete Speech Tokens review](https://arxiv.org/abs/2502.06490)

### Speech Tokenization and Neural Codecs

- [WavTokenizer](https://arxiv.org/abs/2408.16532) and [BigCodec](https://arxiv.org/abs/2409.05377)

### Audio Understanding and Reasoning

- [Qwen-Audio](https://arxiv.org/abs/2311.07919), [Kimi-Audio](https://arxiv.org/abs/2504.18425), [Audio Flamingo 3](https://arxiv.org/abs/2507.08128), and [Qwen3-ASR](https://arxiv.org/abs/2601.21337)

### Real-Time Spoken Dialogue and Omni Models

- [Mini-Omni](https://arxiv.org/abs/2408.16725), [EMOVA](https://arxiv.org/abs/2409.18042), [FunAudioLLM](https://arxiv.org/abs/2407.04051), and [Moshi](https://arxiv.org/abs/2410.00037)
- [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215), [LLaMA-Omni 2](https://arxiv.org/abs/2505.02625), and [Step-Audio 2](https://arxiv.org/abs/2507.16632)

### Speech Generation and Multilingual TTS

- [CosyVoice](https://arxiv.org/abs/2407.05407), [F5-TTS](https://arxiv.org/abs/2410.06885), and [CosyVoice 3](https://arxiv.org/abs/2505.17589)
- [Qwen3-TTS](https://arxiv.org/abs/2601.15621) and [OmniVoice](https://arxiv.org/abs/2604.00688)

### Evaluation, Benchmarks, and Data

- [VoiceBench](https://arxiv.org/abs/2410.17196), [Full-Duplex-Bench](https://arxiv.org/abs/2507.23159), and [ART](https://arxiv.org/abs/2601.19673)
- [WavBench](https://arxiv.org/abs/2602.12135), [HumDial-FDBench](https://arxiv.org/abs/2604.21406), and [DuplexChat](https://arxiv.org/abs/2607.04941)
