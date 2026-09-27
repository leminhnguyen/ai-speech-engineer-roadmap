<div align="center">

# 🐌 AI Speech Engineer Roadmap

**A practical, end-to-end learning path for becoming an AI Speech Engineer.**

From Python and signal-processing fundamentals to ASR, TTS, speaker technologies, voice conversion, and today's audio-language models.

[![Tutorial checks](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/actions/workflows/tutorial.yml/badge.svg)](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/actions/workflows/tutorial.yml)
[![GitHub stars](https://img.shields.io/github/stars/leminhnguyen/ai-speech-engineer-roadmap?style=flat&logo=github&color=yellow)](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/leminhnguyen/ai-speech-engineer-roadmap?style=flat&logo=github)](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/forks)
[![Last commit](https://img.shields.io/github/last-commit/leminhnguyen/ai-speech-engineer-roadmap?logo=git&logoColor=white)](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/commits/master)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

[🗺️ Roadmap](ROADMAP.md) ·
[🛠️ Tutorials](#%EF%B8%8F-hands-on-tutorial-track) ·
[🚀 Getting Started](#-getting-started) ·
[🤝 Contributing](#-contributing) ·
[🐛 Report an Issue](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/issues)

<br/>

[![AI Speech Engineer Roadmap](ai-speech-engineer-roadmap.png)](ROADMAP.md)

</div>

---

## 📖 Overview

Speech AI sits at the intersection of signal processing, deep learning, and linguistics, and the good material is scattered across courses, papers, and library docs. This repository brings it together into one opinionated path, built on six years of hands-on experience shipping speech systems.

It combines two complementary resources:

| | Resource | What you get |
| :---: | --- | --- |
| 🗺️ | **[Curated roadmap](ROADMAP.md)** | Courses, must-read papers (⭐ marked), tools, and Vietnamese speech resources for every stage, from fundamentals to research trends. |
| 🛠️ | **[Hands-on tutorial track](#%EF%B8%8F-hands-on-tutorial-track)** | Build core audio concepts from scratch with NumPy, validate them with tests, then compare with production libraries. |

> **New here?** Start with the [complete roadmap](ROADMAP.md), or jump straight into the [first lesson](curriculum/01-audio-signal-processing/01-sound-and-waveforms/docs/en.md).

## ✨ Highlights

- 🎯 **Opinionated, not exhaustive.** Every phase lists the few resources worth your time, with ⭐ marking the essentials.
- 🧱 **Build before you import.** Tutorials implement each concept in NumPy first, so library calls stop being black boxes.
- ✅ **Reproducible and tested.** Every lesson ships with unit tests, deterministic figures, and an executable notebook, all verified in CI.
- 🔬 **Kept current.** The research section tracks speech tokenization, audio-language models, full-duplex dialogue, and modern benchmarks.
- 🇻🇳 **Vietnamese speech coverage.** Dedicated resources, theses, and VLSP reports for a low-resource language.

## 🗺️ Learning Path

| Phase | Duration | What you will learn |
| --- | :---: | --- |
| 🧠 [**01 · Foundations**](ROADMAP.md#-1-foundations-3-months) | 3 months | Python, machine learning, deep learning, and audio signal processing. |
| 💼 [**02 · Tools & Frameworks**](ROADMAP.md#-2-tools--frameworks-3-months) | 3 months | PyTorch, librosa, torchaudio, audio utilities, and the Hugging Face Audio course. |
| 🌱 [**03 · Core Speech Technologies**](ROADMAP.md#-3-dive-into-speech-core-technologies-12-months) | 12 months | Transformers, ASR, TTS, speaker verification, speaker diarization, and voice conversion. |
| 🔬 [**04 · Research Trends**](ROADMAP.md#-4-research-trends) | Continuous | Speech tokenization, audio-language models, real-time dialogue, generation, and evaluation. |

Durations are guidelines, not deadlines. Move on when you can explain the core idea, reproduce a small implementation, and describe its failure modes.

## 🛠️ Hands-on Tutorial Track

![Progress](https://img.shields.io/badge/progress-6%20%2F%2012%20lessons-orange)

Every lesson follows the same learning loop: **understand → build with NumPy → validate → compare with the library → practice.**

| Status | Lesson | Time | What you will build |
| :---: | --- | :---: | --- |
| ✅ | [01 · Sound and Waveforms](curriculum/01-audio-signal-processing/01-sound-and-waveforms/docs/en.md) | ~45 min | Generate and mix tones, inspect spectra, and round-trip PCM16 WAV audio. |
| ✅ | [02 · Digital Audio](curriculum/01-audio-signal-processing/02-digital-audio/docs/en.md) | ~50 min | Reproduce sampling, aliasing, resampling pitfalls, and PCM quantization. |
| ✅ | [03 · Time-Domain Features](curriculum/01-audio-signal-processing/03-time-domain-features/docs/en.md) | ~50 min | Frame a signal and compute amplitude envelope, RMS energy, and zero-crossing rate. |
| ✅ | [04 · The Discrete Fourier Transform](curriculum/01-audio-signal-processing/04-discrete-fourier-transform/docs/en.md) | ~60 min | Implement the DFT from its definition, read a spectrum, and explain spectral leakage. |
| ✅ | [05 · The Short-Time Fourier Transform](curriculum/01-audio-signal-processing/05-short-time-fourier-transform/docs/en.md) | ~60 min | Window frames, build the STFT and its inverse, and explore the time-frequency trade-off. |
| ✅ | [06 · Mel Spectrograms and MFCCs](curriculum/01-audio-signal-processing/06-mel-spectrograms-and-mfccs/docs/en.md) | ~70 min | Build a mel filterbank, log-mel spectrograms, and MFCCs, and show they track the vowel, not the pitch. |

<details>
<summary><b>What is inside each lesson?</b></summary>
<br/>

```text
NN-lesson-name/
├── docs/en.md            # Learner-facing explanation, pitfalls, exercises, key terms
├── assets/               # Concept diagram and deterministic generated figures
├── code/
│   ├── main.py           # Runnable NumPy implementation
│   └── tests/            # Unit tests for the concept
└── notebook/lesson.ipynb # Interactive companion, executed in CI
```

</details>

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or newer
- `make` and Git

### Installation

```bash
git clone https://github.com/leminhnguyen/ai-speech-engineer-roadmap.git
cd ai-speech-engineer-roadmap

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the tutorials

```bash
make audit      # verify lesson structure
make test       # run unit tests
make figures    # regenerate plots
make notebooks  # execute every notebook from a clean state
```

Or run a single lesson directly:

```bash
python curriculum/01-audio-signal-processing/01-sound-and-waveforms/code/main.py
```

## 📂 Repository Structure

```text
.
├── README.md                        # Project overview and quick start
├── ROADMAP.md                       # Full learning path and references
├── ai-speech-engineer-roadmap.png   # Rendered roadmap diagram
├── curriculum/
│   └── 01-audio-signal-processing/  # One folder per lesson (see above)
├── materials/                       # Selected theses and VLSP reports
├── scripts/audit_curriculum.py      # Lesson structure validation
├── LESSON_TEMPLATE.md               # Contract for new lessons
├── Makefile                         # Reproducible project commands
└── requirements.txt
```

## 🤝 Contributing

Contributions that improve explanations, tests, references, accessibility, or lesson coverage are welcome.

1. **Fix or improve a resource.** Prefer primary papers and official documentation. For roadmap changes, explain the learning objective and why the resource improves the sequence.
2. **Add a lesson.** Follow [LESSON_TEMPLATE.md](LESSON_TEMPLATE.md), keep examples deterministic, and give diagrams descriptive alt text.
3. **Verify before opening a PR.**

   ```bash
   make audit test figures notebooks
   ```

The same checks run in [GitHub Actions](.github/workflows/tutorial.yml) on every push and pull request.

## ⭐ Star History

<a href="https://star-history.com/#leminhnguyen/ai-speech-engineer-roadmap&Date">
  <img src="https://api.star-history.com/svg?repos=leminhnguyen/ai-speech-engineer-roadmap&type=Date" alt="Star History Chart" width="600"/>
</a>

## 📝 Citation

If this roadmap helps your learning, research, or teaching, please consider citing it:

```bibtex
@misc{nguyen2026aispeechroadmap,
  author = {Nguyen, Le Minh},
  title  = {AI Speech Engineer Roadmap},
  year   = {2026},
  url    = {https://github.com/leminhnguyen/ai-speech-engineer-roadmap}
}
```

## 🙏 Acknowledgments

The tutorial sequence was informed by [Audio Signal Processing for Machine Learning](https://github.com/musikalkemist/AudioSignalProcessingForML) and the lesson-first layout by [AI Engineering From Scratch](https://github.com/patchwork-ai/ai-engineering-from-scratch). See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for details.

## 📄 License

Distributed under the [MIT License](LICENSE).

<div align="center">
<br/>

**If you find this useful, give it a ⭐ to help others discover it.**

[⬆ Back to top](#-ai-speech-engineer-roadmap)

</div>
