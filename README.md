# 🐌 AI Speech Engineer Roadmap

[![Tutorial checks](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/actions/workflows/tutorial.yml/badge.svg)](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/actions/workflows/tutorial.yml)
[![GitHub stars](https://img.shields.io/github/stars/leminhnguyen/ai-speech-engineer-roadmap?style=social)](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/leminhnguyen/ai-speech-engineer-roadmap?style=social)](https://github.com/leminhnguyen/ai-speech-engineer-roadmap/forks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A practical learning path for becoming an AI Speech Engineer—from Python and signal processing fundamentals to ASR, TTS, speaker technologies, voice conversion, and current speech-language research.

This repository combines two complementary resources:

- **A curated roadmap** of foundational material, tools, papers, and modern speech systems.
- **A hands-on tutorial track** that builds core audio concepts with NumPy before comparing them with production libraries.

> Start with the [complete learning roadmap](ROADMAP.md), or jump directly into the [first tutorial](curriculum/01-audio-signal-processing/01-sound-and-waveforms/docs/en.md).

## Contents

- [Roadmap](#roadmap)
- [Learning path](#learning-path)
- [Hands-on tutorial track](#hands-on-tutorial-track)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Quality checks](#quality-checks)
- [Contributing](#contributing)

## Roadmap

[![AI Speech Engineer Roadmap](ai-speech-engineer-roadmap.png)](ROADMAP.md)

The [detailed roadmap](ROADMAP.md) contains the recommended courses, papers, implementations, and Vietnamese speech resources for every topic.

## Learning path

| Phase | Suggested duration | Main outcomes |
| --- | ---: | --- |
| [01 · Foundations](ROADMAP.md#1-foundations) | 3 months | Build fluency in Python, machine learning, deep learning, and audio signal processing. |
| [02 · Tools & Frameworks](ROADMAP.md#2-tools--frameworks) | 3 months | Work with PyTorch, librosa, torchaudio, audio utilities, and Hugging Face. |
| [03 · Core Speech Technologies](ROADMAP.md#3-core-speech-technologies) | 12 months | Study transformers, ASR, TTS, speaker verification, diarization, and voice conversion. |
| [04 · Research Trends](ROADMAP.md#4-research-trends) | Continuous | Follow speech tokenization, audio-language models, real-time dialogue, generation, and evaluation. |

The durations are guidelines, not deadlines. Move forward when you can explain the core idea, reproduce a small implementation, and evaluate its failure modes.

## Hands-on tutorial track

**Progress: 2 of 12 lessons available**

Each lesson follows the same learning loop:

1. Develop an intuitive mental model.
2. Implement the core idea with NumPy.
3. Validate it with known cases and unit tests.
4. Compare it with the practical library workflow.
5. Reinforce it through exercises and common pitfalls.

| Status | Lesson | Time | What you will build |
| :---: | --- | ---: | --- |
| ✅ | [01 · Sound and Waveforms](curriculum/01-audio-signal-processing/01-sound-and-waveforms/docs/en.md) | ~45 min | Generate and mix tones, inspect spectra, and round-trip PCM16 WAV audio. |
| ✅ | [02 · Digital Audio](curriculum/01-audio-signal-processing/02-digital-audio/docs/en.md) | ~50 min | Reproduce sampling, aliasing, resampling pitfalls, and PCM quantization. |
| 🚧 | Time-Domain Features | Coming next | Build amplitude envelope, RMS energy, and zero-crossing rate. |

Lessons include an explanation, conceptual diagram, runnable implementation, deterministic figures, tests, and an executable notebook.

## Getting started

### Prerequisites

- Python 3.11 or newer
- `make`
- Git

### Set up the environment

```bash
git clone https://github.com/leminhnguyen/ai-speech-engineer-roadmap.git
cd ai-speech-engineer-roadmap

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

### Run the tutorial suite

```bash
make audit
make test
make figures
make notebooks
```

To run one lesson directly:

```bash
python curriculum/01-audio-signal-processing/01-sound-and-waveforms/code/main.py
```

## Repository structure

```text
.
├── README.md                         # Project overview and quick start
├── ROADMAP.md                        # Full learning path and references
├── ai-speech-engineer-roadmap.png    # Rendered roadmap diagram
├── curriculum/
│   └── 01-audio-signal-processing/
│       └── NN-lesson-name/
│           ├── docs/en.md            # Learner-facing lesson
│           ├── assets/               # Concept art and generated figures
│           ├── code/                 # Implementation and tests
│           └── notebook/lesson.ipynb # Interactive companion
├── materials/                        # Selected theses and VLSP reports
├── scripts/audit_curriculum.py       # Lesson structure validation
├── LESSON_TEMPLATE.md                # Contract for new lessons
└── Makefile                          # Reproducible project commands
```

## Quality checks

| Command | Purpose |
| --- | --- |
| `make audit` | Verify that every lesson contains the required documentation, assets, code, tests, and notebook. |
| `make test` | Run the unit tests for the implemented signal-processing concepts. |
| `make figures` | Regenerate committed plots from source code. |
| `make notebooks` | Execute every lesson notebook from a clean state. |

The same checks run in GitHub Actions for pushes and pull requests.

## Contributing

Contributions that improve explanations, tests, references, accessibility, or lesson coverage are welcome.

For a new lesson:

1. Follow the structure and content contract in [LESSON_TEMPLATE.md](LESSON_TEMPLATE.md).
2. Prefer primary papers and official library documentation.
3. Keep examples deterministic and diagrams accessible with descriptive alt text.
4. Run `make audit test figures notebooks` before opening a pull request.

For substantial roadmap changes, explain the learning objective and why the resource improves the existing sequence.

## Acknowledgments

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for the projects that informed the tutorial sequence and repository structure.

## License

This project is available under the [MIT License](LICENSE).
