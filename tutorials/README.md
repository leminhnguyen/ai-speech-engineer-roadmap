# 🛠️ Hands-on Tutorials

> Build speech fundamentals from scratch, so library calls stop being black boxes.

[← Back to home](../README.md) · [🗺️ Roadmap](../roadmap/README.md) · [📡 SOTA Tracker](../sota/README.md)

Every lesson follows the same loop:

```mermaid
flowchart LR
    A[💡 Understand] --> B[🧱 Build with NumPy] --> C[✅ Validate with tests] --> D[📚 Compare with the library] --> E[🏋️ Practice]
```

## 📚 Modules

| Module | Status | Lessons | Roadmap phase |
| --- | :---: | :---: | --- |
| 🔊 [**01 · Audio Signal Processing**](#-01--audio-signal-processing) | 🟢 Active | 8 / 12 | [Foundations](../roadmap/01-foundations.md#-audio-signal-processing-for-ml) |
| 🎙️ **02 · Speech Recognition** | ⚪ Planned | — | [Core Speech Technologies](../roadmap/03-core-speech-technologies.md) |
| 🗣️ **03 · Speech Synthesis** | ⚪ Planned | — | [Core Speech Technologies](../roadmap/03-core-speech-technologies.md) |

## 🔊 01 · Audio Signal Processing

![Progress](https://img.shields.io/badge/progress-8%20%2F%2012%20lessons-orange)

| | Lesson | Time | What you will build |
| :---: | --- | :---: | --- |
| ✅ | [01 · Sound and Waveforms](01-audio-signal-processing/01-sound-and-waveforms/docs/en.md) | ~45 min | Generate and mix tones, inspect spectra, and round-trip PCM16 WAV audio. |
| ✅ | [02 · Digital Audio](01-audio-signal-processing/02-digital-audio/docs/en.md) | ~50 min | Reproduce sampling, aliasing, resampling pitfalls, and PCM quantization. |
| ✅ | [03 · Time-Domain Features](01-audio-signal-processing/03-time-domain-features/docs/en.md) | ~50 min | Frame a signal and compute amplitude envelope, RMS energy, and zero-crossing rate. |
| ✅ | [04 · The Discrete Fourier Transform](01-audio-signal-processing/04-discrete-fourier-transform/docs/en.md) | ~60 min | Implement the DFT from its definition, read a spectrum, and explain spectral leakage. |
| ✅ | [05 · The Short-Time Fourier Transform](01-audio-signal-processing/05-short-time-fourier-transform/docs/en.md) | ~60 min | Window frames, build the STFT and its inverse, and explore the time-frequency trade-off. |
| ✅ | [06 · Mel Spectrograms and MFCCs](01-audio-signal-processing/06-mel-spectrograms-and-mfccs/docs/en.md) | ~70 min | Build a mel filterbank, log-mel spectrograms, and MFCCs, and show they track the vowel, not the pitch. |
| ✅ | [07 · Frequency-Domain Features](01-audio-signal-processing/07-frequency-domain-features/docs/en.md) | ~60 min | Compute band energy ratio, spectral centroid, bandwidth, rolloff, and flatness, and label vowels, fricatives, and silence. |
| ✅ | [08 · Pitch Estimation](01-audio-signal-processing/08-pitch-estimation/docs/en.md) | ~70 min | Build autocorrelation and YIN pitch trackers, make voicing decisions, and diagnose octave errors. |

## 📦 Inside Each Lesson

```text
NN-lesson-name/
├── docs/en.md            # Explanation, pitfalls, exercises, key terms
├── assets/               # Concept diagram and deterministic generated figures
├── code/
│   ├── main.py           # Runnable NumPy implementation
│   ├── generate_figures.py
│   └── tests/            # Unit tests for the concept
└── notebook/lesson.ipynb # Interactive companion, executed in CI
```

## 🚀 Run the Lessons

From the repository root, after following [Getting Started](../README.md#-getting-started):

```bash
make audit      # verify lesson structure
make test       # run unit tests
make figures    # regenerate plots
make notebooks  # execute every notebook from a clean state
```

Or run one lesson directly:

```bash
python tutorials/01-audio-signal-processing/01-sound-and-waveforms/code/main.py
```

## ✍️ Write a Lesson

Follow [LESSON_TEMPLATE.md](LESSON_TEMPLATE.md), keep examples deterministic, and give every diagram descriptive alt text. New lesson folders are picked up by `make` automatically.
