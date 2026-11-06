# 💼 02 · Tools & Frameworks

> **Duration:** ~3 months · **Goal:** load, transform, and train on audio with the standard toolkit, and inspect audio by ear and by eye.

[← Foundations](01-foundations.md) · [Roadmap](README.md) · [Next: Core Speech Technologies →](03-core-speech-technologies.md)

## 🧰 Frameworks & Libraries

| Tool | Use it for |
| --- | --- |
| ⭐ [PyTorch](https://pytorch.org/) | Building and training models |
| ⭐ [librosa](https://librosa.org/) | Audio analysis and features: STFT, mel spectrograms, MFCCs, pitch |
| ⭐ [torchaudio](https://pytorch.org/audio/) | Audio loading, GPU transforms, and pretrained model wrappers |
| `ffmpeg`, `sox`, `pydub` | Format conversion, resampling, slicing |
| `noisereduce` | Simple noise reduction on raw audio |

## 🖥️ Tools

- [Audacity](https://www.audacityteam.org/) - Free, powerful software for editing and visualizing audio
- 📺 [Audacity Tutorial](https://www.youtube.com/watch?v=vlzOb4OLj94)

## 🤗 Hugging Face

- ⭐ [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) - Tackle a range of audio tasks and gain experience with speech datasets.

## ✅ You Are Ready to Move On When You Can

- Build a PyTorch `Dataset` that loads, resamples, and batches variable-length audio.
- Reproduce a librosa feature with torchaudio and explain any numerical differences.
- Fine-tune a pretrained Hugging Face audio model on a small dataset.
