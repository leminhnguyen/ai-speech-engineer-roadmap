"""Create deterministic plots embedded by Lesson 03."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import amplitude_envelope, frames_to_time, rms_energy, sine_wave, zero_crossing_rate


def synthetic_speech_like(sample_rate: int) -> np.ndarray:
    """A deterministic 'voiced - silence - noisy' sequence for illustrating features."""
    rng = np.random.default_rng(7)
    voiced = sine_wave(150.0, sample_rate, 0.6, amplitude=0.5) + sine_wave(300.0, sample_rate, 0.6, amplitude=0.25)
    fade = np.linspace(0.2, 1.0, len(voiced)) * np.linspace(1.0, 0.4, len(voiced))
    voiced = voiced * fade
    silence = np.zeros(round(sample_rate * 0.3))
    noisy = rng.normal(0.0, 0.18, round(sample_rate * 0.6))
    return np.concatenate([voiced, silence, noisy])


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate, frame_length, hop_length = 16_000, 1024, 512
    signal = synthetic_speech_like(sample_rate)
    time_s = np.arange(len(signal)) / sample_rate

    envelope = amplitude_envelope(signal, frame_length, hop_length)
    rms = rms_energy(signal, frame_length, hop_length)
    zcr = zero_crossing_rate(signal, frame_length, hop_length)
    frame_time = frames_to_time(len(rms), hop_length, sample_rate)

    fig, axis = plt.subplots(figsize=(8, 3.4))
    axis.plot(time_s, signal, color="#2f6b9a", alpha=0.45, linewidth=0.8, label="waveform")
    axis.step(frame_time, envelope, where="post", color="#b45050", linewidth=1.8, label="amplitude envelope")
    axis.step(frame_time, rms, where="post", color="#3f8f5c", linewidth=1.8, label="RMS energy")
    axis.set(title="Envelope follows peaks; RMS follows average loudness", xlabel="Time (s)", ylabel="Amplitude")
    axis.set_ylim(-1.0, 1.0)
    axis.legend(loc="upper right", ncol=3, fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "envelope-and-rms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    axes[0].plot(time_s, signal, color="#2f6b9a", alpha=0.6, linewidth=0.8)
    axes[0].set(title="Zero-crossing rate separates tonal, silent, and noisy regions", ylabel="Amplitude")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[1].step(frame_time, zcr, where="post", color="#9a4f12", linewidth=1.8)
    axes[1].set(xlabel="Time (s)", ylabel="ZCR (fraction)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    for axis, label, x in ((axes[1], "voiced", 0.3), (axes[1], "silence", 0.75), (axes[1], "noise", 1.2)):
        axis.text(x, 0.88, label, ha="center", fontsize=9, color="#40566b")
    fig.tight_layout()
    fig.savefig(output_dir / "zero-crossing-rate.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
