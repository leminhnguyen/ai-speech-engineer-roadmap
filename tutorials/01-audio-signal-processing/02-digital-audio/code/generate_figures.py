"""Create deterministic plots embedded by Lesson 02."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import alias_frequency, dequantize_pcm, naive_downsample, quantize_pcm, sine_wave


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_rate, target_rate, frequency = 24_000, 8_000, 7_000.0
    original = sine_wave(frequency, original_rate, 0.0025)
    decimated = naive_downsample(original, 3)
    original_time = np.arange(len(original)) / original_rate * 1_000
    decimated_time = np.arange(len(decimated)) / target_rate * 1_000
    fig, axis = plt.subplots(figsize=(8, 3.4))
    axis.plot(original_time, original, label="7 kHz at 24 kHz", color="#2f6b9a")
    axis.scatter(decimated_time, decimated, label="kept samples at 8 kHz", color="#b45050", s=22, zorder=3)
    axis.set(title="Sampling too slowly makes 7 kHz indistinguishable from 1 kHz", xlabel="Time (ms)", ylabel="Amplitude")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "aliasing.png", dpi=150)
    plt.close(fig)

    samples = sine_wave(440.0, 16_000, 0.01, amplitude=0.9)
    low_bits = dequantize_pcm(quantize_pcm(samples, bits=3), bits=3)
    high_bits = dequantize_pcm(quantize_pcm(samples, bits=8), bits=8)
    time_ms = np.arange(len(samples)) / 16_000 * 1_000
    fig, axis = plt.subplots(figsize=(8, 3.4))
    axis.plot(time_ms, samples, label="original", color="#2f6b9a", linewidth=2)
    axis.step(time_ms, low_bits, label="3-bit PCM", color="#b45050", where="mid")
    axis.step(time_ms, high_bits, label="8-bit PCM", color="#3f8f5c", where="mid", alpha=0.8)
    axis.set(title="Quantization maps continuous values to finite levels", xlabel="Time (ms)", ylabel="Amplitude")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "quantization.png", dpi=150)
    plt.close(fig)
    assert alias_frequency(frequency, target_rate) == 1_000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
