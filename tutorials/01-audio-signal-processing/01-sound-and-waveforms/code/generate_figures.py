"""Create deterministic plots embedded by Lesson 01."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import mix, sine_wave, time_axis


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    tone = sine_wave(440.0, sample_rate, 0.03)
    time_ms = time_axis(sample_rate, 0.03) * 1_000
    fig, axis = plt.subplots(figsize=(8, 3))
    axis.plot(time_ms, tone, color="#2f6b9a", linewidth=1.5)
    axis.set(title="A 440 Hz sine waveform", xlabel="Time (ms)", ylabel="Amplitude")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sine-wave.png", dpi=150)
    plt.close(fig)

    seconds = 0.04
    signals = [sine_wave(freq, sample_rate, seconds) for freq in (220.0, 440.0, 880.0)]
    chord = mix(signals)
    frequencies = np.fft.rfftfreq(len(chord), 1.0 / sample_rate)
    magnitude = np.abs(np.fft.rfft(chord))
    fig, (wave_axis, spectrum_axis) = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)
    wave_axis.plot(time_axis(sample_rate, seconds) * 1_000, chord, color="#b45050")
    wave_axis.set(title="A mixed waveform", xlabel="Time (ms)", ylabel="Amplitude")
    spectrum_axis.plot(frequencies, magnitude, color="#3f8f5c")
    spectrum_axis.set(xlim=(0, 1_200), title="Its frequency components", xlabel="Frequency (Hz)", ylabel="Magnitude")
    spectrum_axis.grid(alpha=0.25)
    fig.savefig(output_dir / "mixed-wave-and-spectrum.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
