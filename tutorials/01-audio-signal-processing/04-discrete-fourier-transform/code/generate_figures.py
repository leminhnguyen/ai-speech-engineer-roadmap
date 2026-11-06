"""Create deterministic plots embedded by Lesson 04."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import bin_frequencies, dft, magnitude_spectrum, sine_wave


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate, seconds = 8_000, 0.25
    n = round(sample_rate * seconds)
    components = ((440.0, 0.5), (1_000.0, 0.3), (2_400.0, 0.1))
    signal = sum(sine_wave(f, sample_rate, seconds, amplitude=a) for f, a in components)
    frequencies = bin_frequencies(n, sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2))
    time_ms = np.arange(n) / sample_rate * 1_000
    axes[0].plot(time_ms[:120], signal[:120], color="#2f6b9a", linewidth=1.6)
    axes[0].set(title="Time domain: three sines added together", xlabel="Time (ms)", ylabel="Amplitude")
    axes[0].grid(alpha=0.25)
    magnitude = magnitude_spectrum(signal)
    half = frequencies[: n // 2 + 1]
    axes[1].plot(half, magnitude, color="#b45050", linewidth=1.4)
    for f, a in components:
        axes[1].annotate(f"{f:.0f} Hz, A={a}", xy=(f, a), xytext=(f + 120, min(a + 0.03, 0.44)), fontsize=8, color="#40566b")
    axes[1].set(title="Frequency domain: the DFT reveals each ingredient", xlabel="Frequency (Hz)", ylabel="Amplitude")
    axes[1].set_xlim(0, sample_rate / 2)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "signal-and-spectrum.png", dpi=150)
    plt.close(fig)

    # Spectral leakage: an on-grid 440 Hz tone versus an off-grid 442 Hz tone.
    on_grid = magnitude_spectrum(sine_wave(440.0, sample_rate, seconds, amplitude=0.5))
    off_grid = magnitude_spectrum(sine_wave(442.0, sample_rate, seconds, amplitude=0.5))
    window = (half >= 380) & (half <= 500)
    fig, axis = plt.subplots(figsize=(8, 3.4))
    axis.stem(half[window], on_grid[window], linefmt="#2f6b9a", markerfmt="o", basefmt=" ", label="440 Hz (on a bin)")
    axis.stem(half[window] + 1.0, off_grid[window], linefmt="#b45050", markerfmt="s", basefmt=" ", label="442 Hz (between bins)")
    axis.set(title="Bins are 4 Hz apart; a tone between bins leaks into its neighbours", xlabel="Frequency (Hz)", ylabel="Amplitude")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "spectral-leakage.png", dpi=150)
    plt.close(fig)

    assert np.max(np.abs(dft(signal[:256]) - np.fft.fft(signal[:256]))) < 1e-8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
