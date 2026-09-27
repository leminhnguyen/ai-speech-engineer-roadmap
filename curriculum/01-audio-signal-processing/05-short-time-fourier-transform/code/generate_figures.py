"""Create deterministic plots embedded by Lesson 05."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import frames_to_time, hann_window, linear_chirp, power_spectrogram, power_to_db, sine_wave, stft_frequencies


def show_spectrogram(axis, samples, sample_rate, n_fft, hop_length, max_hz):
    db = power_to_db(power_spectrogram(samples, n_fft, hop_length), reference=float(n_fft), top_db=80.0)
    frequencies = stft_frequencies(sample_rate, n_fft)
    times = frames_to_time(db.shape[1], hop_length, sample_rate)
    keep = frequencies <= max_hz
    return axis.pcolormesh(times, frequencies[keep], db[keep], shading="nearest", cmap="magma")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Windowing: the same off-bin 442 Hz tone with a rectangular and a Hann window.
    sample_rate, seconds = 8_000, 0.25
    tone = sine_wave(442.0, sample_rate, seconds, amplitude=0.5)
    window = hann_window(len(tone))
    frequencies = np.fft.rfftfreq(len(tone), d=1.0 / sample_rate)
    rect_db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(tone)) / (len(tone) / 2), 1e-6))
    hann_db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(tone * window)) / (window.sum() / 2), 1e-6))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.4))
    time_ms = np.arange(len(tone)) / sample_rate * 1_000
    axes[0].plot(time_ms, tone, color="#9aa5b1", linewidth=0.8, label="rectangular (frame as cut)")
    axes[0].plot(time_ms, tone * window, color="#2f6b9a", linewidth=1.0, label="Hann-windowed")
    axes[0].plot(time_ms, 0.5 * window, color="#b45050", linewidth=1.4, linestyle="--", label="Hann window × 0.5")
    axes[0].set(title="A Hann window tapers the frame to zero at both edges", xlabel="Time (ms)", ylabel="Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(frequencies, rect_db, color="#9aa5b1", linewidth=1.2, label="rectangular")
    axes[1].plot(frequencies, hann_db, color="#2f6b9a", linewidth=1.4, label="Hann")
    axes[1].set(title="Leakage from a 442 Hz tone: the Hann window confines it near the peak",
                xlabel="Frequency (Hz)", ylabel="Level (dB re. amplitude 1)")
    axes[1].set_xlim(0, 1_200)
    axes[1].set_ylim(-110, 0)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "windowing-and-leakage.png", dpi=150)
    plt.close(fig)

    # A chirp plus a steady tone: one FFT versus the STFT.
    sample_rate = 16_000
    signal = linear_chirp(300.0, 3_000.0, sample_rate, 1.0, amplitude=0.6) + sine_wave(1_000.0, sample_rate, 1.0, amplitude=0.2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 2.4]})
    whole = np.abs(np.fft.rfft(signal * hann_window(len(signal))))
    axes[0].plot(np.fft.rfftfreq(len(signal), d=1.0 / sample_rate), whole / whole.max(), color="#b45050", linewidth=1.0)
    axes[0].set(title="One FFT of the whole second", xlabel="Frequency (Hz)", ylabel="Relative magnitude")
    axes[0].set_xlim(0, 4_000)
    axes[0].grid(alpha=0.25)
    mesh = show_spectrogram(axes[1], signal, sample_rate, 512, 128, 4_000)
    axes[1].set(title="STFT: n_fft=512 (32 ms), hop=128 (8 ms)", xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(mesh, ax=axes[1], label="dB")
    fig.tight_layout()
    fig.savefig(output_dir / "chirp-spectrogram.png", dpi=150)
    plt.close(fig)

    # Time-frequency trade-off: clicks need short frames, close tones need long frames.
    sample_rate, seconds = 8_000, 1.0
    signal = sine_wave(1_000.0, sample_rate, seconds, amplitude=0.3) + sine_wave(1_080.0, sample_rate, seconds, amplitude=0.3)
    signal[[1_000, 3_000, 5_000, 7_000]] += 1.0
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for axis, n_fft in zip(axes, (128, 1_024)):
        show_spectrogram(axis, signal, sample_rate, n_fft, 32, 2_000)
        axis.set(title=f"n_fft={n_fft}: {1_000 * n_fft / sample_rate:.0f} ms frames, {sample_rate / n_fft:.1f} Hz bins",
                 xlabel="Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.suptitle("Short frames resolve the clicks; long frames separate 1000 Hz from 1080 Hz", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "time-frequency-tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
