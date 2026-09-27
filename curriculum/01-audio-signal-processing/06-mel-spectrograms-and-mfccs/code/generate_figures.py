"""Create deterministic plots embedded by Lesson 06."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    apply_filterbank,
    hz_to_mel,
    mel_filterbank,
    mel_frequencies,
    mfcc_from_log_mel,
    power_spectrogram,
    power_to_db,
    synthetic_vowel,
)

SAMPLE_RATE, N_FFT, HOP, N_MELS, N_MFCC = 16_000, 512, 160, 40, 13
VOWEL_A = (730.0, 1_090.0, 2_440.0)
VOWEL_I = (270.0, 2_290.0, 3_010.0)


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bins_hz = np.arange(N_FFT // 2 + 1) * SAMPLE_RATE / N_FFT

    # The mel scale and the triangular filters built on it.
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), gridspec_kw={"height_ratios": [1, 1.3]})
    hz = np.linspace(0, SAMPLE_RATE / 2, 400)
    axes[0].plot(hz, hz_to_mel(hz), color="#2f6b9a", linewidth=1.8, label="mel = 2595 log10(1 + f / 700)")
    axes[0].plot(hz, hz, color="#9aa5b1", linewidth=1.0, linestyle="--", label="identity")
    axes[0].set(title="The mel scale is close to linear below 1 kHz and compresses above", xlabel="Frequency (Hz)", ylabel="Mel")
    axes[0].set_ylim(0, 3_200)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.25)
    filters = mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, normalize=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, N_MELS))
    for row, color in zip(filters, colors):
        axes[1].plot(bins_hz, row, color=color, linewidth=1.0)
    axes[1].set(title=f"{N_MELS} triangular filters, evenly spaced in mels (before area normalisation)",
                xlabel="Frequency (Hz)", ylabel="Weight")
    axes[1].set_xlim(0, SAMPLE_RATE / 2)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mel-filterbank.png", dpi=150)
    plt.close(fig)

    # One frame of the same vowel at two pitches: different harmonics, same envelope.
    filters = mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
    centers = mel_frequencies(N_MELS + 2, 0.0, SAMPLE_RATE / 2)[1:-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for axis, pitch in zip(axes, (120.0, 200.0)):
        power = power_spectrogram(synthetic_vowel(VOWEL_A, pitch, SAMPLE_RATE, 0.3), N_FFT, HOP)[:, 10:11]
        mel_db = power_to_db(apply_filterbank(filters, power), top_db=None)[:, 0]
        # Undo the unit-area scaling so the mel envelope sits on the same dB scale as the bins.
        mel_db -= 10 * np.log10(filters.sum(axis=1))
        axis.plot(bins_hz, power_to_db(power, top_db=None)[:, 0], color="#9aa5b1", linewidth=0.9, label="STFT bins")
        axis.plot(centers, mel_db, color="#b45050", linewidth=1.8, marker="o", markersize=2.5, label="mel bands")
        axis.set(title=f"Vowel /a/ at pitch {pitch:.0f} Hz", xlabel="Frequency (Hz)")
        axis.set_xlim(0, 4_000)
        axis.set_ylim(-40, 70)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Power (dB)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Pitch moves the harmonics; the mel bands keep the formant envelope", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "pitch-and-envelope.png", dpi=150)
    plt.close(fig)

    # /a/ then /i/: spectrogram, log-mel spectrogram, MFCCs.
    signal = np.concatenate([
        synthetic_vowel(VOWEL_A, 120.0, SAMPLE_RATE, 0.5),
        synthetic_vowel(VOWEL_I, 120.0, SAMPLE_RATE, 0.5),
    ])
    power = power_spectrogram(signal, N_FFT, HOP)
    log_mel = power_to_db(apply_filterbank(filters, power))
    coefficients = mfcc_from_log_mel(log_mel, N_MFCC)
    times = np.arange(power.shape[1]) * HOP / SAMPLE_RATE
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)
    axes[0].pcolormesh(times, bins_hz, power_to_db(power), shading="nearest", cmap="magma")
    axes[0].set(title=f"Power spectrogram: {power.shape[0]} bins per frame", ylabel="Frequency (Hz)")
    axes[0].set_ylim(0, 4_000)
    axes[1].pcolormesh(times, np.arange(N_MELS), log_mel, shading="nearest", cmap="magma")
    axes[1].set(title=f"Log-mel spectrogram: {N_MELS} bands per frame", ylabel="Mel band")
    mesh = axes[2].pcolormesh(times, np.arange(1, N_MFCC), coefficients[1:], shading="nearest", cmap="coolwarm",
                              vmin=-np.abs(coefficients[1:]).max(), vmax=np.abs(coefficients[1:]).max())
    axes[2].set(title="MFCCs c1 to c12 (c0, the overall level, omitted)", xlabel="Time (s)", ylabel="Coefficient")
    fig.colorbar(mesh, ax=axes[2], orientation="horizontal", fraction=0.06, pad=0.3)
    for axis in axes:
        axis.axvline(0.5, color="#ffffff", linewidth=0.8, linestyle="--")
    fig.suptitle("Synthetic /a/ then /i/ at the same pitch", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "vowel-features.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
