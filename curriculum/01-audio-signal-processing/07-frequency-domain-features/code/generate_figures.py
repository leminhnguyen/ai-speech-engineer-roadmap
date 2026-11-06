"""Create deterministic plots embedded by Lesson 07."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    asa_utterance,
    band_energy_ratio,
    bin_frequencies,
    highpass_noise,
    label_frames,
    magnitude_spectrogram,
    rms_from_magnitude,
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
    synthetic_vowel,
)

SAMPLE_RATE, N_FFT, HOP = 16_000, 512, 160
LABEL_COLORS = {"silence": "#d9dee4", "voiced": "#b8d4ea", "fricative": "#f3c9a6"}


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bins_hz = bin_frequencies(SAMPLE_RATE, N_FFT)

    # One frame of four sounds, with centroid, +/- bandwidth, and rolloff marked.
    sounds = [
        ("Vowel /a/ (120 Hz pitch)", synthetic_vowel((730.0, 1_090.0, 2_440.0), 120.0, SAMPLE_RATE, 0.3)),
        ("Vowel /i/ (120 Hz pitch)", synthetic_vowel((270.0, 2_290.0, 3_010.0), 120.0, SAMPLE_RATE, 0.3)),
        ("/s/-like hiss (high-pass noise)", highpass_noise(4_000.0, SAMPLE_RATE, 0.3)),
        ("White noise", highpass_noise(0.0, SAMPLE_RATE, 0.3)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), sharex=True, sharey=True)
    for axis, (title, samples) in zip(axes.flat, sounds):
        frame = magnitude_spectrogram(samples, N_FFT, HOP)[:, 15:16]
        centroid = spectral_centroid(frame, SAMPLE_RATE, N_FFT)[0]
        bandwidth = spectral_bandwidth(frame, SAMPLE_RATE, N_FFT)[0]
        rolloff = spectral_rolloff(frame, SAMPLE_RATE, N_FFT)[0]
        flatness = spectral_flatness(frame)[0]
        db = 20.0 * np.log10(np.maximum(frame[:, 0], 1e-5) / frame.max())
        axis.axvspan(max(centroid - bandwidth, 0.0), centroid + bandwidth, color="#2f6b9a", alpha=0.12,
                     label="centroid ± bandwidth")
        axis.plot(bins_hz, db, color="#5b6770", linewidth=0.9)
        axis.axvline(centroid, color="#2f6b9a", linewidth=1.8, label=f"centroid {centroid:,.0f} Hz")
        axis.axvline(rolloff, color="#b45050", linewidth=1.4, linestyle="--", label=f"85% rolloff {rolloff:,.0f} Hz")
        axis.set_title(f"{title}\nflatness {flatness:.3f}", fontsize=9)
        axis.set_ylim(-80, 5)
        axis.set_xlim(0, SAMPLE_RATE / 2)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left", fontsize=7)
    for axis in axes[1]:
        axis.set_xlabel("Frequency (Hz)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Magnitude (dB re. peak)")
    fig.suptitle("Each feature compresses one frame's spectrum to a single number", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "spectral-shapes.png", dpi=150)
    plt.close(fig)

    # A synthetic 'asa': spectrogram with the centroid track, then BER, flatness, and RMS over time.
    utterance, segments = asa_utterance(SAMPLE_RATE)
    magnitude = magnitude_spectrogram(utterance, N_FFT, HOP)
    times = np.arange(magnitude.shape[1]) * HOP / SAMPLE_RATE
    labels = label_frames(magnitude, SAMPLE_RATE, N_FFT)
    rms = rms_from_magnitude(magnitude, N_FFT)
    loud = rms >= 0.01
    centroid = np.where(loud, spectral_centroid(magnitude, SAMPLE_RATE, N_FFT), np.nan)
    ber_db = np.where(loud, 10.0 * np.log10(np.maximum(band_energy_ratio(magnitude ** 2, 2_000.0, SAMPLE_RATE, N_FFT), 1e-10)), np.nan)
    flatness = np.where(loud, spectral_flatness(magnitude), np.nan)

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-5) / magnitude.max())
    axes[0].pcolormesh(times, bins_hz, np.maximum(db, -80.0), shading="nearest", cmap="magma")
    axes[0].plot(times, centroid, color="#7fd3ff", linewidth=1.8, label="spectral centroid")
    axes[0].axhline(2_500.0, color="#ffffff", linewidth=0.8, linestyle=":", label="2.5 kHz decision line")
    axes[0].set(title="Synthetic 'asa': spectrogram with the centroid track", ylabel="Frequency (Hz)")
    axes[0].legend(loc="center right", fontsize=7)
    axes[1].plot(times, ber_db, color="#2f6b9a", linewidth=1.5)
    axes[1].axhline(0.0, color="#5b6770", linewidth=0.8)
    axes[1].set(title="Band energy ratio, below / above 2 kHz", ylabel="dB")
    axes[2].plot(times, flatness, color="#3f8f5c", linewidth=1.5)
    axes[2].set(title="Spectral flatness", ylabel="GM / AM")
    axes[3].plot(times, rms, color="#5b6770", linewidth=1.5)
    axes[3].axhline(0.01, color="#b45050", linewidth=0.8, linestyle="--", label="silence threshold")
    axes[3].set(title="Frame RMS (background: rule-based label per frame)", ylabel="RMS", xlabel="Time (s)")
    axes[3].legend(loc="upper right", fontsize=7)
    for axis in axes[1:]:
        for t, label in zip(times, labels):
            axis.axvspan(t - HOP / SAMPLE_RATE / 2, t + HOP / SAMPLE_RATE / 2, color=LABEL_COLORS[label], linewidth=0)
        axis.grid(alpha=0.25)
    for label, start, end in segments:
        axes[0].text((start + end) / 2, 7_400, label, ha="center", fontsize=8, color="#ffffff")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in LABEL_COLORS.values()]
    axes[1].legend(handles, LABEL_COLORS.keys(), loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "asa-features.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
