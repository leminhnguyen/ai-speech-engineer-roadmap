"""Create deterministic plots embedded by Lesson 08."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    autocorrelation,
    constant_pitch,
    cumulative_mean_normalized_difference,
    difference_function,
    frame_signal,
    harmonic_signal,
    intonation_utterance,
    yin,
)

SAMPLE_RATE, FRAME, HOP = 16_000, 640, 160
TAU_MIN, TAU_MAX = SAMPLE_RATE // 400, int(np.ceil(SAMPLE_RATE / 60))


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME) / FRAME)

    # Why the loudest peak is not the pitch: a vowel and a missing-fundamental tone.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    cases = [
        ("Vowel /a/ at 150 Hz", harmonic_signal(constant_pitch(150.0, SAMPLE_RATE, 0.3), SAMPLE_RATE), 150.0),
        ("Harmonics 2-10 of 200 Hz (no 200 Hz energy)",
         harmonic_signal(constant_pitch(200.0, SAMPLE_RATE, 0.3), SAMPLE_RATE, harmonics=tuple(range(2, 11))), 200.0),
    ]
    frequencies = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    for axis, (title, samples, f0) in zip(axes, cases):
        magnitude = np.abs(np.fft.rfft(frame_signal(samples, FRAME, HOP)[15] * hann))
        db = 20.0 * np.log10(np.maximum(magnitude / magnitude.max(), 1e-5))
        axis.plot(frequencies, db, color="#5b6770", linewidth=0.9)
        peak = frequencies[np.argmax(magnitude)]
        axis.axvline(peak, color="#b45050", linewidth=1.5, linestyle="--", label=f"loudest peak {peak:.0f} Hz")
        axis.axvline(f0, color="#2f6b9a", linewidth=1.8, label=f"pitch (YIN) {np.nanmedian(yin(samples, SAMPLE_RATE)[0]):.0f} Hz")
        for k in range(1, 11):
            axis.axvline(k * f0, color="#2f6b9a", linewidth=0.5, alpha=0.3)
        axis.set(title=title, xlabel="Frequency (Hz)")
        axis.set_xlim(0, 2_200)
        axis.set_ylim(-80, 5)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("Magnitude (dB re. peak)")
    fig.suptitle("Pitch is the spacing of the harmonics, not the loudest one", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "harmonics-and-pitch.png", dpi=150)
    plt.close(fig)

    # One frame through autocorrelation and YIN, at 300 Hz where d' dips at T, 2T, and 3T.
    voice = harmonic_signal(constant_pitch(300.0, SAMPLE_RATE, 0.3), SAMPLE_RATE)
    frames = frame_signal(voice, FRAME, HOP)[15:16]
    r = autocorrelation(frames, TAU_MAX)[0]
    normalized = cumulative_mean_normalized_difference(difference_function(frames, TAU_MAX))[0]
    lags = np.arange(TAU_MAX + 1)
    yin_tau = SAMPLE_RATE / yin(voice, SAMPLE_RATE)[0][15]
    deepest = TAU_MIN + int(np.argmin(normalized[TAU_MIN : TAU_MAX + 1]))

    fig, axes = plt.subplots(3, 1, figsize=(8, 7.2))
    time_ms = 1_000.0 * np.arange(FRAME) / SAMPLE_RATE
    axes[0].plot(time_ms, frames[0], color="#5b6770", linewidth=0.9)
    period_ms = 1_000.0 / 300.0
    for k in range(4):
        axes[0].axvline(10.0 + k * period_ms, color="#2f6b9a", linewidth=0.8, linestyle=":")
    axes[0].set(title="One 40 ms frame of a 300 Hz voice (dotted lines one period apart)", xlabel="Time (ms)", ylabel="Amplitude")
    axes[1].plot(lags, r / r[0], color="#5b6770", linewidth=1.2)
    axes[1].axvspan(0, TAU_MIN, color="#d9dee4")
    axes[1].axvline(SAMPLE_RATE / 300.0, color="#2f6b9a", linewidth=1.5, label="true period T = 53.3 samples")
    axes[1].set(title="Autocorrelation r(τ) / r(0): peaks at T, 2T, 3T, shrinking as the overlap shrinks",
                xlabel="Lag τ (samples)", ylabel="r(τ) / r(0)")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[2].plot(lags[1:], normalized[1:], color="#3f8f5c", linewidth=1.4)
    axes[2].axvspan(0, TAU_MIN, color="#d9dee4", label="outside 60-400 Hz")
    axes[2].axhline(0.1, color="#b45050", linewidth=1.0, linestyle="--", label="threshold 0.1")
    axes[2].plot([yin_tau], [normalized[round(yin_tau)]], "o", color="#2f6b9a", markersize=7,
                 label=f"YIN: first dip below threshold → {SAMPLE_RATE / yin_tau:.0f} Hz")
    axes[2].plot([deepest], [normalized[deepest]], "x", color="#b45050", markersize=9, markeredgewidth=2,
                 label=f"deepest dip → {SAMPLE_RATE / deepest:.0f} Hz")
    axes[2].set(title="Cumulative mean normalised difference d'(τ)", xlabel="Lag τ (samples)", ylabel="d'(τ)")
    axes[2].set_ylim(-0.05, 1.8)
    axes[2].legend(loc="upper right", fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    for axis in axes[1:]:
        axis.set_xlim(0, TAU_MAX)
    fig.tight_layout()
    fig.savefig(output_dir / "autocorrelation-and-yin.png", dpi=150)
    plt.close(fig)

    # A pitch track over the intonation utterance, with two frame lengths.
    utterance, truth = intonation_utterance(SAMPLE_RATE)
    spectrum_frames = frame_signal(utterance, 1_024, HOP) * (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1_024) / 1_024))
    magnitude = np.abs(np.fft.rfft(spectrum_frames, axis=1)).T
    times = np.arange(magnitude.shape[1]) * HOP / SAMPLE_RATE
    spec_frequencies = np.fft.rfftfreq(1_024, 1.0 / SAMPLE_RATE)
    db = 20.0 * np.log10(np.maximum(magnitude / magnitude.max(), 1e-4))
    sample_times = np.arange(len(truth)) / SAMPLE_RATE

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    axes[0].pcolormesh(times, spec_frequencies, db, shading="nearest", cmap="magma")
    axes[0].plot(sample_times, truth, color="#ffffff", linewidth=3.0, alpha=0.6, label="true F0")
    colors = {640: "#7fd3ff", 1_024: "#f3c9a6"}
    for length in (640, 1_024):
        f0, _, aperiodicity = yin(utterance, SAMPLE_RATE, frame_length=length)
        axes[0].plot(times, f0, "o", color=colors[length], markersize=2.5,
                     label=f"YIN, {1_000 * length / SAMPLE_RATE:.0f} ms frames")
        axes[1].plot(times, aperiodicity, color=colors[length], linewidth=1.5,
                     label=f"{1_000 * length / SAMPLE_RATE:.0f} ms frames")
    axes[0].set(title="Rising glide, voiceless hiss, falling glide, silence", ylabel="Frequency (Hz)")
    axes[0].set_ylim(0, 1_000)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[1].axhline(0.1, color="#b45050", linewidth=1.0, linestyle="--", label="voicing threshold 0.1")
    axes[1].set(title="Aperiodicity: d' at the chosen dip (frames below the line are voiced)",
                xlabel="Time (s)", ylabel="d'")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper left", fontsize=7)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pitch-track.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
