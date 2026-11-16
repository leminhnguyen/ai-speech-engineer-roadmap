"""Create deterministic plots embedded by Module 04, Lesson 05."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import main

# Editorial palette shared by every Module 04 figure (paper, ink, one accent, link blue).
PAPER, INK, MUTED, SOFT = "#f5f5f5", "#2d3142", "#4f5d75", "#5a6580"
ACCENT, LINK, SAGE, MUSTARD = "#bf4520", "#2e5aa8", "#7c8f6f", "#b8915a"
RULE = (45 / 255, 49 / 255, 66 / 255, 0.12)


def first_available(families):
    for family in families:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": SOFT, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 9, "legend.frameon": False, "legend.fontsize": 8,
        "font.family": first_available(["Geist", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]),
    })


def eyebrow(fig, text):
    fig.text(0.012, 0.975, text.upper(), color=ACCENT, fontsize=7.5, family="monospace", va="top")


def save(fig, path):
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


SR = main.SAMPLE_RATE
FRAME = 800  # 100 ms ERLE frames


def frame_times(n_frames, frame=FRAME):
    return (np.arange(n_frames) + 0.5) * frame / SR


def plot_erle(axis, erle, **style):
    times = frame_times(len(erle))
    valid = np.isfinite(erle)  # NaN where the far end was silent: nothing to cancel
    axis.plot(times[valid], erle[valid], marker="o", markersize=2.5, **style)


def convergence_figure(output_dir: Path) -> None:
    s = main.scenario(seconds=6.0, path_change_at=3.0)
    residual, _ = main.nlms_echo_canceller(s["far"], s["mic"])
    erle = main.erle_db(s["echo"], residual - s["noise"], frame=FRAME)
    t = np.arange(len(s["mic"])) / SR
    fig, (wave, curve) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    wave.plot(t, s["mic"], color=SOFT, linewidth=0.5, label="microphone: echo of the agent's TTS")
    wave.plot(t, residual, color=ACCENT, linewidth=0.5, label="after the NLMS canceller")
    wave.set(title="The filter learns the echo path, loses it when the path changes, and learns it again",
             ylabel="Amplitude", ylim=(-0.12, 0.12))
    wave.legend(loc="upper right", ncol=2)
    plot_erle(curve, erle, color=ACCENT, linewidth=1.6)
    for axis in (wave, curve):
        axis.axvline(3.0, color=MUTED, linewidth=0.8, linestyle="--")
    curve.text(3.08, 30, "echo path changes\n(handset moved)", color=MUTED, fontsize=8, va="center")
    curve.set(xlabel="Time (s)", ylabel="ERLE (dB)", ylim=(0, 36), xlim=(0, 6))
    eyebrow(fig, "Fig 1 · NLMS convergence")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "erle-convergence.png")


def double_talk_figure(output_dir: Path) -> None:
    s = main.scenario(seconds=4.0, double_talk=(2.0, 3.0))
    flag = main.geigel_double_talk(s["far"], s["mic"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    t = np.arange(len(s["mic"])) / SR
    top.plot(t, s["echo"], color=SOFT, linewidth=0.5, label="echo (agent)")
    top.plot(t, s["near"], color=LINK, linewidth=0.5, label="caller speech")
    top.fill_between(t, -0.3, -0.3 + 0.06 * flag, color=ACCENT, alpha=0.5, linewidth=0, step="mid",
                     label="Geigel: double talk")
    top.set(title="When the caller talks over the agent, adaptation must stop", ylabel="Amplitude", ylim=(-0.32, 0.3))
    top.legend(loc="upper left", ncol=3)
    for freeze, color, label, width in ((None, SOFT, "adapt always", 1.2), (flag, ACCENT, "freeze on double talk", 1.8)):
        residual, _ = main.nlms_echo_canceller(s["far"], s["mic"], freeze=freeze)
        erle = main.erle_db(s["echo"], residual - s["near"] - s["noise"], frame=FRAME)
        plot_erle(bottom, erle, color=color, linewidth=width, label=label)
    for axis in (top, bottom):
        axis.axvspan(2.0, 3.0, color=LINK, alpha=0.06, linewidth=0)
    bottom.set(xlabel="Time (s)", ylabel="ERLE (dB)", ylim=(-25, 36), xlim=(0, 4))
    bottom.axhline(0, color=MUTED, linewidth=0.8)
    bottom.legend(loc="lower left")
    eyebrow(fig, "Fig 2 · Double talk")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "double-talk.png")


def barge_in_figure(output_dir: Path) -> None:
    s = main.scenario(seconds=4.0, double_talk=(2.6, 3.4))
    flag = main.geigel_double_talk(s["far"], s["mic"])
    residual, _ = main.nlms_echo_canceller(s["far"], s["mic"], freeze=flag)
    suppressed = main.residual_echo_suppressor(residual, s["far"], flag)
    frame = 160
    fig, axis = plt.subplots(figsize=(9, 3.6))
    threshold = -40.0
    for signal, color, label, width in ((s["mic"], SOFT, "no echo canceller", 1.0),
                                        (residual, LINK, "NLMS canceller", 1.0),
                                        (suppressed, ACCENT, "NLMS + residual suppressor", 1.6)):
        level = np.maximum(10 * np.log10(main.frame_power(signal, frame) + 1e-12), -78)
        times = frame_times(len(level), frame)
        quiet_caller = times < 2.6
        false_hits = int(np.sum(level[quiet_caller] > threshold))
        axis.plot(times, level, color=color, linewidth=width, label=f"{label}: {false_hits} false speech frames")
    axis.axhline(threshold, color=INK, linewidth=0.8, linestyle="--")
    axis.text(0.05, threshold + 1.5, "energy VAD threshold", color=INK, fontsize=8)
    axis.axvspan(2.6, 3.4, color=LINK, alpha=0.06, linewidth=0)
    axis.text(3.0, -12, "caller\nspeaks", color=MUTED, fontsize=8, ha="center")
    axis.set(title="Echo the canceller misses becomes a false barge-in", xlabel="Time (s)",
             ylabel="Frame level (dBFS)", ylim=(-80, 0), xlim=(0, 4))
    axis.legend(loc="upper left", fontsize=7.5)
    eyebrow(fig, "Fig 3 · Why agents care")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "false-barge-in.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    convergence_figure(output_dir)
    double_talk_figure(output_dir)
    barge_in_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
