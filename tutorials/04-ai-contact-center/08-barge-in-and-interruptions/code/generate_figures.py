"""Create deterministic plots embedded by Module 04, Lesson 08."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
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


AGENT = 6.0
EXAMPLES = [
    ("Real interruption", main.OverlapEvent("interruption", 2.0, 1.6, tuple("khoan đã tôi muốn đổi địa chỉ".split()))),
    ('One word: "không"', main.OverlapEvent("interruption", 2.0, 0.35, ("không",))),
    ('Backchannel: "dạ"', main.OverlapEvent("backchannel", 2.0, 0.3, ("dạ",))),
    ("Cough", main.OverlapEvent("cough", 2.0, 0.6, ())),
    ("Own echo, no AEC", main.OverlapEvent("echo", 0.4, 1.5, tuple(main.AGENT_REPLY[:4]))),
]


def timeline_figure(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4.4))
    vad, adaptive = main.BargeInConfig(), main.BargeInConfig(mode="adaptive")
    for row, (label, event) in enumerate(EXAMPLES):
        y = len(EXAMPLES) - 1 - row
        out = main.simulate_event(event, AGENT, vad)
        color = ACCENT if event.kind == "interruption" else SAGE
        axis.add_patch(Rectangle((event.start, y + 0.12), event.duration, 0.22, color=color, linewidth=0))
        quiet = out.silenced_at if out.silenced_at is not None else AGENT
        axis.add_patch(Rectangle((0, y - 0.3), quiet, 0.3, color=SOFT, alpha=0.85, linewidth=0))
        if out.action == "resumed":
            axis.add_patch(Rectangle((out.silenced_at, y - 0.3), out.resumed_at - out.silenced_at, 0.3,
                                     facecolor="none", edgecolor=MUSTARD, hatch="////", linewidth=0.8))
            axis.add_patch(Rectangle((out.resumed_at, y - 0.3), AGENT - out.silenced_at, 0.3, color=SOFT,
                                     alpha=0.85, linewidth=0))
        elif out.action == "stopped":
            axis.add_patch(Rectangle((quiet, y - 0.3), AGENT - quiet, 0.3, facecolor="none", edgecolor=SOFT,
                                     linestyle="--", linewidth=0.8))
            axis.plot([quiet, quiet], [y - 0.38, y + 0.4], color=INK, linewidth=1.0)
        verdict = f"VAD: {out.action}  ·  adaptive: {main.simulate_event(event, AGENT, adaptive).action}"
        axis.text(-0.15, y, label, ha="right", va="center", fontsize=8.5)
        axis.text(1.01, (y + 0.7) / (len(EXAMPLES) + 0.4), verdict, ha="left", va="center", fontsize=7.5, color=MUTED,
                  family="monospace", transform=axis.transAxes)
    axis.set(xlim=(0, 8.4), ylim=(-0.7, len(EXAMPLES) - 0.3), xlabel="Time since the agent started speaking (s)",
             title="Same controller, five kinds of overlap (VAD mode, LiveKit defaults)")
    axis.set_yticks([])
    axis.set_xticks(range(0, 9))
    axis.spines["left"].set_visible(False)
    axis.grid(axis="y", visible=False)
    handles = [Rectangle((0, 0), 1, 1, color=SOFT, alpha=0.85), Rectangle((0, 0), 1, 1, facecolor="none",
               edgecolor=MUSTARD, hatch="////"), Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=SOFT,
               linestyle="--"), Rectangle((0, 0), 1, 1, color=ACCENT), Rectangle((0, 0), 1, 1, color=SAGE)]
    axis.legend(handles, ["agent audio", "paused", "never played", "caller interrupts", "other overlap"],
                loc="lower center", bbox_to_anchor=(0.55, -0.32), ncol=5)
    eyebrow(fig, "Fig 1 · Overlap timeline")
    fig.subplots_adjust(left=0.17, right=0.70, top=0.86, bottom=0.22)
    save(fig, output_dir / "overlap-timeline.png")


def sweep_figure(output_dir: Path) -> None:
    events = main.make_events(400, seed=0)
    values = np.round(np.arange(0.1, 1.31, 0.1), 2)
    vad = main.sweep_min_duration(events, values)
    adaptive = main.sweep_min_duration(events, values, mode="adaptive")
    fig, (rates, latency) = plt.subplots(1, 2, figsize=(9, 3.6))
    rates.plot(values, 100 * vad["false_stop_rate"], color=ACCENT, linewidth=1.8, label="false stops, VAD mode")
    rates.plot(values, 100 * vad["false_pause_rate"], color=MUSTARD, linewidth=1.4, label="false pauses, VAD mode")
    rates.plot(values, 100 * adaptive["false_stop_rate"], color=SAGE, linewidth=1.4, linestyle="--",
               label="false stops, adaptive")
    rates.plot(values, 100 * vad["missed_rate"], color=LINK, linewidth=1.2, label="missed interruptions")
    rates.set(title="Raising min_duration cannot fix backchannels", xlabel="min_duration (s)",
              ylabel="% of overlaps", ylim=(-2, 78))
    rates.legend(loc="upper right")
    latency.plot(values, vad["median_stop_latency"], color=LINK, linewidth=1.6, label="median")
    latency.plot(values, vad["p90_stop_latency"], color=LINK, linewidth=1.2, linestyle="--", label="90th percentile")
    latency.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    latency.set(title="…but it makes callers talk over the agent longer", xlabel="min_duration (s)",
                ylabel="Caller speech before agent goes quiet (s)", ylim=(0, 2.0))
    latency.legend(loc="upper left")
    eyebrow(fig, "Fig 2 · Parameter sweep")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "min-duration-sweep.png")


def truncation_figure(output_dir: Path) -> None:
    words = main.AGENT_REPLY
    timings = main.word_timings(words, AGENT)
    cut_time = 2.4
    heard, _ = main.truncate_to_heard(words, timings, cut_time)
    n_heard = len(heard.split())
    fig, axis = plt.subplots(figsize=(9, 2.8))
    for i, (word, (start, end)) in enumerate(zip(words, timings)):
        row = i % 2
        y = 0.55 if row == 0 else 0.0
        if i < n_heard:
            axis.add_patch(Rectangle((start, y), end - start - 0.02, 0.4, facecolor=(191 / 255, 69 / 255, 32 / 255, 0.10),
                                     edgecolor=ACCENT, linewidth=0.8))
            color = INK
        else:
            axis.add_patch(Rectangle((start, y), end - start - 0.02, 0.4, facecolor="none", edgecolor=SOFT,
                                     linestyle="--", linewidth=0.7))
            color = SOFT
        axis.text((start + end) / 2, y + 0.2, word, ha="center", va="center", fontsize=7, color=color)
    axis.axvline(cut_time, color=ACCENT, linewidth=1.4)
    axis.text(cut_time + 0.05, 1.08, "caller cuts in at 2.4 s", color=ACCENT, fontsize=8)
    axis.text(0.0, -0.32, "keep in chat history", color=ACCENT, fontsize=8)
    axis.text(cut_time + 0.05, -0.32, "generated but never heard: drop it", color=SOFT, fontsize=8)
    axis.set(xlim=(0, AGENT), ylim=(-0.45, 1.25), xlabel="Playout time (s)",
             title="The chat history must end where the caller stopped hearing")
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.grid(False)
    eyebrow(fig, "Fig 3 · Truncating to what was heard")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, output_dir / "heard-truncation.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    timeline_figure(output_dir)
    sweep_figure(output_dir)
    truncation_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
