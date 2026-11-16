"""Create deterministic plots embedded by Module 04, Lesson 07."""

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


def pauses_figure(output_dir: Path, calls: list) -> None:
    fig, axis = plt.subplots(figsize=(9, 3.6))
    bins = np.arange(0, 2.05, 0.05)
    hesitation = main.within_turn_pauses(calls, "hesitation")
    number = main.within_turn_pauses(calls, "number")
    axis.hist(hesitation, bins=bins, color=SOFT, alpha=0.75, label="hesitation inside a request")
    axis.hist(number, bins=bins, color=ACCENT, alpha=0.75, label="between groups of digits")
    for x, label in ((0.5, "0.5 s"), (1.0, "1.0 s")):
        axis.axvline(x, color=INK, linewidth=0.8, linestyle="--")
        share = 100 * np.mean(np.concatenate([hesitation, number]) > x)
        axis.text(x + 0.03, axis.get_ylim()[1] * 0.9, f"{label} timer:\n{share:.0f}% of pauses\nlonger", fontsize=8,
                  color=INK, va="top")
    axis.set(title="Callers pause inside their turns, longest when reading numbers",
             xlabel="Pause inside a turn (s)", ylabel="Pauses", xlim=(0, 2))
    axis.legend(loc="upper right")
    eyebrow(fig, "Fig 1 · Pauses inside turns")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "pause-distributions.png")


def tradeoff_figure(output_dir: Path, calls: list) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.2))
    fixed = [main.run_policy(main.FixedEndpointing(d), calls) for d in np.arange(0.2, 2.05, 0.1)]
    dynamic = [main.run_policy(main.DynamicEndpointing(0.3, 3.0, margin=m), calls) for m in np.arange(0.8, 4.05, 0.2)]
    semantic = [main.run_policy(main.SemanticEndpointing(d, 3.0), calls) for d in np.arange(0.2, 1.25, 0.1)]
    for rows, color, width, label in ((fixed, SOFT, 1.2, "fixed silence timer (0.2 to 2.0 s)"),
                                      (dynamic, LINK, 1.4, "dynamic: 2 x average pause, 0.3 to 3.0 s"),
                                      (semantic, ACCENT, 1.8, "semantic: min_delay 0.2 to 1.2 s, max_delay 3.0 s")):
        axis.plot([r["mean_wait_s"] for r in rows], [100 * r["cutoff_rate"] for r in rows], color=color,
                  linewidth=width, marker="o", markersize=3, label=label)
    default = main.run_policy(main.FixedEndpointing(0.5), calls)
    axis.scatter([default["mean_wait_s"]], [100 * default["cutoff_rate"]], s=60, facecolor="none",
                 edgecolor=INK, linewidth=1.2, zorder=5)
    axis.annotate("a 0.5 s timer cuts off\nmore than one turn in four", xy=(default["mean_wait_s"], 100 * default["cutoff_rate"]),
                  xytext=(0.75, 40), fontsize=8, color=INK,
                  arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    axis.set(title="Every endpointing policy trades cut-offs for waiting; knowing the words moves the curve",
             xlabel="Average silence the caller waits after finishing (s)", ylabel="Turns cut off (%)",
             xlim=(0, 2.1), ylim=(0, 60))
    axis.legend(loc="upper right")
    eyebrow(fig, "Fig 2 · The trade-off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "endpointing-tradeoff.png")


def trace_figure(output_dir: Path) -> None:
    call = main.simulate_call(n_turns=14, seed=7, tempo=1.8)
    dynamic, semantic = main.DynamicEndpointing(0.5, 3.0), main.SemanticEndpointing(0.5, 3.0)
    pauses, dyn_delays, sem_delays = [], [], []
    for turn in call:
        for pause in turn:
            pauses.append(pause)
            dyn_delays.append(dynamic.delay_for(pause))
            sem_delays.append(semantic.delay_for(pause))
            dynamic.observe(pause)
    x = np.arange(len(pauses))
    lengths = np.array([p.length_s if not p.final else np.nan for p in pauses])
    fig, axis = plt.subplots(figsize=(9, 3.8))
    axis.bar(x[~np.isnan(lengths)], lengths[~np.isnan(lengths)], color=SOFT, alpha=0.6, width=0.7,
             label="pause inside a turn (caller continues)")
    finals = np.array([p.final for p in pauses])
    axis.scatter(x[finals], np.full(finals.sum(), 0.05), marker="|", s=120, color=INK, label="end of turn")
    axis.step(x, dyn_delays, where="mid", color=LINK, linewidth=1.4, label="dynamic timer")
    axis.plot(x, sem_delays, color=ACCENT, linewidth=0, marker="o", markersize=4, label="semantic timer")
    axis.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    axis.set(title="A slow caller: the dynamic timer learns their pauses, the semantic timer reads their words",
             xlabel="Silences in the call, in order", ylabel="Seconds", ylim=(0, 3.9))
    axis.legend(loc="upper left", ncol=4, fontsize=7.5)
    eyebrow(fig, "Fig 3 · Two adaptive timers")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "delay-trace.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    calls = main.simulate_calls()
    pauses_figure(output_dir, calls)
    tradeoff_figure(output_dir, calls)
    trace_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
