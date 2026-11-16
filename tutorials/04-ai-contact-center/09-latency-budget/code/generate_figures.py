"""Create deterministic plots embedded by Module 04, Lesson 09."""

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


STAGE_COLORS = {"uplink": "#9aa3b5", "jitter buffer": "#b7bdca", "end of turn": MUSTARD,
                "llm ttft": LINK, "tts ttfb": SAGE, "downlink": "#9aa3b5"}
STAGE_LABELS = {"uplink": "uplink", "jitter buffer": "jitter", "end of turn": "end of turn",
                "llm ttft": "LLM", "tts ttfb": "TTS", "downlink": "downlink"}


def budget_figure(output_dir: Path) -> None:
    ladder = main.optimization_ladder()
    rows = [ladder[0], ladder[-1]]
    fig, axis = plt.subplots(figsize=(9, 3.2))
    for y, (name, cfg) in zip((1, 0), rows):
        result = main.simulate_turns(cfg)
        left = 0.0
        for stage in main.STAGE_ORDER:
            width = float(np.median(result[stage]))
            axis.barh(y, width, left=left, height=0.5, color=STAGE_COLORS[stage], edgecolor=PAPER, linewidth=1)
            if width > 90:
                axis.text(left + width / 2, y, f"{STAGE_LABELS[stage]}\n{width:.0f}", ha="center", va="center",
                          fontsize=7, color=INK if stage in ("uplink", "jitter buffer", "downlink") else "white")
            left += width
        axis.text(-30, y, name.lstrip("+ ").capitalize() if y == 1 else "Optimized", ha="right", va="center",
                  fontsize=9)
    axis.axvline(1000, color=ACCENT, linewidth=1.4)
    axis.text(1010, 1.42, "1 s target", color=ACCENT, fontsize=8)
    axis.text(1850, 1.0, "grey: network legs\nand jitter buffer", color=MUTED, fontsize=7.5, va="center")
    axis.set(xlim=(0, 2100), ylim=(-0.5, 1.6), xlabel="Milliseconds after the caller stops speaking (median of each stage)",
             title="Where one turn's time goes on a phone line")
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.grid(axis="y", visible=False)
    eyebrow(fig, "Fig 1 · Latency budget")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.18)
    save(fig, output_dir / "latency-budget.png")


def cdf_figure(output_dir: Path) -> None:
    ladder = main.optimization_ladder()
    greys = ["#b7bdca", "#9aa3b5", SOFT, LINK]
    fig, axis = plt.subplots(figsize=(9, 3.8))
    for i, (name, cfg) in enumerate(ladder):
        total = np.sort(main.simulate_turns(cfg)["total"])
        p = np.arange(1, len(total) + 1) / len(total)
        s = main.summarize(total)
        color = ACCENT if i == len(ladder) - 1 else greys[i]
        axis.plot(total, p, color=color, linewidth=1.8 if i == len(ladder) - 1 else 1.3,
                  label=f"{name}  (p50 {s['p50'] / 1000:.2f} s, p95 {s['p95'] / 1000:.2f} s)")
    axis.axvline(1000, color=MUTED, linewidth=0.8, linestyle="--")
    axis.axvline(1500, color=MUTED, linewidth=0.8, linestyle=":")
    axis.axhline(0.5, color=SOFT, linewidth=0.6, linestyle=":")
    axis.axhline(0.95, color=SOFT, linewidth=0.6, linestyle=":")
    axis.text(3480, 0.51, "p50", color=MUTED, fontsize=7.5, ha="right", va="bottom")
    axis.text(3480, 0.96, "p95", color=MUTED, fontsize=7.5, ha="right", va="bottom")
    axis.text(1010, 0.03, "1.0 s", color=MUTED, fontsize=8)
    axis.text(1510, 0.03, "1.5 s", color=MUTED, fontsize=8)
    axis.set(xlim=(300, 3500), ylim=(0, 1.0), xlabel="End of caller speech to first agent audio (ms)",
             ylabel="Share of turns", title="Each optimization shifts the whole distribution, not just the median")
    axis.legend(loc="lower right")
    eyebrow(fig, "Fig 2 · Cumulative distribution")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "latency-cdf.png")


def streaming_figure(output_dir: Path) -> None:
    reply = ("Dạ, đơn hàng của anh sẽ được giao vào thứ năm. Phí giao hàng là ba mươi nghìn đồng, "
             "anh có muốn đổi địa chỉ không?")
    fig, (bars, sweep) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
    combos = [(False, False), (False, True), (True, False), (True, True)]
    labels = ["neither", "TTS only", "LLM only", "LLM + TTS"]
    values = [main.first_audio_ms(reply, llm, tts) for llm, tts in combos]
    colors = [SOFT, SOFT, SOFT, ACCENT]
    bars.bar(labels, values, color=colors, width=0.6)
    for x, v in enumerate(values):
        bars.text(x, v + 40, f"{v:.0f}", ha="center", fontsize=8, color=MUTED)
    bars.set(title="Stream both stages", ylabel="Request to first audio (ms)", ylim=(0, 3100))
    bars.grid(axis="x", visible=False)

    base_cfg = main.optimization_ladder()[1][1]
    base = np.median(main.simulate_turns(base_cfg)["total"])
    probabilities = np.linspace(0, 0.8, 9)
    medians = []
    for p in probabilities:
        cfg = main.PipelineConfig(**{**base_cfg.__dict__, "preemptive": True, "p_transcript_change": float(p)})
        medians.append(np.median(main.simulate_turns(cfg)["total"]))
    sweep.axhline(base, color=SOFT, linewidth=1.2, linestyle="--", label="without preemptive generation")
    sweep.plot(100 * probabilities, medians, color=LINK, linewidth=1.6, marker="o", markersize=3,
               label="with preemptive generation")
    sweep.set(title="Preemption pays while transcripts are stable", xlabel="Turns whose transcript changes (%)",
              ylabel="Median latency (ms)", ylim=(1300, 1850))
    sweep.text(0, 1820, "every changed transcript is a wasted LLM call", color=MUTED, fontsize=7.5, va="top")
    sweep.legend(loc="lower right")
    eyebrow(fig, "Fig 3 · Streaming and preemption")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "streaming-preemption.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    budget_figure(output_dir)
    cdf_figure(output_dir)
    streaming_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
