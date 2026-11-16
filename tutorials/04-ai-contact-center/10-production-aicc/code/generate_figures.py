"""Create deterministic plots embedded by Module 04, Lesson 10."""

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

HUMAN_AHT, AI_AHT, PEAK_CALLS = 300.0, 160.0, 1200


def erlang_figure(output_dir: Path) -> None:
    fig, (sl, asa) = plt.subplots(1, 2, figsize=(9, 3.6))
    loads = [(150, SOFT), (300, ACCENT), (450, LINK)]
    agents = np.arange(10, 56)
    for calls, color in loads:
        traffic = main.offered_load(calls, HUMAN_AHT)
        levels = [100 * main.service_level(traffic, int(m), HUMAN_AHT, 20) for m in agents]
        waits = [main.average_speed_of_answer(traffic, int(m), HUMAN_AHT) for m in agents]
        sl.plot(agents, levels, color=color, linewidth=1.8 if color == ACCENT else 1.3,
                label=f"{calls} transfers/h ({traffic:.1f} Erl)")
        asa.plot(agents, np.where(np.isfinite(waits), waits, np.nan), color=color,
                 linewidth=1.8 if color == ACCENT else 1.3)
        need = main.agents_for_service_level(calls, HUMAN_AHT)
        sl.plot([need], [100 * main.service_level(traffic, need, HUMAN_AHT, 20)], "o", color=color, markersize=4)
    sl.axhline(80, color=MUTED, linewidth=0.8, linestyle="--")
    sl.text(48, 73, "80% in 20 s", color=MUTED, fontsize=8)
    sl.set(title="Service level climbs steeply, then saturates", xlabel="Human agents on shift",
           ylabel="Answered within 20 s (%)", ylim=(0, 102))
    sl.legend(loc="lower right")
    asa.set(title="Average wait explodes near capacity", xlabel="Human agents on shift",
            ylabel="Average speed of answer (s)", ylim=(0, 120))
    eyebrow(fig, "Fig 1 · Erlang C")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "erlang-c-staffing.png")


def staffing_figure(output_dir: Path) -> None:
    profile = main.hourly_profile()
    calls = PEAK_CALLS * profile
    transfer_rate = main.compute_kpis(main.generate_call_log())["transfer_rate"]
    sessions = [main.channels_for_blocking(main.offered_load(c, AI_AHT)) for c in calls]
    humans = [main.agents_for_service_level(c * transfer_rate, HUMAN_AHT) if c * transfer_rate >= 1 else 0
              for c in calls]
    hours = np.arange(24)
    fig, axis = plt.subplots(figsize=(9, 3.6))
    axis.bar(hours, sessions, color=SOFT, alpha=0.35, width=0.8, label="concurrent AI sessions (1% blocking)")
    axis.step(hours, humans, where="mid", color=ACCENT, linewidth=1.8, label="human agents for transfers (80/20)")
    peak = int(np.argmax(sessions))
    axis.text(peak, sessions[peak] + 2, f"{sessions[peak]}", ha="center", fontsize=8, color=MUTED)
    axis.text(peak + 0.45, humans[peak] + 2, f"{humans[peak]}", ha="left", fontsize=8, color=ACCENT)
    axis.set(title=f"One day at {PEAK_CALLS} calls in the peak hour, {transfer_rate:.0%} transferred",
             xlabel="Hour of day", ylabel="Concurrent capacity", xlim=(-0.6, 23.6), ylim=(0, max(sessions) * 1.2))
    axis.set_xticks(range(0, 24, 2))
    axis.legend(loc="upper left")
    axis.grid(axis="x", visible=False)
    eyebrow(fig, "Fig 2 · Capacity by hour")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "capacity-by-hour.png")


def kpi_figure(output_dir: Path) -> None:
    kpis = main.compute_kpis(main.generate_call_log())
    fig, (outcomes, reasons) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 1.15]})
    labels = ["resolved by AI", "transferred", "abandoned", "dropped"]
    values = [kpis["containment_rate"], kpis["transfer_rate"], kpis["abandonment_rate"], kpis["drop_rate"]]
    colors = [ACCENT, LINK, MUSTARD, SOFT]
    y = np.arange(len(labels))[::-1]
    outcomes.barh(y, [100 * v for v in values], color=colors, height=0.6)
    for yy, v in zip(y, values):
        outcomes.text(100 * v + 1, yy, f"{v:.0%}", va="center", fontsize=8, color=MUTED)
    outcomes.set_yticks(y)
    outcomes.set_yticklabels(labels, fontsize=8.5)
    outcomes.set(title=f"Outcomes of {kpis['calls']} calls", xlabel="% of offered calls", xlim=(0, 80))
    outcomes.grid(axis="y", visible=False)
    names = list(kpis["transfer_reasons"])
    counts = [kpis["transfer_reasons"][n] for n in names]
    y2 = np.arange(len(names))[::-1]
    reasons.barh(y2, counts, color=SOFT, height=0.6)
    for yy, c in zip(y2, counts):
        reasons.text(c + 3, yy, str(c), va="center", fontsize=8, color=MUTED)
    reasons.set_yticks(y2)
    reasons.set_yticklabels(names, fontsize=8.5)
    reasons.set(title="Why calls reached a human", xlabel="Transferred calls", xlim=(0, max(counts) * 1.2))
    reasons.grid(axis="y", visible=False)
    eyebrow(fig, "Fig 3 · KPIs from the call log")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "kpi-summary.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    erlang_figure(output_dir)
    staffing_figure(output_dir)
    kpi_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
