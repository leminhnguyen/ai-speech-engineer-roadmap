"""Create deterministic plots embedded by Module 04, Lesson 03."""

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




def jitter_figure(output_dir: Path) -> None:
    n = 3000
    lost = main.gilbert_elliott(n, 0.01, 0.4, seed=1)
    delays = main.network_delays(n, seed=2, step_at=1800, step_ms=60.0)
    t = np.arange(n) * main.PTIME_MS / 1000
    fixed = main.fixed_jitter_buffer(delays, 60, lost)
    adaptive = main.adaptive_jitter_buffer(delays, lost)

    fig, (trace, curve) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    trace.plot(t[~lost], delays[~lost], ".", color=SOFT, markersize=1.6, alpha=0.6, label="packet delay")
    trace.axhline(fixed["mouth_to_ear_ms"], color=LINK, linewidth=1.2, label="fixed 60 ms buffer deadline")
    trace.step(t, adaptive["offsets"], where="post", color=ACCENT, linewidth=1.5, label="adaptive deadline")
    late = fixed["late"]
    trace.plot(t[late], delays[late], "x", color=LINK, markersize=3, label="late for the fixed buffer")
    trace.annotate("route change +60 ms", xy=(36.3, 98), xytext=(40, 70), color=INK, fontsize=8,
                   arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    trace.set(title="Packets arriving after the deadline count as lost", xlabel="Call time (s)",
              ylabel="One-way delay (ms)", ylim=(30, 230))
    trace.legend(loc="upper left", markerscale=3)

    buffers = np.arange(0, 201, 5)
    fixed_runs = [main.fixed_jitter_buffer(delays, b, lost) for b in buffers]
    curve.plot([r["mouth_to_ear_ms"] for r in fixed_runs], [100 * r["late_rate"] for r in fixed_runs],
               color=LINK, linewidth=1.4, label="fixed buffer, best size in hindsight")
    points = [main.adaptive_jitter_buffer(delays, lost, quantile=q) for q in (0.95, 0.97, 0.98, 0.99, 0.995)]
    curve.plot([p["mouth_to_ear_ms"] for p in points], [100 * p["late_rate"] for p in points], "o-",
               color=ACCENT, linewidth=1.4, markersize=4, label="adaptive, chosen online (q 0.95-0.995)")
    curve.set(title="Online adaptation vs. the best fixed size", xlabel="Network + buffer delay (ms)",
              ylabel="Late packets (%)", yscale="log", ylim=(0.1, 100))
    curve.legend(loc="upper right")
    eyebrow(fig, "Fig 1 · Jitter buffer")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "jitter-buffer.png")


def plc_figure(output_dir: Path) -> None:
    x = main.voiced_signal()
    missing = np.zeros(50, dtype=bool)
    missing[[25, 26]] = True
    received = x * ~main.frames_to_samples(missing, 160)
    window = slice(160 * 23, 160 * 29)
    t_ms = np.arange(len(x))[window] / 8
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.2), sharex=True)
    labels = {"zero": "zero fill (silence)", "repeat": "repeat last frame", "pitch": "repeat last pitch period"}
    for axis, method in zip(axes, ("zero", "repeat", "pitch")):
        y = main.conceal(received, missing, 160, method)
        lsd = main.log_spectral_distance(x, y, missing)
        axis.axvspan(25 * 20, 27 * 20, color=ACCENT, alpha=0.06, linewidth=0)
        axis.plot(t_ms, x[window], color=SOFT, linewidth=0.9, alpha=0.7, label="original")
        axis.plot(t_ms, y[window], color=ACCENT if method == "pitch" else LINK, linewidth=1.2, label=labels[method])
        axis.set_ylim(-1, 1.15)
        axis.set_yticks([])
        axis.set_title(f"{labels[method]}: log-spectral distance {lsd:.1f} dB", fontsize=9.5, pad=4)
    axes[0].text(25 * 20 + 2, 0.85, "2 packets lost", color=ACCENT, fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    eyebrow(fig, "Fig 2 · Packet loss concealment")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, output_dir / "plc-comparison.png")


def emodel_figure(output_dir: Path) -> None:
    delays = np.arange(0, 401, 5)
    fig, axis = plt.subplots(figsize=(9, 3.6))
    cases = [("no loss", 0.0, 1.0, SOFT, "-"), ("1% random loss", 0.01, 1.0, SAGE, "-"),
             ("3% random loss", 0.03, 1.0, LINK, "-"), ("3% loss in bursts (BurstR 3)", 0.03, 3.0, ACCENT, "-")]
    for name, loss, burst, color, style in cases:
        mos = [main.mos_from_r(main.r_factor(d, loss, burst)) for d in delays]
        axis.plot(delays, mos, style, color=color, linewidth=1.8 if color == ACCENT else 1.3, label=name)
    for r, text in ((80, "R 80: satisfied"), (70, "R 70: some dissatisfied")):
        level = main.mos_from_r(r)
        axis.axhline(level, color=MUTED, linewidth=0.7, linestyle="--")
        axis.text(402, level, text, va="center", fontsize=7.5, color=MUTED)
    axis.axvline(177.3, color=MUTED, linewidth=0.7, linestyle=":")
    axis.text(182, 2.65, "delay knee 177 ms", fontsize=7.5, color=MUTED)
    axis.set(title="G.711 with PLC: estimated MOS from delay and loss (E-model)", xlabel="Mouth-to-ear delay (ms)",
             ylabel="Estimated MOS", xlim=(0, 400), ylim=(2.5, 4.5))
    axis.legend(loc="lower left")
    eyebrow(fig, "Fig 3 · E-model")
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    save(fig, output_dir / "emodel-mos.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    jitter_figure(output_dir)
    plc_figure(output_dir)
    emodel_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
