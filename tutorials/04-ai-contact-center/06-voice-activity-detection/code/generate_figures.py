"""Create deterministic plots embedded by Module 04, Lesson 06."""

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


def noisy_dialog(kind, snr, dialog):
    return main.mix_at_snr(dialog["signal"], main.noise(kind, len(dialog["signal"])), snr, dialog["speech"])


def shade_utterances(axis, utterances, limit):
    for start, end in utterances:
        if start < limit:
            axis.axvspan(start, min(end, limit), color=LINK, alpha=0.07, linewidth=0)


def decisions_figure(output_dir: Path, dialog: dict) -> None:
    limit = 14.0
    x = noisy_dialog("babble", 10, dialog)
    params = main.VADParams()
    probs = main.speech_probability(x)
    result = main.run_vad(probs, params)
    t = np.arange(len(x)) / SR
    frame_t = (np.arange(len(probs)) + 0.5) * params.frame_duration
    fig, (wave, prob, seg) = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1.2, 0.55]})
    keep = t < limit
    wave.plot(t[keep], x[keep], color=SOFT, linewidth=0.4)
    shade_utterances(wave, dialog["utterances"], limit)
    wave.set(title="Caller speech in 10 dB babble: probability, thresholds, and the segments that come out",
             ylabel="Amplitude")
    keep_f = frame_t < limit
    prob.plot(frame_t[keep_f], probs[keep_f], color=INK, linewidth=0.9, label="speech probability")
    prob.axhline(params.activation_threshold, color=ACCENT, linewidth=1.0, linestyle="--")
    prob.axhline(params.deactivation(), color=MUTED, linewidth=1.0, linestyle=":")
    prob.text(0.05, params.activation_threshold + 0.03, "activation 0.50", color=ACCENT, fontsize=7.5,
              bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1}, zorder=5)
    prob.text(0.05, params.deactivation() + 0.03, "deactivation 0.35", color=MUTED, fontsize=7.5,
              bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1}, zorder=5)
    shade_utterances(prob, dialog["utterances"], limit)
    prob.set(ylabel="p(speech)", ylim=(-0.02, 1.05))
    for start, end in dialog["utterances"]:
        if start < limit:
            seg.broken_barh([(start, min(end, limit) - start)], (0.55, 0.3), color=LINK, alpha=0.45)
    for start, end in result.segments:
        if start < limit:
            seg.broken_barh([(start, min(end, limit) - start)], (0.1, 0.3), color=ACCENT)
    for time, kind in result.events:
        if time < limit:
            seg.plot([time], [0.25], marker="v" if kind == "END_OF_SPEECH" else "^", color=INK, markersize=4)
    seg.set(yticks=[0.25, 0.7], yticklabels=["VAD", "truth"], ylim=(0, 1), xlabel="Time (s)", xlim=(0, limit))
    seg.grid(axis="y", visible=False)
    eyebrow(fig, "Fig 1 · VAD decisions")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "vad-decisions.png")


def noise_sweep_figure(output_dir: Path, dialog: dict) -> None:
    snrs = np.arange(0, 21, 2.5)
    fig, (f1_axis, fa_axis) = plt.subplots(1, 2, figsize=(9, 3.6))
    styles = {"white": (SOFT, 1.2, "--", "line hiss (white)"), "hum": (LINK, 1.2, "-", "fan + tone (hum)"),
              "babble": (ACCENT, 1.8, "-", "other talkers (babble)")}
    for kind, (color, width, dash, label) in styles.items():
        rows = [main.evaluate(noisy_dialog(kind, snr, dialog), dialog) for snr in snrs]
        style = dict(color=color, linewidth=width, linestyle=dash, marker="o", markersize=3, label=label)
        f1_axis.plot(snrs, [r["f1"] for r in rows], **style)
        fa_axis.plot(snrs, [100 * r["false_alarm"] for r in rows], **style)
    f1_axis.set(title="Frame F1 against utterance truth", xlabel="SNR (dB)", ylabel="F1", ylim=(0, 1.02))
    fa_axis.set(title="False alarms: speech reported in silence", xlabel="SNR (dB)", ylabel="% of silent frames",
                ylim=(0, 20))
    f1_axis.legend(loc="lower right")
    eyebrow(fig, "Fig 2 · Noise and SNR")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "vad-noise-sweep.png")


def timers_figure(output_dir: Path, dialog: dict) -> None:
    x = noisy_dialog("white", 20, dialog)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
    silences = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2])
    segments = [main.evaluate(x, dialog, main.VADParams(min_silence_duration=s))["segments"] for s in silences]
    left.plot(silences, segments, color=ACCENT, linewidth=1.8, marker="o", markersize=3.5)
    left.axhline(len(dialog["utterances"]), color=MUTED, linewidth=0.9, linestyle="--")
    left.text(1.2, len(dialog["utterances"]) + 0.6, "true utterances", ha="right", color=MUTED, fontsize=8)
    left.axvline(0.55, color=INK, linewidth=0.8, linestyle=":")
    left.text(0.58, 15.5, "Silero default 0.55 s", color=INK, fontsize=8)
    left.set(title="Short timers split one turn into many", xlabel="min_silence_duration (s)  =  END delay",
             ylabel="Segments in the call", ylim=(0, 18))
    pads = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    for min_speech, color, width in ((0.05, SOFT, 1.2), (0.15, ACCENT, 1.8)):
        lost = [main.evaluate(x, dialog, main.VADParams(prefix_padding_duration=p, min_speech_duration=min_speech))
                ["onset_clipping_ms"] for p in pads]
        right.plot(pads, lost, color=color, linewidth=width, marker="o", markersize=3.5,
                   label=f"min_speech_duration = {min_speech} s")
    right.set(title="Prefix padding gives back the onset", xlabel="prefix_padding_duration (s)",
              ylabel="Onset lost per utterance (ms)", ylim=(-3, 100))
    right.legend(loc="upper right")
    eyebrow(fig, "Fig 3 · Timers")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "vad-timers.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    dialog = main.synthetic_dialog()
    decisions_figure(output_dir, dialog)
    noise_sweep_figure(output_dir, dialog)
    timers_figure(output_dir, dialog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
