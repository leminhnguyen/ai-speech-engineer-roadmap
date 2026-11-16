"""Create deterministic plots embedded by Module 04, Lesson 04."""

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




from matplotlib.colors import LinearSegmentedColormap

SPEC_CMAP = LinearSegmentedColormap.from_list("paper_ink", [PAPER, "#c9ccd4", MUTED, INK])


def show_spec(axis, x, title, vmax):
    power_db = 10 * np.log10(np.abs(main.stft(x)) ** 2 + 1e-10)
    extent = (0, len(x) / main.SR, 0, main.SR / 2000)
    axis.imshow(power_db.T, origin="lower", aspect="auto", cmap=SPEC_CMAP, vmin=vmax - 55, vmax=vmax, extent=extent,
                interpolation="nearest")
    axis.set_title(title, fontsize=9.5, pad=4)
    axis.grid(False)


def spectrogram_figure(output_dir: Path) -> None:
    clean = main.synthetic_speech()
    noisy, _ = main.mix_at_snr(clean, main.make_noise("fan", len(clean)), 5.0)
    sub = main.enhance(noisy, "subtraction", over=1.0, floor=0.0)
    wiener = main.enhance(noisy, "wiener")
    vmax = 10 * np.log10(np.max(np.abs(main.stft(noisy)) ** 2))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True, sharey=True)
    show_spec(axes[0, 0], clean, "Clean speech", vmax)
    show_spec(axes[0, 1], noisy, "With fan noise and mains hum, 5 dB SNR", vmax)
    show_spec(axes[1, 0], sub, f"Spectral subtraction, no floor: {main.snr_db(clean, sub):.1f} dB SNR", vmax)
    show_spec(axes[1, 1], wiener, f"Wiener + decision-directed: {main.snr_db(clean, wiener):.1f} dB SNR", vmax)
    axes[1, 0].annotate("isolated peaks =\n'musical noise'", xy=(1.05, 2.9), xytext=(0.62, 3.45), fontsize=8, color=ACCENT,
                        bbox={"boxstyle": "square,pad=0.15", "fc": PAPER, "ec": "none"},
                        arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": 0.8})
    for axis in axes[1]:
        axis.set_xlabel("Time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("kHz")
    eyebrow(fig, "Fig 1 · Spectral suppression")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, output_dir / "suppression-spectrograms.png")


def tradeoff_figure(output_dir: Path) -> None:
    clean = main.synthetic_speech()
    noisy, noise = main.mix_at_snr(clean, main.make_noise("fan", len(clean)), 5.0)
    spec = main.stft(noisy)
    power = np.abs(spec) ** 2
    psd = main.minimum_statistics(power)
    overs = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    fig, axis = plt.subplots(figsize=(9, 3.8))
    axis.axhspan(-10, 0, color=ACCENT, alpha=0.06, linewidth=0)
    axis.text(0.8, -1.8, "over-suppression: the gain is eating speech", color=ACCENT, fontsize=8)
    for name, rule, color in (("spectral subtraction", main.spectral_subtraction_gain, LINK),
                              ("Wiener + decision-directed", main.wiener_dd_gain, ACCENT)):
        points, snrs = [], []
        for over in overs:
            gain = rule(power, psd, over=over, floor=0.01)
            points.append(main.distortion_and_reduction(clean, noise, gain))
            snrs.append(main.snr_db(clean, main.istft(gain * spec, len(noisy))))
        axis.plot([r for _, r in points], [d for d, _ in points], "o-", color=color, markersize=4,
                  linewidth=1.6 if color == ACCENT else 1.2, label=f"{name}, noise over-estimate 0.5x to 32x")
        if color == ACCENT:
            best = int(np.argmax(snrs))
            for i in (0, best, len(overs) - 1):
                d, r = points[i]
                text = f"{overs[i]:g}x: {snrs[i]:.1f} dB SNR" + (" (best)" if i == best else "")
                axis.annotate(text, (r, d), textcoords="offset points", xytext=(-10, 9), fontsize=7.5, color=INK,
                              ha="right")
    axis.set(title="Pushing suppression harder removes more noise, then more speech", xlabel="Noise reduction (dB)",
             ylabel="Speech distortion (dB)", xlim=(0, 26), ylim=(-30, 0))
    axis.legend(loc="lower right")
    eyebrow(fig, "Fig 2 · Reduction vs distortion")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "reduction-distortion.png")


def noise_type_figure(output_dir: Path) -> None:
    clean = main.synthetic_speech()
    kinds = ("fan", "street", "keyboard", "babble")
    names = ("Fan + mains hum\n(stationary)", "Street\n(slowly varying)", "Keyboard\n(impulsive)", "Background talkers\n(babble)")
    gains = []
    for kind in kinds:
        noisy, _ = main.mix_at_snr(clean, main.make_noise(kind, len(clean)), 5.0)
        gains.append(main.snr_db(clean, main.enhance(noisy)) - main.snr_db(clean, noisy))
    fig, axis = plt.subplots(figsize=(9, 3.4))
    colors = [SOFT, SOFT, SOFT, ACCENT]
    bars = axis.bar(names, gains, color=colors, width=0.55)
    for bar, value in zip(bars, gains):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"+{value:.1f} dB", ha="center", fontsize=8, color=INK)
    axis.text(3, 2.2, "speech-like noise looks like speech\nto the noise tracker: use voice isolation", ha="center",
              fontsize=8, color=ACCENT)
    axis.set(title="The same suppressor, four noises, all at 5 dB SNR", ylabel="SNR improvement (dB)", ylim=(0, 8.5))
    axis.grid(axis="x", visible=False)
    eyebrow(fig, "Fig 3 · Noise types")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "noise-types.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    spectrogram_figure(output_dir)
    tradeoff_figure(output_dir)
    noise_type_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
