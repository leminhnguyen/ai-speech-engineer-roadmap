"""Create deterministic plots embedded by Module 04, Lesson 02."""

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




def spectrum_db(x, sample_rate, n_fft=4096):
    frames = [x[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(x) - n_fft + 1, n_fft // 2)]
    power = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    return np.fft.rfftfreq(n_fft, d=1 / sample_rate), 10 * np.log10(power + 1e-20)


def aliasing_figure(output_dir: Path) -> None:
    fig, (response, result) = plt.subplots(1, 2, figsize=(9, 3.6))
    up, down, h = main.design_resampler(48_000, 8_000)
    freqs = np.fft.rfftfreq(1 << 16, d=1 / 48_000)
    gain = 20 * np.log10(np.abs(np.fft.rfft(h / up, 1 << 16)) + 1e-12)
    response.plot(freqs, gain, color=ACCENT, linewidth=1.4)
    response.axvline(4_000, color=MUTED, linewidth=0.8, linestyle="--")
    response.text(4_150, -20, "new Nyquist\n4 kHz", color=MUTED, fontsize=8)
    response.set(title="Anti-alias filter for 48 kHz to 8 kHz", xlabel="Frequency at 48 kHz (Hz)",
                 ylabel="Gain (dB)", xlim=(0, 8_000), ylim=(-140, 5))

    x = main.tone(1_000, 48_000, 1.0) + main.tone(6_000, 48_000, 1.0)
    f_good, p_good = spectrum_db(main.resample(x, 48_000, 8_000), 8_000, 2048)
    f_bad, p_bad = spectrum_db(main.naive_resample(x, 48_000, 8_000), 8_000, 2048)
    ref = p_good.max()
    result.plot(f_bad, p_bad - ref, color=LINK, linewidth=1.0, label="linear interpolation")
    result.plot(f_good, p_good - ref, color=ACCENT, linewidth=1.4, label="windowed-sinc resampler")
    result.annotate("6 kHz folds to 2 kHz", xy=(2_000, -6), xytext=(2_300, 0), color=INK, fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    result.set(title="A 1 kHz + 6 kHz signal after downsampling", xlabel="Frequency at 8 kHz (Hz)",
               ylabel="Power (dB, relative)", xlim=(0, 4_000), ylim=(-160, 12))
    result.legend(loc="lower left")
    eyebrow(fig, "Fig 1 · Rate conversion")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "resampler-aliasing.png")


def empty_band_figure(output_dir: Path) -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(16_000)
    taps = main.kaiser_lowpass(3_400 / 8_000, 129) - main.kaiser_lowpass(300 / 8_000, 129)
    phone = np.convolve(noise, taps, mode="same")  # "speech" occupying the 300-3400 Hz phone band
    upsampled = main.resample(phone, 8_000, 16_000)
    native = rng.standard_normal(32_000)  # what a 16 kHz model saw in training: energy up to 8 kHz
    fig, axis = plt.subplots(figsize=(9, 3.4))
    f_up, p_up = spectrum_db(upsampled, 16_000)
    f_nat, p_nat = spectrum_db(native * np.std(upsampled), 16_000)
    ref = p_nat.max()
    axis.axvspan(4_000, 8_000, color=ACCENT, alpha=0.06, linewidth=0)
    axis.plot(f_nat, p_nat - ref, color=SOFT, linewidth=1.0, label="wideband 16 kHz input")
    axis.plot(f_up, p_up - ref, color=ACCENT, linewidth=1.4, label="phone audio resampled 8 kHz to 16 kHz")
    axis.text(6_000, -40, "no energy here:\nresampling cannot restore it", ha="center", color=INK, fontsize=8)
    axis.set(title="Upsampling changes the sample rate, not the bandwidth", xlabel="Frequency (Hz)",
             ylabel="Power (dB, relative)", xlim=(0, 8_000), ylim=(-140, 15))
    axis.legend(loc="lower left")
    eyebrow(fig, "Fig 2 · Bandwidth is not sample rate")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "upsampling-empty-band.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    aliasing_figure(output_dir)
    empty_band_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
