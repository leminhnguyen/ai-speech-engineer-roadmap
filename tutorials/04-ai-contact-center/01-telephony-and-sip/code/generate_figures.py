"""Create deterministic plots embedded by Module 04, Lesson 01."""

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


def companding_figure(output_dir: Path) -> None:
    fig, (curve, snr) = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.linspace(-1, 1, 2001)
    codes = main.mulaw_encode(main.float_to_pcm16(x)).astype(np.int32)
    signed_code = np.where(codes >= 128, 255 - codes, codes - 127)  # 0xFF=0, 0x80=+127, 0x00=-127
    curve.plot(x, main.mulaw_compress(x), color=SOFT, linewidth=1.2, label="continuous mu-law, mu = 255")
    curve.step(x, signed_code / 128, color=ACCENT, linewidth=1.4, where="mid", label="G.711 8-bit code")
    curve.plot(x, x, color=MUTED, linewidth=0.9, linestyle="--", label="linear")
    curve.set(title="Small amplitudes get most of the codes", xlabel="Input amplitude", ylabel="Code (normalized)")
    curve.legend(loc="upper left")

    levels_db = np.arange(-60, 1, 2)
    t = np.arange(8000) / 8000
    mu_snr, lin_snr = [], []
    for level in levels_db:
        tone = 10 ** (level / 20) * np.sin(2 * np.pi * 1013 * t)
        mu_snr.append(main.snr_db(tone, main.mulaw_roundtrip(tone)))
        lin_snr.append(main.snr_db(tone, main.quantize_linear(tone, 8)))
    snr.plot(levels_db, mu_snr, color=ACCENT, linewidth=1.8, label="G.711 mu-law, 8 bits")
    snr.plot(levels_db, lin_snr, color=LINK, linewidth=1.4, label="linear PCM, 8 bits")
    snr.axvspan(-40, -10, color=SAGE, alpha=0.12, linewidth=0)
    snr.text(-25, 51, "typical speech levels", color=MUTED, fontsize=8, ha="center")
    snr.set(title="Same 64 kbps, steady SNR across loudness", xlabel="Tone level (dBFS)", ylabel="SNR (dB)",
            ylim=(0, 55))
    snr.legend(loc="lower right")
    eyebrow(fig, "Fig 1 · G.711 companding")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "g711-companding.png")


def spectrum_db(x, sample_rate, n_fft=4096):
    frames = [x[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(x) - n_fft + 1, n_fft // 2)]
    power = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    return np.fft.rfftfreq(n_fft, d=1 / sample_rate), 10 * np.log10(power + 1e-12)


def narrowband_figure(output_dir: Path) -> None:
    wide = main.synthetic_utterance()
    phone = main.phone_channel(wide)
    fig, axis = plt.subplots(figsize=(9, 3.6))
    f_wide, p_wide = spectrum_db(wide, 16_000)
    f_phone, p_phone = spectrum_db(phone, 8000, n_fft=2048)
    reference = p_wide.max()
    axis.axvspan(3400, 8000, color=ACCENT, alpha=0.06, linewidth=0)
    axis.axvspan(0, 300, color=ACCENT, alpha=0.06, linewidth=0)
    axis.plot(f_wide, p_wide - reference, color=SOFT, linewidth=1.0, label="wideband 16 kHz (browser, app)")
    axis.plot(f_phone, p_phone - reference, color=ACCENT, linewidth=1.4, label="after the phone line (8 kHz, G.711)")
    axis.annotate("/s/ energy above 3.4 kHz is gone", xy=(6000, -42), xytext=(4400, -78), color=INK, fontsize=8,
                  arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    axis.annotate("F0 = 120 Hz removed;\nharmonics keep the pitch", xy=(150, -70), xytext=(700, -92), color=INK,
                  fontsize=8, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    axis.set(title="A phone line keeps 300-3400 Hz", xlabel="Frequency (Hz)", ylabel="Power (dB, relative)",
             xlim=(0, 8000), ylim=(-110, 5))
    axis.legend(loc="upper right")
    eyebrow(fig, "Fig 2 · Narrowband speech")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir / "narrowband-speech.png")


def dtmf_figure(output_dir: Path) -> None:
    tone = main.mulaw_roundtrip(main.dtmf_tone("8"))
    freqs = main.DTMF_LOW + main.DTMF_HIGH
    power = np.array([main.goertzel_power(tone, f, 8000) for f in freqs])
    power_db = 10 * np.log10(power / power.max() + 1e-12)
    fig, axis = plt.subplots(figsize=(9, 3.2))
    colors = [ACCENT if p > -6 else SOFT for p in power_db]
    bars = axis.bar([f"{int(f)}" for f in freqs], power_db + 80, bottom=-80, color=colors, width=0.6)
    for bar, value in zip(bars, power_db):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}" if value < -0.5 else "0", ha="center", fontsize=7.5,
                  color=MUTED)
    axis.axvline(3.5, color=MUTED, linewidth=0.8, linestyle="--")
    axis.text(-0.3, 9, "ROW TONES", ha="left", family="monospace", color=MUTED, fontsize=8)
    axis.text(3.7, 9, "COLUMN TONES", ha="left", family="monospace", color=MUTED, fontsize=8)
    axis.set(title="Key 8 = 852 Hz + 1336 Hz, found with eight Goertzel filters", xlabel="Goertzel frequency (Hz)",
             ylabel="Energy (dB, relative)", ylim=(-80, 14))
    axis.grid(axis="x", visible=False)
    eyebrow(fig, "Fig 3 · DTMF detection")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir / "dtmf-goertzel.png")


def save_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    companding_figure(output_dir)
    narrowband_figure(output_dir)
    dtmf_figure(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    save_figures(parser.parse_args().output_dir)
