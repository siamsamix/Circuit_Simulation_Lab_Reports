#!/usr/bin/env python3
"""
Plot the LTspice transient simulation exported in exp1/exp1_2.txt.

Reads the tab-separated file (columns: time, V(N001,N002), V(n001),
V(n002)) and renders a print-ready figure suitable for direct inclusion
in a lab report, saved next to the data as exp1_2.png (raster, for
Word/Docs) and exp1_2.pdf (vector, for LaTeX).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np

DATA_FILE = Path(__file__).resolve().parent / "exp1_2.txt"
OUT_PNG = DATA_FILE.with_suffix(".png")
OUT_PDF = DATA_FILE.with_suffix(".pdf")

# ---------------------------------------------------------------------------
# Edit these per experiment / figure.
# ---------------------------------------------------------------------------
TITLE = "Transient Response of RC circuit using a pulse input"
CAPTION = None  # e.g. "Figure 2. Transient response of the RC network."
LABEL_1 = r"Voltage across resistor"
LABEL_2 = r"$V_{in}$"
LABEL_3 = r"Voltage across capacitor"
XLABEL = "Time (ms)"
YLABEL = "Voltage (V)"
N_MARKERS = 16  # markers per curve across the full time span

# Standard, print-safe, colorblind-legible colors.
RED = "#B22222"     # V(N001, N002)
GREEN = "#2E7D32"   # V(n001)
BLUE = "#1F4E8C"    # V(n002)
GRAY = "#7F7F7F"

def _ensure_font_registered(family_name: str) -> bool:
    """Register a font with Matplotlib even if it was installed after
    Matplotlib built its font cache.

    Matplotlib scans installed fonts once per process and caches the
    result on disk, so a font installed afterward stays invisible to
    ``font.serif`` until it is registered directly like this.
    """
    have = {f.name for f in fm.fontManager.ttflist}
    if family_name in have:
        return True
    for path in fm.findSystemFonts():
        try:
            name = fm.FontProperties(fname=path).get_name()
        except Exception:
            continue
        if name == family_name:
            fm.fontManager.addfont(path)
            return True
    return False


if not _ensure_font_registered("Times New Roman"):
    print("Note: 'Times New Roman' was not found on this system, so this "
          "figure will fall back to the next serif font in font.serif "
          "below. If you just installed it, confirm Matplotlib can see "
          "it (it should now if it's installed for this user/OS) and "
          "rerun this script.")


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
})


def load_data(path: Path):
    raw = np.genfromtxt(path, delimiter="\t", skip_header=1)
    raw = raw[np.isfinite(raw).all(axis=1)]        # drop blank/short lines
    return raw[:, 0] * 1e3, raw[:, 1], raw[:, 2], raw[:, 3]  # time in ms


def marker_indices(t_ms, n_markers, t_start, t_end):
    """Indices of the samples closest to evenly spaced times.

    LTspice exports cluster many samples near t=0, so a point-count-based
    markevery would bunch the markers at the left edge.
    """
    idx = np.searchsorted(t_ms, np.linspace(t_start, t_end, n_markers))
    return np.clip(idx, 0, len(t_ms) - 1)


def main() -> None:
    t_ms, v_diff, v_n001, v_n002 = load_data(DATA_FILE)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90,
                         bottom=0.22 if CAPTION else 0.13)

    span = t_ms[-1] - t_ms[0]
    step = span / N_MARKERS
    idx_a = marker_indices(t_ms, N_MARKERS + 1, t_ms[0], t_ms[-1])
    idx_b = marker_indices(t_ms, N_MARKERS,
                            t_ms[0] + step / 3, t_ms[-1] - step / 3)
    idx_c = marker_indices(t_ms, N_MARKERS,
                            t_ms[0] + 2 * step / 3, t_ms[-1] - 2 * step / 3)

    ax.plot(t_ms, v_diff, color=RED, lw=1.4, label=LABEL_1,
            marker="s", ms=5.5, markevery=idx_a,
            markerfacecolor="white", markeredgecolor=RED, markeredgewidth=1.2,
            zorder=3)
    ax.plot(t_ms, v_n001, color=GREEN, lw=1.4, label=LABEL_2,
            marker="o", ms=5, markevery=idx_b,
            markerfacecolor="white", markeredgecolor=GREEN, markeredgewidth=1.2,
            zorder=3)
    ax.plot(t_ms, v_n002, color=BLUE, lw=1.4, label=LABEL_3,
            marker="D", ms=5, markevery=idx_c,
            markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.2,
            zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=1)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.margins(y=0.08)

    # Classic boxed scientific-plot style: ticks in, box fully closed.
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.grid(True, which="minor", color="#ECECEC", linewidth=0.5, zorder=0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5.5)
    ax.tick_params(which="minor", length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    ax.legend(frameon=True, edgecolor="#BFBFBF", framealpha=1.0,
              fancybox=False, loc="best")

    ax.set_title(TITLE, pad=10)

    if CAPTION:
        fig.text(0.5, 0.03, CAPTION, ha="center", va="bottom", fontsize=10)

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
