#!/usr/bin/env python3
"""
Plot the LTspice transient simulation exported in exp1/exp1_4.txt.

Reads the tab-separated file (columns: time, -V(N002)*I(C1)) and
renders a print-ready figure suitable for direct inclusion in a lab
report, saved next to the data as exp1_4.png (raster, for Word/Docs)
and exp1_4.pdf (vector, for LaTeX).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_FILE = Path(__file__).resolve().parent / "exp1_4.txt"
OUT_PNG = DATA_FILE.with_suffix(".png")
OUT_PDF = DATA_FILE.with_suffix(".pdf")

# ---------------------------------------------------------------------------
# Edit these per experiment / figure.
# ---------------------------------------------------------------------------
TITLE = "Instantaneous Power \u2013 Experiment 1.4"
CAPTION = None  # e.g. "Figure 4. Instantaneous power delivered to C1."
LABEL_1 = r"$-V_{\mathrm{N002}} \cdot I_{C1}$"
XLABEL = "Time (ms)"
YLABEL = "Power (W)"
N_MARKERS = 16  # markers across the full time span

# Standard, print-safe color.
RED = "#B22222"    # -V(N002)*I(C1)
GRAY = "#7F7F7F"

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
    return raw[:, 0] * 1e3, raw[:, 1]               # time in ms


def marker_indices(t_ms, n_markers, t_start, t_end):
    """Indices of the samples closest to evenly spaced times.

    LTspice exports cluster many samples near t=0, so a point-count-based
    markevery would bunch the markers at the left edge.
    """
    idx = np.searchsorted(t_ms, np.linspace(t_start, t_end, n_markers))
    return np.clip(idx, 0, len(t_ms) - 1)


def main() -> None:
    t_ms, power = load_data(DATA_FILE)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90,
                         bottom=0.22 if CAPTION else 0.13)

    idx = marker_indices(t_ms, N_MARKERS + 1, t_ms[0], t_ms[-1])

    ax.plot(t_ms, power, color=RED, lw=1.4, label=LABEL_1,
            marker="s", ms=5.5, markevery=idx,
            markerfacecolor="white", markeredgecolor=RED, markeredgewidth=1.2,
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
