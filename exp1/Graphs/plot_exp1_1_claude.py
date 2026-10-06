#!/usr/bin/env python3
"""
Plot the LTspice transient simulation exported in exp1/exp1_1.txt.

Reads the tab-separated file (columns: time, V(N005,N003), V(n003)) and
renders a print-ready figure suitable for direct inclusion in a lab
report, saved next to the data as exp1_1.png (raster, for Word/Docs)
and exp1_1.pdf (vector, for LaTeX).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import numpy as np

DATA_FILE = Path(__file__).resolve().parent / "exp1_1.txt"
OUT_PNG = DATA_FILE.with_suffix(".png")
OUT_PDF = DATA_FILE.with_suffix(".pdf")

# ---------------------------------------------------------------------------
# Edit these per experiment / figure.
# ---------------------------------------------------------------------------
TITLE = "Transient Response of the RC circuit using switches"
CAPTION = None  # e.g. "Figure 1. Transient response of the RC network."
LABEL_1 = r"Voltage across resistor"
LABEL_2 = r"Voltage across capacitor"
XLABEL = "Time (ms)"
YLABEL = "Voltage (V)"
N_MARKERS = 16  # markers per curve across the full time span

# Standard, print-safe, colorblind-legible colors.
RED = "#B22222"    # V(N005, N003)
BLUE = "#1F4E8C"   # V(n003)
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
    "font.serif": ["Times New Roman"],#, "Nimbus Roman", "DejaVu Serif"],
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
    return raw[:, 0] * 1e3, raw[:, 1], raw[:, 2]    # time in ms


def marker_indices(t_ms, n_markers, t_start, t_end):
    """Indices of the samples closest to evenly spaced times.

    LTspice exports cluster many samples near t=0, so a point-count-based
    markevery would bunch the markers at the left edge.
    """
    idx = np.searchsorted(t_ms, np.linspace(t_start, t_end, n_markers))
    return np.clip(idx, 0, len(t_ms) - 1)


def ltspice_legend(ax):
    """LTspice-style legend: a frameless horizontal row of hollow marker
    symbols and trace names just below the x tick labels, with the x
    axis label moved below the row, as in an LTspice plot pane."""
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()
    proxies = [Line2D([], [], linestyle="none", marker=h.get_marker(),
                      markerfacecolor=h.get_markerfacecolor(),
                      markeredgecolor=h.get_markeredgecolor(),
                      markeredgewidth=h.get_markeredgewidth(),
                      markersize=h.get_markersize())
               for h in handles]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bottom = ax.get_window_extent(renderer).y0
    tick_labels = [lbl for lbl in ax.get_xticklabels() if lbl.get_text()]
    ticks_bottom = min(lbl.get_window_extent(renderer).y0
                       for lbl in tick_labels)
    # Drop the row just below the tick-label band.
    drop_pt = (ax_bottom - ticks_bottom) * 72 / fig.dpi + 5
    offset = mtransforms.ScaledTranslation(0, -drop_pt / 72,
                                           fig.dpi_scale_trans)

    ncol = max(len(proxies), 1)
    while True:
        leg = ax.legend(proxies, labels, loc="upper left", ncol=ncol,
                        bbox_to_anchor=(0, 0),
                        bbox_transform=ax.transAxes + offset,
                        frameon=False, borderaxespad=0,
                        handlelength=1.0, handletextpad=0.4,
                        columnspacing=1.8)
        fig.canvas.draw()
        if (ncol == 1 or leg.get_window_extent(renderer).x1
                <= ax.get_window_extent(renderer).x1):
            break
        ncol -= 1

    # Push the x label below the legend row.
    leg_bottom = leg.get_window_extent(renderer).y0
    labelpad = (ticks_bottom - leg_bottom) * 72 / fig.dpi + 7
    ax.set_xlabel(ax.get_xlabel(), labelpad=labelpad)
    for txt in leg.get_texts():
        txt.set_color(GRAY)

    # Make room at the bottom of the figure for the extra rows.
    fig.canvas.draw()
    depth_pt = (ax_bottom - ax.xaxis.label.get_window_extent(renderer).y0) \
        * 72 / fig.dpi + 8
    fig_h_pt = fig.get_figheight() * 72
    bottom = fig.subplotpars.bottom \
        + (depth_pt + (14 if CAPTION else 0)) / fig_h_pt
    fig.subplots_adjust(bottom=bottom)
    return leg


def main() -> None:
    t_ms, v_diff, v_n003 = load_data(DATA_FILE)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90,
                         bottom=0.22 if CAPTION else 0.13)

    step = (t_ms[-1] - t_ms[0]) / N_MARKERS
    sq_idx = marker_indices(t_ms, N_MARKERS + 1, t_ms[0], t_ms[-1])
    di_idx = marker_indices(t_ms, N_MARKERS,
                             t_ms[0] + step / 2, t_ms[-1] - step / 2)

    ax.plot(t_ms, v_diff, color=RED, lw=1, label=LABEL_1,
            marker="s", ms=5.5, markevery=sq_idx,
            markerfacecolor="white", markeredgecolor=RED, markeredgewidth=1.2,
            zorder=3)
    ax.plot(t_ms, v_n003, color=BLUE, lw=1, label=LABEL_2,
            marker="D", ms=5, markevery=di_idx,
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

    ltspice_legend(ax)

    ax.set_title(TITLE, pad=10)

    if CAPTION:
        fig.text(0.5, 0.03, CAPTION, ha="center", va="bottom", fontsize=10)

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
