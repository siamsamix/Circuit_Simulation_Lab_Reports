#!/usr/bin/env python3
"""
Plot the LTspice transient simulation exported in exp1/exp1_1.txt.

Reads the tab-separated file (columns: time, V(N005,N003), V(n003))
and renders a polished figure with both voltages overlaid,
saved next to it as exp1_1.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import numpy as np

DATA_FILE = Path(__file__).resolve().parent / "exp1_1.txt"
OUT_FILE = DATA_FILE.with_suffix(".png")

RED = "#C73E3A"    # V(N005, N003)
BLUE = "#2E86AB"   # V(n003)
GRAY = "#9A9A9A"

plt.rcParams.update({
    "figure.facecolor": "#FDFDFD",
    "axes.facecolor": "white",
    "axes.edgecolor": "#B5B5B5",
    "axes.labelcolor": "#404040",
    "xtick.color": "#606060",
    "ytick.color": "#606060",
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.linewidth": 0.9,
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
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + depth_pt / fig_h_pt)
    return leg


def main() -> None:
    t_ms, v_diff, v_n003 = load_data(DATA_FILE)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.12)

    step = (t_ms[-1] - t_ms[0]) / 40
    sq_idx = marker_indices(t_ms, 41, t_ms[0], t_ms[-1])
    di_idx = marker_indices(t_ms, 40, t_ms[0] + step / 2, t_ms[-1] - step / 2)

    ax.plot(t_ms, v_diff, color=RED, lw=1.5, label="V(N005, N003)",
            marker="s", ms=6, markevery=sq_idx,
            markerfacecolor="white", markeredgecolor=RED, markeredgewidth=1.3)
    ax.plot(t_ms, v_n003, color=BLUE, lw=1.5, label="V(n003)",
            marker="D", ms=5.5, markevery=di_idx,
            markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.3)
    ax.axhline(0, color=GRAY, lw=0.9)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, color="#E3E3E3", linewidth=0.8)
    ax.grid(True, which="minor", color="#F1F1F1", linewidth=0.6)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.06)
    ltspice_legend(ax)

    fig.suptitle("Transient response - exp1_1",
                 x=0.09, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.09, 0.945,
             f"Source: {DATA_FILE.name}  |  {len(t_ms)} samples  |  "
             f"{t_ms[0]:.0f}-{t_ms[-1]:.0f} ms",
             fontsize=9.5, color="#8A8A8A")

    fig.savefig(OUT_FILE, dpi=200)
    print(f"Saved: {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
