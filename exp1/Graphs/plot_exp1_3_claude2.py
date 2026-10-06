#!/usr/bin/env python3
"""
Plot the LTspice transient simulation exported in exp1/exp1_3.txt.

Reads the tab-separated file (columns: time, V(n001), V(n002),
-V(N002)) and renders a print-ready figure suitable for direct
inclusion in a lab report, saved next to the data as exp1_3.png
(raster, for Word/Docs) and exp1_3.pdf (vector, for LaTeX).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import numpy as np

DATA_FILE = Path(__file__).resolve().parent / "exp1_3.txt"
OUT_PNG = DATA_FILE.with_suffix(".png")
OUT_PDF = DATA_FILE.with_suffix(".pdf")

# ---------------------------------------------------------------------------
# Edit these per experiment / figure.
# ---------------------------------------------------------------------------
TITLE = "Transient Response of the RC circuit using a sinusoidal input"
CAPTION = None  # e.g. "Figure 3. Input and output waveforms."
LABEL_1 = r"$V_{in}$"
LABEL_2 = r"$V(C1:2)$"
LABEL_3 = r"-$V(C1:2)$"
# Text shown in the on-plot boxes next to each curve. Independent of
# LABEL_1/2/3 above (which feed the legend row) - edit freely.
BOX_LABEL_1 = r"$V_{in}$"
BOX_LABEL_2 = r"$V(C1:2)$"
BOX_LABEL_3 = r"-$V(C1:2)$"
XLABEL = "Time (ms)"
YLABEL = "Voltage (V)"
N_MARKERS = 16  # markers per curve across the full time span

# Standard, print-safe, colorblind-legible colors.
RED = "#B22222"      # V(n001)
BLUE = "#1F4E8C"      # V(n002)
ORANGE = "#C7791A"    # -V(N002)
GRAY = "#7F7F7F"
GREEN = "#008000"

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


def _nearest_index(x, target):
    """Index of the sample nearest a target x value."""
    idx = np.searchsorted(x, target)
    return int(np.clip(idx, 0, len(x) - 1))


def _flattest_index(x, y, target, half_window, xlo, xhi, edge_margin):
    """Index near `target` anchored to the *middle* of whichever locally
    flat stretch of the curve (plateau, peak, trough) lies closest to
    `target`. Centering on the run - not just snapping to whichever
    single sample has minimum slope - keeps label leader lines away from
    steep transitions on both sides (e.g. a switch edge or pulse front
    just before the anchor, another one just after).

    Curves with no real plateau nearby (a smooth monotonic decay tail,
    say) have no genuine flat run - only ever-decreasing slope - so
    chasing the single flattest sample there just drags the anchor to
    the edge of the search window. In that case (or if the run found
    is too close to the axis edge to safely fit a box) this falls back
    to the plain nearest-sample-to-target choice instead."""
    fallback = _nearest_index(x, target)
    lo, hi = target - half_window, target + half_window
    i0 = max(np.searchsorted(x, lo), 1)
    i1 = min(np.searchsorted(x, hi), len(x) - 2)
    if i1 - i0 < 3:
        return fallback
    seg_x = x[i0:i1 + 1]
    seg_y = y[i0:i1 + 1]
    slopes = np.gradient(seg_y, seg_x)
    max_slope = np.abs(slopes).max() + 1e-12
    flat = np.abs(slopes) <= 0.08 * max_slope
    target_k = int(np.clip(np.searchsorted(seg_x, target), 0, len(seg_x) - 1))
    if not flat.any():
        return fallback

    runs, start = [], None
    for k, f in enumerate(flat):
        if f and start is None:
            start = k
        elif not f and start is not None:
            runs.append((start, k - 1))
            start = None
    if start is not None:
        runs.append((start, len(flat) - 1))

    def run_dist(r):
        s, e = r
        if s <= target_k <= e:
            return 0
        return min(abs(target_k - s), abs(target_k - e))

    s, e = min(runs, key=run_dist)
    idx = i0 + (s + e) // 2
    safe_lo, safe_hi = xlo + edge_margin, xhi - edge_margin
    if not (safe_lo <= x[idx] <= safe_hi):
        return fallback
    return idx


def curve_textboxes(ax, labels, x_fracs=None,
                     base_pt=26.0, min_pt=14.0, max_pt=34.0):
    """Individual bordered text boxes, one per curve, colored to match
    it. The text in each box comes from ``labels`` (matched to curves by
    plot order) and is completely independent of that curve's legend
    label - editing one has no effect on the other.

    Each box is anchored to a settled point on its curve - the middle of
    a nearby plateau/peak/trough where possible, so the leader line
    doesn't sit next to a switch edge or pulse front - and nudged a
    short, fixed, consistent distance (in points, not data units, so it
    looks right regardless of the y-axis scale) toward whichever side -
    above or below - has more clear space relative to the other curves
    and the axes edges. After placement, any box that would spill past
    the edge of the figure is nudged back in, sliding the box (and
    turning its leader line diagonal) while keeping the arrow anchored
    to the exact data point."""
    lines = [ln for ln in ax.get_lines()
             if ln.get_label() and not ln.get_label().startswith("_")]
    n = len(lines)
    if n == 0:
        return []
    if len(labels) != n:
        raise ValueError(
            f"curve_textboxes: got {len(labels)} labels for {n} curves "
            "- pass one box label per plotted curve, in plot order.")

    if x_fracs is None:
        x_fracs = np.linspace(0.18, 0.82, n) if n > 1 else [0.5]

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax_bbox = ax.get_window_extent(renderer)
    px_per_data_x = ax_bbox.width / (xhi - xlo) if xhi > xlo else 1.0
    px_per_data_y = ax_bbox.height / (yhi - ylo) if yhi > ylo else 1.0
    px_per_pt = fig.dpi / 72.0

    search_half_x = 0.15 * (xhi - xlo)
    edge_margin = 0.05 * (xhi - xlo)

    boxes = []
    for line, label, frac in zip(lines, labels, x_fracs):
        x, y = line.get_xdata(), line.get_ydata()
        target = xlo + frac * (xhi - xlo)
        idx = _flattest_index(x, y, target, search_half_x, xlo, xhi, edge_margin)
        xp, yp = x[idx], y[idx]
        color = line.get_color()

        other_vals = [np.interp(xp, other.get_xdata(), other.get_ydata())
                      for other in lines if other is not line]
        above = [v for v in other_vals if v > yp]
        below = [v for v in other_vals if v < yp]
        room_up_data = (min(above) - yp) if above else (yhi - yp)
        room_down_data = (yp - max(below)) if below else (yp - ylo)
        room_up_px = room_up_data * px_per_data_y
        room_down_px = room_down_data * px_per_data_y

        if room_up_px >= room_down_px:
            sign, va, room_px = 1, "bottom", room_up_px
        else:
            sign, va, room_px = -1, "top", room_down_px

        offset_px = min(max_pt * px_per_pt,
                         max(min_pt * px_per_pt, room_px * 0.55))
        offset_px = max(min(offset_px, room_px * 0.85), 10.0 * px_per_pt)

        yt = yp + sign * offset_px / px_per_data_y
        ann = ax.annotate(label, xy=(xp, yp), xytext=(xp, yt),
                          textcoords="data",
                          ha="center", va=va, color=color, fontsize=9.5,
                          bbox=dict(boxstyle="round,pad=0.28",
                                    facecolor="white", edgecolor=color,
                                    linewidth=1.0, alpha=0.92),
                          arrowprops=dict(arrowstyle="-", color=color,
                                          lw=0.8, shrinkA=0, shrinkB=3),
                          zorder=6, annotation_clip=False)
        boxes.append(ann)

    # Second pass: nudge any box that spills past the figure edge back
    # in, horizontally, keeping its arrow anchored to the true point.
    fig.canvas.draw()
    fig_bbox = fig.bbox
    pad_px = 4.0
    for ann in boxes:
        bb = ann.get_window_extent(renderer)
        dx_px = 0.0
        if bb.x0 < fig_bbox.x0 + pad_px:
            dx_px = (fig_bbox.x0 + pad_px) - bb.x0
        elif bb.x1 > fig_bbox.x1 - pad_px:
            dx_px = (fig_bbox.x1 - pad_px) - bb.x1
        if dx_px:
            xt, yt = ann.xyann
            ann.set_position((xt + dx_px / px_per_data_x, yt))
    if boxes:
        fig.canvas.draw()

    return boxes


def main() -> None:
    t_ms, v_n001, v_n002, v_neg_n002 = load_data(DATA_FILE)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90,
                        bottom=0.22 if CAPTION else 0.13)

    span = t_ms[-1] - t_ms[0]
    step = span / N_MARKERS
    idx_a = marker_indices(t_ms, N_MARKERS + 1, t_ms[0], t_ms[-1])
    idx_b = marker_indices(t_ms, N_MARKERS,
                            t_ms[0] + step / 3, t_ms[-1] - step / 3)
    idx_c = marker_indices(t_ms, N_MARKERS,
                            t_ms[0] + 2 * step / 3, t_ms[-1] - 2 * step / 3)

    ax.plot(t_ms, v_n001, color=RED, lw=1, label=LABEL_1,
            marker="s", ms=5.5, markevery=idx_a,
            markerfacecolor="white", markeredgecolor=RED, markeredgewidth=1.2,
            zorder=3)
    ax.plot(t_ms, v_n002, color=BLUE, lw=1, label=LABEL_2,
            marker="D", ms=5, markevery=idx_b,
            markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.2,
            zorder=3)
    ax.plot(t_ms, v_neg_n002, color=GREEN, lw=1, label=LABEL_3,
            marker="o", ms=4.5, markevery=idx_c,
            markerfacecolor="white", markeredgecolor=GREEN, markeredgewidth=1.1,
            zorder=2)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=1)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.margins(y=0.16)

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
    curve_textboxes(ax, [BOX_LABEL_1, BOX_LABEL_2, BOX_LABEL_3])

    ax.set_title(TITLE, pad=10)

    if CAPTION:
        fig.text(0.5, 0.03, CAPTION, ha="center", va="bottom", fontsize=10)

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
