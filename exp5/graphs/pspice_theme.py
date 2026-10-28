#!/usr/bin/env python3
"""
Shared PSpice/Probe-style theme for the exp3 plot scripts.

One 991 x 423 pt white page, green trace with white square markers (green rims) spaced at a
constant on-screen distance along the curve, solid black major grid, dotted minor
grid on the y axis, PSpice-style tick labels ("0V", "5mA", "-100.5V", ...), a
hollow-square legend and centred axis label underneath the axes, a white serif
description box that dodges the trace, and thin black cursor crosshairs with
grey-bordered "(x,y)" read-outs.

Scripts build their figures from:

    header, data = load_table(datafile)
    fig, ax, frame = new_page(xlim, ylim, ylabels)
    style_axes(ax, xticks, xlabels, yticks, ylabels, yminor)
    curve = dense_curve(x, y, frame)
    draw_trace(ax, x, y, frame)
    add_footer(fig, trace_name, axis_label)
    add_note(ax, note, frame, curve, avoid_rects)
    add_footer(fig, trace_name, axis_label)
    add_y_label(fig, y_label)
    add_note(ax, note, frame, curve, avoid_rects)

`frame` (see Frame) converts data coordinates to page points, which the marker
spacing and cursor helpers work in.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

# --------------------------------------------------------------------------- #
# Layout constants (same 991 x 423 pt page as the reference)
# --------------------------------------------------------------------------- #
FIG_W_PT, FIG_H_PT = 991.0, 423.0
AX_RECT = (0.0257, 0.1132, 0.9539, 0.8822)       # left, bottom, width, height
AX_LEFT_PT   = AX_RECT[0] * FIG_W_PT
AX_RIGHT_PT  = (AX_RECT[0] + AX_RECT[2]) * FIG_W_PT
AX_BOTTOM_PT = AX_RECT[1] * FIG_H_PT
AX_TOP_PT    = (AX_RECT[1] + AX_RECT[3]) * FIG_H_PT
AX_H_PT      = AX_RECT[3] * FIG_H_PT
TICK_LEN_PT, TICK_PAD_PT = 4.7, 4.0
Y_LABEL_PAD_PT = 22.0                            # strip on the far left reserved for the y-title
AX_PAD_PT = 5.0

GREEN = "#00aa00"                                # PSpice trace colour
NOTE_EDGE = "#999999"
NOTE_SERIF = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "Tinos", "DejaVu Serif"]
READOUT_PAD_PT = 2.0

MARKER_FIRST_PT = 222.0                          # first marker: distance along the curve
MARKER_SPACING_PT = 106.5                        # spacing between markers along the curve


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
def load_table(path):
    """Read a tab/space/comma-separated numeric export; returns (header, data)."""
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    header = re.split(r"\s+", lines[0])

    def _to_float(tok):
        try:
            return float(tok)
        except ValueError:          # "-nan(ind)" and other sim artefacts
            return float("nan")

    data = np.array([[_to_float(tok) for tok in re.split(r"[\s,]+", ln)] for ln in lines[1:]])
    return header, data


def spice_format(value):
    """Format a number the way PSpice prints cursor read-outs (6 sig. digits + SI suffix)."""
    scales = [(1e9, "G"), (1e6, "Meg"), (1e3, "K"), (1.0, ""), (1e-3, "m"),
              (1e-6, "u"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]
    if value == 0:
        return "0"
    for scale, suffix in scales:
        if abs(value) >= scale:
            return f"{value / scale:.6g}{suffix}"
    return f"{value:.6g}"


# --------------------------------------------------------------------------- #
# Tick and label helpers
# --------------------------------------------------------------------------- #
def qty_label(value, unit):
    """Tick label in PSpice style: plain "0V", "5.0V" for volts, SI-prefixed "5mA" for amps."""
    value = round(float(value), 6) + 0.0
    if unit == "A":
        return spice_format(value) + "A"
    return f"{value:g}{unit}"


def major_ticks(lo, hi, step):
    """Multiples of `step` filling [lo, hi] (float-noise free)."""
    first = np.ceil(lo / step - 1e-9) * step
    return [round(v, 12) for v in np.arange(first, hi + step * 1e-9, step)]


def minor_between(majors, sub):
    """`sub` - 1 evenly spaced minor positions inside every major interval."""
    out = []
    for a, b in zip(majors[:-1], majors[1:]):
        out += [round(a + (b - a) * k / sub, 12) for k in range(1, sub)]
    return out


def text_width(text, size, prop=None):
    return TextPath((0, 0), text, size=size, prop=prop or FontProperties()).get_extents().width


def label_widths_pt(texts, size):
    """Width (points) of each text as matplotlib really renders it."""
    fig = plt.figure()
    try:
        renderer = fig.canvas.get_renderer()
        prop = FontProperties()
        prop.set_size(size)
        return [renderer.get_text_width_height_descent(t, prop, False)[0] * 72.0 / fig.dpi
                for t in texts]
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
class Frame:
    """Axes limits (data units) and axes size (points); converts data to points."""

    def __init__(self, xmin, xmax, ymin, ymax, w_pt, h_pt):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.w, self.h = w_pt, h_pt

    def to_pt(self, x, y):
        x_pt = (np.asarray(x) - self.xmin) / (self.xmax - self.xmin) * self.w
        y_pt = (np.asarray(y) - self.ymin) / (self.ymax - self.ymin) * self.h
        return x_pt, y_pt


def marker_positions(t, y, frame):
    """Points spaced at a constant on-screen distance along the plotted curve."""
    x_pt, y_pt = frame.to_pt(t, y)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_pt), np.diff(y_pt)))])
    marks = np.arange(MARKER_FIRST_PT, s[-1], MARKER_SPACING_PT)
    return np.interp(marks, s, t), np.interp(marks, s, y)


def dense_curve(t, y, frame):
    """Curve in points, resampled finely so no segment can slip through a collision test."""
    x_pt, y_pt = frame.to_pt(t, y)
    seg = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_pt), np.diff(y_pt)))])
    at = np.linspace(0.0, seg[-1], 6001)
    return np.interp(at, seg, x_pt), np.interp(at, seg, y_pt)


def rect_is_free(rect, curve_pts, avoid_rects, margin, frame):
    """True if rect (x0, y0, x1, y1) is inside the axes and clear of the curve and other rects."""
    x0, y0, x1, y1 = rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin
    if x0 < 0 or y0 < 0 or x1 > frame.w or y1 > frame.h:
        return False
    cx, cy = curve_pts
    if np.any((cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)):
        return False
    return not any(x0 < r[2] and x1 > r[0] and y0 < r[3] and y1 > r[1] for r in avoid_rects)


def place_readout(px, py, text_w, fsize, frame, gap=8.3):
    """Centre the read-out above the point (below if there is no room), kept inside the axes."""
    m = 1.5
    if py + gap + fsize <= frame.h - m:
        cy = py + gap + fsize / 2
    elif py - gap - fsize >= m:
        cy = py - gap - fsize / 2
    else:
        cy = min(max(py + gap + fsize / 2, m + fsize / 2), frame.h - m - fsize / 2)
    dx = -text_w / 2
    x0, x1 = px + dx, px + dx + text_w
    if x0 < m:
        dx += m - x0
    elif x1 > frame.w - m:
        dx -= x1 - (frame.w - m)
    return dx, cy - py


def add_note(ax, note, frame, curve_pts, avoid_rects):
    """White, thin-framed, serif text box, moved to the free spot nearest the upper right."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in NOTE_SERIF if f in installed), "serif")
    prop = FontProperties(family=family)
    fontsize = 8.0 * (0.2256 * AX_RECT[2] * FIG_W_PT) / text_width("Voltage across capacitor", 10, prop)

    box_w = text_width(note, fontsize, prop) + 2 * 0.0140 * AX_RECT[2] * FIG_W_PT
    box_h = 0.0925 * AX_H_PT
    m = 3.0

    def free(cx, cy):
        return rect_is_free((cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2),
                            curve_pts, avoid_rects, m, frame)

    px, py = 0.75 * frame.w, 0.6083 * frame.h          # preferred spot (as in the reference)
    cx, cy = px, py
    if not free(px, py):
        xs = np.arange(box_w / 2 + m, frame.w - box_w / 2 - m + 1e-9, 3.0)
        ys = np.arange(box_h / 2 + m, frame.h - box_h / 2 - m + 1e-9, 3.0)
        gx, gy = np.meshgrid(xs, ys)
        for i in np.argsort(np.hypot(gx - px, gy - py), axis=None):
            if free(gx.flat[i], gy.flat[i]):
                cx, cy = gx.flat[i], gy.flat[i]
                break

    fx, fy, fw, fh = cx / frame.w, cy / frame.h, box_w / frame.w, box_h / frame.h
    ax.add_patch(FancyBboxPatch((fx - fw / 2, fy - fh / 2), fw, fh,
                                boxstyle=f"round,pad=0,rounding_size={2.0 / frame.w}",
                                mutation_aspect=frame.w / frame.h, transform=ax.transAxes,
                                facecolor="white", edgecolor=NOTE_EDGE, linewidth=1.0,
                                zorder=10, clip_on=False))
    ax.text(fx, fy, note, transform=ax.transAxes, ha="center", va="center_baseline",
            fontproperties=prop, fontsize=fontsize, color="black", zorder=11)


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #
def choose_font():
    """Noto Sans (the reference font) if present, otherwise a fallback."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for fam, size in (("Noto Sans", 11.6), ("Open Sans", 11.4), ("DejaVu Sans", 10.6)):
        if fam in installed:
            return fam, size
    return "sans-serif", 10.6


def new_page(xlim, ylim, ylabels, reserve_y_label=True):
    """White 991 x 423 pt page with a PSpice-styled axes; returns (fig, ax, frame).

    The plot area is inset by ``AX_PAD_PT`` on every side so the figure reads as
    slightly "zoomed out"; when ``reserve_y_label`` is true (the default) an
    extra ``Y_LABEL_PAD_PT``-point strip is kept clear on the far left so
    ``add_y_label`` can place the rotated y-axis title clear of the tick labels.
    """
    family, fsize = choose_font()
    plt.rcParams["font.family"] = family
    plt.rcParams["font.size"] = fsize

    widest = max(label_widths_pt(ylabels, fsize))
    left_pt = widest + TICK_LEN_PT + TICK_PAD_PT
    if reserve_y_label:
        left_pt += Y_LABEL_PAD_PT
    left_pt = max(AX_PAD_PT, left_pt)

    right_pt  = AX_RIGHT_PT  - AX_PAD_PT
    bottom_pt = AX_BOTTOM_PT + AX_PAD_PT
    top_pt    = AX_TOP_PT    - AX_PAD_PT
    width_pt  = right_pt - left_pt
    height_pt = top_pt - bottom_pt

    frame = Frame(xlim[0], xlim[1], ylim[0], ylim[1], width_pt, height_pt)

    fig = plt.figure(figsize=(FIG_W_PT / 72.0, FIG_H_PT / 72.0), facecolor="white")
    ax = fig.add_axes((left_pt / FIG_W_PT, bottom_pt / FIG_H_PT,
                       width_pt / FIG_W_PT, height_pt / FIG_H_PT))
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    return fig, ax, frame

def style_axes(ax, xticks, xlabels, yticks, ylabels, yminor=()):
    """PSpice grid: solid black majors, dotted minors on y only; plain outer ticks."""
    ax.xaxis.set_major_locator(FixedLocator(list(xticks)))
    ax.set_xticklabels(list(xlabels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(list(yticks))
    ax.set_yticklabels(list(ylabels))
    if len(yminor):
        ax.set_yticks(list(yminor), minor=True)
        ax.grid(which="minor", axis="y", color="black", linewidth=0.6, linestyle=(0, (1.2, 1.2)))
    ax.tick_params(axis="both", which="major", length=TICK_LEN_PT, width=0.8, pad=TICK_PAD_PT,
                   color="black")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.grid(which="major", axis="both", color="black", linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")


def draw_trace(ax, x, y, frame, linestyle="-"):
    """Green trace with white square markers (green rims) spaced along the curve in page
    points; the opaque faces sit on top of the trace, cutting it at each marker.

    ``linestyle`` is the matplotlib line style of the trace itself
    (``'-'``, ``'--'``, ``'-.'``, ``':'``, ``'solid'``, ``'dashed'``, …);
    the markers are always drawn solid."""
    ax.plot(x, y, color=GREEN, linewidth=2.6, solid_joinstyle="round",
            linestyle=linestyle, zorder=4)
    mx, my = marker_positions(x, y, frame)
    ax.plot(mx, my, linestyle="none", marker="s", markersize=4.4, markeredgewidth=1.1,
            markerfacecolor="white", markeredgecolor=GREEN, zorder=4)

def add_footer(fig, trace_name, axis_label, ax=None):
    """Hollow-square legend at bottom-left and the axis label centred underneath.

    The legend is anchored at the left edge of the axes (the zero position of
    the x-axis) rather than a fixed page coordinate, so it always lines up with
    the y-axis.  If ``ax`` is not supplied the first axes on the figure is used.
    """
    if ax is None:
        axs = fig.get_axes()
        ax = axs[0] if axs else None
    x0 = ax.get_position().x0 if ax is not None else 0.0349

    y = 0.0314
    fig.add_artist(Line2D([x0], [y], marker="s", markersize=4.4, markeredgewidth=1.1,
                          markerfacecolor="white", markeredgecolor=GREEN, linestyle="none",
                          transform=fig.transFigure))
    fig.text(x0 + 0.0084, y, trace_name, ha="left", va="center", color="black")
    fig.text(0.491, 0.0137, axis_label, ha="center", va="center", color="black")

def add_y_label(fig, label, ax=None, gap_pt=8.0):
    """Rotated y-axis title, vertically centred on the axes.

    With an axes available (either passed in or discovered from the figure) the
    title is placed by measuring where the tick labels really start (right edge
    of the title ``gap_pt`` points left of them), so it can never overlap them.
    """
    if ax is None:
        axs = fig.get_axes()
        ax = axs[0] if axs else None
    if ax is None:
        # Bare figure with no axes: fall back to the nominal page centre.
        y = AX_RECT[1] + AX_RECT[3] * 0.5
        fig.text((Y_LABEL_PAD_PT / 2.0 - 2.0) / FIG_W_PT, y, label,
                 rotation=90, ha="center", va="center", color="black")
        return

    pos = ax.get_position()
    y = pos.y0 + pos.height * 0.5
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = [t for t in ax.get_yticklabels() if t.get_text()]
    left_px = min(t.get_window_extent(renderer).x0 for t in ticks)
    x_pt = left_px * 72.0 / fig.dpi - gap_pt
    txt = fig.text(max(x_pt, 0.0) / FIG_W_PT, y, label, rotation=90,
                   ha="right", va="center", color="black")
    # keep it on the page if the margin is ever too small
    x0 = txt.get_window_extent(renderer).x0
    if x0 < 2.0 * fig.dpi / 72.0:
        txt.set_x(txt.get_position()[0] + (2.0 * fig.dpi / 72.0 - x0) / fig.dpi * 72.0 / FIG_W_PT)
