#!/usr/bin/env python3
"""
PSpice/Probe-style transient plots for the full-wave bridge rectifier (exp4).

One PNG per data file: the voltage difference across R1, V(R1:1,R1:2), drawn in
green with white square markers (green rims) over the source voltage V(1) in
violet with white triangle markers (violet rims), both on the shared
pspice_theme page (solid black major grid, dotted minor grid on y, PSpice-style
tick labels, hollow-marker legend and centred axis label underneath the axes).
Framed serif notes — "Voltage across resistor" on the green trace, "Vin" on the
violet one — are dodged clear of both curves.

Usage : python plot_full_bridge.py [--dpi 260]
Output: full_bridge.png, full_bridge_with_capacitor.png (next to this script)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from pspice_theme import (AX_H_PT, AX_RECT, FIG_W_PT, GREEN, MARKER_FIRST_PT,
                          MARKER_SPACING_PT, NOTE_EDGE, NOTE_SERIF, choose_font,
                          dense_curve, label_widths_pt, load_table, major_ticks,
                          minor_between, new_page, qty_label, rect_is_free,
                          style_axes, text_width)

PURPLE = "#5400ff"                    # second Probe trace colour (as in the reference)

VDIFF, VIN = "V(R1:1,R1:2)", "V(1)"   # column names in the LTspice exports
NOTE_VDIFF = "Voltage across resistor"
NOTE_VIN = "Vin"
X_LABEL = "Time (ms)"

# preferred note spots (axes-fraction x, y, force): "Voltage across resistor" is pinned
# onto a feature of the green trace (overlap allowed), "Vin" anchors inside a sine
# trough; unless force is set, the search only moves a note if a curve is in the way
PLOTS = (
    ("full_bridge.txt", "full_bridge.png",
     {NOTE_VDIFF: (0.40, 0.873, True), NOTE_VIN: (0.30, 0.182, False)}, 0),
    ("full_bridge_with_capacitor.txt", "full_bridge_with_capacitor.png",
     {NOTE_VDIFF: (0.30, 0.84, False), NOTE_VIN: (0.52, 0.72, False)}, 200e-3),
    ("full_bridge_with_capacitor_changed.txt", "full_bridge_with_capacitor_changed.png",
     {NOTE_VDIFF: (0.30, 0.84, False), NOTE_VIN: (0.52, 0.72, False)}, 200e-3),
)


# --------------------------------------------------------------------------- #
# Time-axis helpers (as in the exp3 transient script)
# --------------------------------------------------------------------------- #
X_NICE = (1, 2, 5, 10)


def time_ticks(tmin, tmax, target=6):
    """Evenly spaced 1-2-5 tick positions from tmin to tmax (roughly `target` intervals)."""
    raw = (tmax - tmin) / target
    base = 10.0 ** np.floor(np.log10(raw))
    step = next(m * base for m in X_NICE if m * base >= raw * (1 - 1e-9))
    first = np.ceil(tmin / step - 1e-9) * step
    n = int(np.floor((tmax - first) / step + 1e-9)) + 1
    return [round(first + k * step, 15) for k in range(n)]


def time_label(value, unit_scale):
    """Tick label in Probe style, all ticks sharing one unit: 0ms, 10ms, 20ms, ..."""
    scale, suffix = unit_scale
    return f"{round(value / scale, 6):g}{suffix}"


def pick_time_unit(tmax):
    for scale, suffix in ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns")):
        if tmax >= scale:
            return scale, suffix
    return 1e-9, "ns"


# --------------------------------------------------------------------------- #
# Traces, notes, legend
# --------------------------------------------------------------------------- #
def marker_positions(t, y, frame, first_pt):
    """Points spaced at a constant on-screen distance along the plotted curve."""
    x_pt, y_pt = frame.to_pt(t, y)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_pt), np.diff(y_pt)))])
    marks = np.arange(first_pt, s[-1], MARKER_SPACING_PT)
    return np.interp(marks, s, t), np.interp(marks, s, y)


def draw_trace(ax, t, v, frame, color, marker, zorder, first_pt):
    """Solid trace with hollow white markers (coloured rims) spaced along the curve."""
    ax.plot(t, v, color=color, linewidth=2.6, solid_joinstyle="round", zorder=zorder)
    mx, my = marker_positions(t, v, frame, first_pt)
    ax.plot(mx, my, linestyle="none", marker=marker, markersize=4.4, markeredgewidth=1.1,
            markerfacecolor="white", markeredgecolor=color, zorder=zorder)


def note_fontsize(prop):
    """Font size that spans the reference width, exactly as the theme's add_note."""
    return 8.0 * (0.2256 * AX_RECT[2] * FIG_W_PT) / text_width("Voltage across capacitor", 10, prop)


def add_note_at(ax, note, frame, curve_pts, avoid_rects, prefer, fsize, prop, force=False):
    """White thin-framed serif note box at `prefer`; moved to the nearest free spot unless
    force is set (overlap with the traces allowed)."""
    box_w = text_width(note, fsize, prop) + 2 * 0.0140 * AX_RECT[2] * FIG_W_PT
    box_h = 0.0925 * AX_H_PT
    m = 3.0

    def free(cx, cy):
        return rect_is_free((cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2),
                            curve_pts, avoid_rects, m, frame)

    cx, cy = prefer[0] * frame.w, prefer[1] * frame.h
    if not force and not free(cx, cy):
        placed = False
        xs = np.arange(box_w / 2 + m, frame.w - box_w / 2 - m + 1e-9, 3.0)
        ys = np.arange(box_h / 2 + m, frame.h - box_h / 2 - m + 1e-9, 3.0)
        gx, gy = np.meshgrid(xs, ys)
        for i in np.argsort(np.hypot(gx - cx, gy - cy), axis=None):
            if free(gx.flat[i], gy.flat[i]):
                cx, cy = gx.flat[i], gy.flat[i]
                placed = True
                break
        if not placed:
            print(f"  warning: no free spot for note {note!r}; keeping the preferred one")

    fx, fy, fw, fh = cx / frame.w, cy / frame.h, box_w / frame.w, box_h / frame.h
    ax.add_patch(FancyBboxPatch((fx - fw / 2, fy - fh / 2), fw, fh,
                                boxstyle=f"round,pad=0,rounding_size={2.0 / frame.w}",
                                mutation_aspect=frame.w / frame.h, transform=ax.transAxes,
                                facecolor="white", edgecolor=NOTE_EDGE, linewidth=1.0,
                                zorder=10, clip_on=False))
    ax.text(fx, fy, note, transform=ax.transAxes, ha="center", va="center_baseline",
            fontproperties=prop, fontsize=fsize, color="black", zorder=11)

    rect = (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2)
    to_data = lambda px, py: (frame.xmin + px / frame.w * (frame.xmax - frame.xmin),
                              frame.ymin + py / frame.h * (frame.ymax - frame.ymin))
    x_d, y_d = to_data(cx, cy)
    print(f"  note {note!r} centred at ({x_d * 1e3:.1f} ms, {y_d:.1f} V)")
    return rect


def add_legend_footer(fig, entries, axis_label):
    """Hollow-marker legend row bottom-left and the axis label centred underneath."""
    fsize = choose_font()[1]
    widths = label_widths_pt([name for _, _, name in entries], fsize)
    x = 0.0349
    for (marker, edge, name), w in zip(entries, widths):
        fig.add_artist(Line2D([x], [0.0314], marker=marker, markersize=4.4, markeredgewidth=1.1,
                              markerfacecolor="white", markeredgecolor=edge, linestyle="none",
                              transform=fig.transFigure))
        fig.text(x + 0.0084, 0.0314, name, ha="left", va="center", color="black")
        x += 0.0084 + w / FIG_W_PT + 0.020
    fig.text(0.491, 0.0137, axis_label, ha="center", va="center", color="black")


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def make_figure(datafile, output, prefs, dpi, cut):
    header, data = load_table(datafile)
    data = data[~np.isnan(data).any(axis=1)]   # drop truncated trailing rows
    names = [h.lower() for h in header]
    try:
        c_vdiff, c_vin = names.index(VDIFF.lower()), names.index(VIN.lower())
    except ValueError:
        raise SystemExit(f"expected columns {VDIFF} and {VIN}; found: {', '.join(header)}")

    t, v_diff, v_in = data[:, 0], data[:, c_vdiff], data[:, c_vin]
    tmin, tmax = float(t.min()), float(t.max()-cut)

    # ---- axes: symmetric y around 0 with ~10% headroom (110 V for the 100 V
    # source, 20 V for the 10 V one); major step 50/10/5 V so ~9 labels fit ----
    span = max(float(np.abs(v_diff).max()), float(np.abs(v_in).max()))
    lim = float(np.ceil(span * 1.1 / 10.0) * 10.0)
    y_step = 50.0 if lim >= 75.0 else (10.0 if lim >= 30.0 else 5.0)
    y_majors = major_ticks(-lim, lim, y_step)
    y_minors = minor_between(y_majors, 5)
    y_labels = [qty_label(v, "V") for v in y_majors]

    tunit = pick_time_unit(tmax)
    xticks = time_ticks(tmin, tmax)
    xlabels = [time_label(x, tunit) for x in xticks]

    fig, ax, frame = new_page((tmin, tmax), (-lim, lim), y_labels)
    style_axes(ax, xticks, xlabels, y_majors, y_labels, y_minors)

    # ---- traces: V(1) underneath, the voltage difference on top --------------------
    draw_trace(ax, t, v_in, frame, PURPLE, "v", 3, MARKER_FIRST_PT + MARKER_SPACING_PT / 2)
    draw_trace(ax, t, v_diff, frame, GREEN, "s", 4, MARKER_FIRST_PT)
    print(f"{Path(datafile).name}: {VDIFF} {v_diff.min():.4g}..{v_diff.max():.4g} V, "
          f"{VIN} {v_in.min():.4g}..{v_in.max():.4g} V, y-axis +/-{lim:g} V")

    # ---- notes dodged clear of both curves ------------------------------------------
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    prop = FontProperties(family=next((f for f in NOTE_SERIF if f in installed), "serif"))
    fsize = note_fontsize(prop)
    curve_g = dense_curve(t, v_diff, frame)
    curve_p = dense_curve(t, v_in, frame)
    both = (np.concatenate([curve_g[0], curve_p[0]]), np.concatenate([curve_g[1], curve_p[1]]))
    fx, fy, force = prefs[NOTE_VDIFF]
    rect = add_note_at(ax, NOTE_VDIFF, frame, both, [], (fx, fy), fsize, prop, force=force)
    fx, fy, force = prefs[NOTE_VIN]
    add_note_at(ax, NOTE_VIN, frame, both, [rect], (fx, fy), fsize, prop, force=force)

    add_legend_footer(fig, [("s", GREEN, VDIFF), ("v", PURPLE, VIN)], X_LABEL)

    fig.savefig(output, dpi=dpi, facecolor="white")
    print(f"  saved {output}")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PSpice-style full-wave bridge rectifier plots")
    parser.add_argument("--dpi", type=int, default=260, help="output resolution (default: 260)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    for datafile, output, prefs, cut in PLOTS:
        make_figure(str(here / datafile), str(here / output), prefs, args.dpi, cut)


if __name__ == "__main__":
    main()
