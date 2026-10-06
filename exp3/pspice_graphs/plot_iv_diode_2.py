#!/usr/bin/env python3
"""
Diode I-V characteristic zooms on the shared LTspice/Probe theme (ltspice_theme.py):
991 x 423 pt page, green trace with hollow square markers, solid black major grid, dotted
minor grid, legend and axis label underneath, serif description box, and cursor crosshairs
with "(x,y)" read-outs joined by a dV/dI right triangle.

Reads the exported simulation data (columns: V1, V(n002), I(D1)) and creates
  1) diode_forward_zoom.png : forward-bias zoom, knee voltage + 2 read-out points
  2) diode_reverse_zoom.png : reverse-bias zoom, breakdown voltage + 2 read-out points

V(n002) is the voltage across the diode (V1 sits behind a 1k series resistor).

Knee voltage      : x-intercept of the straight line fitted to the steep forward region
                    (standard tangent / linear-extrapolation method).
Breakdown voltage : same method on the steep reverse region.

Usage:
    python plot_iv_diode_2.py [data_file] [--out folder] [--seed N]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ltspice_theme import (NOTE_EDGE, add_footer, add_note, choose_font, dense_curve,
                           draw_trace, load_table, major_ticks, minor_between, new_page,
                           qty_label, spice_format, style_axes, text_width)

NOTE = "Current through diode"


# ----------------------------------------------------------------- helpers
def mark_point(ax, frame, x0, y0, label, fsize, dx=0.0, dy=0.0, ha="left", va="bottom"):
    """Theme cursor: thin black crosshair and a grey-bordered '(x,y)' read-out box.
    Returns the box rectangle in page points (for the note-box collision test)."""
    xmin, ymin = ax.get_xlim()[0], ax.get_ylim()[0]
    ax.plot([xmin, x0], [y0, y0], color="black", linewidth=0.8, zorder=2)
    ax.plot([x0, x0], [ymin, y0], color="black", linewidth=0.55, zorder=2)
    readout = f"({spice_format(x0)},{spice_format(y0)})"
    if label:
        readout = f"{label}\n{readout}"
    ax.annotate(readout, xy=(x0, y0), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, ma="left", fontsize=fsize, color="black", zorder=7,
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=NOTE_EDGE, lw=1.0))

    lines = readout.split("\n")
    w = max(text_width(ln, fsize) for ln in lines)
    h = 1.2 * fsize * len(lines)
    pad = 0.3 * fsize                                   # boxstyle pad, in points
    px, py = (float(c) for c in frame.to_pt(x0, y0))
    bx0, bx1 = ((px + dx - pad, px + dx + w + pad) if ha == "left"
                else (px + dx - w - pad, px + dx + pad))
    by0, by1 = ((py + dy - pad, py + dy + h + pad) if va == "bottom"
                else (py + dy - h - pad, py + dy + pad))
    return (bx0, by0, bx1, by1)


def tangent_intercept(v, i, mask):
    """x-intercept (I = 0) of the least-squares line through v[mask], i[mask]."""
    slope, icpt = np.polyfit(v[mask], i[mask], 1)
    return -icpt / slope, slope, icpt


def pick_random(v, i, lo, hi, k, rng, min_sep):
    """k random data points with lo <= v <= hi, at least min_sep apart in voltage."""
    cand = np.where((v >= lo) & (v <= hi))[0]
    if cand.size == 0:
        return []
    chosen = []
    for _ in range(1000):
        j = int(rng.choice(cand))
        if all(abs(v[j] - v[c]) >= min_sep for c in chosen):
            chosen.append(j)
        if len(chosen) == k:
            break
    return sorted(chosen, key=lambda j: v[j])


def draw_rtriangle(ax, xa, ya, xb, yb):
    """Triangle outline joining the two read-out points: axis-aligned legs (dV horizontal,
    dI vertical) meeting at the corner where the two crosshairs intersect, closed by the
    chord between the points (the dynamic-resistance construction). Drawn above the trace
    so the chord stays visible where it rides the nearly straight breakdown curve."""
    x1, x2 = sorted((xa, xb))
    y1, y2 = (ya, yb) if xa <= xb else (yb, ya)
    ax.plot([x1, x2, x2, x1], [y1, y1, y2, y1], color="black", linewidth=1.2, zorder=5)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Diode forward/reverse zoom plots, LTspice theme")
    ap.add_argument("data_file", nargs="?", default="Draft1.txt",
                    help="tab-separated export with columns V1, V(n002), I(D1)")
    ap.add_argument("--out", default=".", help="output folder (default: current folder)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the random points so runs are repeatable")
    ap.add_argument("--dpi", type=int, default=260, help="output resolution (default: 260)")
    ap.add_argument("--note", default=None, help="text of the description box")
    ap.add_argument("--no-note", action="store_true", help="do not draw the description box")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    fsize = choose_font()[1]

    header, data = load_table(a.data_file)
    vcol = "V(n002)" if "V(n002)" in header else header[0]     # diode voltage
    icol = "I(D1)" if "I(D1)" in header else header[-1]
    order = np.argsort(data[:, header.index(vcol)])
    v = data[order, header.index(vcol)]
    i = data[order, header.index(icol)]
    note = NOTE if a.note is None and not a.no_note else a.note

    # ======================= 1) FORWARD BIAS ==============================
    fwd = v > 0
    knee_mask = fwd & (i > 0.2 * i[fwd].max()) & (i < 0.9 * i[fwd].max())
    v_knee, slope_f, icpt_f = tangent_intercept(v, i, knee_mask)

    xlo, xhi, ylo, yhi = 0.3, 0.85, -10e-3, 100e-3
    xticks = major_ticks(xlo, xhi, 0.05)
    yticks = major_ticks(ylo, yhi, 10e-3)
    fig, ax, frame = new_page((xlo, xhi), (ylo, yhi), [qty_label(y, "A") for y in yticks])
    style_axes(ax, xticks, [qty_label(x, "V") for x in xticks],
               yticks, [qty_label(y, "A") for y in yticks], minor_between(yticks, 2))

    m = (v >= xlo) & (v <= xhi) & (i >= ylo) & (i <= yhi)
    curve = dense_curve(v[m], i[m], frame)
    draw_trace(ax, v[m], i[m], frame)

    # extrapolated tangent used to locate the knee
    xt = np.array([v_knee, 0.78])
    # ax.plot(xt, slope_f * xt + icpt_f, color="#555555", lw=1.0, ls="-.", zorder=2)

    # two random points (one below, one above the knee), joined by a dV/dI triangle
    p1 = pick_random(v, i, 0.45, 0.62, 1, rng, 0.0)
    p2 = pick_random(v, i, 0.74, 0.82, 1, rng, 0.0)
    if p1 and p2:
        draw_rtriangle(ax, v[p1[0]], i[p1[0]], v[p2[0]], i[p2[0]])
    rects = [mark_point(ax, frame, v[j], i[j], "", fsize, dx=-6, dy=6, ha="right", va="bottom")
             for j in p1 + p2]

    # knee voltage (on the curve at that voltage, I ~ 0 on the tangent axis)
    i_knee = float(np.interp(v_knee, v, i))   # point on the curve
    rects.append(mark_point(ax, frame, v_knee, i_knee, "Knee", fsize,
                            dx=-6, dy=6, ha="right", va="bottom"))

    add_footer(fig, icol, "V1(D1)")
    if note:
        add_note(ax, note, frame, curve, rects)

    f1 = os.path.join(a.out, "diode_forward_zoom.png")
    fig.savefig(f1, dpi=a.dpi, facecolor="white")
    plt.close(fig)
    print(f"Knee voltage (tangent method): {v_knee:.4f} V   -> {f1}")

    # ======================= 2) REVERSE BIAS ==============================
    rev = v < 0
    bd_mask = rev & (i < -0.2 * abs(i[rev]).max()) & (i > -0.9 * abs(i[rev]).max())
    v_bd, slope_r, icpt_r = tangent_intercept(v, i, bd_mask)

    xlo, xhi, ylo, yhi = -100.5, -99.5, -100e-3, 10e-3
    xticks = major_ticks(xlo, xhi, 0.1)
    yticks = major_ticks(ylo, yhi, 10e-3)
    fig, ax, frame = new_page((xlo, xhi), (ylo, yhi), [qty_label(y, "A") for y in yticks])
    style_axes(ax, xticks, [qty_label(x, "V") for x in xticks],
               yticks, [qty_label(y, "A") for y in yticks], minor_between(yticks, 2))

    m = (v >= xlo) & (v <= xhi) & (i >= ylo) & (i <= yhi)
    curve = dense_curve(v[m], i[m], frame)
    draw_trace(ax, v[m], i[m], frame)

    xt = np.array([v_bd, -100.22])
    # ax.plot(xt, slope_r * xt + icpt_r, color="#555555", lw=1.0, ls="-.", zorder=2)

    # two random points on the steep breakdown region, joined by a dV/dI triangle
    r1 = pick_random(v, i, -100.22, -100.12, 1, rng, 0.0)
    r2 = pick_random(v, i, -100.23, -100.225, 1, rng, 0.0)
    if r1 and r2:
        draw_rtriangle(ax, v[r1[0]], i[r1[0]], v[r2[0]], i[r2[0]])
    rects = [mark_point(ax, frame, v[j], i[j], "", fsize, dx=6, dy=-6, ha="left", va="top")
             for j in r1 + r2]

    i_bd = float(np.interp(v_bd, v, i))       # point on the curve
    rects.append(mark_point(ax, frame, v_bd, i_bd, "V(BR)", fsize,
                            dx=6, dy=-6, ha="left", va="top"))

    add_footer(fig, icol, "V1(D1)")
    if note:
        add_note(ax, note, frame, curve, rects)

    f2 = os.path.join(a.out, "diode_reverse_zoom.png")
    fig.savefig(f2, dpi=a.dpi, facecolor="white")
    plt.close(fig)
    print(f"Reverse breakdown voltage (tangent method): {v_bd:.4f} V   -> {f2}")


if __name__ == "__main__":
    main()
