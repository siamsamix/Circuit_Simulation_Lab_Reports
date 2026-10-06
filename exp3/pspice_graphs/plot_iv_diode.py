#!/usr/bin/env python3
"""
Diode I-V characteristic on the shared LTspice/Probe theme (ltspice_theme.py): 991 x 423 pt
page, green trace with hollow square markers spaced along the curve, solid black major grid,
dotted minor grid, legend and axis label underneath, serif description box.

Plots I(D1) versus V(n002) from the exported sweep (columns: V1, V(n002), I(D1));
V(n002) is the voltage across the diode (V1 sits behind a 1k series resistor).

Usage:
    python plot_iv_diode.py [data_file] [output_image] [--no-show]
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ltspice_theme import (add_footer, add_note, dense_curve, draw_trace, load_table,
                           major_ticks, new_page, qty_label, style_axes)

XSPAN = (-120.0, 20.0)          # diode-voltage window, V
XSTEP = 20.0                    # major tick step, V
YSPAN = (-0.1, 0.1)             # diode-current window, A
YSTEP = 0.05                    # major tick step, A
NOTE = "Current through diode"


def main():
    ap = argparse.ArgumentParser(description="Diode I-V characteristic, LTspice theme")
    ap.add_argument("data_file", nargs="?", default="Draft1.txt",
                    help="tab-separated export with columns V1, V(n002), I(D1)")
    ap.add_argument("output", nargs="?", default="iv_diode.png", help="image file to write")
    ap.add_argument("--dpi", type=int, default=260, help="output resolution (default: 260)")
    ap.add_argument("--note", default=None, help="text of the description box")
    ap.add_argument("--no-note", action="store_true", help="do not draw the description box")
    ap.add_argument("--no-show", action="store_true", help="save only, do not open a window")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    header, data = load_table(args.data_file)
    vcol = "V(n002)" if "V(n002)" in header else header[0]     # diode voltage
    icol = "I(D1)" if "I(D1)" in header else header[-1]
    order = np.argsort(data[:, header.index(vcol)])
    v = data[order, header.index(vcol)]
    i = data[order, header.index(icol)]

    xticks = major_ticks(*XSPAN, XSTEP)
    yticks = major_ticks(*YSPAN, YSTEP)
    yminor = [y for y in np.arange(YSPAN[0], YSPAN[1] + 1e-9, (YSPAN[1] - YSPAN[0]) / 10)
              if not any(np.isclose(y, yticks))]

    fig, ax, frame = new_page(XSPAN, YSPAN, [qty_label(y, "A") for y in yticks])
    style_axes(ax, xticks, [qty_label(x, "V") for x in xticks],
               yticks, [qty_label(y, "A") for y in yticks], yminor)

    m = (v >= XSPAN[0]) & (v <= XSPAN[1]) & (i >= YSPAN[0]) & (i <= YSPAN[1])
    curve = dense_curve(v[m], i[m], frame)
    draw_trace(ax, v[m], i[m], frame)

    add_footer(fig, icol, "V1(D1)")
    note = NOTE if args.note is None and not args.no_note else args.note
    if note:
        add_note(ax, note, frame, curve, [])

    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    print(f"saved {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
