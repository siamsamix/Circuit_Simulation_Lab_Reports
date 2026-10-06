#!/usr/bin/env python3
"""
LTspice-style TRANSIENT plot: a voltage or current trace (y-axis) versus time (x-axis).

Same look as plot_spice_filter.py (theme shared via ltspice_theme.py): green trace with
hollow square markers, black solid major grid, dotted minor grid, legend and axis label
underneath the plot, LTspice-style tick labels ("0V", "5.0V", "10V", "0ms", "10ms", ...)
and cursor read-outs such as "(5.0019m,9.31343)".

Input : a text file exported from LTspice ("Export data as text") with the layout

            time<TAB>V(1)<TAB>V(2)
            0.000000e+00<TAB>0.000000e+00<TAB>0.000000e+00
            ...

Output: one PNG per run (the data trace is drawn as-is; the axes follow the data like LTspice).
        --column picks the trace: voltage columns get volt-style tick labels ("0V", "5.0V"),
        current columns I(...) get ampere-style ones ("0A", "5mA", "10mA").
        The positive peak of the trace is found automatically and marked with cursor lines and a
        read-out.  Transient plots carry an automatic framed description ("Voltage across
        resistor" or "Current through resistor", matching the plotted quantity); --note replaces
        its text and --no-note removes the box.
        For transient exports (x-axis "time") a rectifier performance metrics table is printed to
        the terminal, averaged over whole cycles: peak/average/RMS output voltage and current,
        DC and AC load power, rectification efficiency, ripple factor, form factor, peak inverse
        voltage, peak factor and output frequency.  The table always describes the rectifier
        output: its voltage column defaults to the plotted column, or to the highest-numbered
        V(n) node when plotting a current (--voltage overrides); the load-current column is
        auto-detected (prefers I(RL), --current overrides) and the remaining V(n) node provides
        the PIV.

Usage : python plot_spice_transient.py Draft3.txt [-o output.png] [--column V(2)] [--no-show]
                                       [--label "V(2)"] [--note "Half-wave output"] [--no-peak]
                                       [--no-note] [--current I(RL)] [--voltage V(2)]
"""

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ltspice_theme import (NOTE_EDGE, READOUT_PAD_PT, add_footer, add_note, choose_font,
                           dense_curve, draw_trace, new_page, place_readout, qty_label,
                           spice_format, style_axes, text_width)

# --------------------------------------------------------------------------- #
# Nice-step tables for the auto-fitted axes
# --------------------------------------------------------------------------- #
Y_NICE = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)
X_NICE = (1, 2, 5, 10)                           # tick steps for the time axis


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
def load_transient(path, column="V(2)"):
    """Return (header, data, column_index) from an LTspice transient text export."""
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    header = re.split(r"\s+", lines[0])
    names = [h.lower() for h in header]
    if column.lower() not in names:
        raise SystemExit(f"column {column!r} not found; available: {', '.join(header)}")
    col = names.index(column.lower())

    rows = [[float(tok) for tok in re.split(r"[\s,]+", ln)] for ln in lines[1:]]
    return header, np.array(rows), col


def column_unit(name):
    """Plot unit for a column name: currents I(...) in amperes, everything else in volts."""
    return "A" if name.lower().startswith("i(") else "V"


# --------------------------------------------------------------------------- #
# Rectifier performance metrics
# --------------------------------------------------------------------------- #
try:
    _trapz = np.trapezoid                          # numpy 2.x renamed trapz -> trapezoid
except AttributeError:
    _trapz = np.trapz


def upward_crossings(t, y):
    """Times where y rises through zero, interpolated onto y = 0."""
    i = np.where((y[:-1] <= 0) & (y[1:] > 0))[0]
    if len(i) == 0:
        return np.array([])
    frac = -y[i] / (y[i + 1] - y[i])
    return t[i] + frac * (t[i + 1] - t[i])


def cycle_bounds(t, y):
    """(first crossing, last closing crossing, mean period, cycle count) from upward crossings."""
    x = upward_crossings(t, y)
    if len(x) < 2:
        return None
    period = (x[-1] - x[0]) / (len(x) - 1)
    n = int(np.floor((t[-1] - x[0]) / period + 1e-6))   # whole cycles that fit in the record
    if n < 1:
        return None
    return x[0], x[0] + n * period, period, n


def output_frequency(t, y):
    """Dominant frequency of the output: median spacing of its significant local maxima."""
    thr = 0.05 * (float(y.max()) - float(y.min()))      # ignore wiggles in the flat off-region
    i = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > thr))[0] + 1
    if len(i) < 2:
        return None
    spacing = np.diff(t[i])
    spacing = spacing[spacing > 0]
    if len(spacing) == 0:
        return None
    return 1.0 / float(np.median(spacing))


def pick_output_voltage(header):
    """Best-guess output-voltage column when plotting a current: the highest-numbered V(n)
    node (the driven node is usually V(1), the tapped output V(2))."""
    best, best_n = None, -1
    for i, h in enumerate(header):
        m = re.fullmatch(r"v\((\d+)\)", h.strip().lower())
        if m and int(m.group(1)) > best_n:
            best, best_n = i, int(m.group(1))
    return best


def pick_metric_columns(header, out_col):
    """Indices of the load-current and source-voltage columns for the metrics table."""
    lower = [h.lower() for h in header]
    currents = [i for i, n in enumerate(lower) if n.startswith("i(")]
    cur = next((i for i in currents if "rl" in lower[i]), None)
    if cur is None:
        cur = next((i for i in currents
                    if not any(s in lower[i] for s in ("vin", "v1", "vsource"))), None)
    if cur is None and currents:
        cur = currents[0]
    vin = next((i for i, n in enumerate(lower)
                if n.startswith("v(") and i != out_col), None)
    return cur, vin


def print_metrics(header, data, out_col, cur_col, in_col):
    """Rectifier performance table, averaged over whole cycles of the source waveform."""
    t_all = data[:, 0]
    v_out = data[:, out_col].astype(float)

    bounds = cycle_bounds(t_all, data[:, in_col].astype(float)) if in_col is not None else None
    if bounds is None:
        bounds = cycle_bounds(t_all, v_out)
    mask = np.ones(len(t_all), dtype=bool)
    cycles = period = None
    if bounds is not None:
        lo, hi, period, cycles = bounds
        mask = (t_all >= lo) & (t_all <= hi)
    t, v = t_all[mask], v_out[mask]
    # LTspice exports are non-uniformly sampled (waveform compression), so averages must be
    # time-weighted; divide by the nominal window length (whole cycles) rather than t[-1].
    span = (period * cycles if cycles is not None and period is not None
            else float(t[-1] - t[0])) if len(t) > 1 else None

    def time_avg(y):
        return float(_trapz(y, t) / span)

    v_m = float(v.max())
    v_dc = time_avg(v)
    v_rms = float(np.sqrt(time_avg(v ** 2)))

    i_m = i_dc = i_rms = None
    flipped = False
    if cur_col is not None and span:
        i = data[mask, cur_col].astype(float)
        i_dc = time_avg(i)
        if v_dc >= 0 and i_dc < 0:              # LTspice pin-order sign convention
            i, i_dc, flipped = -i, -i_dc, True
        i_m = float(i.max())
        i_rms = float(np.sqrt(time_avg(i ** 2)))

    p_dc = p_ac = eta = None
    if i_dc is not None:
        p_dc, p_ac = v_dc * i_dc, v_rms * i_rms
        if p_ac > 0:
            eta = 100.0 * p_dc / p_ac

    v_ripple = np.sqrt(max(v_rms ** 2 - v_dc ** 2, 0.0))
    r = v_ripple / v_dc if abs(v_dc) > 1e-12 else None
    ff = v_rms / v_dc if abs(v_dc) > 1e-12 else None
    pf = v_m / v_rms if v_rms > 0 else None
    piv = float(np.max(np.abs(data[mask, in_col] - v))) if in_col is not None else None
    f_o = output_frequency(t, v)

    rows = [
        ("Peak output voltage", "V_m", v_m, "V"),
        ("Peak output current", "I_m", i_m, "A"),
        ("Average output voltage", "V_DC", v_dc, "V"),
        ("Average output current", "I_DC", i_dc, "A"),
        ("RMS output voltage", "V_RMS", v_rms, "V"),
        ("RMS output current", "I_RMS", i_rms, "A"),
        ("DC load power", "P_DC", p_dc, "W"),
        ("AC load power", "P_AC", p_ac, "W"),
        ("Rectification efficiency", "η", eta, "%"),
        ("Ripple factor", "r", r, ""),
        ("Form factor", "FF", ff, ""),
        ("Peak inverse voltage", "PIV", piv, "V"),
        ("Peak factor", "PF", pf, ""),
        ("Output frequency", "f_o", f_o, "Hz"),
    ]

    def cell(value, unit):
        if value is None:
            return "n/a"
        s = spice_format(value)
        if unit and s[-1].isalpha():                    # already carries an SI suffix
            return s + unit
        return s + (f" {unit}" if unit else "")

    w_desc = max(len(d) for d, *_ in rows)
    w_sym = max(len(s) for _, s, *_ in rows)
    what = f"output {header[out_col]}"
    load = f", load current {header[cur_col]}" if cur_col is not None else ""
    src = f", source {header[in_col]}" if in_col is not None else ""
    if cycles:
        scope = (f"{cycles} complete cycles ({t[0] * 1e3:g}–{t[-1] * 1e3:g} ms), "
                 f"T = {period * 1e3:g} ms")
    else:
        scope = "whole record (no complete cycles found)"
    print(f"\nRectifier performance metrics — {what}{load}{src}; {scope}")
    if flipped:
        print(f"  (sign of {header[cur_col]} flipped: LTspice measured it against the "
              f"load-current direction)")
    for d, s, val, unit in rows:
        print(f"  {d:<{w_desc}}  {s:>{w_sym}} = {cell(val, unit)}")


# --------------------------------------------------------------------------- #
# Axis helpers (transient-specific auto-fitting)
# --------------------------------------------------------------------------- #
def nice_ceiling(value):
    """Smallest 'round' number (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^n) not below value."""
    base = 10.0 ** np.floor(np.log10(value))
    for m in Y_NICE:
        if m * base >= value * (1 - 1e-6):
            return m * base
    return 10.0 * base


def time_ticks(tmin, tmax, target=6):
    """Evenly spaced 1-2-5 tick positions from tmin to tmax (roughly `target` intervals)."""
    raw = (tmax - tmin) / target
    base = 10.0 ** np.floor(np.log10(raw))
    step = next(m * base for m in X_NICE if m * base >= raw * (1 - 1e-9))
    first = np.ceil(tmin / step - 1e-9) * step
    n = int(np.floor((tmax - first) / step + 1e-9)) + 1
    return [round(first + k * step, 15) for k in range(n)]


def time_label(value, unit_scale):
    """Tick label in LTspice style, all ticks sharing one unit: 0ms, 10ms, 20ms, ..."""
    scale, suffix = unit_scale
    return f"{round(value / scale, 6):g}{suffix}"


def pick_time_unit(tmax):
    for scale, suffix in ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns")):
        if tmax >= scale:
            return scale, suffix
    return 1e-9, "ns"


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_figure(t, v, trace_name="V(2)", note=None, show_peak=True, unit=None):
    unit = unit or column_unit(trace_name)
    tmin, tmax = float(t.min()), float(t.max())

    # ---- y-axis: 0 .. round top; symmetric if the signal goes clearly negative ----
    top = nice_ceiling(max(abs(v.max()), abs(v.min())))
    # LTspice compression noise dips ~1% below zero on flat stretches; only flip the axis
    # for swings past 2% so such wiggles cannot force a symmetric scale
    bottom = 0.0 if v.min() > -0.02 * top else -top
    y_major = [bottom, top] if bottom == 0.0 else [bottom, 0.0, top]
    if bottom == 0.0:
        y_major = [0.0, top / 2, top]
    y_minor = [v_ for v_ in np.arange(bottom, top + 1e-9, (top - bottom) / 10)
               if not any(np.isclose(v_, y_major))]
    y_labels = [qty_label(y_, unit) for y_ in y_major]

    # ---- x-axis ticks -------------------------------------------------------------
    tunit = pick_time_unit(tmax)
    ticks = time_ticks(tmin, tmax)
    x_labels = [time_label(x, tunit) for x in ticks]

    fig, ax, frame = new_page((tmin, tmax), (bottom, top), y_labels)
    fsize = choose_font()[1]
    style_axes(ax, ticks, x_labels, sorted(y_major),
               [qty_label(y_, unit) for y_ in sorted(y_major)], sorted(y_minor))
    curve = dense_curve(t, v, frame)

    # ---- peak cursor: lines + read-out ----------------------------------------------
    avoid = []
    if show_peak:
        i = int(np.argmax(v))
        tp, vp = float(t[i]), float(v[i])
        # sub-sample refinement with a parabola through the peak sample and its neighbours
        if 0 < i < len(t) - 1:
            x3, y3 = t[i - 1:i + 2], v[i - 1:i + 2]
            c = np.polyfit(x3 - x3[1], y3, 2)
            if c[0] < 0:
                dt = -c[1] / (2 * c[0])
                if x3[0] - x3[1] <= dt <= x3[2] - x3[1]:
                    tp, vp = float(x3[1] + dt), float(np.polyval(c, dt))
        px, py = (float(a) for a in frame.to_pt(tp, vp))

        ax.plot([tmin, tp], [vp, vp], color="black", linewidth=0.8, zorder=2)
        ax.plot([tp, tp], [bottom, vp], color="black", linewidth=0.55, zorder=2)
        avoid += [(0.0, py - 0.5, px, py + 0.5), (px - 0.4, 0.0, px + 0.4, py)]

        readout = f"({spice_format(tp)},{spice_format(vp)})"
        tw = text_width(readout, fsize)
        dx, dy = place_readout(px, py, tw, fsize, frame)
        x0, y0 = px + dx, py + dy - fsize / 2
        pad = READOUT_PAD_PT
        avoid.append((x0 - pad, y0 - pad, x0 + tw + pad, y0 + fsize + pad))
        ax.add_patch(Rectangle(((x0 - pad) / frame.w, (y0 - pad) / frame.h),
                               (tw + 2 * pad) / frame.w, (fsize + 2 * pad) / frame.h,
                               transform=ax.transAxes, facecolor="white", edgecolor=NOTE_EDGE,
                               zorder=4.5, clip_on=False))
        ax.annotate(readout, xy=(tp, vp), xytext=(dx, dy), textcoords="offset points",
                    ha="left", va="center", color="black", zorder=5)
        print(f"peak of {trace_name}: {vp:.5f} {unit} at t = {tp * 1e3:.4f} ms")

    print(f"{trace_name}: min = {v.min():.6g} {unit}, max = {v.max():.6g} {unit}, "
          f"span = {tmin * 1e3:g} ms .. {tmax * 1e3:g} ms")

    # ---- data trace + footer ---------------------------------------------------------
    draw_trace(ax, t, v, frame)
    add_footer(fig, trace_name, "Time (s)")

    if note:
        add_note(ax, note, frame, curve, avoid)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="LTspice-style transient plot: voltage V(2) versus time")
    parser.add_argument("datafile", help="LTspice text export (time<TAB>V(1)<TAB>V(2))")
    parser.add_argument("-o", "--output", default=None,
                        help="image file to write (png/pdf/svg; default: <datafile>_V2.png)")
    parser.add_argument("--column", default="V(2)", help="column to plot (default: V(2))")
    parser.add_argument("--current", default=None,
                        help="current column for the metrics table "
                             "(default: auto-detect, prefers I(RL))")
    parser.add_argument("--voltage", default=None,
                        help="voltage column for the metrics table (default: the plotted "
                             "column, or the highest-numbered V(n) node when plotting a current)")
    parser.add_argument("--label", default=None, help="legend text (default: the column name)")
    parser.add_argument("--note", default=None,
                        help="text of the framed description box (default: 'Voltage across "
                             "resistor' or 'Current through resistor' for transient plots)")
    parser.add_argument("--no-note", action="store_true",
                        help="do not draw the framed description box")
    parser.add_argument("--no-peak", action="store_true",
                        help="do not mark the positive peak with cursor lines and a read-out")
    parser.add_argument("--dpi", type=int, default=260, help="output resolution (default: 260)")
    parser.add_argument("--no-show", action="store_true", help="save only, do not open a window")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    header, data, col = load_transient(args.datafile, args.column)
    name = header[col]
    unit = column_unit(name)
    if args.note:
        note = args.note
    elif not args.no_note and header[0].lower() in ("time", "t"):
        note = "Voltage across resistor" if unit == "V" else "Current through resistor"
    else:
        note = None
    fig = make_figure(data[:, 0], data[:, col], trace_name=args.label or name, note=note,
                      show_peak=not args.no_peak, unit=unit)

    output = args.output or f"{Path(args.datafile).stem}_{name.replace('(', '').replace(')', '')}.png"
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    print(f"saved {output}")

    if header[0].lower() in ("time", "t"):
        names = [h.lower() for h in header]
        vcol = col
        if args.voltage:
            if args.voltage.lower() not in names:
                raise SystemExit(f"column {args.voltage!r} not found; "
                                 f"available: {', '.join(header)}")
            vcol = names.index(args.voltage.lower())
        elif unit == "A":
            vcol = pick_output_voltage(header)
        if vcol is None:
            print("metrics table: skipped (no voltage column found)")
        else:
            cur_col, in_col = pick_metric_columns(header, vcol)
            if args.current:
                if args.current.lower() not in names:
                    raise SystemExit(f"column {args.current!r} not found; "
                                     f"available: {', '.join(header)}")
                cur_col = names.index(args.current.lower())
            print_metrics(header, data, vcol, cur_col, in_col)
    else:
        print("metrics table: skipped (x-axis is not time)")


if __name__ == "__main__":
    main()
