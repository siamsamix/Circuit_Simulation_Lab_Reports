#!/usr/bin/env python3
"""
Run an ngspice netlist and plot the results in PSpice/Probe style.

The .cir file is simulated with ngspice in batch mode (ngspice -b -r), the
binary raw file is parsed, optional expressions such as v(1)-v(2),
v(out)/v(in) or 20*log10(mag(v(out))) are evaluated on the waveforms, and all
requested traces are drawn on ONE shared pspice_theme page, styled like the
report figures made by plot_full_bridge.py / simulate_full_bridge.py:

  * trace 1 in green with white square markers, trace 2 in violet with white
    triangle markers (further traces: red diamonds, blue circles, ...), the
    first trace drawn on top and the markers staggered along the curves;
    line styles are solid by default and can be overridden per trace with
    --linestyle (e.g. --linestyle solid,dashed,dotted);
  * solid black major grid, dotted minor grid on y, PSpice-style tick labels
    ("0V", "5mA", "10ms"); a time axis uses one common unit for every tick
    and is titled "Time (ms)", the y axis "Voltage (V)" / "Current (A)";
  * hollow-marker legend with the centred x-axis title at the foot of the
    page and the rotated y-axis title on the left;
  * optional framed serif notes (--notes / --note / --title), attached
    per trace by position or by trace name, set in the trace colour and
    moved clear of every curve;
  * dashed black clip-level lines, with "+3.549 V"-style value labels and
    a dot at the plateau centre, drawn at the flat (clipped) max/min of
    every trace, as in the clipped-waveform report figures; plain sine
    crests are never flagged, and the marks can be turned off with
    --no-clip-levels.

Traces are plotted on linear axes. Without -x the first vector of the plot is
the x axis (time, frequency, or the swept source); without -y every real
vector is drawn (complex AC vectors have to be requested as mag(...),
phase(...), real(...) or imag(...), e.g. -y "mag(v(out))").

Output: one PNG per run, <fig-dir>/<circuit name>.png (default fig-dir
ngspice_figs/), or the file given with -o. The ngspice raw file and log live
in a temporary directory unless --keep puts them in ngspice_runs/<circuit>/.

Requires ngspice on PATH (tested with ngspice-42) plus matplotlib, with the
companion pspice_theme.py and plot_full_bridge.py modules beside this script.
"""

import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

from plot_full_bridge import (PURPLE, add_legend_footer, add_note_at, draw_trace,
                              note_fontsize, pick_time_unit, time_label,
                              time_ticks)
from pspice_theme import (GREEN, MARKER_FIRST_PT, MARKER_SPACING_PT, NOTE_SERIF,
                          add_y_label, dense_curve, major_ticks, minor_between,
                          new_page, qty_label, rect_is_free, spice_format,
                          style_axes, text_width)

# Trace colour and marker per curve: the first two are the reference figures'
# green squares and violet triangles; the rest are Probe-like extras.
TRACE_STYLES = ((GREEN, "s"), (PURPLE, "v"), ("#ff0000", "D"), ("#0000ff", "o"),
                ("#00aaaa", "^"), ("#aa5500", "p"))

# Preferred note spots (axes-fraction x, y), one per trace, as in the
# with-capacitor graphs of plot_full_bridge.py; add_note_at moves a note to
# the nearest free spot if a curve is in the way.
NOTE_SPOTS = ((0.30, 0.5), (0.60, -0.55),
              (0.40, 0), (0.60, 0.55))


# Line styles the CLI accepts.  Everything matplotlib understands as a
# string is allowed, so users can write '-,-.,:' or 'solid,dashed,dotted'.
_VALID_LINESTYLES = {"-", "--", "-.", ":", "solid", "dashed", "dashdot",
                     "dotted", "none"}


def parse_linestyles(spec, n_traces):
    """Split a ``--linestyle`` value into ``n_traces`` validated styles.

    Comma-separated; short lists cycle so every trace gets a style.  When
    no value is given, all traces are solid."""
    if not spec:
        return ["-"] * n_traces
    styles = [s.strip() for s in spec.split(",") if s.strip()]
    for s in styles:
        if s not in _VALID_LINESTYLES:
            sys.exit(f"invalid linestyle '{s}'; valid: "
                     f"{', '.join(sorted(_VALID_LINESTYLES))}")
    return [styles[k % len(styles)] for k in range(n_traces)]


def _apply_trace_linestyle(new_lines, linestyle):
    """Apply ``linestyle`` to the just-drawn trace line(s), skipping markers.

    ``plot_full_bridge.draw_trace`` draws two Line2D objects per trace — the
    curve itself (no marker) and a marker-only overlay (``linestyle='none'``).
    Only the curve carries no marker, so checking ``get_marker()`` picks it
    out cleanly without touching the white squares/triangles/etc."""
    if linestyle == "-":
        return
    for line in new_lines:
        if line.get_marker() in (None, "None", "none", "", " "):
            line.set_linestyle(linestyle)

Y_TITLES = {"V": "Voltage (V)", "A": "Current (A)"}

# SPICE unit suffixes, case-insensitive ('M' means milli, 'MEG' means mega).
SPICE_SUFFIXES = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3,
                  "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}


# --------------------------------------------------------------------------- #
# ngspice run and raw-file parsing
# --------------------------------------------------------------------------- #
def run_ngspice(cir_file, run_dir):
    """Run ngspice in batch mode on cir_file; return the path of the raw file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    raw, log = run_dir / "simulation.raw", run_dir / "ngspice.log"
    try:
        res = subprocess.run(["ngspice", "-b", "-r", str(raw), "-o", str(log),
                              str(cir_file)],
                             cwd=cir_file.parent, capture_output=True,
                             text=True, timeout=60)
    except subprocess.TimeoutExpired:
        sys.exit("ngspice timed out after 60 s")
    if not raw.exists() or raw.stat().st_size == 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-12:]) \
            if log.exists() else res.stderr
        sys.exit(f"ngspice failed (exit {res.returncode}), log tail:\n{tail}")
    return raw


def parse_raw_file(raw_file):
    """Parse an ngspice binary raw file into {plot name: {vector: values}}.

    Complex (AC) vectors also get mag(), phase() (degrees), real() and imag()
    companions; frequency and time are reduced to their real part."""
    with open(raw_file, "rb") as fh:
        content = fh.read()

    # Find the "Binary:" marker (case insensitive)
    binary_idx = content.lower().find(b"binary:")
    if binary_idx == -1:
        raise ValueError("Could not find 'Binary:' marker in raw file")

    # Find the newline after "Binary:"
    newline_idx = content.find(b"\n", binary_idx)
    if newline_idx == -1:
        raise ValueError("Malformed raw file: no newline after 'Binary:'")

    text = content[:binary_idx].decode("ascii", errors="ignore")
    binary_data = content[newline_idx + 1:]

    plot_name = None
    variables = []
    num_vars = num_points = 0
    flags = ""

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Plotname:"):
            plot_name = line.split(":", 1)[1].strip()
        elif line.startswith("Flags:"):
            flags = line.split(":", 1)[1].strip()
        elif line.startswith("No. Variables:"):
            num_vars = int(line.split(":", 1)[1].strip())
        elif line.startswith("No. Points:"):
            num_points = int(line.split(":", 1)[1].strip())
        elif re.match(r"^\s*\d+\s", line):
            parts = line.split()
            if len(parts) >= 3:
                variables.append(parts[1])

    is_complex = "complex" in flags.lower()
    bytes_per_value = 16 if is_complex else 8

    expected_size = num_points * num_vars * bytes_per_value
    if len(binary_data) < expected_size:
        print(f"Warning: Binary data size ({len(binary_data)}) smaller than "
              f"expected ({expected_size})")

    data = {}
    if plot_name:
        data[plot_name] = {var: [] for var in variables}
        for point_idx in range(num_points):
            for var_idx, var in enumerate(variables):
                offset = (point_idx * num_vars + var_idx) * bytes_per_value
                if offset + bytes_per_value > len(binary_data):
                    continue
                if is_complex:      # pairs of doubles (real, imag)
                    real, imag = struct.unpack("dd", binary_data[offset:offset + 16])
                    data[plot_name][var].append(complex(real, imag))
                else:               # single doubles
                    data[plot_name][var].append(
                        struct.unpack("d", binary_data[offset:offset + 8])[0])

    # For complex data, also add magnitude and phase as derived variables
    if is_complex and plot_name:
        plot_vars = data[plot_name]
        for var in list(plot_vars.keys()):
            values = plot_vars[var]
            if values and isinstance(values[0], complex):
                arr = np.array(values)
                # frequency and other known real variables: take real part
                if var.lower() in ("frequency", "time"):
                    plot_vars[var] = np.real(arr).tolist()
                else:
                    plot_vars[f"mag({var})"] = np.abs(arr).tolist()
                    plot_vars[f"phase({var})"] = np.angle(arr, deg=True).tolist()
                    plot_vars[f"real({var})"] = np.real(arr).tolist()
                    plot_vars[f"imag({var})"] = np.imag(arr).tolist()

    return data


# --------------------------------------------------------------------------- #
# Expressions
# --------------------------------------------------------------------------- #
def evaluate_expression(expr, variables_dict):
    """Evaluate a custom expression on the available vectors.

    Supports basic arithmetic (+, -, *, /, **, parentheses) and numpy
    functions; vectors are referenced as v(1), v(2), i(v1), mag(v(out)), ..."""
    namespace = {
        "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "log10": np.log10,
        "sqrt": np.sqrt, "abs": np.abs,
        "pi": np.pi, "e": np.e,
        "mag": np.abs,
        "phase": lambda x: np.angle(x, deg=True),
        "real": np.real,
        "imag": np.imag,
    }

    # Map vector names like "v(1)" to Python identifiers like "v_1"
    var_mapping = {}
    for name, values in variables_dict.items():
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        safe_name = re.sub(r"_+", "_", safe_name).strip("_")
        if not safe_name or safe_name[0].isdigit():
            safe_name = "var_" + safe_name
        namespace[safe_name] = np.array(values)
        var_mapping[name] = safe_name

    # Longest names first so nested names are replaced correctly
    safe_expr = expr
    for var in sorted(var_mapping, key=len, reverse=True):
        safe_expr = safe_expr.replace(var, var_mapping[var])

    try:
        return np.array(eval(safe_expr, namespace, namespace))
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expr}': {e}")


def compute_expressions(data, plot_name, y_vars):
    """Compute custom expressions and add them to the plot's vectors."""
    if plot_name not in data:
        return data

    plot_vars = data[plot_name]
    variables_dict = {k: np.array(v) for k, v in plot_vars.items()}

    for expr in y_vars:
        is_expression = (any(op in expr for op in ["+", "-", "*", "/", "**"])
                         and expr not in plot_vars)
        if is_expression:
            try:
                result = evaluate_expression(expr, variables_dict)
                plot_vars[expr] = result.tolist()
                print(f"Computed expression: {expr} ({len(result)} points)")
            except Exception as e:
                print(f"Warning: Could not compute expression '{expr}': {e}")

    return data


# --------------------------------------------------------------------------- #
# Axis helpers (units, ticks, limits)
# --------------------------------------------------------------------------- #
_V = r"v\([^()]*\)"
_I = r"i\([^()]*\)"


def guess_unit(name):
    """Best-effort unit of a vector / expression from its name."""
    n = name.strip().lower()
    if n == "time":
        return "s"
    if n == "frequency":
        return "Hz"
    wrap = r"(?:(?:mag|real|imag)\()?"
    if re.fullmatch(rf"{wrap}{_V}(?:\s*[-+]\s*{_V})*\)?", n):
        return "V"
    if re.fullmatch(rf"{wrap}{_I}(?:\s*[-+]\s*{_I})*\)?", n):
        return "A"
    return ""


def probe_name(name):
    """Legend name in Probe style: v(out) -> V(out), v(1)-v(2) -> V(1)-V(2)."""
    return re.sub(r"\b([vi])\(", lambda m: m.group(1).upper() + "(", name)


def nice_step(span, target=8):
    """1/2/5 x 10^n step giving roughly `target` divisions across `span`."""
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag * (1 + 1e-9):
            return m * mag
    return 10 * mag


def tick_labels(ticks, unit):
    """PSpice-style tick labels ("0V", "5V", "500mV", "1KHz", ...)."""
    if unit == "V" and max(abs(t) for t in ticks) >= 1:
        return [qty_label(t, "V") for t in ticks]
    return [("0" if t == 0 else spice_format(t)) + unit for t in ticks]


def y_limits(arrays):
    """Symmetric +/-10 % headroom for signals that cross zero (as in the
    bridge graphs), otherwise 10 % of the span above and below."""
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    if lo < 0.0 < hi:
        lim = 1.5 * max(-lo, hi)
        return -lim, lim
    span = hi - lo
    if span <= 1e-12 * max(abs(lo), abs(hi), 1e-30):      # flat trace
        half = abs(hi) * 0.1 or 1.0
        return lo - half, hi + half
    return lo - 0.1 * span, hi + 0.1 * span


def spice_float(text):
    """Parse a SPICE-style value such as '1meg', '100u' or '10.5'."""
    if isinstance(text, (int, float)):
        return float(text)
    text = text.strip()
    for suf in sorted(SPICE_SUFFIXES, key=len, reverse=True):
        if text.lower().endswith(suf):
            return float(text[:-len(suf)]) * SPICE_SUFFIXES[suf]
    return float(text)


# Clip/clamp detection tolerances.  Real diode clamps plateau with a few
# percent of variation in the span, and multi-cycle records split the
# flat run into one short plateau per period, so the original numbers
# (0.2 % tolerance, 4 % dwell, 2 % run, 15 % band) could never fire on
# anything but an ideal clamp in a one- or two-cycle record.
CLIP_TOL_REL  = 0.015     # half-band around a candidate level (was 0.002)
CLIP_MIN_FRAC = 0.09      # total dwell near a level             (was 0.04)
CLIP_MIN_RUN  = 0.001     # longest single plateau run           (was 0.02)
CLIP_BAND_REL = 0.20      # level must be within this of ymin/ymax (was 0.15)
CLIP_COLOR = "black"

def find_extrema(x, y):
    """Global max and min of a trace, tagged by side."""
    y = np.asarray(y, dtype=float)
    if y.size < 2 or np.ptp(y) == 0.0:
        return []
    x = np.asarray(x, dtype=float)
    imax, imin = int(np.argmax(y)), int(np.argmin(y))
    return [("max", float(y[imax]), float(x[imax]), float(x[imax])),
            ("min", float(y[imin]), float(x[imin]), float(x[imin]))]

def find_clips(x, y, tol_rel=CLIP_TOL_REL, min_frac=CLIP_MIN_FRAC,
               min_run=CLIP_MIN_RUN, band_rel=CLIP_BAND_REL):
    """Clipped (flat-plateau) levels of one trace: [("max"/"min", level, x0, x1)].

    A level is a value within ``tol_rel`` of which the trace dwells on at
    least ``min_frac`` of all samples, gathered into runs of which the
    longest holds ``min_run`` of them (a plain sine crest holds under 1 %
    at its flat top, a clamp plateau 3 % or more); ``level`` is the
    median over that cluster of plateau runs, and only levels within
    ``band_rel`` of the trace's max or min are kept, so a resting trace
    (both diodes off, say) is skipped while a clamp level that sits just
    under turn-off spikes is still found.  `x0`/`x1` span its longest run."""
    y = np.asarray(y, dtype=float)
    n = y.size
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span = ymax - ymin
    if span <= 0.0:
        return []
    tol = tol_rel * span
    need = int(np.ceil(min_frac * n))
    run_need = max(2, int(np.ceil(min_run * n)))

    # ---- candidate levels: windows of >= `need` sorted values within 2*tol --
    ys = np.sort(y)
    cands = []
    j = 0
    for i in range(n - need + 1):
        if j < i + need:
            j = i + need
        if j > n:
            break
        while j < n and ys[j] - ys[i] <= 2.0 * tol:
            j += 1
        if j - i >= need:
            cands.append(0.5 * (float(ys[i]) + float(ys[j - 1])))

    # ---- refine each candidate: total dwell + longest-run, chain merged ------
    clusters = []      # per plateau: [hi level, member samples, x0, x1, len]
    for i, cand in enumerate(cands):
        if i and ys[i] == ys[i - 1]:            # same-value start: same window
            continue
        idx = np.flatnonzero(np.abs(y - cand) <= tol)
        if len(idx) < need:
            continue
        pbest = max(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1),
                    key=len)
        if len(pbest) < run_need:
            continue
        level = float(np.median(y[pbest]))
        if clusters and level - clusters[-1][0] <= tol:         # same plateau
            g = clusters[-1]
            g[0] = max(g[0], level)                 # level walks up the drift
            g[1].append(np.asarray(y[pbest]))
            if len(pbest) > g[4]:
                g[2], g[3], g[4] = (float(x[pbest[0]]), float(x[pbest[-1]]),
                                    len(pbest))
        else:
            clusters.append([level, [np.asarray(y[pbest])],
                             float(x[pbest[0]]), float(x[pbest[-1]]),
                             len(pbest)])

    # ---- one level per cluster, kept near the extremes, tagged by side ------
    out = []
    for _, samples, x0, x1, _ in clusters:
        level = float(np.median(np.concatenate(samples)))
        if level >= ymax - band_rel * span:
            out.append(("max", level, x0, x1))
        elif level <= ymin + band_rel * span:
            out.append(("min", level, x0, x1))
    return out


def _draw_levels(ax, xmin, curves, frame, prop, fsize, kind):
    """Dashed level lines, plateau-centre dots and labels at flat plateaus.

    ``kind`` is either ``"clip"`` (clipped waveform plateau) or ``"clamp"``
    (clamper clamp level).  Only the label text and the printed message differ.
    Returns the labels' bounding rects in frame points so the note boxes can
    dodge them."""
    rects = []
    fsize = max(9.5, 0.62 * fsize)
    axspan = frame.ymax - frame.ymin
    gap = 0.016
    for name, cx, cy in curves:
        for level_kind, level, x0, x1 in find_extrema(cx, cy):
            if kind == "clip":
                label = f"{round(level, 3) + 0.0:+.3f} V"
            else:
                label = f"clamp {round(level, 3) + 0.0:+.3f} V"
            ax.axhline(level, color=CLIP_COLOR, linewidth=1.1,
                       linestyle=(0, (5.5, 2.8)), zorder=1.3)
            xd = 0.5 * (float(x0) + float(x1))
            ax.plot([xd], [level], marker="o", markersize=3.2,
                    markerfacecolor=CLIP_COLOR, markeredgecolor="none",
                    linestyle="none", zorder=7)
            room = (frame.ymax - level) / axspan if level_kind == "max" \
                else (level - frame.ymin) / axspan
            if room >= gap + fsize / frame.h:           # normal side: outward
                y_lab = level + (gap if level_kind == "max" else -gap) * axspan
                va = "bottom" if level_kind == "max" else "top"
            else:                                       # no headroom: inward
                y_lab = level - (gap if level_kind == "max" else -gap) * axspan
                va = "top" if level_kind == "max" else "bottom"
            half_w = 0.5 * text_width(label, fsize, prop)
            if half_w > 0.0:                            # keep the label inside
                wx = half_w / frame.w * (frame.xmax - frame.xmin)
                xd = float(np.clip(xd, frame.xmin + wx, frame.xmax - wx))
            px, py = (float(v) for v in frame.to_pt(xd, y_lab))
            lo_off = 0.2 * fsize if va == "bottom" else -1.1 * fsize
            rects.append((px - half_w + 0.2, py - lo_off,
                          px + half_w + 0.2,
                          py + (1.1 * fsize if va == "bottom" else 0.2 * fsize)))
            ax.text(xd, y_lab, label, fontproperties=prop, fontsize=fsize,
                    ha="center", va=va, color=CLIP_COLOR, zorder=7)
            print(f"{kind} level {label} ({level_kind}) on {probe_name(name)}")
    return rects


def draw_clip_levels(ax, xmin, curves, frame, prop, note_fsize):
    """Dashed clip-level lines, plateau-centre dots and "+3.549 V" labels at every
    flat (clipped) max/min of every trace, as in the clipper figures."""
    return _draw_levels(ax, xmin, curves, frame, prop, note_fsize, "clip")


def draw_clamp_levels(ax, xmin, curves, frame, prop, note_fsize):
    """Dashed clamp-level lines, plateau-centre dots and "clamp +X.XXX V" labels at
    the flat (clamped) level of every trace, as in the clamper figures."""
    return _draw_levels(ax, xmin, curves, frame, prop, note_fsize, "clamp")


def _text_colour_for(color):
    """Black or white text depending on the perceived brightness of ``color``."""
    r, g, b, _ = to_rgba(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.60 else "black"


def draw_trace_boxes(ax, curves, frame, avoid_rects, prop, fsize):
    """Small coloured box next to the end of each trace, carrying its name.

    Each box sits at the right edge of the axes, vertically centred on its
    trace's end value, in the trace's own colour.  Boxes are stacked so they
    never overlap each other or the clip-level labels (``avoid_rects``), and
    the boxes' page-point rects are returned so the notes can dodge them."""
    rects = []
    if not curves:
        return rects
    fsize = max(8.5, 0.55 * note_fontsize(prop))
    pad_x, pad_y = 2.0, 3.0
    box_h = 1.85 * fsize
    lo_cy, hi_cy = box_h / 2.0, frame.h - box_h / 2.0

    def free(cx, cy):
        return rect_is_free((cx - box_w / 2, cy - box_h / 2,
                             cx + box_w / 2, cy + box_h / 2),
                            (np.array([]), np.array([])), avoid_rects, 2.0, frame)

    placed = []          # accepted box y-centres (page points), for stacking
    for k, (name, cx, cy) in enumerate(curves):
        color = TRACE_STYLES[k % len(TRACE_STYLES)][0]
        label = probe_name(name)
        box_w = text_width(label, fsize, prop) + 2.0 * pad_x
        x_pt = frame.w - pad_x - 0.5 * box_w    # box's right edge at the frame edge
        cy_pt = (float(cy[-1]) - frame.ymin) / (frame.ymax - frame.ymin) * frame.h
        cy_pt = min(max(cy_pt, lo_cy), hi_cy)
        # stack: shift up past any earlier box we would overlap
        for py in placed:
            if abs(cy_pt - py) < box_h + pad_y:
                cy_pt = min(py + box_h + pad_y, hi_cy)
        # if we landed on a clip label, hunt the nearest free spot along y
        if not free(x_pt, cy_pt):
            best, best_d = None, None
            for trial in np.arange(lo_cy, hi_cy + 1e-9, 2.0):
                if free(x_pt, trial) and all(abs(trial - p) >= box_h + pad_y
                                             for p in placed):
                    d = abs(trial - cy_pt)
                    if best_d is None or d < best_d:
                        best, best_d = trial, d
            if best is not None:
                cy_pt = best
        placed.append(cy_pt)
        fx, fy = x_pt / frame.w, cy_pt / frame.h
        fw, fh = box_w / frame.w, box_h / frame.h
        ax.add_patch(FancyBboxPatch((fx - fw / 2, fy - fh / 2), fw, fh,
                                    boxstyle=f"round,pad=0,rounding_size={2.0 / frame.w}",
                                    mutation_aspect=frame.w / frame.h,
                                    transform=ax.transAxes,
                                    facecolor=color, edgecolor="black",
                                    linewidth=0.8, zorder=6, clip_on=False))
        ax.text(fx, fy, label, transform=ax.transAxes, ha="center",
                va="center_baseline", fontproperties=prop, fontsize=fsize,
                color=_text_colour_for(color), zorder=7)
        rects.append((x_pt - box_w / 2, cy_pt - box_h / 2,
                      x_pt + box_w / 2, cy_pt + box_h / 2))
        print(f"  trace box {label} at end of {probe_name(name)}")
    return rects


# --------------------------------------------------------------------------- #
# Figure assembly (same page layout as plot_full_bridge.make_figure)
# --------------------------------------------------------------------------- #
def make_graph(x, traces, x_var, outpath, dpi, notes=(), xlim=(None, None),
               x_label=None, y_label=None, linestyles=None, clip_levels=True,
               trace_boxes=True, clamp_levels=True):
    """Draw every (name, y) trace of `traces` against x on one Probe page.

    `notes` is a list aligned with `traces`; each element is either a note
    string (drawn as a framed serif note in the trace colour) or None (no
    note for that trace)."""
    x = np.real(np.asarray(x)).astype(float)
    keep = np.isfinite(x)
    if xlim[0] is not None:
        keep &= x >= xlim[0]
    if xlim[1] is not None:
        keep &= x <= xlim[1]

    curves = []                                 # (name, x, y) of the finite points
    for name, y in traces:
        y = np.asarray(y)
        if np.iscomplexobj(y):
            print(f"Warning: '{name}' is complex; plotting its real part")
            y = np.real(y)
        y = y.astype(float)
        n = min(len(x), len(y))
        ok = keep[:n] & np.isfinite(y[:n])
        if ok.sum() < 2:
            print(f"Warning: not enough finite points to plot '{name}', skipping")
            continue
        curves.append((name, x[:n][ok], y[:n][ok]))
    if not curves:
        sys.exit("nothing to plot")

    # ---- axes: x spans the data, y gets headroom; labels as in Probe ---------
    xmin = min(float(c[1].min()) for c in curves)
    xmax = max(float(c[1].max()) for c in curves)
    if xmax == xmin:
        xmax = xmin + 1.0
    if x_var.lower() == "time":
        tunit = pick_time_unit(xmax)
        xticks = time_ticks(xmin, xmax, 20)
        xlabels = [time_label(v, tunit) for v in xticks]
        x_label = x_label or f"Time ({tunit[1]})"
    else:
        x_unit = guess_unit(x_var)
        xticks = major_ticks(xmin, xmax, nice_step(xmax - xmin, 10))
        xlabels = tick_labels(xticks, x_unit)
        x_label = x_label or (f"{x_var.capitalize()} ({x_unit})" if x_unit else x_var)

    units = {guess_unit(c[0]) for c in curves}
    y_unit = units.pop() if len(units) == 1 else ""
    ylo, yhi = y_limits([c[2] for c in curves])
    y_majors = major_ticks(ylo, yhi, nice_step(yhi - ylo, 10))
    y_minors = minor_between(y_majors, 5)
    y_labels = tick_labels(y_majors, y_unit)

    fig, ax, frame = new_page((xmin, xmax), (ylo, yhi), y_labels)
    style_axes(ax, xticks, xlabels, y_majors, y_labels, y_minors)

    # ---- traces: the first one on top, markers staggered along the curves ----
    count = len(curves)
    if not linestyles:
        linestyles = ["-"] * count

    for k in reversed(range(count)):
        color, marker = TRACE_STYLES[k % len(TRACE_STYLES)]
        _, cx, cy = curves[k]
        n_before = len(ax.lines)
        draw_trace(ax, cx, cy, frame, color, marker, max(4 - k, 1),
                   MARKER_FIRST_PT + k * MARKER_SPACING_PT / count)
        _apply_trace_linestyle(ax.lines[n_before:], linestyles[k % len(linestyles)])
    for name, _, cy in curves:
        print(f"  {probe_name(name)}: {cy.min():.6g}..{cy.max():.6g}{y_unit}")

    # ---- shared serif family/size of the clip labels and the notes ----------
    installed = {f.name for f in font_manager.fontManager.ttflist}
    prop = FontProperties(family=next(
        (f for f in NOTE_SERIF if f in installed), "serif"))
    fsize = note_fontsize(prop)

    # ---- dashed clip-level lines with value labels at flat extremes ---------
    clip_rects = draw_clip_levels(ax, xmin, curves, frame, prop, fsize) \
        if clip_levels else []

    # ---- dashed clamp-level lines with value labels at clamp levels ---------
    clamp_rects = draw_clamp_levels(ax, xmin, curves, frame, prop, fsize) \
        if clamp_levels else []

    # ---- small coloured boxes next to each trace end, carrying its name -----
    box_rects = draw_trace_boxes(ax, curves, frame,
                                 clip_rects + clamp_rects, prop, fsize) \
        if trace_boxes else []

    # ---- notes dodged clear of every curve, clip label, clamp label ---------
    # ---- and trace box ------------------------------------------------------
    if any(notes):
        pts = [dense_curve(cx, cy, frame) for _, cx, cy in curves]
        every = (np.concatenate([p[0] for p in pts]),
                 np.concatenate([p[1] for p in pts]))
        rects = list(clip_rects) + list(clamp_rects) + list(box_rects)
        for k, note in enumerate(notes[:count]):
            if not note:
                continue
            rects.append(add_note_at(ax, note, frame, every, rects,
                                     NOTE_SPOTS[k % len(NOTE_SPOTS)], fsize, prop,
                                     TRACE_STYLES[k % len(TRACE_STYLES)][0]))
        if len(notes) > count:
            print(f"Warning: {len(notes)} notes for {count} traces; "
                  f"ignoring the extra ones")

    entries = [(TRACE_STYLES[k % len(TRACE_STYLES)][1],
                TRACE_STYLES[k % len(TRACE_STYLES)][0], probe_name(c[0]))
               for k, c in enumerate(curves)]
    add_legend_footer(fig, entries, x_label, ax=ax)
    y_label = y_label if y_label is not None else Y_TITLES.get(y_unit, "")
    if y_label:
        add_y_label(fig, y_label, ax)

    fig.savefig(outpath, dpi=dpi, facecolor="white")
    return fig

def build_notes(traces, positional_notes, keyed_notes):
    """Assemble the per-trace note list from -n/--notes and --note.

    `positional_notes` are matched to `traces` in order; `keyed_notes` are
    "TRACE=TEXT" specs whose TRACE may be a trace name (the same string you
    passed to -y) or its 0-based index.  Keyed notes win.  Returns a list of
    the same length as `traces`, each element a note string or None."""
    notes = [None] * len(traces)
    for k, text in enumerate(list(positional_notes or ())[:len(traces)]):
        notes[k] = text
    for spec in keyed_notes or ():
        key, sep, text = spec.partition("=")
        if not sep:
            sys.exit(f"--note expects TRACE=TEXT, got '{spec}'")
        key, text = key.strip(), text.strip()
        idx = None
        if key.lstrip("-").isdigit():
            idx = int(key)
            if not 0 <= idx < len(traces):
                sys.exit(f"--note index {idx} out of range "
                         f"(0..{len(traces) - 1})")
        else:
            for k, (name, _) in enumerate(traces):
                if name == key:
                    idx = k
                    break
            if idx is None:                     # case-insensitive fallback
                low = key.lower()
                for k, (name, _) in enumerate(traces):
                    if name.lower() == low:
                        idx = k
                        break
        if idx is None:
            sys.exit(f"--note trace '{key}' not found; available: "
                     f"{[t[0] for t in traces]}")
        notes[idx] = text
    return notes

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cir_file", help="path to the .cir circuit file")
    ap.add_argument("-x", "--xvar", default=None,
                    help="x-axis vector (default: the first vector of the "
                         "plot, e.g. time or frequency)")
    ap.add_argument("-y", "--yvars", nargs="+",
                    help="vectors or expressions to plot, e.g. 'v(out)' "
                         "'v(1)-v(2)' 'v(out)/v(in)' '20*log10(mag(v(out)))' "
                         "(default: every real vector)")
    ap.add_argument("-p", "--plot", help="specific plot name to use")
    ap.add_argument("-n", "--notes", nargs="+", default=None,
                    help="framed serif note for each trace, in -y order "
                         "(e.g. 'Voltage across resistor' 'Vin')")
    ap.add_argument("-t", "--title", default=None,
                    help="shorthand for a single note on the first trace")
    ap.add_argument("--xmin", type=str, default=None,
                    help="lower x limit shown, SPICE suffixes allowed "
                         "(default: start of the data)")
    ap.add_argument("--xmax", type=str, default=None,
                    help="upper x limit shown, SPICE suffixes allowed, e.g. "
                         "50m (default: end of the data)")
    ap.add_argument("--x-label", default=None,
                    help="x-axis title (default: derived, e.g. 'Time (ms)')")
    ap.add_argument("--y-label", default=None,
                    help="y-axis title (default: 'Voltage (V)' / 'Current (A)' "
                         "from the traces)")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="output PNG (default <fig-dir>/<circuit name>.png)")
    ap.add_argument("--fig-dir", type=str, default="ngspice_figs",
                    help="directory for the generated graph "
                         "(default ngspice_figs)")
    ap.add_argument("--dpi", type=int, default=260,
                    help="graph output resolution (default 260)")
    ap.add_argument("--list", action="store_true",
                    help="list the available vectors and exit")
    ap.add_argument("--keep", action="store_true",
                    help="keep the raw file and ngspice log in "
                         "ngspice_runs/<circuit name>/ instead of a temp "
                         "directory")
    ap.add_argument("--show", action="store_true",
                    help="also open the graph in a window")
    ap.add_argument("--linestyle", dest="linestyle", default=None,
                metavar="STYLE[,STYLE...]",
                help="comma-separated matplotlib line style for each trace "
                     "in -y order, e.g. 'solid,dashed,dotted' or "
                     "'-,-.,:'; default: all solid.  When a style starts "
                     "with '-' use the '=' form: --linestyle=-,--,:")
    ap.add_argument("--clip-levels", action=argparse.BooleanOptionalAction,
                default=True,
                help="draw a dashed line, a plateau-centre dot and a "
                     "'+X.XXX V' label at the flat (clipped) max/min of "
                     "every trace (default: on; --no-clip-levels disables)")
    ap.add_argument("--trace-boxes", action=argparse.BooleanOptionalAction,
                default=True,
                help="draw a small coloured box next to the end of each trace, "
                     "carrying the trace name (default: on; "
                     "--no-trace-boxes disables)")
    ap.add_argument("--clamp-levels", action=argparse.BooleanOptionalAction,
                default=True,
                help="draw a dashed line, a plateau-centre dot and a "
                     "'clamp +X.XXX V' label at the flat (clamped) level of "
                     "every trace (default: on; --no-clamp-levels disables)")
    ap.add_argument("--note", action="append", default=None,
                    metavar="TRACE=TEXT",
                    help="framed note attached to one specific trace; TRACE "
                         "is the trace name given to -y (e.g. v(out) or "
                         "v(1)-v(2)) or its 0-based index in -y order, e.g. "
                         "--note 'v(out)=Rectified output' "
                         "--note '1=Input voltage'.  Repeatable and "
                         "overrides the matching -n/--notes entry.")
    p = ap.parse_args()
    xlim = tuple(spice_float(v) if v is not None else None
                 for v in (p.xmin, p.xmax))

    ngspice = shutil.which("ngspice")
    if not ngspice:
        sys.exit("ngspice not found on PATH")
    cir = Path(p.cir_file).resolve()
    if not cir.is_file():
        sys.exit(f"circuit file not found: {cir}")

    run_dir = Path("ngspice_runs") / cir.stem if p.keep else Path(
        tempfile.mkdtemp(prefix="ngspice_plot_"))
    print(f"ngspice at {ngspice}; simulating {cir.name}")
    try:
        data = parse_raw_file(run_ngspice(cir, run_dir))
    finally:
        if not p.keep:
            shutil.rmtree(run_dir, ignore_errors=True)
    if p.keep:
        print(f"raw file and log kept in {run_dir.resolve()}")
    if not data:
        sys.exit("no plot found in the ngspice raw file")

    if p.list:
        for plot_name, vectors in data.items():
            print(f"\nPlot: {plot_name}")
            for var in vectors:
                print(f"  {var}: {len(vectors[var])} points")
        return

    plot_name = p.plot or next(iter(data))
    if plot_name not in data:
        sys.exit(f"plot '{plot_name}' not found. Available: {list(data)}")
    vectors = data[plot_name]
    x_var = p.xvar or next(iter(vectors))
    if x_var not in vectors:
        sys.exit(f"X variable '{x_var}' not found. Available: {list(vectors)}")

    if p.yvars:
        data = compute_expressions(data, plot_name, p.yvars)
        vectors = data[plot_name]
        y_vars = p.yvars
    else:                       # every real vector (complex ones need mag() etc.)
        y_vars = [v for v, vals in vectors.items()
                  if v != x_var and not (len(vals) and isinstance(vals[0], complex))]

    traces = []
    for y_var in y_vars:
        if y_var not in vectors:
            print(f"Warning: '{y_var}' not found in data, skipping")
        else:
            traces.append((y_var, vectors[y_var]))
    if not traces:
        sys.exit("none of the requested traces were found (see --list)")

    positional = list(p.notes) if p.notes else []
    if p.title and not positional:
        positional = [p.title]
    notes = build_notes(traces, positional, p.note)

    if p.show:      # plot_full_bridge forces the Agg backend; restore the default
        plt.switch_backend(matplotlib.rcParamsOrig["backend"])

    out = Path(p.output) if p.output else Path(p.fig_dir) / f"{cir.stem}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    linestyles = parse_linestyles(p.linestyle, len(traces))
    fig = make_graph(vectors[x_var], traces, x_var, out, p.dpi, notes, xlim,
                     p.x_label, p.y_label, linestyles, p.clip_levels,
                     p.trace_boxes, p.clamp_levels)
    print(f"  saved graph {out}")
    if p.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
