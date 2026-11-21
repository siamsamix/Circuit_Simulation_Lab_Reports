#!/usr/bin/env python3
"""
simulate_bjt.py - ngspice simulation and PSpice/Probe-style plotting of the
input and output characteristics of BJTs in CB, CE and CC configurations.

Experiment 7, Circuit Simulation Laboratory (0714-EEE-2102).

For every configuration and every curve of its characteristic family the
script writes a netlist, runs ngspice in batch mode, reads the raw file and
draws one figure per family (six families for the NPN 2N3904 and six for the
PNP 2N3906).  It also extracts the current gain, input resistance and output
resistance from the simulated curves, compares them with the theoretical
values and writes everything to results.json.

Usage : python simulate_bjt.py [--dpi 300] [--netlists ../netlist] [--out .]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from plot_full_bridge import draw_trace
from pspice_theme import (FIG_W_PT, GREEN, MARKER_FIRST_PT,
                          MARKER_SPACING_PT, add_y_label, choose_font,
                          label_widths_pt, major_ticks, new_page,
                          spice_format, style_axes)
from simulate_and_plot import nice_step, parse_raw_file, run_ngspice

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Transistor models (PSpice-compatible Gummel-Poon cards)
# --------------------------------------------------------------------------- #
MODEL_NPN = (
    ".MODEL Q2N3904 NPN(IS=6.734F XTI=3 EG=1.11 VAF=74.03 BF=416.4 NE=1.259\n"
    "+ ISE=6.734F IKF=.06678 XTB=1.5 BR=.7371 NC=2 ISC=0 IKR=0 RC=1\n"
    "+ CJC=3.638P MJC=.3085 VJC=.75 FC=.5 CJE=4.493P MJE=.2593 VJE=.75\n"
    "+ TR=239.5N TF=301.2P ITF=.4 VTF=4 XTF=2 RB=10)"
)
MODEL_PNP = (
    ".MODEL Q2N3906 PNP(IS=1.41F XTI=3 EG=1.11 VAF=18.7 BF=180.7 NE=1.5\n"
    "+ ISE=0 IKF=80m XTB=1.5 BR=4.977 NC=2 ISC=0 IKR=0 RC=2.5\n"
    "+ CJC=9.728P MJC=.5776 VJC=.75 FC=.5 CJE=8.063P MJE=.3677 VJE=.75\n"
    "+ TR=33.42N TF=179.3P ITF=.4 VTF=4 XTF=6 RB=10)"
)

SAVE = ".SAVE all @q1[ic] @q1[ib] @q1[ie]"

TRACE_STYLES = ((GREEN, "s"), ("#5400ff", "v"), ("#ff0000", "D"),
                ("#0000ff", "o"), ("#00aaaa", "^"), ("#aa5500", "p"))

# Datasheet-typical small-signal current gain of the 2N3904 / 2N3906 at
# I_C = 10 mA (specified range 100-300), used for the theoretical values.
HFE_TYP = 200.0
VAF = {"npn": 74.03, "pnp": 18.7}
VT = 0.02585                       # thermal voltage at 27 degC (V)


# --------------------------------------------------------------------------- #
# Netlist builders.  Each returns a complete, self-contained .cir text.
# --------------------------------------------------------------------------- #
def cb_input_npn(vcb):
    return f"""* Common-base input characteristic (NPN 2N3904)
* V_CB = {vcb:g} V
VEE e 0 0
Q1 c 0 e Q2N3904
VCB c 0 {vcb:g}
{MODEL_NPN}
{SAVE}
.DC VEE 0 -0.72 -0.005
.PROBE
.END
"""


def cb_output_npn(ie):
    return f"""* Common-base output characteristic (NPN 2N3904)
* I_E = {ie*1e3:g} mA
IEE e 0 {ie:g}
Q1 c 0 e Q2N3904
VCB c 0 0
{MODEL_NPN}
{SAVE}
.DC VCB -0.75 10 0.05
.PROBE
.END
"""


def ce_input_npn(vce):
    return f"""* Common-emitter input characteristic (NPN 2N3904)
* V_CE = {vce:g} V
VBB b 0 0
Q1 c b 0 Q2N3904
VCE c 0 {vce:g}
{MODEL_NPN}
{SAVE}
.DC VBB 0 0.75 0.005
.PROBE
.END
"""


def ce_output_npn(ib):
    return f"""* Common-emitter output characteristic (NPN 2N3904)
* I_B = {ib*1e6:g} uA
IB 0 b {ib:g}
Q1 c b 0 Q2N3904
VCE c 0 0
{MODEL_NPN}
{SAVE}
.DC VCE 0 10 0.05
.PROBE
.END
"""


def cc_input_npn(vce):
    return f"""* Common-collector input characteristic (NPN 2N3904)
* collector common, V_CE = {vce:g} V
VBB b 0 {-vce+0.5:g}
Q1 0 b e Q2N3904
VCE e 0 {-vce:g}
{MODEL_NPN}
{SAVE}
.DC VBB {-vce+0.30:g} {-vce+0.75:g} 0.005
.PROBE
.END
"""


def cc_output_npn(ib):
    return f"""* Common-collector output characteristic (NPN 2N3904)
* I_B = {ib*1e6:g} uA
IB 0 b {ib:g}
Q1 c b e Q2N3904
VC c 0 5
VE e 0 0
{MODEL_NPN}
{SAVE}
.DC VE 0 5 0.025
.PROBE
.END
"""


def cb_input_pnp(veb):
    return f"""* Common-base input characteristic (PNP 2N3906)
* V_EB = {veb:g} V
VEE e 0 0
Q1 c 0 e Q2N3906
VCB c 0 {-veb:g}
{MODEL_PNP}
{SAVE}
.DC VEE 0 0.76 0.005
.PROBE
.END
"""


def cb_output_pnp(ie):
    return f"""* Common-base output characteristic (PNP 2N3906)
* I_E = {ie*1e3:g} mA
IEE 0 e {ie:g}
Q1 c 0 e Q2N3906
VCB c 0 0
{MODEL_PNP}
{SAVE}
.DC VCB 0.75 -10 -0.05
.PROBE
.END
"""


def ce_input_pnp(vce):
    return f"""* Common-emitter input characteristic (PNP 2N3906)
* V_CE = {vce:g} V
VBB b 0 0
Q1 c b 0 Q2N3906
VCE c 0 {-vce:g}
{MODEL_PNP}
{SAVE}
.DC VBB 0 -0.78 -0.005
.PROBE
.END
"""


def ce_output_pnp(ib):
    return f"""* Common-emitter output characteristic (PNP 2N3906)
* I_B = {ib*1e6:g} uA
IB b 0 {ib:g}
Q1 c b 0 Q2N3906
VCE c 0 0
{MODEL_PNP}
{SAVE}
.DC VCE 0 -10 -0.05
.PROBE
.END
"""


def cc_input_pnp(vce):
    return f"""* Common-collector input characteristic (PNP 2N3906)
* collector common, V_CE = {vce:g} V
VBB b 0 {vce-0.5:g}
Q1 0 b e Q2N3906
VCE e 0 {vce:g}
{MODEL_PNP}
{SAVE}
.DC VBB {vce-0.78:g} {vce-0.30:g} 0.005
.PROBE
.END
"""


def cc_output_pnp(ib):
    return f"""* Common-collector output characteristic (PNP 2N3906)
* I_B = {ib*1e6:g} uA
IB b 0 {ib:g}
Q1 c b e Q2N3906
VC c 0 -5
VE e 0 0
{MODEL_PNP}
{SAVE}
.DC VE 0 -5 -0.025
.PROBE
.END
"""


# --------------------------------------------------------------------------- #
# Vector extraction: map raw vectors to the plotted x and y arrays
# --------------------------------------------------------------------------- #
def _asc(x, y):
    order = np.argsort(np.asarray(x))
    return (np.asarray(x)[order], np.asarray(y)[order])


def extract(key, v):
    """Return (x, y) for one simulated curve, in SI units."""
    ic = np.abs(np.asarray(v["i(@q1[ic])"]))
    ib = np.abs(np.asarray(v["i(@q1[ib])"]))
    ie = np.abs(np.asarray(v["i(@q1[ie])"]))
    vb = np.asarray(v.get("v(b)", 0.0))
    vc = np.asarray(v.get("v(c)", 0.0))
    ve = np.asarray(v.get("v(e)", 0.0))

    if key == "cb_input_npn":
        return _asc(-ve, ie)
    if key == "cb_output_npn":
        return _asc(vc, ic)
    if key == "ce_input_npn":
        return _asc(vb, ib)
    if key == "ce_output_npn":
        return _asc(vc, ic)
    if key == "cc_input_npn":
        return _asc(vb - ve, ib)
    if key == "cc_output_npn":
        return _asc(vc - ve, ie)
    if key == "cb_input_pnp":
        return _asc(ve, ie)
    if key == "cb_output_pnp":
        return _asc(-vc, ic)
    if key == "ce_input_pnp":
        return _asc(-vb, ib)
    if key == "ce_output_pnp":
        return _asc(-vc, ic)
    if key == "cc_input_pnp":
        return _asc(ve - vb, ib)
    if key == "cc_output_pnp":
        return _asc(ve - vc, ie)
    raise KeyError(key)


# --------------------------------------------------------------------------- #
# Family definitions
# --------------------------------------------------------------------------- #
FAMILIES = [
    dict(key="cb_input_npn", build=cb_input_npn, params=[5, 10],
         legend=lambda p: f"$V_{{CB}}$={p:g} V",
         xlabel="Emitter-base voltage $V_{EB}$ (V)",
         ylabel="Emitter current $I_E$", fname="cb_input_npn",
         xlim=(-0.02, 0.74)),
    dict(key="cb_output_npn", build=cb_output_npn,
         params=[0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3],
         legend=lambda p: f"$I_E$={p*1e3:g} mA",
         xlabel="Collector-base voltage $V_{CB}$ (V)",
         ylabel="Collector current $I_C$", fname="cb_output_npn",
         xlim=(-0.75, 10.2)),
    dict(key="ce_input_npn", build=ce_input_npn, params=[5, 20],
         legend=lambda p: f"$V_{{CE}}$={p:g} V",
         xlabel="Base-emitter voltage $V_{BE}$ (V)",
         ylabel="Base current $I_B$", fname="ce_input_npn",
         xlim=(-0.02, 0.78)),
    dict(key="ce_output_npn", build=ce_output_npn,
         params=[0, 20e-6, 40e-6, 60e-6, 80e-6],
         legend=lambda p: f"$I_B$={p*1e6:g} $\\mu$A",
         xlabel="Collector-emitter voltage $V_{CE}$ (V)",
         ylabel="Collector current $I_C$", fname="ce_output_npn",
         xlim=(-0.2, 10.2)),
    dict(key="cc_input_npn", build=cc_input_npn, params=[1, 2, 5, 10, 20],
         legend=lambda p: f"$V_{{CE}}$={p:g} V",
         xlabel="Base-emitter voltage $V_{BE}$ (V)",
         ylabel="Base current $I_B$", fname="cc_input_npn",
         xlim=(0.20, 0.85)),
    dict(key="cc_output_npn", build=cc_output_npn,
         params=[0, 20e-6, 40e-6, 60e-6, 80e-6],
         legend=lambda p: f"$I_B$={p*1e6:g} $\\mu$A",
         xlabel="Collector-emitter voltage $V_{CE}$ (V)",
         ylabel="Emitter current $I_E$", fname="cc_output_npn",
         xlim=(-0.2, 5.2)),
    dict(key="cb_input_pnp", build=cb_input_pnp, params=[5, 10],
         legend=lambda p: f"$|V_{{CB}}|$={p:g} V",
         xlabel="Emitter-base voltage $|V_{EB}|$ (V)",
         ylabel="Emitter current $|I_E|$", fname="cb_input_pnp",
         xlim=(-0.02, 0.78)),
    dict(key="cb_output_pnp", build=cb_output_pnp,
         params=[0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3],
         legend=lambda p: f"$|I_E|$={p*1e3:g} mA",
         xlabel="Collector-base voltage $|V_{CB}|$ (V)",
         ylabel="Collector current $|I_C|$", fname="cb_output_pnp",
         xlim=(-0.75, 10.2)),
    dict(key="ce_input_pnp", build=ce_input_pnp, params=[5, 20],
         legend=lambda p: f"$|V_{{CE}}|$={p:g} V",
         xlabel="Base-emitter voltage $|V_{BE}|$ (V)",
         ylabel="Base current $|I_B|$", fname="ce_input_pnp",
         xlim=(-0.02, 0.80)),
    dict(key="ce_output_pnp", build=ce_output_pnp,
         params=[0, 20e-6, 40e-6, 60e-6, 80e-6],
         legend=lambda p: f"$|I_B|$={p*1e6:g} $\\mu$A",
         xlabel="Collector-emitter voltage $|V_{CE}|$ (V)",
         ylabel="Collector current $|I_C|$", fname="ce_output_pnp",
         xlim=(-0.2, 10.2)),
    dict(key="cc_input_pnp", build=cc_input_pnp, params=[1, 2, 5, 10, 20],
         legend=lambda p: f"$|V_{{CE}}|$={p:g} V",
         xlabel="Base-emitter voltage $|V_{BE}|$ (V)",
         ylabel="Base current $|I_B|$", fname="cc_input_pnp",
         xlim=(0.20, 0.85)),
    dict(key="cc_output_pnp", build=cc_output_pnp,
         params=[0, 20e-6, 40e-6, 60e-6, 80e-6],
         legend=lambda p: f"$|I_B|$={p*1e6:g} $\\mu$A",
         xlabel="Collector-emitter voltage $|V_{CE}|$ (V)",
         ylabel="Emitter current $|I_E|$", fname="cc_output_pnp",
         xlim=(-0.2, 5.2)),
]


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate_text(text, name):
    """Run one netlist text through ngspice and return its vector dictionary."""
    with tempfile.TemporaryDirectory(prefix="bjt_run_") as tmp:
        tmp = Path(tmp)
        cir = tmp / f"{name}.cir"
        cir.write_text(text)
        try:
            raw = run_ngspice(cir, tmp)
        except SystemExit as exc:
            sys.exit(f"{name}: {exc}")
        data = parse_raw_file(raw)
    if not data:
        sys.exit(f"{name}: no plot in the raw file")
    return next(iter(data.values()))


# --------------------------------------------------------------------------- #
# Family plotting
# --------------------------------------------------------------------------- #
def tick_label(value, unit):
    value = round(float(value), 9) + 0.0          # kill -0.0 and 1e-16 noise
    if unit == "V":
        return f"{value:g}V"
    return ("0" if value == 0 else spice_format(value)) + unit


def add_family_footer(fig, entries, axis_label, ax):
    """Hollow-marker legend row plus the centred axis label, on two clear rows."""
    x0 = ax.get_position().x0
    y_leg, y_lab = 0.050, 0.016
    fsize = choose_font()[1]
    widths = label_widths_pt([name for _, _, name in entries], fsize)
    x = x0
    for (marker, edge, name), w in zip(entries, widths):
        fig.add_artist(Line2D([x], [y_leg], marker=marker, markersize=4.4,
                              markeredgewidth=1.1, markerfacecolor="white",
                              markeredgecolor=edge, linestyle="none",
                              transform=fig.transFigure))
        fig.text(x + 0.0084, y_leg, name, ha="left", va="center", color="black")
        x += 0.0084 + w / FIG_W_PT + 0.013
    fig.text(0.491, y_lab, axis_label, ha="center", va="center", color="black")


def plot_family(curves, xlabel, ylabel, outpath, xlim=None, dpi=300):
    """Draw one characteristic family on a PSpice/Probe-style page."""
    xs = np.concatenate([c[1] for c in curves])
    ys = np.concatenate([c[2] for c in curves])
    if xlim is None:
        xlim = (float(xs.min()), float(xs.max()))
    ylo = min(0.0, float(ys.min()))
    yhi = float(ys.max()) * 1.08
    step = nice_step(yhi - ylo, 9)
    ylo = np.floor(ylo / step) * step
    yhi = np.ceil(yhi / step) * step
    ylim = (ylo, yhi)

    ystep = nice_step(ylim[1] - ylim[0], 9)
    yticks = major_ticks(ylim[0], ylim[1], ystep)
    ylabels = [tick_label(t, "A") for t in yticks]
    xstep = nice_step(xlim[1] - xlim[0], 10)
    xticks = major_ticks(xlim[0], xlim[1], xstep)
    xlabels = [tick_label(t, "V") for t in xticks]

    fig, ax, frame = new_page(xlim, ylim, ylabels)
    style_axes(ax, xticks, xlabels, yticks, ylabels)

    for k in reversed(range(len(curves))):
        color, marker = TRACE_STYLES[k % len(TRACE_STYLES)]
        _, x, y = curves[k]
        draw_trace(ax, x, y, frame, color, marker, max(4 - k, 1),
                   MARKER_FIRST_PT + k * MARKER_SPACING_PT / len(curves))

    entries = [(TRACE_STYLES[k % len(TRACE_STYLES)][1],
                TRACE_STYLES[k % len(TRACE_STYLES)][0], c[0])
               for k, c in enumerate(curves)]
    add_family_footer(fig, entries, xlabel, ax)
    add_y_label(fig, ylabel, ax)
    fig.savefig(outpath, dpi=dpi, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Parameter extraction from the simulated curves
# --------------------------------------------------------------------------- #
def x_at_y(y_target, x, y):
    """x where the monotonic curve reaches y_target (linear interpolation)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    return float(np.interp(y_target, y[order], x[order]))


def y_at_x(x_target, x, y):
    return float(np.interp(x_target, np.asarray(x, dtype=float),
                           np.asarray(y, dtype=float)))


def measure_polarity(d, pol):
    """Gain and port resistances from one polarity's curve families."""
    cb_in = {p: (x, y) for p, x, y in d["cb_input_" + pol]}
    cb_out = {p: (x, y) for p, x, y in d["cb_output_" + pol]}
    ce_in = {p: (x, y) for p, x, y in d["ce_input_" + pol]}
    ce_out = {p: (x, y) for p, x, y in d["ce_output_" + pol]}
    cc_in = {p: (x, y) for p, x, y in d["cc_input_" + pol]}
    cc_out = {p: (x, y) for p, x, y in d["cc_output_" + pol]}

    # CB: alpha from the output family between I_E = 4 and 5 mA
    ic4 = y_at_x(5.0, *cb_out[4e-3])
    ic5 = y_at_x(5.0, *cb_out[5e-3])
    alpha = (ic5 - ic4) / 1e-3
    # CB input resistance between I_E = 3 and 5 mA at V_CB = 5 V
    veb3 = x_at_y(3e-3, *cb_in[5])
    veb5 = x_at_y(5e-3, *cb_in[5])
    rin_cb = (veb5 - veb3) / 2e-3
    # CB output resistance at I_E = 4 mA, V_CB = 5 -> 10 V
    ic5v = y_at_x(5.0, *cb_out[4e-3])
    ic10v = y_at_x(10.0, *cb_out[4e-3])
    rout_cb = 5.0 / (ic10v - ic5v)

    # CE: beta between I_B = 40 and 60 uA at V_CE = 5 V
    ic40 = y_at_x(5.0, *ce_out[40e-6])
    ic60 = y_at_x(5.0, *ce_out[60e-6])
    beta = (ic60 - ic40) / 20e-6
    vbe40 = x_at_y(40e-6, *ce_in[5])
    vbe60 = x_at_y(60e-6, *ce_in[5])
    rin_ce = (vbe60 - vbe40) / 20e-6
    ic5v = y_at_x(5.0, *ce_out[60e-6])
    ic10v = y_at_x(10.0, *ce_out[60e-6])
    rout_ce = 5.0 / (ic10v - ic5v)

    # CC: gamma between I_B = 40 and 60 uA at V_CE = 5 V (same V_CE as beta)
    ie40 = y_at_x(5.0, *cc_out[40e-6])
    ie60 = y_at_x(5.0, *cc_out[60e-6])
    gamma = (ie60 - ie40) / 20e-6
    vbe40 = x_at_y(40e-6, *cc_in[5])
    vbe60 = x_at_y(60e-6, *cc_in[5])
    rin_cc = (vbe60 - vbe40) / 20e-6
    ie3 = y_at_x(3.0, *cc_out[60e-6])
    ie5 = y_at_x(5.0, *cc_out[60e-6])
    rout_cc = 2.0 / (ie5 - ie3)

    sim = dict(alpha=alpha, beta=beta, gamma=gamma,
               rin_cb=rin_cb, rin_ce=rin_ce, rin_cc=rin_cc,
               rout_cb=rout_cb, rout_ce=rout_ce, rout_cc=rout_cc)

    # ----- theoretical values at the same bias points -----
    vaf = VAF[pol]

    def alpha_at(vcb):
        """Ideal alpha with base-width modulation: (1+x)/(1+x+1/beta)."""
        x = vcb / vaf
        return (1 + x) / (1 + x + 1 / HFE_TYP)

    # CE/CC output resistance at the middle of the measured voltage span
    ic_mid_ce = y_at_x(7.5, *ce_out[60e-6])
    ie_mid_cc = y_at_x(4.0, *cc_out[60e-6])
    th = dict(
        alpha=HFE_TYP / (1 + HFE_TYP),
        beta=HFE_TYP,
        gamma=1 + HFE_TYP,
        # chord slopes of the ideal exponential input characteristic
        rin_cb=VT * np.log(5 / 3) / 2e-3,
        rin_ce=VT * np.log(60 / 40) / 20e-6,
        rin_cc=VT * np.log(60 / 40) / 20e-6,
        # CB at fixed emitter current: d(alpha)/dV is divided by I_E
        rout_cb=5.0 / (4e-3 * (alpha_at(10.0) - alpha_at(5.0))),
        alpha5=alpha_at(5.0), alpha10=alpha_at(10.0),
        rout_ce=(vaf + 7.5) / ic_mid_ce,
        rout_cc=(vaf + 4.0) / ie_mid_cc,
    )
    # alpha5/alpha10 are helper terms of rout_cb, not measured parameters
    err = {k: (th[k] - sim[k]) / th[k] * 100 for k in sim}
    pts = dict(ic_cb4=ic4, ic_cb5=ic5,
               ic_cb4_5v=y_at_x(5.0, *cb_out[4e-3]),
               ic_cb4_10v=y_at_x(10.0, *cb_out[4e-3]),
               veb_3m=veb3, veb_5m=veb5,
               ic_ce40_5v=y_at_x(5.0, *ce_out[40e-6]),
               ic_ce60_5v=y_at_x(5.0, *ce_out[60e-6]),
               ic_ce60_7v5=y_at_x(7.5, *ce_out[60e-6]),
               ic_ce60_10v=y_at_x(10.0, *ce_out[60e-6]),
               vbe_40u=vbe40, vbe_60u=vbe60,
               ie_cc40_3v=y_at_x(3.0, *cc_out[40e-6]),
               ie_cc60_3v=y_at_x(3.0, *cc_out[60e-6]),
               ie_cc40_5v=y_at_x(5.0, *cc_out[40e-6]),
               ie_cc60_5v=y_at_x(5.0, *cc_out[60e-6]),
               ie_cc60_4v=y_at_x(4.0, *cc_out[60e-6]))
    def knee(curve, ref_x, frac=0.9):
        """Saturation knee: the lowest x at which the trace has reached frac of its
        value at ref_x (just inside the active region)."""
        x, y = np.asarray(curve[0], float), np.asarray(curve[1], float)
        o = np.argsort(x)
        x, y = x[o], y[o]
        target = frac * np.interp(ref_x, x, y)
        return float(x[np.where(y >= target)[0][0]])

    regions = dict(
        sat_cb=knee(cb_out[4e-3], 0.5),          # V_CB at the CB saturation knee
        sat_ce=knee(ce_out[60e-6], 1.0),         # V_CE at the CE saturation knee
        sat_cc=knee(cc_out[60e-6], 1.0),         # V_CE at the CC saturation knee
        ib_cutoff_ce=y_at_x(5.0, *ce_out[0]),      # collector leakage with I_B = 0, V_CE = 5 V
        ie_cutoff_cb=y_at_x(5.0, *cb_out[0]),      # collector leakage with I_E = 0, V_CB = 5 V
        vbe_on_ce=x_at_y(10e-6, *ce_in[5]),      # V_BE at I_B = 10 uA
        veb_on_cb=x_at_y(10e-6 * 100, *cb_in[5]),  # V_EB at I_E = 1 mA
    )
    return dict(simulated=sim, theoretical=th, error=err, points=pts, regions=regions,
                bias=dict(ie_cb=4.5e-3, ic_cb=ic4, ib_ce=50e-6,
                          ic_ce=ic60, ib_cc=50e-6, ie_cc=ie60))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--netlists", default=str(HERE.parent / "netlist"))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    netdir = Path(args.netlists)
    outdir = Path(args.out)
    netdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for fam in FAMILIES:
        key = fam["key"]
        curves = []
        for p in fam["params"]:
            vectors = simulate_text(fam["build"](p), f"{key}_{p:g}")
            x, y = extract(key, vectors)
            curves.append((p, x, y))
            print(f"  {key:14s} param={p:<10g} {len(x):4d} pts  "
                  f"y: {np.min(y):.4g} .. {np.max(y):.4g}")
        all_data[key] = curves

        # representative netlist for the first curve
        # (for the output families the first parameter is the I = 0 cutoff
        #  trace, so the netlist on file is the first non-zero one)
        rep = next(p for p in fam["params"] if p != 0)
        (netdir / f"{fam['fname']}.cir").write_text(fam["build"](rep))

        if not args.no_plots:
            curves_lab = [(fam["legend"](p), x, y) for p, x, y in curves]
            plot_family(curves_lab, fam["xlabel"], fam["ylabel"],
                        outdir / f"{fam['fname']}.png",
                        xlim=fam["xlim"], dpi=args.dpi)
            print(f"    -> {fam['fname']}.png")

    res = {pol: measure_polarity(all_data, pol) for pol in ("npn", "pnp")}
    (outdir / "results.json").write_text(json.dumps(res, indent=2))

    for pol in ("npn", "pnp"):
        s, t, e = res[pol]["simulated"], res[pol]["theoretical"], res[pol]["error"]
        print(f"\n{pol.upper()}")
        print(f"  {'':10s} {'sim':>10s} {'theory':>10s} {'err %':>8s}")
        for k in ("alpha", "beta", "gamma", "rin_cb", "rin_ce", "rin_cc",
                  "rout_cb", "rout_ce", "rout_cc"):
            print(f"  {k:10s} {s[k]:10.4g} {t[k]:10.4g} {e[k]:8.2f}")


if __name__ == "__main__":
    main()
