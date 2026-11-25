#!/usr/bin/env python3
"""
simulate_bjt.py - ngspice simulation and PSpice/Probe-style plotting of the
input and output characteristics of BJTs in CB, CE and CC configurations.

Experiment 7, Circuit Simulation Laboratory (0714-EEE-2102).

Every characteristic family is produced by ONE netlist with a nested DC sweep
of BOTH voltage sources (".DC <inner source> ... <outer source> ...").  The
inner source sweeps the x axis, the outer source steps through the curves of
the family.  No current sources are used anywhere: every source reaches the
transistor through a series resistor (R_B in the base lead, R_C in the
collector lead, R_E in the emitter lead) and the base / emitter current of the
output families is set by V_BB / R_B or V_EE / R_E.  All plotted voltages and
currents are taken at the transistor terminals, i.e. after the resistors.

The common-collector circuits are wired exactly as in the textbook figure: the
collector is the common (ground) node, V_BB feeds the base resistor from the
collector rail, and the emitter supply V_EE feeds the emitter resistor.  In the
output families I_B therefore changes along each curve (I_B = (V_BB + V_CE -
V_BE)/R_B); the measurements use the simulated I_B of each curve.

Curve families (as in the experiment-7 reference figures)
  CB input   I_E vs V_EB    V_CB = 5, 25, 45 V   (Early effect shifts the curves)
  CB output  I_C vs V_CB    I_E stepped by V_EE: ~0, 0.4, 1.3, 2.3, 3.3, 4.3 mA
  CE input   I_B vs V_BE    V_CE = 0.025, 0.05, 0.1, 0.2 V (octave sweep, see below)
  CE output  I_C vs V_CE    I_B stepped by V_BB: 0, 14, 33, 53, 73 uA
  CC input   I_B vs V_BC    V_CE = 1, 2, 3, 4 V
  CC output  I_E vs V_CE    I_B ~ 0, 14, 34, 54, 74 uA (V_BB = -6 ... 74 V, R_B = 1 MOhm)

Note on the CE input family: in the Gummel-Poon model the base current does not
depend on V_CE once V_CE is above ~0.3 V, so the manual's V_CE = 5 V and 20 V
curves would be one and the same line.  The curves are therefore taken at the
edge of saturation with an OCTAVE sweep of V_CE (0.025, 0.05, 0.1, 0.2 V), which
spaces them almost evenly.  The CE input resistance is measured on a separate,
not plotted V_CE = 5 V run (ce_input_ref_*.cir).

Decade / octave sweeps: PSpice's ".DC OCT/DEC" has no ngspice counterpart, so
geometric_sweep() emulates it (stepped index source + behavioural voltage source).
Change CE_IN_SWEEP to try other start/stop/points values or kind="dec".

Each figure follows the Probe style of the reference report figures: every curve
of a family has the same colour and marker, there is a single legend entry (the
plotted current, e.g. IB(Q1)) and the stepped source value is written beside
each curve ("VCE=0V", "5V", "10V", ...).

The script draws one figure per family (six for the NPN 2N3904 and six for the
PNP 2N3906), extracts the current gain, input resistance and output resistance
from the simulated curves, compares them with the theoretical values and writes
everything to results.json.

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
from pspice_theme import (FIG_W_PT, MARKER_FIRST_PT,
                          MARKER_SPACING_PT, add_y_label, choose_font,
                          label_widths_pt, major_ticks, new_page,
                          spice_format, style_axes)
from simulate_and_plot import (nice_step, parse_raw_file, place_sweep_labels,
                               run_ngspice)

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

# Series resistors between the voltage sources and the transistor terminals.
#   R_B : base lead       (100 kOhm in the CE output families -> V_BB steps of 2 V
#                          give I_B steps of ~20 uA, and V_BB = 0 gives I_B = 0)
#   R_C : collector lead  (small, so V_CE / V_CB stay close to the source value)
#   R_E : emitter lead in CB (1 kOhm -> V_EE steps of 1 V give I_E steps of ~1 mA)
#         and in CC (small, so V_CE stays close to the source value)
RB_CE_IN = "10k"      # CE INPUT families: V_BB 0..2 V -> I_B up to ~120 uA
RB_CE_OUT = "100k"    # CE OUTPUT families: I_B = (V_BB - V_BE)/R_B = 0, 14, 33, 53, 73 uA
RC_CE = "10"          # CE OUTPUT collector resistor
RC_CB = "100"         # CB OUTPUT collector resistor
RC_IN = "0.1"         # collector resistor of the CB / CE INPUT families (V_CB, V_CE = source)
RE_CB = "1k"          # CB emitter resistor
RE_CC_OUT = "1"       # CC OUTPUT emitter resistor
RE_CC_IN = "0.1"      # CC INPUT emitter resistor (V_CE = source value)
RB_CC_IN = "22k"      # CC INPUT base resistor
RB_CC_OUT = "1Meg"    # CC OUTPUT base resistor.  With V_BB in the tens of volts the
                      # base current V/R_B hardly depends on V_CE, so every curve of
                      # the family is nearly a constant-I_B curve, as in the textbook.

# Every curve of a family has the SAME colour and marker (violet squares, as in
# the reference Probe figures); the curves are told apart by the value of the
# stepped source written next to each one, and one legend entry names the
# plotted quantity.
CURVE_COLOR, CURVE_MARKER = "#5400ff", "s"

# Datasheet-typical small-signal current gain of the 2N3904 / 2N3906 at
# I_C = 10 mA (specified range 100-300), used for the theoretical values.
HFE_TYP = 200.0
VAF = {"npn": 74.03, "pnp": 18.7}
VT = 0.02585                       # thermal voltage at 27 degC (V)

# --------------------------------------------------------------------------- #
# Decade / octave DC sweeps.
#
# PSpice can step a source geometrically (".DC OCT <src> <start> <stop> <points>"
# and ".DC DEC ...").  ngspice's .DC only knows linear steps, so the same sweep is
# emulated: a linear "index" source VK (1, 2, ..., N) is stepped by the ordinary
# nested .DC and a behavioural voltage source turns it into the geometric values
#     octave:  start * 2^((k-1)/points)        decade:  start * 10^((k-1)/points)
# so the netlist still contains one nested .DC and nothing but voltage sources.
# --------------------------------------------------------------------------- #
def geometric_sweep(kind, start, stop, points=1):
    """Return (expression in v(k), N) for an octave / decade sweep.

    kind   "oct" or "dec";  points = steps per octave / per decade;
    start, stop = first and last value (stop is reached only if it lies on the grid).
    """
    ratio = {"oct": 2.0, "dec": 10.0}[kind[:3].lower()]
    n = int(np.floor(points * np.log(stop / start) / np.log(ratio) + 1e-9)) + 1
    expr = f"{start:g}*{ratio:g}^((v(k)-1)/{points:g})"
    return expr, n


def list_sweep(values):
    """Same interface for an explicit list of values (no PSpice counterpart)."""
    expr = " : ".join(f"v(k) < {i + 1.5} ? {v:g}" for i, v in enumerate(values[:-1])) + f" : {values[-1]:g}"
    return expr, len(values)


# V_CE steps of the CE INPUT families.  In the Gummel-Poon model I_B changes with
# V_CE only below ~0.2 V (the base-collector junction starts to conduct), and the
# shift of the curve grows roughly with the logarithm of V_CE, so an OCTAVE sweep
# (0.025, 0.05, 0.1, 0.2 V) spaces the curves almost evenly.
#   kind "oct" / "dec" : geometric sweep from ``start`` to ``stop``
#   kind "list"        : explicit values (the earlier 0.05/0.1/0.15/5 V version)
CE_IN_SWEEP = {
    "npn": dict(kind="oct", start=0.025, stop=0.2, points=1),
    "pnp": dict(kind="oct", start=0.025, stop=0.2, points=1),
    # "npn": dict(kind="list", values=(0.05, 0.1, 0.15, 5)),
    # "pnp": dict(kind="list", values=(0.02, 0.05, 0.1, 5)),
}


def ce_in_sweep(pol):
    """(BVCE expression, number of V_CE steps) of the CE input family of ``pol``."""
    spec = dict(CE_IN_SWEEP[pol])
    kind = spec.pop("kind")
    return list_sweep(spec["values"]) if kind == "list" else geometric_sweep(kind, **spec)


# --------------------------------------------------------------------------- #
# Netlist builders.  Each returns the complete, self-contained .cir text of one
# characteristic family: a nested .DC sweep of two voltage sources
# (first source = inner / x axis, second source = outer / curve parameter).
# --------------------------------------------------------------------------- #
def cb_input_npn():
    return f"""* Common-base input characteristic (NPN 2N3904)
* inner sweep: VEE (emitter supply)   outer sweep: VCB = 5, 25, 45 V
* (the Early effect shifts the curves left by a few mV per 20 V of V_CB)
VEE 0 nes DC 0
RE nes e {RE_CB}
Q1 c 0 e Q2N3904
RC ncs c {RC_IN}
VCC ncs 0 DC 5
{MODEL_NPN}
{SAVE}
.DC VEE 0 5.3 0.001 VCC 5 45 20
.PROBE
.END
"""


def cb_output_npn():
    return f"""* Common-base output characteristic (NPN 2N3904)
* inner sweep: VCB (collector supply)   outer sweep: VEE = 0 ... 5 V
* (I_E = (VEE - V_EB)/RE, i.e. roughly 0, 0.4, 1.3, 2.3, 3.3, 4.3 mA)
VCC ncs 0 0
RC ncs c {RC_CB}
Q1 c 0 e Q2N3904
RE nes e {RE_CB}
VEE 0 nes 0
{MODEL_NPN}
{SAVE}
.DC VCC -2 12 0.02 VEE 0 5 1
.PROBE
.END
"""


def ce_input_npn():
    expr, n = ce_in_sweep("npn")
    return f"""* Common-emitter input characteristic (NPN 2N3904)
* inner sweep: VBB (base supply)   outer sweep: VCE = 0.025, 0.05, 0.1, 0.2 V
* This is an OCTAVE sweep of V_CE (PSpice: .DC LIN VBB 0 2 0.001 OCT VCC 0.025 0.2 1).
* ngspice has no OCT/DEC option, so the index source VK = 1 ... N is stepped
* linearly and the behavioural voltage source BVCE turns it into the geometric
* V_CE values.  (In the Gummel-Poon model I_B does not depend on V_CE once
* V_CE > ~0.3 V, so the manual's 5 V / 20 V curves would be one and the same line.)
VBB nbs 0 0
RB nbs b {RB_CE_IN}
Q1 c b 0 Q2N3904
RC ncs c {RC_IN}
VK k 0 1
BVCE ncs 0 V = {expr}
{MODEL_NPN}
{SAVE}
.DC VBB 0 2 0.001 VK 1 {n} 1
.PROBE
.END
"""


def ce_input_ref(pol):
    """Single CE input curve at V_CE = 5 V (active region), NOT plotted.

    The plotted CE input family stops at 0.2 V (the curves above that would merge),
    so the input resistance is measured on this separate V_CE = 5 V run instead."""
    npn = pol == "npn"
    return f"""* CE input characteristic at VCE = 5 V, measurement reference ({'NPN 2N3904' if npn else 'PNP 2N3906'})
VBB {'nbs 0' if npn else '0 nbs'} 0
RB nbs b {RB_CE_IN}
Q1 c b 0 {'Q2N3904' if npn else 'Q2N3906'}
RC ncs c {RC_IN}
VCC {'ncs 0' if npn else '0 ncs'} 5
{MODEL_NPN if npn else MODEL_PNP}
{SAVE}
.DC VBB 0 2 0.001 VCC 5 5 1
.PROBE
.END
"""


def ce_output_npn():
    return f"""* Common-emitter output characteristic (NPN 2N3904)
* inner sweep: VCE (collector supply)   outer sweep: VBB = 0, 2, ... 8 V
* (I_B = (VBB - V_BE)/RB, i.e. roughly 0, 14, 33, 53, 73 uA)
VBB nbs 0 0
RB nbs b {RB_CE_OUT}
Q1 c b 0 Q2N3904
RC ncs c {RC_CE}
VCC ncs 0 0
{MODEL_NPN}
{SAVE}
.DC VCC 0 10.4 0.01 VBB 0 8 2
.PROBE
.END
"""


def cc_input_npn():
    return f"""* Common-collector input characteristic (NPN 2N3904)
* collector at the common (ground) node, emitter supply VEE = -1 ... -4 V
* (VCE = |VEE|).  inner sweep: VBB (referred to ground, so x = VBC)
* outer sweep: VEE (emitter supply)
VEE nes 0 0
RE nes e {RE_CC_IN}
Q1 0 b e Q2N3904
VBB nbs 0 0
RB nbs b {RB_CC_IN}
{MODEL_NPN}
{SAVE}
.DC VBB -5 3 0.01 VEE -1 -4 -1
.PROBE
.END
"""


def cc_output_npn():
    return f"""* Common-collector output characteristic (NPN 2N3904), textbook wiring
* collector = common (ground) node, no collector resistor.
* VBB (+) -> RB -> base, VBB (-) on the collector rail.
* emitter supply VEE = -VCE feeds the emitter through RE.
* inner sweep: VEE (0.2 -> -5.2 V, i.e. VCE = -0.2 ... 5.2 V)
* outer sweep: VBB = -6, 14, 34, 54, 74 V.  With RB = 1 MOhm the base current
* (VBB + VCE - VBE)/RB is ~0, 14, 34, 54, 74 uA and drifts only a few uA with VCE.
VEE nes 0 0
RE nes e {RE_CC_OUT}
Q1 0 b e Q2N3904
VBB nbs 0 0
RB nbs b {RB_CC_OUT}
{MODEL_NPN}
{SAVE}
.DC VEE 0.2 -5.2 -0.025 VBB -6 74 20
.PROBE
.END
"""


def cb_input_pnp():
    return f"""* Common-base input characteristic (PNP 2N3906)
* inner sweep: VEE (emitter supply)   outer sweep: VCB = -5, -25, -45 V
VEE nes 0 DC 0
RE nes e {RE_CB}
Q1 c 0 e Q2N3906
RC ncs c {RC_IN}
VCC 0 ncs DC 5
{MODEL_PNP}
{SAVE}
.DC VEE 0 5.3 0.001 VCC 5 45 20
.PROBE
.END
"""


def cb_output_pnp():
    return f"""* Common-base output characteristic (PNP 2N3906)
* inner sweep: VCB (collector supply)   outer sweep: VEE = 0 ... 5 V
VCC 0 ncs 0
RC ncs c {RC_CB}
Q1 c 0 e Q2N3906
RE nes e {RE_CB}
VEE nes 0 0
{MODEL_PNP}
{SAVE}
.DC VCC -2 12 0.02 VEE 0 5 1
.PROBE
.END
"""


def ce_input_pnp():
    expr, n = ce_in_sweep("pnp")
    return f"""* Common-emitter input characteristic (PNP 2N3906)
* inner sweep: VBB (base supply)   outer sweep: VCE = -0.025, -0.05, -0.1, -0.2 V
* octave sweep of V_CE, emulated with the index source VK and the behavioural
* source BVCE exactly as in the NPN netlist.
VBB 0 nbs 0
RB nbs b {RB_CE_IN}
Q1 c b 0 Q2N3906
RC ncs c {RC_IN}
VK k 0 1
BVCE 0 ncs V = {expr}
{MODEL_PNP}
{SAVE}
.DC VBB 0 2 0.001 VK 1 {n} 1
.PROBE
.END
"""


def ce_output_pnp():
    return f"""* Common-emitter output characteristic (PNP 2N3906)
* inner sweep: VCE (collector supply)   outer sweep: VBB = 0, -2, ... -8 V
VBB 0 nbs 0
RB nbs b {RB_CE_OUT}
Q1 c b 0 Q2N3906
RC ncs c {RC_CE}
VCC 0 ncs 0
{MODEL_PNP}
{SAVE}
.DC VCC 0 10.4 0.01 VBB 0 8 2
.PROBE
.END
"""


def cc_input_pnp():
    return f"""* Common-collector input characteristic (PNP 2N3906)
* collector at the common (ground) node, emitter supply VEE = 1 ... 4 V
* (VCE = VEE).  inner sweep: VBB (referred to ground, swept 5 -> -3 V)
* outer sweep: VEE (emitter supply)
VEE nes 0 0
RE nes e {RE_CC_IN}
Q1 0 b e Q2N3906
VBB nbs 0 0
RB nbs b {RB_CC_IN}
{MODEL_PNP}
{SAVE}
.DC VBB 5 -3 -0.01 VEE 1 4 1
.PROBE
.END
"""


def cc_output_pnp():
    return f"""* Common-collector output characteristic (PNP 2N3906), textbook wiring
* collector = common (ground) node, no collector resistor.
* VBB (-) -> RB -> base, VBB (+) on the collector rail (base at -VBB).
* emitter supply VEE = +VCE feeds the emitter through RE.
* inner sweep: VEE (-0.2 -> 5.2 V = VCE)
* outer sweep: VBB = -6, 14, 34, 54, 74 V (I_B ~ 0, 14, 34, 54, 74 uA)
VEE nes 0 0
RE nes e {RE_CC_OUT}
Q1 0 b e Q2N3906
VBB 0 nbs 0
RB nbs b {RB_CC_OUT}
{MODEL_PNP}
{SAVE}
.DC VEE -0.2 5.2 0.025 VBB -6 74 20
.PROBE
.END
"""


# --------------------------------------------------------------------------- #
# Vector extraction: map raw vectors to the plotted x and y arrays
# --------------------------------------------------------------------------- #
def _vec(v, term):
    """Device current vector @q1[term] (the raw-file name differs by version)."""
    for name in (f"i(@q1[{term}])", f"@q1[{term}]"):
        if name in v:
            return np.asarray(v[name], dtype=float)
    raise KeyError(f"vector for @q1[{term}] not found in the raw file")


def _asc(x, *others):
    """Sort x ascending and reorder the other arrays alongside it."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    return (x[order],) + tuple(np.asarray(o, dtype=float)[order] for o in others)


def extract(key, v):
    """Return (x, y, drive) for one simulated curve, in SI units.

    drive is the actual base (CE, CC) or emitter (CB) current of the curve:
    with voltage-source bias it is set by V/R and is not exactly a round
    number, so the measurements use the simulated value.
    Currents are signed so that normal (active-region) operation is positive;
    reverse currents (e.g. CB output curve at V_CB < 0) and the pA leakage
    noise are clipped to zero, as only the first quadrant is plotted.
    """
    s = 1.0 if key.endswith("npn") else -1.0          # NPN: Ic, Ib > 0, Ie < 0
    ic = np.clip(s * _vec(v, "ic"), 0.0, None)
    ib = np.clip(s * _vec(v, "ib"), 0.0, None)
    ie = np.clip(-s * _vec(v, "ie"), 0.0, None)
    vb = np.asarray(v.get("v(b)", 0.0), dtype=float)
    vc = np.asarray(v.get("v(c)", 0.0), dtype=float)
    ve = np.asarray(v.get("v(e)", 0.0), dtype=float)

    if key == "cb_input_npn":
        return _asc(-ve, ie, ie)
    if key == "cb_output_npn":
        return _asc(vc, ic, ie)
    if key == "ce_input_npn":
        return _asc(vb, ib, ib)
    if key == "ce_output_npn":
        return _asc(vc, ic, ib)
    if key == "cc_input_npn":
        return _asc(vb, ib, ib)
    if key == "cc_output_npn":
        return _asc(vc - ve, ie, ib)
    if key == "cb_input_pnp":
        return _asc(ve, ie, ie)
    if key == "cb_output_pnp":
        return _asc(-vc, ic, ie)
    if key == "ce_input_pnp":
        return _asc(-vb, ib, ib)
    if key == "ce_output_pnp":
        return _asc(-vc, ic, ib)
    if key == "cc_input_pnp":
        return _asc(-vb, ib, ib)
    if key == "cc_output_pnp":
        return _asc(ve - vc, ie, ib)
    raise KeyError(key)


def split_family(key, v, outer, signed=False):
    """Split the single nested-sweep result into its curves.

    ngspice writes a nested sweep as one long plot (outer source varies
    slowest); the outer source value is constant along each curve, so the
    curves are the contiguous blocks over which outer(v) does not change.
    Returns [(outer value, x, y, drive), ...] in sweep order; the outer value
    is |value| unless ``signed`` is set (CC output families, V_BB = -6 V).
    """
    v = {k: (np.asarray(a) if np.ndim(a) else a) for k, a in v.items()}
    o = np.round(np.asarray(outer(v), dtype=float), 6) + 0.0
    cuts = np.flatnonzero(np.diff(o) != 0) + 1
    curves = []
    for blk in np.split(np.arange(o.size), cuts):
        sub = {k: (np.asarray(a)[blk] if np.ndim(a) else a) for k, a in v.items()}
        x, y, d = extract(key, sub)
        val = float(o[blk[0]])
        curves.append((val if signed else abs(val), x, y, d))
    return curves


# --------------------------------------------------------------------------- #
# Family definitions.  "outer" returns the outer-sweep source voltage from the
# raw vectors (it identifies the curves); the legend shows that source voltage.
# --------------------------------------------------------------------------- #
FAMILIES = [
    dict(key="cb_input_npn", build=cb_input_npn,
         outer=lambda v: v["v(ncs)"],
         param="VCB", trace="IE(Q1)",
         xlabel="Emitter-base voltage $V_{EB}$ (V)",
         ylabel="Emitter current $I_E$", fname="cb_input_npn",
         xlim=(0.58, 0.72)),
    dict(key="cb_output_npn", build=cb_output_npn,
         outer=lambda v: v["v(nes)"],
         param="VEE", trace="IC(Q1)",
         xlabel="Collector-base voltage $V_{CB}$ (V)",
         ylabel="Collector current $I_C$", fname="cb_output_npn",
         xlim=(-1, 10.2)),
    dict(key="ce_input_npn", build=ce_input_npn,
         outer=lambda v: v["v(ncs)"],
         param="VCE", trace="IB(Q1)",
         xlabel="Base-emitter voltage $V_{BE}$ (V)",
         ylabel="Base current $I_B$", fname="ce_input_npn",
         xlim=(0.52, 0.78), ylim=(0.0, 100e-6)),
    dict(key="ce_output_npn", build=ce_output_npn,
         outer=lambda v: v["v(nbs)"],
         param="VBB", trace="IC(Q1)",
         xlabel="Collector-emitter voltage $V_{CE}$ (V)",
         ylabel="Collector current $I_C$", fname="ce_output_npn",
         xlim=(-0.2, 10.2)),
    dict(key="cc_input_npn", build=cc_input_npn,
         outer=lambda v: v["v(nes)"],
         param="VCE", trace="IB(Q1)",
         xlabel="Base-collector voltage $V_{BC}$ (V)",
         ylabel="Base current $I_B$", fname="cc_input_npn",
         xlim=(-4, 0.5), ylim=(0.0, 120e-6), trim=True),
    dict(key="cc_output_npn", build=cc_output_npn,
         outer=lambda v: np.asarray(v["v(nbs)"], dtype=float),
         param="VBB", trace="IE(Q1)", signed=True,
         xlabel="Collector-emitter voltage $V_{CE}$ (V)",
         ylabel="Emitter current $I_E$", fname="cc_output_npn",
         xlim=(-0.2, 5.2)),
    dict(key="cb_input_pnp", build=cb_input_pnp,
         outer=lambda v: v["v(ncs)"],
         param="|VCB|", trace="|IE(Q1)|",
         xlabel="Emitter-base voltage $|V_{EB}|$ (V)",
         ylabel="Emitter current $|I_E|$", fname="cb_input_pnp",
         xlim=(0.58, 0.80)),
    dict(key="cb_output_pnp", build=cb_output_pnp,
         outer=lambda v: v["v(nes)"],
         param="|VEE|", trace="|IC(Q1)|",
         xlabel="Collector-base voltage $|V_{CB}|$ (V)",
         ylabel="Collector current $|I_C|$", fname="cb_output_pnp",
         xlim=(-1, 10.2)),
    dict(key="ce_input_pnp", build=ce_input_pnp,
         outer=lambda v: v["v(ncs)"],
         param="|VCE|", trace="|IB(Q1)|",
         xlabel="Base-emitter voltage $|V_{BE}|$ (V)",
         ylabel="Base current $|I_B|$", fname="ce_input_pnp",
         xlim=(0.52, 0.80), ylim=(0.0, 100e-6)),
    dict(key="ce_output_pnp", build=ce_output_pnp,
         outer=lambda v: v["v(nbs)"],
         param="|VBB|", trace="|IC(Q1)|",
         xlabel="Collector-emitter voltage $|V_{CE}|$ (V)",
         ylabel="Collector current $|I_C|$", fname="ce_output_pnp",
         xlim=(-0.2, 10.2)),
    dict(key="cc_input_pnp", build=cc_input_pnp,
         outer=lambda v: v["v(nes)"],
         param="|VCE|", trace="|IB(Q1)|",
         xlabel="Base-collector voltage $V_{CB}$ (V)",
         ylabel="Base current $|I_B|$", fname="cc_input_pnp",
         xlim=(-4, 0.5), ylim=(0.0, 120e-6), trim=True),
    dict(key="cc_output_pnp", build=cc_output_pnp,
         outer=lambda v: -np.asarray(v["v(nbs)"], dtype=float),    # = +VBB for the pnp
         param="VBB", trace="|IE(Q1)|", signed=True,
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


def plot_family(curves, param, trace, xlabel, ylabel, outpath, xlim=None, dpi=300,
                ylim=None, trim=False):
    """Draw one characteristic family on a PSpice/Probe-style page.

    ``curves`` is [(stepped source value, x, y), ...].  All curves share one
    colour and marker, a single legend entry names ``trace`` (e.g. "IB(Q1)"),
    and the value of the stepped source is written beside every curve:
    "VCE=0V" on the first one, "5V", "10V", ... on the others."""
    # the sweeps run slightly beyond the plotted window: keep only what is shown
    if xlim is not None:
        cropped = []
        for val, x, y in curves:
            m = (x >= xlim[0]) & (x <= xlim[1])
            if ylim is not None:                  # stop the curve at the top of the window
                m &= y <= ylim[1]
            x, y = x[m], y[m]
            if trim and len(y) > 2:               # drop the flat I = 0 run before turn-on
                on = np.flatnonzero(y > 1e-7)
                if on.size:
                    i0 = max(int(on[0]) - 1, 0)
                    x, y = x[i0:], y[i0:]
            cropped.append((val, x, y))
        curves = cropped
    curves = [c for c in curves if len(c[1]) >= 2]
    if not curves:
        sys.exit("plot_family: no curve has 2+ points inside xlim; check the "
                 "outer-sweep source (curve splitting) and the x-limits")
    xs = np.concatenate([c[1] for c in curves])
    ys = np.concatenate([c[2] for c in curves])
    if xlim is None:
        xlim = (float(xs.min()), float(xs.max()))
    if ylim is None:
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

    # same colour + marker everywhere; the first curve on top, markers staggered
    for k in reversed(range(len(curves))):
        _, x, y = curves[k]
        draw_trace(ax, x, y, frame, CURVE_COLOR, CURVE_MARKER, max(4 - k, 1),
                   MARKER_FIRST_PT + k * MARKER_SPACING_PT / len(curves))

    # source value beside each curve, in a free spot next to the curve's end
    items = [(x, y, (f"{param}=" if k == 0 else "") + f"{val:g}V", "black")
             for k, (val, x, y) in enumerate(curves)]
    place_sweep_labels(fig, ax, [(x, y) for _, x, y in curves], items)

    # one legend entry for the whole family
    add_family_footer(fig, [(CURVE_MARKER, CURVE_COLOR, trace)], xlabel, ax)
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
    x = np.asarray(x, dtype=float)
    if not (x.min() <= x_target <= x.max()):
        print(f"  WARNING: x = {x_target:g} V is outside the simulated range "
              f"[{x.min():.3g}, {x.max():.3g}] V; value is clamped",
              file=sys.stderr)
    return float(np.interp(x_target, x, np.asarray(y, dtype=float)))


def _at(curve, x_ref):
    """(y, drive current) of one curve at x = x_ref."""
    _, x, y, d = curve
    return y_at_x(x_ref, x, y), y_at_x(x_ref, x, d)


def _xy(curve):
    return curve[1], curve[2]


def curve_at(curves, val, tol=0.5):
    """The curve whose stepped source value is closest to val (warns if far)."""
    c = min(curves, key=lambda c: abs(abs(c[0]) - val))
    if abs(abs(c[0]) - val) > tol:
        print(f"  WARNING: no curve at {val:g} V; using the {c[0]:g} V curve",
              file=sys.stderr)
    return c


def measure_polarity(d, pol):
    """Gain and port resistances from one polarity's curve families.

    The output families are stepped by a voltage source, so the base / emitter
    current of each curve is whatever V/R gives (not a round number).  The
    measurements therefore use the *simulated* drive current of the two curves
    they compare: the curves are picked by position in the family (the last
    two curves for CB, the 3rd and 4th for CE / CC) and the drive currents
    are read from the data at the evaluation voltage.
    """
    cb_in, cb_out = d["cb_input_" + pol], d["cb_output_" + pol]
    ce_in, ce_out = d["ce_input_" + pol], d["ce_output_" + pol]
    cc_in, cc_out = d["cc_input_" + pol], d["cc_output_" + pol]
    # reference input curves: V_CB = 5 V (CB), V_CE = 5 V (CE, separate reference
    # run because the plotted CE family stops at 0.2 V) and V_CE = 4 V (CC)
    cb_in5 = curve_at(cb_in, 5.0)
    ce_in5 = curve_at(d["ce_input_ref_" + pol], 5.0)
    cc_in5 = curve_at(cc_in, 4.0)      # the VCE = 4 V curve

    # CB: alpha from the output family between the last two curves at V_CB = 5 V
    ca, cb_ = cb_out[-2], cb_out[-1]
    ic_a, ie_a = _at(ca, 5.0)
    ic_b, ie_b = _at(cb_, 5.0)
    alpha = (ic_b - ic_a) / (ie_b - ie_a)
    # CB input resistance between the same two emitter currents at V_CB = 5 V
    veb_a = x_at_y(ie_a, *_xy(cb_in5))
    veb_b = x_at_y(ie_b, *_xy(cb_in5))
    rin_cb = (veb_b - veb_a) / (ie_b - ie_a)
    # CB output resistance on the highest-I_E curve, V_CB 5 -> 10 V, at constant
    # I_E: with voltage-source bias I_E drifts slightly with V_CB, so the change
    # of I_C is taken as I_E * d(alpha) with alpha = I_C / I_E
    ic5v, ie5v = _at(cb_, 5.0)
    ic10v, ie10v = _at(cb_, 10.0)
    rout_cb = 5.0 / (ie5v * (ic10v / ie10v - ic5v / ie5v))

    # CE: beta between the 3rd and 4th curves at V_CE = 5 V
    ca_e, cb_e = ce_out[2], ce_out[3]
    ic_a_e, ib_a_e = _at(ca_e, 5.0)
    ic_b_e, ib_b_e = _at(cb_e, 5.0)
    beta = (ic_b_e - ic_a_e) / (ib_b_e - ib_a_e)
    vbe_a = x_at_y(ib_a_e, *_xy(ce_in5))
    vbe_b = x_at_y(ib_b_e, *_xy(ce_in5))
    rin_ce = (vbe_b - vbe_a) / (ib_b_e - ib_a_e)
    # CE output resistance on the 4th curve, V_CE 5 -> 10 V, at constant I_B
    # (change of I_C taken as I_B * d(beta), beta = I_C / I_B)
    ic5v_e, ib5v_e = _at(cb_e, 5.0)
    ic10v_e, ib10v_e = _at(cb_e, 10.0)
    rout_ce = 5.0 / (ib5v_e * (ic10v_e / ib10v_e - ic5v_e / ib5v_e))

    # CC: gamma between the 3rd and 4th curves at V_CE = 5 V (same as beta)
    ca_c, cb_c = cc_out[2], cc_out[3]
    ie_a_c, ib_a_c = _at(ca_c, 5.0)
    ie_b_c, ib_b_c = _at(cb_c, 5.0)
    gamma = (ie_b_c - ie_a_c) / (ib_b_c - ib_a_c)
    vbe_a_c = x_at_y(ib_a_c, *_xy(cc_in5))
    vbe_b_c = x_at_y(ib_b_c, *_xy(cc_in5))
    rin_cc = (vbe_b_c - vbe_a_c) / (ib_b_c - ib_a_c)
    # CC output resistance on the 4th curve, V_CE 3 -> 5 V, at constant I_B
    # (change of I_E taken as I_B * d(gamma), gamma = I_E / I_B)
    ie3, ib3 = _at(cb_c, 3.0)
    ie5, ib5 = _at(cb_c, 5.0)
    rout_cc = 2.0 / (ib5 * (ie5 / ib5 - ie3 / ib3))

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
    ic_mid_ce = y_at_x(7.5, *_xy(cb_e))
    ie_mid_cc = y_at_x(4.0, *_xy(cb_c))
    th = dict(
        alpha=HFE_TYP / (1 + HFE_TYP),
        beta=HFE_TYP,
        gamma=1 + HFE_TYP,
        # chord slopes of the ideal exponential input characteristic
        rin_cb=VT * np.log(ie_b / ie_a) / (ie_b - ie_a),
        rin_ce=VT * np.log(ib_b_e / ib_a_e) / (ib_b_e - ib_a_e),
        rin_cc=VT * np.log(ib_b_c / ib_a_c) / (ib_b_c - ib_a_c),
        # CB at fixed emitter current: d(alpha)/dV is divided by I_E
        rout_cb=5.0 / (ie_b * (alpha_at(10.0) - alpha_at(5.0))),
        alpha5=alpha_at(5.0), alpha10=alpha_at(10.0),
        rout_ce=(vaf + 7.5) / ic_mid_ce,
        rout_cc=(vaf + 4.0) / ie_mid_cc,
    )
    # alpha5/alpha10 are helper terms of rout_cb, not measured parameters
    err = {k: (th[k] - sim[k]) / th[k] * 100 for k in sim}

    # evaluation points (key names kept from the earlier version of the script;
    # "_a"/"_b" are the two curves compared, whose drive currents are in "bias")
    pts = dict(ic_cb4=ic_a, ic_cb5=ic_b,
               ic_cb4_5v=ic_a, ic_cb4_10v=ic10v,
               veb_3m=veb_a, veb_5m=veb_b,
               ic_ce40_5v=ic_a_e, ic_ce60_5v=ic_b_e,
               ic_ce60_7v5=ic_mid_ce, ic_ce60_10v=ic10v_e,
               vbe_40u=vbe_a_c, vbe_60u=vbe_b_c,
               ie_cc40_3v=y_at_x(3.0, *_xy(ca_c)),
               ie_cc60_3v=ie3,
               ie_cc40_5v=ie_a_c, ie_cc60_5v=ie_b_c,
               ie_cc60_4v=ie_mid_cc)

    def knee(curve, ref_x, frac=0.9):
        """Saturation knee: the lowest x at which the trace has reached frac of its
        value at ref_x (just inside the active region)."""
        x, y = np.asarray(curve[0], float), np.asarray(curve[1], float)
        o = np.argsort(x)
        x, y = x[o], y[o]
        target = frac * np.interp(ref_x, x, y)
        return float(x[np.where(y >= target)[0][0]])

    regions = dict(
        sat_cb=knee(_xy(cb_), 0.5),               # V_CB at the CB saturation knee
        sat_ce=knee(_xy(cb_e), 1.0),              # V_CE at the CE saturation knee
        sat_cc=knee(_xy(cb_c), 1.0),              # V_CE at the CC saturation knee
        ib_cutoff_ce=y_at_x(5.0, *_xy(ce_out[0])),  # collector leakage, V_BB = 0, V_CE = 5 V
        ie_cutoff_cb=y_at_x(5.0, *_xy(cb_out[0])),  # collector leakage, V_EE = 0, V_CB = 5 V
        vbe_on_ce=x_at_y(10e-6, *_xy(ce_in5)),    # V_BE at I_B = 10 uA
        veb_on_cb=x_at_y(1e-3, *_xy(cb_in5)),     # V_EB at I_E = 1 mA
    )
    return dict(simulated=sim, theoretical=th, error=err, points=pts, regions=regions,
                bias=dict(ie_cb=(ie_a + ie_b) / 2, ic_cb=ic_a,
                          ib_ce=(ib_a_e + ib_b_e) / 2, ic_ce=ic_b_e,
                          ib_cc=(ib_a_c + ib_b_c) / 2, ie_cc=ie_b_c,
                          ie_cb_a=ie_a, ie_cb_b=ie_b,
                          ib_ce_a=ib_a_e, ib_ce_b=ib_b_e,
                          ib_cc_a=ib_a_c, ib_cc_b=ib_b_c))


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
        text = fam["build"]()
        vectors = simulate_text(text, key)          # one nested-sweep run
        curves = split_family(key, vectors, fam["outer"], signed=fam.get("signed", False))
        for p, x, y, dr in curves:
            print(f"  {key:14s} source={p:<6g} {len(x):4d} pts  "
                  f"y: {np.min(y):.4g} .. {np.max(y):.4g}   "
                  f"drive: {np.max(dr):.4g}")
        all_data[key] = curves

        # the netlist on file is the complete nested-sweep netlist of the family
        (netdir / f"{fam['fname']}.cir").write_text(text)

        if not args.no_plots:
            curves_lab = [(p, x, y) for p, x, y, _ in curves]
            plot_family(curves_lab, fam["param"], fam["trace"],
                        fam["xlabel"], fam["ylabel"],
                        outdir / f"{fam['fname']}.png",
                        xlim=fam["xlim"], dpi=args.dpi,
                        ylim=fam.get("ylim"), trim=fam.get("trim", False))
            print(f"    -> {fam['fname']}.png")

    # V_CE = 5 V reference curves of the CE input characteristic (measurements only)
    for pol in ("npn", "pnp"):
        text = ce_input_ref(pol)
        vectors = simulate_text(text, f"ce_input_ref_{pol}")
        all_data[f"ce_input_ref_{pol}"] = split_family(
            f"ce_input_{pol}", vectors, lambda v: v["v(ncs)"])
        (netdir / f"ce_input_ref_{pol}.cir").write_text(text)

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
