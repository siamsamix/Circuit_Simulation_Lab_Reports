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
  CE input   I_B vs V_BE    V_CE = 0.01, 0.1, 1 V (decade sweep, see below)
  CE output  I_C vs V_CE    I_B stepped by V_BB: 0, 14, 33, 53, 73 uA
  CC input   I_B vs V_BC    V_CE = 1, 2, 3, 4, 5 V
  CC output  I_E vs V_CE    I_B ~ 0, 14, 34, 54, 74 uA (V_BB = -6 ... 74 V, R_B = 1 MOhm)

Note on the CE input family: in the Gummel-Poon model the base current does not
depend on V_CE once V_CE is above ~0.3 V, so the manual's V_CE = 5 V and 20 V
curves would be one and the same line.  The curves are therefore taken at the
edge of saturation with a DECADE sweep of V_CE (CE_IN_SWEEP: 0.01, 0.1, 1 V), which
spaces them almost evenly.  The CE input resistance is measured on the highest-V_CE
curve of that plotted family; the separate, not plotted V_CE = 5 V run
(ce_input_ref_*.cir) only supplies the turn-on voltage.

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

Calculation points
  Every graph shows the TWO points that the resistance calculation of that graph
  uses (PSpice cursor style, see mark_points / draw_cursor_*):
    input graphs   the two points on the reference input curve between which
                   R_in = dV / dI is formed (V_CB = 5 V for CB, the highest-V_CE curve,
                   1 V, for CE, V_CE = 5 V for CC = the V_CE of the output-graph points);
    output graphs  the two points on the curve used for R_out = dV / dI
                   (V_CB / V_CE = 5 V and 10 V, or V_CE = 3 V and 5 V for CC).
  Each point has a horizontal line to the y axis and a vertical line to the x
  axis, the two points are joined by a hypotenuse, and the horizontal and
  vertical lines through them close a 90-degree triangle whose legs are dV and dI.
  The coordinates of both points are written in grey-framed read-out boxes.
  The output graphs ALSO mark the two points of the current-gain calculation
  (same V, two neighbouring curves: a vertical segment; one of them coincides
  with a point of the R_out pair), so every output graph shows three points:
    CB  (V_CB, I_C) = (5 V, 4 V step), (5 V, 5 V step), (10 V, 5 V step)
    CE  (V_CE, I_C) = (5 V, 4 V step), (5 V, 6 V step), (10 V, 6 V step)
    CC  (V_CE, I_E) = (5 V, 34 V step), (5 V, 54 V step), (3 V, 54 V step)
  The read-out box of each output-graph point has two lines: the point
  (x,y) and the input current of that curve at that x (I_E for CB, I_B for
  CE and CC), e.g. "IE=3.30422m".  --no-gain-points drops the gain pair again.

Operating regions (output graphs)
  Every output graph also shows where the operating regions lie: a vertical dashed
  line at the saturation knee separates SATURATION (to the left) from ACTIVE (to the
  right), each region is bracketed along the top of the graph, and CUTOFF is written
  just above the lowest curve (I_E = 0 for CB, I_B = 0 for CE / CC).  The knee is the
  one found in measure_polarity (regions["sat_cb" / "sat_ce" / "sat_cc"], also in
  results.json).  --no-regions drops the region markers again.
  The knee itself is read out too: a ring is drawn where the knee line meets the curve it
  was found on (CB: highest-I_E curve, CE / CC: 4th curve) and a red-framed box
  "Knee (90%) / (V, I)" gives its coordinates (regions["sat_*_ic" / "sat_cc_ie"] in results.json).
  Definition (KNEE_FRAC, KNEE_REF): the knee is the lowest voltage at which the plotted
  current reaches 90 % of its value at a reference voltage inside the active region.  The
  sentence to quote in the report is stored in results.json ("knee_definition") and printed.

Output resistance
  The change of the output current is referred to constant input current with
  the measured gains of the task sheet: dIc|Ie = dIc - alpha*dIe (CB),
  dIc|Ib = dIc - beta*dIb (CE), dIe|Ib = dIe - gamma*dIb (CC).

Usage : python simulate_bjt.py [--dpi 300] [--netlists ../netlist] [--out .]
                               [--no-gain-points] [--no-cursors] [--no-regions]
                               [--show-negative]
  --show-negative  the six output graphs also show the reverse (negative-current) part
                   (wider V sweeps, y axis ~12 % below zero); files get a _neg suffix.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

from plot_full_bridge import add_note_at, draw_trace, note_fontsize
from pspice_theme import (AX_H_PT, AX_RECT, FIG_W_PT, MARKER_FIRST_PT,
                          MARKER_SPACING_PT, NOTE_EDGE, NOTE_SERIF, add_y_label,
                          choose_font, dense_curve, label_widths_pt,
                          major_ticks, minor_between, new_page, rect_is_free, spice_format,
                          style_axes, text_width)
from simulate_and_plot import (nice_step, parse_raw_file, place_note_near_trace,
                               place_sweep_labels, run_ngspice)

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

# --show-negative: output graphs also show the reverse (negative-current) part.
# main() sets these three before anything is simulated; defaults = original graphs.
SHOW_NEG = False
CE_OUT_VMIN = "0"       # start of the CE output V_CE sweep (-2 with --show-negative)
CC_OUT_VMIN = "-0.2"    # start of the CC output V_CE sweep (-1.5 with --show-negative)

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

# Calculation points (PSpice cursor style): thin black drops to the axes, thicker
# black legs + hypotenuse of the right triangle, grey-framed coordinate boxes.
CURSOR_COLOR = "black"
CURSOR_EDGE_LW = 1.5       # legs and hypotenuse of the triangle
CURSOR_DROP_LW = 0.9       # lines from the points to the axes
CURSOR_DOT_PT = 4.2        # dot on each of the two points (0 = no dots)
CURSOR_GAP_PT = 3.0        # clearance of a read-out box from labels, notes, other boxes
CURSOR_CURVE_COST = 0.5    # placement cost per pt of curve a read-out box would hide
CURSOR_LINE_COST = 1.5     # ... and per pt of cursor line it would hide
CURSOR_HALO_PT = 5.0       # output graphs: keep read-outs this far from the drawn curves
                           # (covers the 3 pt trace and the 8 pt markers, not just the centre line)

# Operating-region markers of the output graphs: dashed knee line + brackets + labels.
REGION_COLOR = "#b3261e"
REGION_LW = 1.4
REGION_TICK_PT = 4.0       # half-height of the bracket end bars
REGION_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]   # keeps names legible over grid lines
CUTOFF_RAISE_PT = 9.0      # "Cutoff" is written this far above the I = 0 curve (clears its markers)

# PSpice/Probe grid: solid black major lines plus dotted minor lines on the y axis
# (Y_MINOR_SUBDIV dotted lines per major interval).  Set SOFT_GRID = True to recolour
# both light grey so the curves and the black calculation cursors stand out more.
SOFT_GRID = False
GRID_COLOR, GRID_LW = "#c9c9c9", 0.8
Y_MINOR_SUBDIV = 10

# Reference current gain for the theoretical values: the middle of the datasheet
# h_FE range of the 2N3904 / 2N3906 at I_C = 10 mA (100 ... 300).
HFE_TYP = 200.0
VAF = {"npn": 74.03, "pnp": 18.7}
VT = 0.02585                       # thermal voltage at 27 degC (V)

# Saturation knee of the output graphs: the lowest voltage at which the plotted current has
# reached KNEE_FRAC of its value at the reference voltage KNEE_REF (inside the active region).
KNEE_FRAC = 0.9
KNEE_REF = dict(cb=0.5, ce=1.0, cc=1.0)           # V_CB (CB) / V_CE (CE, CC) in volts
KNEE_TEXT = (f"saturation knee = lowest voltage at which the plotted output current reaches "
             f"{KNEE_FRAC:.0%} of its value at V_CB = {KNEE_REF['cb']:g} V (CB, highest-I_E curve), "
             f"V_CE = {KNEE_REF['ce']:g} V (CE, 4th curve) or V_CE = {KNEE_REF['cc']:g} V (CC, 4th curve)")

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
    "npn": dict(kind="dec", start=0.01, stop=1, points=1),
    "pnp": dict(kind="dec", start=0.01, stop=1, points=1),
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
VCC ncs 0 DC 0
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
* inner sweep: VBB (base supply)   outer sweep: VCE = 0.01, 0.1, 1 V (CE_IN_SWEEP)
* This is a DECADE sweep of V_CE (PSpice: .DC LIN VBB 0 2 0.001 DEC VCC 0.01 1 1).
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

    Only used for the turn-on voltage (vbe_on_ce); R_in is measured on the plotted family."""
    npn = pol == "npn"
    return f"""* CE input characteristic at VCE = 5 V, measurement reference ({'NPN 2N3904' if npn else 'PNP 2N3906'})
VBB {'nbs 0' if npn else '0 nbs'} 0
RB nbs b {RB_CE_IN}
Q1 c b 0 {'Q2N3904' if npn else 'Q2N3906'}
RC ncs c {RC_IN}
VCC {'ncs 0' if npn else '0 ncs'} 0
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
.DC VCC {CE_OUT_VMIN} 10.4 0.01 VBB 0 8 2
.PROBE
.END
"""


def cc_input_npn():
    return f"""* Common-collector input characteristic (NPN 2N3904)
* collector at the common (ground) node, emitter supply VEE = -1 ... -5 V
* (VCE = |VEE|).  inner sweep: VBB (referred to ground, so x = VBC)
* outer sweep: VEE (emitter supply)
VEE 0 nes 0
RE nes e {RE_CC_IN}
Q1 0 b e Q2N3904
VBB nbs 0 0
RB nbs b {RB_CC_IN}
{MODEL_NPN}
{SAVE}
.DC VBB -6 3 0.01 VEE 1 5 1
.PROBE
.END
"""


def cc_output_npn():
    return f"""* Common-collector output characteristic (NPN 2N3904), textbook wiring
* collector = common (ground) node, no collector resistor.
* VBB (+) -> RB -> base, VBB (-) on the collector rail.
* emitter supply VEE = -VCE feeds the emitter through RE.
* inner sweep: VEE (VCE = {CC_OUT_VMIN} ... 5.2 V)
* outer sweep: VBB = -6, 14, 34, 54, 74 V.  With RB = 1 MOhm the base current
* (VBB + VCE - VBE)/RB is ~0, 14, 34, 54, 74 uA and drifts only a few uA with VCE.
VEE 0 nes 0
RE nes e {RE_CC_OUT}
Q1 0 b e Q2N3904
VBB nbs 0 0
RB nbs b {RB_CC_OUT}
{MODEL_NPN}
{SAVE}
.DC VEE {CC_OUT_VMIN} 5.2 0.025 VBB -6 74 20
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
VCC 0 ncs DC 0
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
* inner sweep: VBB (base supply)   outer sweep: VCE = -0.01, -0.1, -1 V (CE_IN_SWEEP)
* decade sweep of V_CE, emulated with the index source VK and the behavioural
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
.DC VCC {CE_OUT_VMIN} 10.4 0.01 VBB 0 8 2
.PROBE
.END
"""


def cc_input_pnp():
    return f"""* Common-collector input characteristic (PNP 2N3906)
* collector at the common (ground) node, emitter supply VEE = 1 ... 5 V
* (VCE = VEE).  inner sweep: VBB (referred to ground, swept 6 -> -3 V)
* outer sweep: VEE (emitter supply)
VEE nes 0 0
RE nes e {RE_CC_IN}
Q1 0 b e Q2N3906
VBB 0 nbs 0
RB nbs b {RB_CC_IN}
{MODEL_PNP}
{SAVE}
.DC VBB -6 3 0.01 VEE 1 5 1
.PROBE
.END
"""


def cc_output_pnp():
    return f"""* Common-collector output characteristic (PNP 2N3906), textbook wiring
* collector = common (ground) node, no collector resistor.
* VBB (-) -> RB -> base, VBB (+) on the collector rail (base at -VBB).
* emitter supply VEE = +VCE feeds the emitter through RE.
* inner sweep: VEE (-0.2 -> 5.2 V = VCE)
* outer sweep: VBB = -6, 14, 34, 54, 74 V as set on the source, i.e. the base supply is
* +6, -14, -34, -54, -74 V (I_B ~ 0, 14, 34, 54, 74 uA)
VEE nes 0 0
RE nes e {RE_CC_OUT}
Q1 0 b e Q2N3906
VBB 0 nbs 0
RB nbs b {RB_CC_OUT}
{MODEL_PNP}
{SAVE}
.DC VEE {CC_OUT_VMIN} 5.2 0.025 VBB -6 74 20
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
    ib_drive, ie_drive = ib, ie                      # drive currents always stay clipped
    if SHOW_NEG and "_output_" in key:
        # --show-negative: keep the reverse (negative) part of the plotted current
        ic = s * _vec(v, "ic")
        ie = -s * _vec(v, "ie")
    vb = np.asarray(v.get("v(b)", 0.0), dtype=float)
    vc = np.asarray(v.get("v(c)", 0.0), dtype=float)
    ve = np.asarray(v.get("v(e)", 0.0), dtype=float)

    if key == "cb_input_npn":
        return _asc(-ve, ie, ie)
    if key == "cb_output_npn":
        return _asc(vc, ic, ie_drive)
    if key == "ce_input_npn":
        return _asc(vb, ib, ib)
    if key == "ce_output_npn":
        return _asc(vc, ic, ib_drive)
    if key == "cc_input_npn":
        return _asc(vb, ib, ib)
    if key == "cc_output_npn":
        return _asc(vc - ve, ie, ib_drive)
    if key == "cb_input_pnp":
        return _asc(ve, ie, ie)
    if key == "cb_output_pnp":
        return _asc(-vc, ic, ie_drive)
    if key == "ce_input_pnp":
        return _asc(-vb, ib, ib)
    if key == "ce_output_pnp":
        return _asc(-vc, ic, ib_drive)
    if key == "cc_input_pnp":
        return _asc(-vb, ib, ib)
    if key == "cc_output_pnp":
        return _asc(ve - vc, ie, ib_drive)
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
# Fraction of the largest plotted current that the y axis of an OUTPUT graph extends
# below zero (the reverse / negative part of the curves: V_CB, V_CE < 0).
Y_NEG = 0.12

FAMILIES = [
    dict(key="cb_input_npn", build=cb_input_npn,
         outer=lambda v: v["v(ncs)"],
         param="VCB", trace="IE(Q1)",
         xlabel="Emitter-base voltage $V_{EB}$ (V)",
         ylabel="Emitter current $I_E$", fname="cb_input_npn", resistor="RE",
         xlim=(0.58, 0.72)),
    dict(key="cb_output_npn", build=cb_output_npn,
         outer=lambda v: v["v(nes)"],
         param="VEE", trace="IC(Q1)",
         xlabel="Collector-base voltage $V_{CB}$ (V)",
         ylabel="Collector current $I_C$", fname="cb_output_npn", resistor="RC",
         xlim=(-1, 10.2),
         neg=dict(xlim=(-2, 10.2), y_neg=Y_NEG)),
    dict(key="ce_input_npn", build=ce_input_npn,
         outer=lambda v: v["v(ncs)"],
         param="VCE", trace="IB(Q1)",
         xlabel="Base-emitter voltage $V_{BE}$ (V)",
         ylabel="Base current $I_B$", fname="ce_input_npn", resistor="RB",
         xlim=(0.32, 0.78), ylim=(0.0, 100e-6)),
    dict(key="ce_output_npn", build=ce_output_npn,
         outer=lambda v: v["v(nbs)"],
         param="VBB", trace="IC(Q1)",
         xlabel="Collector-emitter voltage $V_{CE}$ (V)",
         ylabel="Collector current $I_C$", fname="ce_output_npn", resistor="RC",
         xlim=(-0.2, 10.2),
         neg=dict(xlim=(-2, 10.2), y_neg=Y_NEG)),
    dict(key="cc_input_npn", build=cc_input_npn,
         outer=lambda v: v["v(nes)"],
         param="VCE", trace="IB(Q1)",
         xlabel="Base-collector voltage $V_{BC}$ (V)",
         ylabel="Base current $I_B$", fname="cc_input_npn", resistor="RB",
         xlim=(-5, 0.5), ylim=(0.0, 120e-6), trim=True),
    dict(key="cc_output_npn", build=cc_output_npn,
         outer=lambda v: np.asarray(v["v(nbs)"], dtype=float),
         param="VBB", trace="IE(Q1)", signed=True,
         xlabel="Collector-emitter voltage $V_{CE}$ (V)",
         ylabel="Emitter current $I_E$", fname="cc_output_npn", resistor="RE",
         xlim=(-0.2, 5.2),
         neg=dict(xlim=(-1.5, 5.2), y_neg=Y_NEG)),
    dict(key="cb_input_pnp", build=cb_input_pnp,
         outer=lambda v: v["v(ncs)"],
         param="|VCB|", trace="|IE(Q1)|",
         xlabel="Emitter-base voltage $|V_{EB}|$ (V)",
         ylabel="Emitter current $|I_E|$", fname="cb_input_pnp", resistor="RE",
         xlim=(0.58, 0.80)),
    dict(key="cb_output_pnp", build=cb_output_pnp,
         outer=lambda v: v["v(nes)"],
         param="|VEE|", trace="|IC(Q1)|",
         xlabel="Collector-base voltage $|V_{CB}|$ (V)",
         ylabel="Collector current $|I_C|$", fname="cb_output_pnp", resistor="RC",
         xlim=(-1, 10.2),
         neg=dict(xlim=(-2, 10.2), y_neg=Y_NEG)),
    dict(key="ce_input_pnp", build=ce_input_pnp,
         outer=lambda v: v["v(ncs)"],
         param="|VCE|", trace="|IB(Q1)|",
         xlabel="Base-emitter voltage $|V_{BE}|$ (V)",
         ylabel="Base current $|I_B|$", fname="ce_input_pnp", resistor="RB",
         xlim=(0.42, 0.80), ylim=(0.0, 100e-6)),
    dict(key="ce_output_pnp", build=ce_output_pnp,
         outer=lambda v: v["v(nbs)"],
         param="|VBB|", trace="|IC(Q1)|",
         xlabel="Collector-emitter voltage $|V_{CE}|$ (V)",
         ylabel="Collector current $|I_C|$", fname="ce_output_pnp", resistor="RC",
         xlim=(-0.2, 10.2), ylim=(0.0, 20e-3),
         neg=dict(xlim=(-2, 10.2), ylim=(-2.4e-3, 20e-3))),
    dict(key="cc_input_pnp", build=cc_input_pnp,
         outer=lambda v: v["v(nes)"],
         param="|VCE|", trace="|IB(Q1)|",
         xlabel="Collector-base voltage $V_{CB}$ (V)",
         ylabel="Base current $|I_B|$", fname="cc_input_pnp", resistor="RB",
         xlim=(-5, 0.5), ylim=(0.0, 120e-6), trim=True),
    dict(key="cc_output_pnp", build=cc_output_pnp,
         outer=lambda v: np.asarray(v["v(nbs)"], dtype=float),     # real base-supply voltage: +6, -14 ... -74 V
         param="VBB", trace="|IE(Q1)|", signed=True,
         xlabel="Collector-emitter voltage $|V_{CE}|$ (V)",
         ylabel="Emitter current $|I_E|$", fname="cc_output_pnp", resistor="RE",
         xlim=(-0.2, 5.2),
         neg=dict(xlim=(-1.5, 5.2), y_neg=Y_NEG)),
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


_PREFIXES = ((1e-12, "p"), (1e-9, "n"), (1e-6, "\u00b5"), (1e-3, "m"), (1.0, ""), (1e3, "k"))


def eng(value, unit, digits=5):
    """Value with an SI prefix and its unit, e.g. 0.702974 -> "702.97 mV", 58.273e-6 -> "58.273 \u00b5A"."""
    value = float(value)
    if abs(value) < 1e-15:
        return f"0 {unit}"
    scale, prefix = _PREFIXES[0]
    for s, p in _PREFIXES:
        if abs(value) >= s * 0.9995:
            scale, prefix = s, p
    return f"{value / scale:.{digits}g} {prefix}{unit}"


def soften_grid(ax):
    """Recolour the grid light grey.  Call right after style_axes, before anything is drawn,
    so the only full-width lines on the axes are the grid lines."""
    if not SOFT_GRID:
        return
    minor = [t.gridline for t in ax.yaxis.get_minor_ticks()]
    for gl in ax.get_xgridlines() + ax.get_ygridlines() + minor:
        gl.set_color(GRID_COLOR)
        gl.set_linewidth(GRID_LW if gl not in minor else 0.6)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for ln in ax.lines:                       # grid drawn as plain lines by the theme
        xs, ys = (np.asarray(a, dtype=float) for a in ln.get_data())
        if xs.size < 2:
            continue
        spans_x = np.ptp(xs) >= 0.9 * abs(x1 - x0) and np.ptp(ys) == 0
        spans_y = np.ptp(ys) >= 0.9 * abs(y1 - y0) and np.ptp(xs) == 0
        if spans_x or spans_y:
            ln.set_color(GRID_COLOR)
            ln.set_linewidth(GRID_LW)
    for coll in ax.collections:               # ... or as vlines / hlines
        if isinstance(coll, LineCollection):
            coll.set_color(GRID_COLOR)
            coll.set_linewidth(GRID_LW)


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


def add_current_note(ax, text, curves, label_boxes, frame, above=False, extra=(), anchor=None):
    """Framed serif note ("Current through RC") placed right next to the family's
    curves, clear of every curve and of the stepped-source value labels.

    ``curves`` is [(value, x, y), ...]; ``label_boxes`` are the value-label boxes
    returned by place_sweep_labels (data coordinates); ``extra`` are further
    obstacles in page points (the cursor lines).  Returns the note's rect in page
    points."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    prop = FontProperties(family=next((f for f in NOTE_SERIF if f in installed), "serif"))
    fsize = note_fontsize(prop)
    dense = [dense_curve(x, y, frame) for _, x, y in curves]
    extra = list(extra)
    avoid = []
    for x0, y0, x1, y1 in label_boxes:
        (a, b), (c, d) = (tuple(float(v) for v in frame.to_pt(x0, y0)),
                          tuple(float(v) for v in frame.to_pt(x1, y1)))
        avoid.append((min(a, c), min(b, d), max(a, c), max(b, d)))
    if above:
        # input families: the free spot closest to the curves that lies ABOVE all of
        # them (the upper-left side of the rising curves)
        every = (np.concatenate([d[0] for d in dense + extra]),
                 np.concatenate([d[1] for d in dense + extra]))
        box_w = text_width(text, fsize, prop) + 2 * 0.0140 * AX_RECT[2] * FIG_W_PT
        box_h = 0.0925 * AX_H_PT
        gap = 4.0
        gxs = np.linspace(0.0, frame.w, 800)
        env = np.full_like(gxs, -np.inf)                 # upper envelope of the family
        for dx, dy in dense:
            o = np.argsort(dx)
            env = np.maximum(env, np.interp(gxs, dx[o], dy[o], left=-np.inf, right=-np.inf))
        ok = np.isfinite(env)
        ex, ey = gxs[ok], env[ok]
        xs = np.arange(box_w / 2, frame.w - box_w / 2 + 1e-9, 3.0)
        ys = np.arange(box_h / 2, frame.h - box_h / 2 + 1e-9, 3.0)
        gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
        top_at = np.interp(gx, ex, ey) if ex.size else np.full(gx.shape, -np.inf)
        # reference point: where the top curve is about half way up its own range, so
        # the note sits beside the body of the curves, not down on the flat foot
        j = int(np.argmin(np.abs(ey - 0.55 * ey.max()))) if ex.size else 0
        rx, ry = (ex[j], ey[j]) if ex.size else (frame.w / 2, frame.h / 2)
        if anchor is not None:                 # caller names the spot to stay close to
            rx, ry = anchor
        ddx = np.maximum(np.maximum(gx - box_w / 2 - rx, rx - gx - box_w / 2), 0.0)
        ddy = np.maximum(np.maximum(gy - box_h / 2 - ry, ry - gy - box_h / 2), 0.0)
        cost = np.hypot(ddx, ddy)
        cost = cost + 0.02 * np.hypot(gx - frame.w / 2, gy - frame.h / 2)
        for i in np.argsort(cost):
            if gy[i] > top_at[i] and rect_is_free(
                    (gx[i] - box_w / 2, gy[i] - box_h / 2, gx[i] + box_w / 2, gy[i] + box_h / 2),
                    every, avoid, gap, frame):
                return add_note_at(ax, text, frame, every, avoid,
                                   (gx[i] / frame.w, gy[i] / frame.h), fsize, prop,
                                   CURVE_COLOR, force=True)
    named = [(text, x, y) for _, x, y in curves]
    # attach the note to the middle curve of the family so it sits among the curves
    return place_note_near_trace(ax, text, len(curves) // 2, named, dense + extra,
                                 frame, avoid, prop, fsize, CURVE_COLOR)


# --------------------------------------------------------------------------- #
# Calculation points: two points per graph, their drops to the axes and the
# right triangle they span (PSpice cursor style)
# --------------------------------------------------------------------------- #
def cursor_geometry(pair, xlim, ylim):
    """Lines of one point pair, in data coordinates.

    P1 is the left point, P2 the right one and C = (x2, y1) the corner of the
    right angle.  ``drops``: P1 and P2 to the y axis (horizontal lines, the one
    through P1 runs on to the corner) and to the x axis (vertical lines, the one
    through P2 runs up from the axis through C).  ``edges``: the horizontal leg
    P1-C (= dV), the vertical leg C-P2 (= dI) and the hypotenuse P1-P2."""
    # a point is (x, y) or (x, y, input current of its curve at x)
    (xa, ya, da), (xb, yb, db) = sorted(
        ((float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else None) for p in pair),
        key=lambda p: (p[0], p[1]))
    drops = [([xlim[0], xb], [ya, ya]),
             ([xlim[0], xb], [yb, yb]),
             ([xa, xa], [ylim[0], ya]),
             ([xb, xb], [ylim[0], yb])]
    edges = [([xa, xb], [ya, ya]),
             ([xb, xb], [ya, yb]),
             ([xa, xb], [ya, yb])]
    return dict(points=((xa, ya), (xb, yb)), drive=(da, db), drops=drops, edges=edges)


def draw_cursor_lines(ax, geoms):
    """Draw the drops, the triangle and the dots of every point pair."""
    for g in geoms:
        for xs, ys in g["drops"]:
            ax.plot(xs, ys, color=CURSOR_COLOR, linewidth=CURSOR_DROP_LW,
                    solid_capstyle="butt", zorder=5)
        for xs, ys in g["edges"]:
            ax.plot(xs, ys, color=CURSOR_COLOR, linewidth=CURSOR_EDGE_LW,
                    solid_capstyle="butt", zorder=5)
        if CURSOR_DOT_PT > 0:
            ax.plot([g["points"][0][0], g["points"][1][0]],
                    [g["points"][0][1], g["points"][1][1]], linestyle="none",
                    marker="o", markersize=CURSOR_DOT_PT, markeredgewidth=0,
                    color=CURSOR_COLOR, zorder=6)


def draw_regions(ax, frame, knee, xlim, ylim, point_xs=(), point_ys=()):
    """Operating regions of an output graph.

    A dashed vertical line at the saturation knee ``knee`` (data x) separates
    SATURATION (left) from ACTIVE (right); each region has a bracket with its name
    along the top of the graph.  If the saturation region is too narrow to hold its
    name, the name is written vertically inside the region (at a height that no
    cursor line crosses).  CUTOFF is written just above the lowest curve, in the
    widest gap between the knee and the calculation points.

    Returns the drawn items as data-coordinate polylines [(xs, ys), ...] so that
    labels, note and read-outs can keep clear of them.
    """
    family, fsize = choose_font()
    prop = FontProperties(family=family)
    xspan, yspan = xlim[1] - xlim[0], ylim[1] - ylim[0]
    X = lambda pt: xlim[0] + pt / frame.w * xspan          # page points -> data x
    Y = lambda pt: ylim[0] + pt / frame.h * yspan          # page points -> data y
    kx = (knee - xlim[0]) / xspan * frame.w                # knee position in points
    hy = [(py - ylim[0]) / yspan * frame.h for py in point_ys]
    polys = []
    style = dict(color=REGION_COLOR, linewidth=REGION_LW, solid_capstyle="butt", zorder=4)

    ax.plot([knee, knee], [ylim[0], ylim[1]], linestyle=(0, (5, 3)), **style)
    polys.append(([knee, knee], [ylim[0], ylim[1]]))

    row = frame.h - (1.3 * fsize + 7.0)                    # height of the bracket row
    gap, tick = 2.5, REGION_TICK_PT
    for name, a, b in (("Saturation", 2.0, kx - gap), ("Active", kx + gap, frame.w - 2.0)):
        if b - a > 2.0:
            ax.plot([X(a), X(b)], [Y(row)] * 2, **style)
            for e in (a, b):
                ax.plot([X(e)] * 2, [Y(row - tick), Y(row + tick)], **style)
            polys.append(([X(a), X(b)], [Y(row)] * 2))
        tw = text_width(name, fsize, prop)
        cx = (a + b) / 2
        if tw + 6.0 <= b - a:
            ax.text(X(cx), Y(row + 3.0), name, ha="center", va="bottom", fontsize=fsize,
                    color=REGION_COLOR, zorder=11, path_effects=REGION_HALO)
            polys.append(([X(cx - tw / 2), X(cx + tw / 2)], [Y(row + 3.0 + 0.6 * fsize)] * 2))
        else:
            top = row - tick - 3.0                         # vertical label under the bracket
            while top - tw > 0.45 * frame.h and any(top - tw - 3.0 <= h <= top + 3.0 for h in hy):
                top -= 3.0
            ax.text(X(cx), Y(top), name, rotation=90, ha="center", va="top",
                    fontsize=fsize, color=REGION_COLOR, zorder=11, path_effects=REGION_HALO)
            polys.append(([X(cx)] * 2, [Y(top - tw), Y(top)]))

    # CUTOFF: above the lowest curve, in the widest gap free of vertical cursor lines
    edges = sorted([kx, frame.w] + [(p - xlim[0]) / xspan * frame.w for p in point_xs
                                    if knee < p < xlim[1]])
    a, b = max(zip(edges[:-1], edges[1:]), key=lambda s: s[1] - s[0])
    tw = text_width("Cutoff", fsize, prop)
    if b - a >= tw + 6.0:
        cx = (a + b) / 2
        ax.text(X(cx), Y(CUTOFF_RAISE_PT), "Cutoff", ha="center", va="bottom", fontsize=fsize,
                color=REGION_COLOR, zorder=11, path_effects=REGION_HALO)
        polys.append(([X(cx - tw / 2), X(cx + tw / 2)], [Y(CUTOFF_RAISE_PT + 0.6 * fsize)] * 2))
    return polys


def polyline_pt(frame, xs, ys, step=1.5):
    """Straight data-coordinate segment as a dense point cloud in page points."""
    x, y = frame.to_pt(np.asarray(xs, float), np.asarray(ys, float))
    n = max(int(np.hypot(x[-1] - x[0], y[-1] - y[0]) / step), 1) + 1
    return np.linspace(x[0], x[-1], n), np.linspace(y[0], y[-1], n)


def _summed_area(frame, clouds, rects):
    """Summed-area table of the page (1 pt cells) marking obstacle points and rects."""
    w, h = int(np.ceil(frame.w)) + 2, int(np.ceil(frame.h)) + 2
    occ = np.zeros((h, w), dtype=np.int32)
    for cx, cy in clouds:
        ix = np.clip(np.round(cx).astype(int), 0, w - 1)
        iy = np.clip(np.round(cy).astype(int), 0, h - 1)
        occ[iy, ix] = 1
    for x0, y0, x1, y1 in rects:
        occ[max(int(np.floor(y0)), 0):min(int(np.ceil(y1)) + 1, h),
            max(int(np.floor(x0)), 0):min(int(np.ceil(x1)) + 1, w)] = 1
    return np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _box_is_empty(sat, x0, y0, x1, y1):
    h, w = sat.shape[0] - 1, sat.shape[1] - 1
    ix0, ix1 = int(np.clip(np.floor(x0), 0, w)), int(np.clip(np.ceil(x1) + 1, 0, w))
    iy0, iy1 = int(np.clip(np.floor(y0), 0, h)), int(np.clip(np.ceil(y1) + 1, 0, h))
    return sat[iy1, ix1] - sat[iy0, ix1] - sat[iy1, ix0] + sat[iy0, ix0] == 0


def _box_sums(sat, x0, y0, x1, y1):
    """Number of marked cells inside each rectangle (arrays of corners, page points)."""
    h, w = sat.shape[0] - 1, sat.shape[1] - 1
    ix0 = np.clip(np.floor(x0), 0, w).astype(int)
    ix1 = np.clip(np.ceil(x1) + 1, 0, w).astype(int)
    iy0 = np.clip(np.floor(y0), 0, h).astype(int)
    iy1 = np.clip(np.ceil(y1) + 1, 0, h).astype(int)
    return sat[iy1, ix1] - sat[iy0, ix1] - sat[iy1, ix0] + sat[iy0, ix0]


def add_cursor_readouts(ax, frame, geoms, curve_clouds, line_clouds, avoid_rects,
                        drive_name=None, curve_halo=0.0):
    """Grey-framed "(x,y)" read-out box right next to each of the points.

    Like a Probe cursor read-out, the box is opaque and sits as close to its point
    as possible.  Every candidate spot costs its distance from the point plus a
    penalty for the length of curve (CURSOR_CURVE_COST per pt) and of cursor line
    (CURSOR_LINE_COST per pt) it would hide; spots that touch a sweep label, the
    note, another box or either point are not allowed.  The cheapest spot wins, so
    the box moves away from the point only as far as needed to keep the curves and
    the point's own lines mostly visible.

    ``curve_halo`` (pt) thickens the curves for the placement cost, so a box keeps clear
    of the trace and its markers instead of lying on them.

    A point that carries an input current gets a second line "<drive_name>=<value>"
    (e.g. "IB=52.7598u"): the base / emitter current of its curve at that x."""
    family, fsize = choose_font()
    prop = FontProperties(family=family)
    blocked = [tuple(r) for r in avoid_rects]
    for g in geoms:
        for px, py in g["points"]:
            x, y = (float(v) for v in frame.to_pt(px, py))
            blocked.append((x - 5.0, y - 5.0, x + 5.0, y + 5.0))
    if curve_halo > 0:
        d = curve_halo
        curve_clouds = [(np.concatenate([cx + ox for ox in (-d, 0, 0, 0, d)]),
                         np.concatenate([cy + oy for oy in (0, -d, 0, d, 0)]))
                        for cx, cy in curve_clouds]
    sat_curve = _summed_area(frame, list(curve_clouds), [])
    sat_line = _summed_area(frame, list(line_clouds), [])

    seen = []
    for gi, g in enumerate(geoms):
        for k, (px, py) in enumerate(g["points"]):
            if any(np.isclose(px, qx, rtol=1e-9, atol=1e-12)
                   and np.isclose(py, qy, rtol=1e-9, atol=1e-15) for qx, qy in seen):
                continue                      # a point shared by two pairs: one box only
            seen.append((px, py))
            text = f"({eng(px, 'V')}, {eng(py, 'A')})"
            tag = g.get("tag")                # e.g. "Knee": name written above the reading
            lines = [tag, text] if tag else [text]
            drive = g["drive"][k]
            if drive is not None and drive_name:
                lines.append(f"{drive_name} = {eng(drive, 'A')}")
            line_h = 1.45 * fsize
            w = max(text_width(t, fsize, prop) for t in lines) + 1.3 * fsize
            h = (1.75 + 1.45 * (len(lines) - 1)) * fsize
            ptx, pty = (float(v) for v in frame.to_pt(px, py))

            xs = np.arange(w / 2, frame.w - w / 2 + 1e-9, 2.0)
            ys = np.arange(h / 2, frame.h - h / 2 + 1e-9, 2.0)
            gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
            x0, x1, y0, y1 = gx - w / 2, gx + w / 2, gy - h / 2, gy + h / 2

            m = CURSOR_GAP_PT
            sat_block = _summed_area(frame, [], blocked)
            ok = _box_sums(sat_block, x0 - m, y0 - m, x1 + m, y1 + m) == 0
            dist = np.hypot(np.maximum.reduce([x0 - ptx, ptx - x1, np.zeros_like(x0)]),
                            np.maximum.reduce([y0 - pty, pty - y1, np.zeros_like(y0)]))
            cost = (dist + CURSOR_CURVE_COST * _box_sums(sat_curve, x0, y0, x1, y1)
                    + CURSOR_LINE_COST * _box_sums(sat_line, x0, y0, x1, y1))
            cost[~ok] = np.inf
            i = int(np.argmin(cost))
            if not np.isfinite(cost[i]):
                print(f"  warning: no free spot for read-out {text}; overlapping")
                cx = float(np.clip(ptx - 0.6 * w, w / 2, frame.w - w / 2))
                cy = float(np.clip(pty + h, h / 2, frame.h - h / 2))
            else:
                cx, cy = float(gx[i]), float(gy[i])
            blocked.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            ax.add_patch(Rectangle(((cx - w / 2) / frame.w, (cy - h / 2) / frame.h),
                                   w / frame.w, h / frame.h, transform=ax.transAxes,
                                   facecolor="white",
                                   edgecolor=g.get("edge", NOTE_EDGE),
                                   linewidth=1.4, zorder=10, clip_on=False))
            for li, t in enumerate(lines):
                ty = cy + ((len(lines) - 1) / 2 - li) * line_h
                ax.text(cx / frame.w, ty / frame.h, t, transform=ax.transAxes,
                        ha="center", va="center_baseline", fontsize=fsize,
                        color=(g.get("edge", "black") if (tag and li == 0) else "black"),
                        zorder=11)
            extra = (f", {drive_name} = {drive:.6g} A"
                     if drive is not None and drive_name else "")
            print(f"  {tag.lower() if tag else 'point'} {gi * 2 + k + 1}: "
                  f"({px:.6g} V, {py:.6g} A{extra})  box {dist[i]:.0f} pt from it")


def plot_family(curves, param, trace, xlabel, ylabel, outpath, xlim=None, dpi=300,
                ylim=None, trim=False, note=None, note_above=False, marks=None,
                drive_name=None, region_knee=None, knee_y=None, y_neg=None):
    """Draw one characteristic family on a PSpice/Probe-style page.

    ``curves`` is [(stepped source value, x, y), ...].  All curves share one
    colour and marker, a single legend entry names ``trace`` (e.g. "IB(Q1)"),
    and the value of the stepped source is written beside every curve:
    "VCE=0V" on the first one, "5V", "10V", ... on the others.

    ``marks`` is a list of point pairs [((x1, y1), (x2, y2)), ...]: the points used
    for the calculation, drawn with drops to the axes, a right triangle and
    coordinate read-outs (see cursor_geometry).  A point given as (x, y, drive)
    also shows its input current as "<drive_name>=<drive>" in its read-out.

    ``region_knee`` (x of the saturation knee) adds the operating-region markers of an
    output graph (see draw_regions) and leaves head-room above the curves for them.
    ``y_neg`` (fraction of the largest current, e.g. 0.12) lets the auto y axis extend
    below zero by that fraction, so the negative part of an output graph is shown; the
    curves are cut where they leave the window.
    ``knee_y`` (output current at the knee, on the plotted curve) additionally marks the
    knee point itself with a dot and a red-framed "Knee (x,y)" read-out."""
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
        if y_neg:
            ylo = max(ylo, -y_neg * float(ys.max()))   # only a stub of the negative side
        yhi = float(ys.max()) * (1.22 if region_knee is not None else 1.08)
        step = nice_step(yhi - ylo, 9)
        ylo = np.floor(ylo / step) * step
        yhi = np.ceil(yhi / step) * step
        ylim = (ylo, yhi)

    if ylim[0] < 0:                               # stop curves at the bottom of the window too
        cropped = []
        for val, x, y in curves:
            m = y >= ylim[0]
            if m.sum() >= 2:
                cropped.append((val, x[m], y[m]))
        curves = cropped

    ystep = nice_step(ylim[1] - ylim[0], 9)
    yticks = major_ticks(ylim[0], ylim[1], ystep)
    ylabels = [tick_label(t, "A") for t in yticks]
    xstep = nice_step(xlim[1] - xlim[0], 10)
    xticks = major_ticks(xlim[0], xlim[1], xstep)
    xlabels = [tick_label(t, "V") for t in xticks]

    fig, ax, frame = new_page(xlim, ylim, ylabels)
    style_axes(ax, xticks, xlabels, yticks, ylabels)
               #minor_between(yticks, Y_MINOR_SUBDIV))      # dotted PSpice minor grid on y
    soften_grid(ax)

    # same colour + marker everywhere; the first curve on top, markers staggered
    for k in reversed(range(len(curves))):
        _, x, y = curves[k]
        draw_trace(ax, x, y, frame, CURVE_COLOR, CURVE_MARKER, max(4 - k, 1),
                   MARKER_FIRST_PT + k * MARKER_SPACING_PT / len(curves))

    # the two calculation points: drops to the axes and the right triangle
    geoms = [cursor_geometry(pair, xlim, ylim) for pair in (marks or [])]
    draw_cursor_lines(ax, geoms)
    line_polys = [seg for g in geoms for seg in g["drops"] + g["edges"]]
    if region_knee is not None:
        line_polys += draw_regions(
            ax, frame, region_knee, xlim, ylim,
            point_xs=[p[0] for g in geoms for p in g["points"]],
            point_ys=[p[1] for g in geoms for p in g["points"]])
    line_clouds = [polyline_pt(frame, xs, ys) for xs, ys in line_polys]

    # the knee reading: a dot where the knee line meets its curve + a read-out box
    if region_knee is not None and knee_y is not None:
        ax.plot([region_knee], [knee_y], linestyle="none", marker="o",
                markersize=CURSOR_DOT_PT + 1.5, markerfacecolor="white",
                markeredgecolor=REGION_COLOR, markeredgewidth=1.6, zorder=7)
        geoms.append(dict(points=((region_knee, knee_y),), drive=(None,),
                          drops=[], edges=[], tag=f"Knee ({KNEE_FRAC:.0%})", edge=REGION_COLOR))

    # source value beside each curve, in a free spot next to the curve's end
    items = [(x, y, (f"{param}=" if k == 0 else "") + f"{val:g}V", "black")
             for k, (val, x, y) in enumerate(curves)]
    known = {id(t) for t in list(ax.texts) + list(fig.texts)}
    label_boxes = place_sweep_labels(fig, ax, [(x, y) for _, x, y in curves] + line_polys,
                                     items)
    for t in list(ax.texts) + list(fig.texts):          # white halo: no grid line through a label
        if id(t) not in known:
            t.set_path_effects(REGION_HALO)
            t.set_zorder(12)
    label_rects = []
    for x0, y0, x1, y1 in label_boxes:
        (a, b), (c, d) = (tuple(float(v) for v in frame.to_pt(x0, y0)),
                          tuple(float(v) for v in frame.to_pt(x1, y1)))
        label_rects.append((min(a, c), min(b, d), max(a, c), max(b, d)))

    # "Current through RC" etc. beside the curves
    note_rect = None
    if note:
        anchor = None
        if region_knee is not None:
            # free band between the top curve and the region brackets, in the middle of
            # the active region
            kx = (region_knee - xlim[0]) / (xlim[1] - xlim[0]) * frame.w
            anchor = (kx + 0.5 * (frame.w - kx),
                      frame.h - (1.3 * choose_font()[1] + 7.0) - 0.07 * frame.h)
        note_rect = add_current_note(ax, note, curves, label_boxes, frame,
                                     note_above or region_knee is not None,
                                     extra=line_clouds, anchor=anchor)

    # coordinate read-outs of the two points
    if geoms:
        add_cursor_readouts(ax, frame, geoms,
                            [dense_curve(x, y, frame) for _, x, y in curves],
                            line_clouds,
                            label_rects + ([note_rect] if note_rect else []),
                            drive_name=drive_name,
                            curve_halo=CURSOR_HALO_PT if drive_name else 0.0)

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
    # input curves used for R_in, all of them plotted: V_CB = 5 V (CB), the highest
    # V_CE of the CE sweep, and V_CE = 5 V (CC) - the same V_CE as the CC output points,
    # so the I_B values read on the input and on the output graph belong together
    cb_in5 = curve_at(cb_in, 5.0)
    ce_in_r = max(ce_in, key=lambda c: c[0])      # highest V_CE of the plotted sweep
    cc_in5 = curve_at(cc_in, 5.0)                 # the VCE = 5 V curve
    # V_CE = 5 V reference curve (not plotted), only for the turn-on voltage
    ce_in5 = curve_at(d["ce_input_ref_" + pol], 5.0)

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
    # of I_C is referred to constant I_E with the measured alpha = dI_C / dI_E:
    #   dI_C|IE = dI_C - alpha * dI_E
    ic5v, ie5v = _at(cb_, 5.0)
    ic10v, ie10v = _at(cb_, 10.0)
    rout_cb = 5.0 / ((ic10v - ic5v) - alpha * (ie10v - ie5v))

    # CE: beta between the 3rd and 4th curves at V_CE = 5 V
    ca_e, cb_e = ce_out[2], ce_out[3]
    ic_a_e, ib_a_e = _at(ca_e, 5.0)
    ic_b_e, ib_b_e = _at(cb_e, 5.0)
    beta = (ic_b_e - ic_a_e) / (ib_b_e - ib_a_e)
    vbe_a = x_at_y(ib_a_e, *_xy(ce_in_r))
    vbe_b = x_at_y(ib_b_e, *_xy(ce_in_r))
    rin_ce = (vbe_b - vbe_a) / (ib_b_e - ib_a_e)
    # CE output resistance on the 4th curve, V_CE 5 -> 10 V, at constant I_B
    # (dI_C|IB = dI_C - beta * dI_B with the measured beta = dI_C / dI_B)
    ic5v_e, ib5v_e = _at(cb_e, 5.0)
    ic10v_e, ib10v_e = _at(cb_e, 10.0)
    rout_ce = 5.0 / ((ic10v_e - ic5v_e) - beta * (ib10v_e - ib5v_e))

    # CC: gamma between the 3rd and 4th curves at V_CE = 5 V (same as beta)
    ca_c, cb_c = cc_out[2], cc_out[3]
    ie_a_c, ib_a_c = _at(ca_c, 5.0)
    ie_b_c, ib_b_c = _at(cb_c, 5.0)
    gamma = (ie_b_c - ie_a_c) / (ib_b_c - ib_a_c)
    vbe_a_c = x_at_y(ib_a_c, *_xy(cc_in5))
    vbe_b_c = x_at_y(ib_b_c, *_xy(cc_in5))
    rin_cc = (vbe_b_c - vbe_a_c) / (ib_b_c - ib_a_c)
    # CC output resistance on the 4th curve, V_CE 3 -> 5 V, at constant I_B
    # (dI_E|IB = dI_E - gamma * dI_B with the measured gamma = dI_E / dI_B)
    ie3, ib3 = _at(cb_c, 3.0)
    ie5, ib5 = _at(cb_c, 5.0)
    rout_cc = 2.0 / ((ie5 - ie3) - gamma * (ib5 - ib3))

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
    ie_mid_cc, ib_mid_cc = _at(cb_c, 4.0)
    ic_mid_cc = ie_mid_cc - ib_mid_cc
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
        rout_cc=(vaf + 4.0) / ic_mid_cc,
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

    def knee(curve, ref_x, frac=KNEE_FRAC):
        """Saturation knee: the lowest x at which the trace has reached frac of its
        value at ref_x (just inside the active region).  Returns (x, y) of that
        sample, so the knee point lies exactly on the plotted curve."""
        x, y = np.asarray(curve[0], float), np.asarray(curve[1], float)
        o = np.argsort(x)
        x, y = x[o], y[o]
        target = frac * np.interp(ref_x, x, y)
        i = int(np.where(y >= target)[0][0])
        return float(x[i]), float(y[i])

    knee_cb = knee(_xy(cb_), KNEE_REF["cb"])
    knee_ce = knee(_xy(cb_e), KNEE_REF["ce"])
    knee_cc = knee(_xy(cb_c), KNEE_REF["cc"])
    regions = dict(
        sat_cb=knee_cb[0],                        # V_CB at the CB saturation knee
        sat_ce=knee_ce[0],                        # V_CE at the CE saturation knee
        sat_cc=knee_cc[0],                        # V_CE at the CC saturation knee
        sat_cb_ic=knee_cb[1],                     # I_C at the CB knee (on the plotted curve)
        sat_ce_ic=knee_ce[1],                     # I_C at the CE knee
        sat_cc_ie=knee_cc[1],                     # I_E at the CC knee
        ib_cutoff_ce=y_at_x(5.0, *_xy(ce_out[0])),  # collector leakage, V_BB = 0, V_CE = 5 V
        ie_cutoff_cb=y_at_x(5.0, *_xy(cb_out[0])),  # collector leakage, V_EE = 0, V_CB = 5 V
        vbe_on_ce=x_at_y(10e-6, *_xy(ce_in5)),    # V_BE at I_B = 10 uA
        veb_on_cb=x_at_y(1e-3, *_xy(cb_in5)),     # V_EB at I_E = 1 mA
        knee_definition=KNEE_TEXT,
    )
    # The two points of every graph that the calculations use (x in V, y in A, in the
    # coordinates of the plot).  "res": R_in (input graphs) / R_out (output graphs);
    # "gain": the two points of alpha / beta / gamma (same V on two neighbouring curves).
    marks = {
        "cb_input": dict(res=((veb_a, ie_a), (veb_b, ie_b))),
        "ce_input": dict(res=((vbe_a, ib_a_e), (vbe_b, ib_b_e))),
        "cc_input": dict(res=((vbe_a_c, ib_a_c), (vbe_b_c, ib_b_c))),
        # output graphs: (V, output current, input current of that curve at V)
        "cb_output": dict(res=((5.0, ic5v, ie5v), (10.0, ic10v, ie10v)),
                          gain=((5.0, ic_a, ie_a), (5.0, ic_b, ie_b))),
        "ce_output": dict(res=((5.0, ic5v_e, ib5v_e), (10.0, ic10v_e, ib10v_e)),
                          gain=((5.0, ic_a_e, ib_a_e), (5.0, ic_b_e, ib_b_e))),
        "cc_output": dict(res=((3.0, ie3, ib3), (5.0, ie5, ib5)),
                          gain=((5.0, ie_a_c, ib_a_c), (5.0, ie_b_c, ib_b_c))),
    }
    return dict(simulated=sim, theoretical=th, error=err, points=pts, regions=regions,
                marks=marks,
                bias=dict(ie_cb=(ie_a + ie_b) / 2, ic_cb=ic_a,
                          ib_ce=(ib_a_e + ib_b_e) / 2, ic_ce=ic_b_e,
                          ib_cc=(ib_a_c + ib_b_c) / 2, ie_cc=ie_b_c,
                          ie_cb_a=ie_a, ie_cb_b=ie_b,
                          ib_ce_a=ib_a_e, ib_ce_b=ib_b_e,
                          ib_cc_a=ib_a_c, ib_cc_b=ib_b_c))


# --------------------------------------------------------------------------- #
def drive_label(base, pol):
    """Name of the input current written in the read-outs of an output graph
    (I_E for CB, I_B for CE / CC; |.| for the PNP, whose currents are plotted
    as magnitudes); None for the input graphs."""
    if base.endswith("_input"):
        return None
    name = "IE" if base == "cb_output" else "IB"
    return name if pol == "npn" else f"|{name}|"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--netlists", default=str(HERE.parent / "netlist"))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--show-negative", action="store_true",
                    help="output graphs also show the reverse (negative-current) part; files "
                         "get a _neg suffix so they do not overwrite the standard graphs")
    ap.add_argument("--no-cursors", action="store_true",
                    help="do not mark the calculation points on the graphs")
    ap.add_argument("--no-regions", action="store_true",
                    help="do not mark saturation / active / cutoff on the output graphs")
    ap.add_argument("--gain-points", action=argparse.BooleanOptionalAction, default=True,
                    help="also mark the two points of the alpha / beta / gamma calculation "
                         "on the output graphs (default: on; --no-gain-points to drop them)")
    args = ap.parse_args()

    global SHOW_NEG, CE_OUT_VMIN, CC_OUT_VMIN
    sfx = ""
    if args.show_negative:
        SHOW_NEG, CE_OUT_VMIN, CC_OUT_VMIN, sfx = True, "-2", "-1.5", "_neg"

    netdir = Path(args.netlists)
    outdir = Path(args.out)
    netdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. simulate every family (one nested-sweep run each)
    all_data = {}
    for fam in FAMILIES:
        key = fam["key"]
        text = fam["build"]()
        vectors = simulate_text(text, key)
        curves = split_family(key, vectors, fam["outer"], signed=fam.get("signed", False))
        for p, x, y, dr in curves:
            print(f"  {key:14s} source={p:<6g} {len(x):4d} pts  "
                  f"y: {np.min(y):.4g} .. {np.max(y):.4g}   "
                  f"drive: {np.max(dr):.4g}")
        all_data[key] = curves
        # the netlist on file is the complete nested-sweep netlist of the family
        (netdir / f"{fam['fname']}{sfx}.cir").write_text(text)

    # V_CE = 5 V reference curves of the CE input characteristic
    for pol in ("npn", "pnp"):
        text = ce_input_ref(pol)
        vectors = simulate_text(text, f"ce_input_ref_{pol}")
        all_data[f"ce_input_ref_{pol}"] = split_family(
            f"ce_input_{pol}", vectors, lambda v: v["v(ncs)"])
        (netdir / f"ce_input_ref_{pol}{sfx}.cir").write_text(text)

    # 2. choose the two points of every graph and do the calculations with them
    res = {pol: measure_polarity(all_data, pol) for pol in ("npn", "pnp")}
    (outdir / f"results{sfx}.json").write_text(json.dumps(res, indent=2))

    # 3. plot, with the two calculation points drawn on every graph
    if not args.no_plots:
        for fam in FAMILIES:
            if args.show_negative and "neg" in fam:
                fam = {**fam, **fam["neg"]}
            key = fam["key"]
            base, pol = key.rsplit("_", 1)
            curves_lab = [(p, x, y) for p, x, y, _ in all_data[key]]
            pairs = None
            if not args.no_cursors:
                m = res[pol]["marks"][base]
                pairs = [m["res"]]
                if args.gain_points and "gain" in m:
                    pairs.append(m["gain"])
            knee = knee_y = None
            if base.endswith("_output") and not args.no_regions:
                knee = res[pol]["regions"][f"sat_{base[:2]}"]
                knee_y = res[pol]["regions"][f"sat_{base[:2]}_{'ie' if base[:2] == 'cc' else 'ic'}"]
                lo, hi = fam["xlim"]
                if not lo < knee < hi:
                    print(f"  warning: saturation knee {knee:g} V outside the plot; no region markers")
                    knee = None
                else:
                    print(f"  saturation knee at {knee:.4g} V")
            print(f"{key}:")
            plot_family(curves_lab, fam["param"], fam["trace"],
                        fam["xlabel"], fam["ylabel"],
                        outdir / f"{fam['fname']}{sfx}.png",
                        xlim=fam["xlim"], dpi=args.dpi,
                        ylim=fam.get("ylim"), trim=fam.get("trim", False),
                        note=f"Current through {fam['resistor']}",
                        # input families: note on the upper side of the curves
                        note_above=base.endswith("_input"),
                        marks=pairs,
                        # input current shown in the read-outs of the output graphs
                        drive_name=drive_label(base, pol),
                        region_knee=knee, knee_y=knee_y,
                        y_neg=fam.get("y_neg"))
            print(f"    -> {fam['fname']}{sfx}.png")

    for pol in ("npn", "pnp"):
        s, t, e = res[pol]["simulated"], res[pol]["theoretical"], res[pol]["error"]
        print(f"\n{pol.upper()}")
        print(f"  {'':10s} {'sim':>10s} {'theory':>10s} {'err %':>8s}")
        for k in ("alpha", "beta", "gamma", "rin_cb", "rin_ce", "rin_cc",
                  "rout_cb", "rout_ce", "rout_cc"):
            print(f"  {k:10s} {s[k]:10.4g} {t[k]:10.4g} {e[k]:8.2f}")
    print("\nFor the report - " + KNEE_TEXT + ".")


if __name__ == "__main__":
    main()
