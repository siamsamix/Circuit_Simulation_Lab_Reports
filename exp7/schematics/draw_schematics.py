#!/usr/bin/env python3
"""
draw_schematics.py - draw the twelve measurement circuits of Experiment 7.

The drawings follow the netlists in ../netlist one-for-one: the transistor is
drawn vertically (collector up, emitter down, base left) and every source
branch returns to a common ground rail, so the collector, emitter and base
connections are unambiguous.  The capture-style colours match the schematic
figures of the earlier experiments.

Usage : python draw_schematics.py [--out ../schematics] [--dpi 300]
"""

import argparse
from pathlib import Path

import schemdraw
import schemdraw.elements as elm

GREEN = "#00aa00"
BLUE = "#0000cc"
U = 1.55                     # drawing unit
RAIL = -3.15                 # y of the ground rail
SRC_LEN = 1.45               # length of a source symbol


def new_drawing(path, dpi):
    d = schemdraw.Drawing(show=False)
    d.config(unit=U, fontsize=12, color=GREEN, lw=1.7, margin=0.18)
    return d


def line(d, p1, p2):
    d.add(elm.Line().at(p1).to(p2))


def text(d, x, y, txt, halign="left"):
    d.add(elm.Label().at((x, y)).label(txt, halign=halign, valign="center",
                                       color=BLUE, fontsize=12))


def source(d, x, y_top, y_bot, name, value, kind, reverse, side="right"):
    """Vertical source between (x, y_top) and (x, y_bot), with its label."""
    ymid = (y_top + y_bot) / 2
    top = (x, ymid + SRC_LEN / 2)
    bot = (x, ymid - SRC_LEN / 2)
    line(d, (x, y_top), top)
    el = elm.Battery() if kind == "battery" else elm.SourceI()
    # draw in the direction of the source so its polarity marks are correct
    if reverse:
        d.add(el.at(bot).up().length(SRC_LEN))
    else:
        d.add(el.at(top).down().length(SRC_LEN))
    line(d, bot, (x, y_bot))
    if kind == "battery":
        y_plus = (ymid + SRC_LEN / 2 - 0.10) if reverse else (ymid - SRC_LEN / 2 + 0.10)
        text(d, x + 0.14, y_plus, "+", halign="left")
    off = 0.72 if kind == "current" else 0.48
    if side == "right":
        text(d, x + off, ymid, f"{name}\n{value}", halign="left")
    else:
        text(d, x - off, ymid, f"{name}\n{value}", halign="right")


def transistor(d, npn, name):
    q = elm.BjtNpn(circle=True) if npn else elm.BjtPnp(circle=True)
    Q = d.add(q.anchor("base").at((0, 0)))
    text(d, 1.18, 0.12, f"Q1\n{name}")
    return Q


def ground(d, x, y):
    d.add(elm.Ground().at((x, y)).color(BLUE))


def circuit(npn, base, emitter, collector, outpath, dpi):
    """One measurement circuit.

    base/emitter/collector: None for a plain wire to the rail, otherwise a
    tuple (name, value, kind, reverse) for the source in that branch.
    """
    model = "2N3904" if npn else "2N3906"
    d = new_drawing(outpath, dpi)
    Q = transistor(d, npn, model)
    xb, yb = Q.absanchors["base"].x, Q.absanchors["base"].y
    xe, ye = Q.absanchors["emitter"].x, Q.absanchors["emitter"].y
    xc, yc = Q.absanchors["collector"].x, Q.absanchors["collector"].y

    # base branch: left, then down to the rail
    x_b = -2.1
    line(d, (xb, yb), (x_b, yb))
    if base is None:
        line(d, (x_b, yb), (x_b, RAIL))
    else:
        source(d, x_b, yb, RAIL, *base, side="left")

    # emitter branch: down to the rail
    if emitter is None:
        line(d, (xe, ye), (xe, RAIL))
    else:
        source(d, xe, ye, RAIL, *emitter, side="right")

    # collector branch: up, right, then down to the rail
    x_c = 3.4
    y_top = 1.55
    line(d, (xc, yc), (xc, y_top))
    line(d, (xc, y_top), (x_c, y_top))
    if collector is None:
        line(d, (x_c, y_top), (x_c, RAIL))
    else:
        source(d, x_c, y_top, RAIL, *collector, side="right")

    # ground rail and joints
    line(d, (x_b, RAIL), (x_c, RAIL))
    ground(d, x_b, RAIL)
    if emitter is not None:
        d.add(elm.Dot().at((xe, RAIL)).color(BLUE))
    if collector is not None:
        d.add(elm.Dot().at((x_c, RAIL)).color(BLUE))
    d.save(str(outpath), dpi=dpi)


NP, PN = True, False

# key: (npn, base source, emitter source, collector source)
# source tuple: (name, value, kind, reverse)
CIRCUITS = {
    # ----- NPN -----
    "cb_input_npn": (NP, None,
                     ("VEE", "-0.70 V", "battery", False),
                     ("VCB", "5 V", "battery", True)),
    "cb_output_npn": (NP, None,
                      ("IEE", "1 mA", "current", False),
                      ("VCB", "0 V", "battery", True)),
    "ce_input_npn": (NP, ("VBB", "0.65 V", "battery", True), None,
                     ("VCE", "5 V", "battery", True)),
    "ce_output_npn": (NP, ("IB", "20 uA", "current", True), None,
                      ("VCE", "0 V", "battery", True)),
    "cc_input_npn": (NP, ("VBB", "-0.50 V", "battery", False),
                     ("VCE", "-1 V", "battery", False), None),
    "cc_output_npn": (NP, ("IB", "20 uA", "current", True),
                      ("VE", "0 V", "battery", False),
                      ("VC", "5 V", "battery", True)),
    # ----- PNP -----
    "cb_input_pnp": (PN, None,
                     ("VEE", "0.70 V", "battery", True),
                     ("VCB", "-5 V", "battery", False)),
    "cb_output_pnp": (PN, None,
                      ("IEE", "1 mA", "current", True),
                      ("VCB", "0 V", "battery", False)),
    "ce_input_pnp": (PN, ("VBB", "-0.65 V", "battery", False), None,
                     ("VCE", "-5 V", "battery", False)),
    "ce_output_pnp": (PN, ("IB", "20 uA", "current", False), None,
                      ("VCE", "0 V", "battery", False)),
    "cc_input_pnp": (PN, ("VBB", "0.50 V", "battery", True),
                     ("VCE", "1 V", "battery", True), None),
    "cc_output_pnp": (PN, ("IB", "20 uA", "current", False),
                      ("VE", "0 V", "battery", False),
                      ("VC", "-5 V", "battery", False)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "schematics"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key, (npn, b, e, c) in CIRCUITS.items():
        circuit(npn, b, e, c, out / f"{key}.png", args.dpi)
        print(f"  {key}.png")


if __name__ == "__main__":
    main()