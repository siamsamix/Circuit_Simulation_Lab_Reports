#!/usr/bin/env python3
"""
netlist2schematic.py - read a SPICE netlist and draw the circuit diagram.

Usage
-----
    pip install schemdraw matplotlib
    python netlist2schematic.py circuit.cir                 # writes circuit.png
    python netlist2schematic.py circuit.cir -o circuit.svg  # svg / pdf / png
    python netlist2schematic.py circuit.cir --show          # also open a window
    python netlist2schematic.py circuit.cir --theme capture # green wires / blue pins
    cat circuit.cir | python netlist2schematic.py - -o out.png

Supported SPICE elements
------------------------
    R, C, L          resistor, capacitor, inductor      Rname n1 n2 value
    V, I             independent sources                Vname n+ n- [DC] value | SIN(...) | PULSE(...)
    D                diode                              Dname anode cathode model
    Q                BJT (NPN/PNP from .model)          Qname c b e [substrate] model
    M                MOSFET (NMOS/PMOS from .model)     Mname d g s b model [W= L=]
    E, G, F, H       controlled sources (drawn as 2-terminal output)
    X                subcircuit call: op-amp symbol if the subckt name looks like an
                     op-amp (3 nodes: in+ in- out | 5 nodes: in+ in- vcc vee out),
                     otherwise a labelled box
    Nodes "0", "gnd" and "gnd!" are ground.
    Handled syntax: '*' comment lines, inline ';' and '$' comments, '+' continuation
    lines, .model cards, .subckt ... .ends bodies (skipped), .end.

Themes (--theme)
----------------
    default   plain black schematic
    capture   schematic-capture look: green wires and symbol bodies, blue pin stubs,
              blue labels / junction discs / ground / source symbol, no background
              grid (set grid="#808080" in the theme to bring the grey dot grid back). AC sources get a sine-wave symbol
              with green +/- marks and show only their name; part names sit above a
              symbol, values (as written in the netlist) below. No title, no node
              names (use --title / --node-labels to bring them back). All grounds
              are tied together on one ground bus with a single ground symbol.
    Add your own by creating another Theme(...) entry in THEMES below.

How the layout works
--------------------
Automatic schematic placement is a hard problem, so this uses a simple, predictable
"ladder" layout that suits most small analog circuits:

  * Every non-ground node sits on a horizontal "rail" at y = 0. Nodes are ordered by a
    breadth-first walk starting at the first voltage source.
  * Two-terminal parts between neighbouring nodes sit on the rail.
  * Two-terminal parts to ground hang down from their node to a ground symbol.
  * In circuits with a signal source (SIN/PULSE/AC...), a DC bias source that touches
    just one other part is stacked with that part into one vertical branch, as in
    biased clipper circuits: series element on the rail, then diode/resistor + bias
    source below the output node.
  * Parts joining non-neighbouring nodes (or extra parallel parts) are drawn as arcs
    above the rail, in as many lanes as needed.
  * Bridge-type circuits (one source, and the rest of the parts forming three or more
    parallel chains between two nodes, e.g. a diode bridge rectifier with its load, or
    a Wheatstone bridge) use a second, "parallel rows" layout instead: each chain is a
    horizontal row between two vertical columns, and the source sits on the far left,
    wired round the outside to its nodes.
  * Transistors / op-amps / subcircuits are placed in a row above the rail and wired
    to their nodes with orthogonal wires. Wires that cross without a dot are NOT
    connected, exactly as in a hand-drawn schematic.
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

try:
    import schemdraw
    import schemdraw.elements as elm
    from schemdraw.segments import Segment
except ImportError:  # pragma: no cover
    sys.exit("This script needs schemdraw:  pip install schemdraw matplotlib")


# ---------------------------------------------------------------------------
# Netlist model + parser
# ---------------------------------------------------------------------------

GROUND_NAMES = {"0", "gnd", "gnd!"}
TWO_TERMINAL = set("RCLVIDEGFH")
DEVICES = set("QMX")

NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?"
VALUE_RE = re.compile(rf"^({NUM})(meg|mil|[tgkmunpfa])?([a-z]*)$", re.IGNORECASE)
OPAMP_HINTS = ("opamp", "op_amp", "op-amp", "ua741", "lm741", "lm358", "lm324",
               "tl07", "tl08", "op07")


@dataclass
class Component:
    name: str
    kind: str                      # upper-case first letter of the name
    nodes: list
    value: str = ""
    model: str = ""


@dataclass
class Circuit:
    title: str = ""
    components: list = field(default_factory=list)
    models: dict = field(default_factory=dict)    # model name (lower) -> TYPE (upper)
    warnings: list = field(default_factory=list)


def is_ground(node):
    return node.lower() in GROUND_NAMES


def _norm(node):
    """Canonical node key: every ground alias becomes '0', everything else lower-case."""
    return "0" if is_ground(node) else node.lower()


def _value_like(tok):
    return bool(VALUE_RE.match(tok)) or tok.startswith("{") or "=" in tok


def _logical_lines(text):
    """Strip comments, join '+' continuation lines. Returns list of strings."""
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("*"):
            continue
        line = re.split(r"\s;|\s\$|;", line, maxsplit=1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("+") and out:
            out[-1] += " " + line.lstrip()[1:].strip()
        else:
            out.append(line.strip())
    return out


def parse_element(line, models):
    """Parse one element line. Returns a Component, or None if it is not one."""
    # Keep SIN(0 1 1k) style groups together as single tokens.
    flat = re.sub(r"\(([^)]*)\)", lambda m: "(" + m.group(1).replace(" ", ",") + ")", line)
    tok = flat.split()
    if not tok or not re.match(r"^[A-Za-z]", tok[0]):
        return None
    name, kind, t = tok[0], tok[0][0].upper(), tok[1:]
    t = [x.replace(",", " ") if x.startswith(("SIN(", "PULSE(", "PWL(", "EXP(", "sin(",
                                                "pulse(", "pwl(", "exp("))
         else x for x in t]

    if kind in "RCL":
        if len(t) >= 3 and _value_like(t[2]):
            return Component(name, kind, t[:2], value=t[2])
    elif kind in "VI":
        if len(t) >= 2:
            return Component(name, kind, t[:2], value=" ".join(t[2:]))
    elif kind == "D":
        if len(t) >= 3:
            return Component(name, kind, t[:2], model=t[2])
    elif kind == "Q":
        if len(t) >= 4:
            if len(t) >= 5 and t[4].lower() in models and t[3].lower() not in models:
                return Component(name, kind, t[:3], model=t[4])      # 4-node form
            return Component(name, kind, t[:3], model=t[3])
    elif kind == "M":
        if len(t) >= 5:
            return Component(name, kind, t[:4], model=t[4])
    elif kind in "EG":
        if len(t) >= 5:
            return Component(name, kind, t[:2], value=t[4])
    elif kind in "FH":
        if len(t) >= 4:
            return Component(name, kind, t[:2], value=t[3])
    elif kind == "X":
        pos = [x for x in t if "=" not in x]
        if len(pos) >= 2:
            return Component(name, kind, pos[:-1], model=pos[-1])
    return None


def parse_netlist(text, first_line="auto"):
    circ = Circuit()
    raw_lines = text.splitlines()
    # First physical line is the title in SPICE, unless it is clearly an element.
    title_line = None
    if raw_lines and first_line != "element":
        first = raw_lines[0].strip()
        if first_line == "title":
            title_line = first
        elif first.startswith("*"):
            title_line = first.lstrip("* ").strip()
        elif first and not first.startswith("."):
            toks = first.split()
            kind0 = toks[0][0].upper()
            # A real element name has a digit (R1, Q3) or is a short R/C/L/V/I name (Vin).
            plausible_name = bool(re.search(r"\d", toks[0])) or \
                (len(toks[0]) <= 4 and kind0 in "RCLVI")
            comp = parse_element(first, {})
            strict = comp is not None and plausible_name and all(
                re.fullmatch(r"[\w!#.+\-]+", n) for n in comp.nodes)
            if strict and kind0 in "VI" and len(toks) < 4:
                strict = False
            if not strict:
                title_line = first
        if title_line is not None:
            raw_lines = raw_lines[1:]
            circ.title = title_line

    lines = _logical_lines("\n".join(raw_lines))

    # Pass 1: .model cards
    for ln in lines:
        m = re.match(r"^\.model\s+(\S+)\s+([A-Za-z0-9_]+)", ln, re.IGNORECASE)
        if m:
            circ.models[m.group(1).lower()] = m.group(2).upper()

    # Pass 2: elements
    depth = 0
    for ln in lines:
        low = ln.lower()
        if low.startswith(".subckt"):
            depth += 1
            continue
        if low.startswith(".ends"):
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if low.startswith(".end") and not low.startswith(".ends"):
            break
        if ln.startswith("."):
            continue
        comp = parse_element(ln, circ.models)
        if comp is None:
            circ.warnings.append(f"Skipped unsupported/unparsable line: {ln!r}")
        else:
            circ.components.append(comp)
    return circ


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

_PREFIX = {"meg": "M", "mil": "mil", "t": "T", "g": "G", "k": "k", "m": "m",
           "u": "µ", "n": "n", "p": "p", "f": "f", "a": "a"}
_UNIT = {"R": "Ω", "C": "F", "L": "H"}


def fmt_value(value, unit):
    m = VALUE_RE.match(value)
    if not m:
        return value
    number, prefix, rest = m.groups()
    if rest:                                   # user already wrote units, e.g. 10uF
        return value.replace("u", "µ") if value[-1:].isalpha() else value
    return number + _PREFIX.get((prefix or "").lower(), "") + unit


def fmt_source(spec, unit):
    spec = spec.strip()
    spec = re.sub(r"^(?i:dc)\s+", "", spec)
    first = spec.split()[0] if spec else ""
    if VALUE_RE.match(first):
        spec = fmt_value(first, unit) + spec[len(first):]
    return spec if len(spec) <= 26 else spec[:25] + "…"


def comp_label(c, units=True):
    if c.kind in "RCL":
        value = fmt_value(c.value, _UNIT[c.kind]) if units else c.value
        return f"{c.name}\n{value}"
    if c.kind == "V":
        return f"{c.name}\n{fmt_source(c.value, 'V')}".rstrip()
    if c.kind == "I":
        return f"{c.name}\n{fmt_source(c.value, 'A')}".rstrip()
    if c.kind == "D":
        return f"{c.name}\n{c.model}"
    return f"{c.name}\n{c.value}".rstrip()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

ELEN = 3.0        # length of a 2-terminal symbol
MARGIN = 1.5      # wire either side of a symbol on the rail
GX = 2 * MARGIN + ELEN   # spacing between several ground-hanging branches on one node;
                         # equal to the rail pitch so columns are evenly spaced
LANE_H = 3.5      # vertical spacing of arc lanes above the rail
HANG_PRE = 1.0    # wire between rail and a hanging part
HANG_POST = 1.0   # wire between a hanging part and its ground symbol
STUB = 0.8        # wire stub leaving a transistor pin


@dataclass(frozen=True)
class Theme:
    name: str = "default"
    wire: str = "black"          # connecting wires
    body: str = "black"          # component symbol bodies
    source: str = None           # V/I source symbols (None -> body colour)
    pin: str = None              # colour of the symbol's lead stubs (None -> no overlay)
    marks: str = None            # +/- marks on voltage sources (capture style only)
    text: str = "black"          # part labels / captions / title
    node_text: str = "#555555"   # node-name labels
    dot: str = "black"           # junction discs
    ground: str = "black"
    dot_radius: float = 0.1
    lw: float = 2.0
    fontsize: float = 11
    fonts: tuple = ()            # preferred font families, first installed one wins
    grid: str = None             # colour of the dotted background grid (None -> no grid)
    grid_pitch: float = 0.5
    split_labels: bool = False   # name above a horizontal part, value below
    ac_symbols: bool = False     # sine / pulse symbol for AC, SIN and PULSE sources
    dc_battery: bool = False     # 4-plate battery symbol for DC voltage sources
    lead: float = None           # fixed lead length on each side of a symbol (None -> long)
    units: bool = True           # append Ω / F / H to R, C, L values
    show_title: bool = True
    node_labels: bool = True
    ground_bus: bool = False     # join all grounds on one bus with a single ground symbol
    dpi: int = 150               # default raster resolution


THEMES = {
    "default": Theme(),
    # Palette (0,128,0) green / (0,0,255) blue / (128,128,128) grey, sans-serif labels.
    "capture": Theme(
        name="capture", wire="#008000", body="#008000", source="#0000ff", pin="#008000",
        marks="#008000", text="#0000ff", node_text="#0000ff", dot="#0000ff",
        ground="#0000ff", dot_radius=0.1, lw=0.6, fontsize=14,
        fonts=("Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"),
        grid=None, grid_pitch=0.5, split_labels=True, ac_symbols=True, dc_battery=True,
        lead=0.5, units=False, show_title=False, node_labels=False, ground_bus=True,
        dpi=176),     # 176 dpi -> 0.5-unit grid pitch = 44 px, like the reference capture
}


def _pick_font(names):
    if not names:
        return None
    try:
        from matplotlib import font_manager as fm
        have = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        return None
    return next((n for n in names if n in have), None)


_LEADS = {}


def _lead_length(cls):
    """Length of the plain lead on each side of a symbol body at length ELEN."""
    if cls not in _LEADS:
        try:
            probe = schemdraw.Drawing(show=False)
            e = cls().right().at((0, 0)).length(ELEN)
            probe += e
            lead = -float(e.segments[0].path[0][0])
            _LEADS[cls] = lead if 0.05 < lead < ELEN / 2 else None
        except Exception:
            _LEADS[cls] = None
    return _LEADS[cls]


class _SineSource(elm.Source):
    """Source circle holding one small sine cycle (flatter than schemdraw's SourceSin)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pts = []
        for i in range(41):                # time axis runs across the element
            s = -0.3 + 0.6 * i / 40
            pts.append((0.5 + 0.14 * math.sin(2 * math.pi * s / 0.6), -s))
        self.segments.append(Segment(pts))


class _BatterySource(elm.Element2Term):
    """DC voltage source drawn like the capture tool: long, short, long, short plates
    (long plate = + end) on a 2.0-unit symbol. Plates are 1.0 / 0.5 wide."""
    symbol_len = 2.0
    own_leads = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Leads are 0.70 long at the + end and 0.80 at the - end (as in the capture):
        # symmetric 0.70 leads around a 0.6 body, plus a 0.1 stub down to the lowest plate.
        b = 0.6
        self.segments.append(Segment([(0, 0), (math.nan, math.nan), (b, 0)]))
        self.segments.append(Segment([(0, 0), (0.1, 0)]))
        for x, half in ((0.1, 0.25), (0.3, 0.5), (0.45, 0.25), (b, 0.5)):
            self.segments.append(Segment([(x, half), (x, -half)]))


def _simple_two_terminal_class(c):
    return {
        "R": elm.Resistor, "C": elm.Capacitor, "L": elm.Inductor2,
        "V": elm.SourceV, "I": elm.SourceI, "D": elm.Diode,
        "E": elm.SourceControlledV, "H": elm.SourceControlledV,
        "G": elm.SourceControlledI, "F": elm.SourceControlledI,
    }[c.kind]


def _logical_start_is_first(kind):
    """Which SPICE node is the symbol's 'start'?  (V/E/H have + at the end of the
    symbol, so start is the negative node; I/D/G/F point start -> end.)"""
    return kind not in "VEH"


def place(cls, a, b, labels=(), color=None, lead_color=None, **kw):
    """Axis-aligned element from a to b. Explicit direction+length is used because
    schemdraw sources ignore .to().  labels = [(text, loc, colour), ...]"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    e = cls(**kw).at(a)
    if abs(dx) > abs(dy):
        e = e.right() if dx > 0 else e.left()
    else:
        e = e.up() if dy > 0 else e.down()
    e = e.length(length)
    if color:
        e = e.color(color)
    if lead_color:
        # Source symbols keep their leads in their own first segment(s): recolour just
        # those, so the leads are drawn once, in the wire colour, at the wire's width.
        nlead = 2 if cls is _BatterySource else 1
        for seg in e.segments[:nlead]:
            seg.color = lead_color
    for text, loc, lcolor in labels:
        opts = {"loc": loc}
        if lcolor:
            opts["color"] = lcolor
        e = e.label(text, **opts)
    return e


class Builder:
    def __init__(self, circ, theme=None):
        self.circ = circ
        self.t = theme or THEMES["default"]
        self.d = schemdraw.Drawing(show=False)
        cfg = dict(unit=3, fontsize=self.t.fontsize, lw=self.t.lw)
        font = _pick_font(self.t.fonts)
        if font:
            cfg["font"] = font
        self.d.config(**cfg)
        self.disp = {}                       # normalised node -> display name
        self.degree = defaultdict(int)       # connection count per node (for dots)
        self.pin_counter = 0
        self._texts = []                     # exact-position text, drawn in save()
        self.warn = circ.warnings

    # -- tiny helpers ------------------------------------------------------
    def line(self, a, b):
        if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
            return
        self.d += elm.Line().at(a).to(b).color(self.t.wire)

    def polyline(self, pts):
        for a, b in zip(pts, pts[1:]):
            self.line(a, b)

    def ground(self, p):
        self.d += elm.Ground().at(p).color(self.t.ground)

    # -- theme-aware helpers -------------------------------------------------
    def _cls(self, c):
        if self.t.ac_symbols and c.kind == "V":
            spec = c.value.lower()
            if re.search(r"\b(sin|ac)\b", spec):
                return _SineSource
            if re.search(r"\bpulse\b", spec):
                return elm.SourcePulse
            return _BatterySource if self.t.dc_battery else elm.Source
        return _simple_two_terminal_class(c)

    def _symbol_len(self, cls):
        """Overall length of a 2-terminal symbol (body + two leads)."""
        if hasattr(cls, "symbol_len"):
            return cls.symbol_len
        lead = _lead_length(cls)
        if self.t.lead is None or lead is None:
            return ELEN
        return (ELEN - 2 * lead) + 2 * self.t.lead

    def _dot(self, p):
        self.d += elm.Dot(radius=self.t.dot_radius).at(p).color(self.t.dot)

    def _draw_part(self, c, p1, p2, node1, centre=None):
        """Draw 2-terminal part c in the axis-aligned slot p1 -> p2: wires plus a symbol
        centred in the slot (or on `centre`). node1 is the normalised node at p1."""
        cls = self._cls(c)
        total = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        ux, uy = (p2[0] - p1[0]) / total, (p2[1] - p1[1]) / total
        length = self._symbol_len(cls)
        t_mid = total / 2 if centre is None else \
            (centre[0] - p1[0]) * ux + (centre[1] - p1[1]) * uy
        a = (p1[0] + ux * (t_mid - length / 2), p1[1] + uy * (t_mid - length / 2))
        b = (p1[0] + ux * (t_mid + length / 2), p1[1] + uy * (t_mid + length / 2))
        start_at_p1 = (_norm(c.nodes[0]) == node1) == _logical_start_is_first(c.kind)
        p_from, p_to = (a, b) if start_at_p1 else (b, a)
        self.line(p1, a)
        self.line(b, p2)
        lead_color = self.t.pin if (c.kind in "VI" and self.t.pin) else None
        self.d += place(cls, p_from, p_to, self._labels(c, abs(uy) < 1e-9), self._body_color(c),
                        lead_color=lead_color)
        self._decorate(c, cls, p_from, p_to)

    def _battery_extras(self, c, p_from, p_to, ux, uy):
        """+/- strokes beside the leads, name on the right, value on the left (all
        positions measured from the reference capture, in drawing units)."""
        T, B = p_to, p_from                        # + end, - end
        left = (-uy, ux)                           # left of the +direction
        col = self.t.marks or self.t.text
        lw = max(self.t.lw * 1.6, 1.0)

        def at(base, along, side):
            return (base[0] + ux * along + left[0] * side, base[1] + uy * along + left[1] * side)

        def stroke(a, b):
            self.d += elm.Line().at(a).to(b).color(col).linewidth(lw)

        px, py = at(T, -0.25, 0.30)                # plus: centre 0.25 in from the + end
        stroke((px - 0.07, py), (px + 0.07, py))
        stroke((px, py - 0.07), (px, py + 0.07))
        mx, my = at(B, 0.24, 0.25)                 # minus: centre 0.24 in from the - end
        stroke((mx - 0.033, my), (mx + 0.033, my))

        name = c.name
        value = fmt_source(c.value, "V")
        if abs(uy) >= abs(ux):                     # vertical: name right, value left
            for text, along, side in ((name, -0.21, -0.456), (value, -0.76, 0.811)):
                if not text:
                    continue
                x, y = at(T, along, side)
                self._texts.append(dict(x=x, y=y, text=text, color=self.t.text,
                                        ha="left" if left[0] * side > 0 else "right"))
        else:                                      # horizontal: name/value above/below
            cx, cy = (T[0] + B[0]) / 2, (T[1] + B[1]) / 2
            for text, side in ((name, -0.7), (value, 0.7)):
                if text:
                    self._texts.append(dict(x=cx + left[0] * side, y=cy + left[1] * side,
                                            text=text, color=self.t.text, ha="center"))

    def _body_color(self, c):
        return self.t.source if (c.kind in "VI" and self.t.source) else self.t.body

    def _label_text(self, c):
        if c.kind == "V" and self._cls(c) in (_SineSource, elm.SourcePulse):
            return c.name                       # AC source: the symbol says it all
        return comp_label(c, self.t.units)

    def _labels(self, c, horizontal):
        if self._cls(c) is _BatterySource:
            return []                           # name/value are placed by _battery_extras
        text = self._label_text(c)
        if horizontal and self.t.split_labels and "\n" in text:
            name, rest = text.split("\n", 1)
            return [(name, "top", self.t.text), (rest, "bottom", self.t.text)]
        return [(text, "top" if horizontal else "bottom", self.t.text)]

    def _decorate(self, c, cls, p_from, p_to):
        """Blue lead stubs over the symbol's leads and +/- marks on voltage sources."""
        length = math.hypot(p_to[0] - p_from[0], p_to[1] - p_from[1])
        ux, uy = (p_to[0] - p_from[0]) / length, (p_to[1] - p_from[1]) / length
        if cls is _BatterySource:
            self._battery_extras(c, p_from, p_to, ux, uy)
            return
        lead = _lead_length(cls)
        if self.t.pin and lead and c.kind not in "VI" and self.t.pin != self.t.body:
            lead = self.t.lead if self.t.lead is not None else lead
            for a, sgn in ((p_from, 1), (p_to, -1)):
                b = (a[0] + sgn * ux * lead, a[1] + sgn * uy * lead)
                self.d += elm.Line().at(a).to(b).color(self.t.pin)
        if self.t.ac_symbols and c.kind == "V":
            # +/- marks drawn as strokes (not text) so they sit exactly on the axis;
            # + is at the symbol's end. Plus arm = 0.16, minus = 0.12 units.
            cx, cy = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
            col = self.t.marks or self.t.text
            lw = max(self.t.lw * 1.6, 1.0)
            for sign, off in (("+", 0.33), ("-", -0.33)):
                mx, my = cx + ux * off, cy + uy * off
                arms = [((mx - 0.08, my), (mx + 0.08, my))]
                if sign == "+":
                    arms.append(((mx, my - 0.08), (mx, my + 0.08)))
                else:
                    arms = [((mx - 0.06, my), (mx + 0.06, my))]
                for a, b in arms:
                    self.d += elm.Line().at(a).to(b).color(col).linewidth(lw)

    # -- main --------------------------------------------------------------
    def build(self):
        circ = self.circ
        for c in circ.components:
            for n in c.nodes:
                self.disp.setdefault(n.lower(), n)

        two, devs = [], []
        for c in circ.components:
            if c.kind in TWO_TERMINAL:
                a, b = c.nodes[0].lower(), c.nodes[1].lower()
                if a == b:
                    self.warn.append(f"{c.name}: both terminals on node '{c.nodes[0]}', skipped")
                    continue
                if is_ground(a) and is_ground(b):
                    self.warn.append(f"{c.name}: connects ground to ground, skipped")
                    continue
                two.append(c)
            elif c.kind in DEVICES:
                devs.append(c)

        if self._rows_layout(two, devs):
            return self.d

        folds = self._fold_bias_chains(circ.components, two)
        if folds:
            folded = {id(c) for _, chain in folds for c, _ in chain}
            rest = {_norm(n) for c in circ.components if id(c) not in folded for n in c.nodes}
            folds = [f for f in folds if f[0] in rest]       # branch must hang off a rail node
        folded = {id(c) for _, chain in folds for c, _ in chain}
        two = [c for c in two if id(c) not in folded]
        rail_comps = [c for c in circ.components if id(c) not in folded]

        order = self._order_nodes(rail_comps, two, devs)
        if not order:
            raise SystemExit("No drawable (non-ground) nodes found in the netlist.")
        idx = {n: i for i, n in enumerate(order)}

        # Hanging branches (chains of parts ending on ground) and rail/arc connections
        hang = defaultdict(list)                  # node -> [chain, ...]; chain = [(part, top node)]
        conns = defaultdict(list)
        for c in two:
            a, b = c.nodes[0].lower(), c.nodes[1].lower()
            if is_ground(a) or is_ground(b):
                node = b if is_ground(a) else a
                hang[node].append([(c, node)])
                self.degree[node] += 1
            else:
                i, j = sorted((idx[a], idx[b]))
                conns[(i, j)].append(c)
                self.degree[a] += 1
                self.degree[b] += 1

        # Bias branches go to the front so they sit inside, next to the series element;
        # the plain load (e.g. RL) is then the outermost leg on the right-hand edge.
        for top, chain in folds:
            hang[top].insert(0, chain)
            self.degree[top] += 1

        # x positions
        ext = [max(0, len(hang[n]) - 1) * GX for n in order]
        xs = [0.0]
        for i in range(len(order) - 1):
            xs.append(xs[-1] + ext[i] + 2 * MARGIN + ELEN)

        # Split connections into rail parts and arcs
        rail, arcs = {}, []
        for (i, j), comps in sorted(conns.items(), key=lambda kv: (kv[0][1] - kv[0][0], kv[0])):
            for k, c in enumerate(comps):
                if k == 0 and j == i + 1:
                    rail[i] = c
                else:
                    arcs.append([i, j, c, 1])
        arcs.sort(key=lambda a: (a[1] - a[0]))
        placed = []
        for arc in arcs:
            lanes = [p[3] for p in placed if not (arc[1] <= p[0] or arc[0] >= p[1])]
            arc[3] = 1 + max(lanes, default=0)
            placed.append(arc)
        max_lane = max((a[3] for a in arcs), default=0)

        # Which nodes have risers reaching which lane (to keep symbols off the risers)
        riser_top = defaultdict(int)
        for i, j, _, lane in arcs:
            riser_top[i] = max(riser_top[i], lane)
            riser_top[j] = max(riser_top[j], lane)

        # --- draw rail parts ---
        for i, c in rail.items():
            s = xs[i] + ext[i] + MARGIN
            self._draw_between(c, (xs[i], 0), (xs[i + 1], 0), s, s + ELEN, 0, order[i], idx, order)

        # --- draw arcs ---
        for i, j, c, lane in arcs:
            y = lane * LANE_H
            lo, hi = xs[i] + ext[i] + ELEN / 2 + 0.6, xs[j] - ELEN / 2 - 0.6
            blocked = [xs[k] for k in range(i + 1, j) if riser_top[k] > lane]
            mid = (xs[i] + xs[j]) / 2
            if lo >= hi:
                centre = mid
            else:
                best, centre = -1e9, mid
                steps = max(1, int((hi - lo) / 0.25))
                for s in range(steps + 1):
                    cand = lo + (hi - lo) * s / steps
                    clear = min((abs(cand - bx) for bx in blocked), default=99)
                    score = min(clear, 4) - 0.02 * abs(cand - mid)
                    if score > best:
                        best, centre = score, cand
            self.line((xs[i], 0), (xs[i], y))
            self.line((xs[j], 0), (xs[j], y))
            self._draw_between(c, (xs[i], y), (xs[j], y), centre - ELEN / 2, centre + ELEN / 2,
                               y, order[i], idx, order)

        # --- draw hanging parts ---
        legs = []
        longest = max((len(ch) for chains in hang.values() for ch in chains), default=1)
        yg = -(HANG_PRE + longest * ELEN + HANG_POST)
        for k, n in enumerate(order):
            chains = hang[n]
            if len(chains) > 1:
                self.line((xs[k], 0), (xs[k] + ext[k], 0))
            for m, chain in enumerate(chains):
                legs.append(xs[k] + m * GX)
                self._draw_chain(chain, xs[k] + m * GX, yg)
        self._ground_bus(legs, yg)

        # --- devices (above everything else) ---
        if devs:
            npins = sum(len(c.nodes) for c in devs)
            y_dev = math.ceil((max_lane * LANE_H + 3.5 + 0.6 * npins) * 2) / 2
            self._draw_devices(devs, xs, idx, y_dev)

        # --- node dots / labels ---
        for k, n in enumerate(order):
            p = (xs[k], 0)
            deg = self.degree[n]
            if deg >= 3:
                self.d += elm.Dot(radius=self.t.dot_radius).at(p).color(self.t.dot)
            elif deg <= 1:
                self.d += elm.Dot(open=True).at(p).color(self.t.dot)
            if self.t.node_labels:
                self.d += elm.Label().at((p[0] - 0.15, p[1] + 0.25)).label(
                    self.disp[n], fontsize=self.t.fontsize - 2, color=self.t.node_text,
                    halign="right", valign="bottom")

        if self.circ.title and self.t.show_title:
            ytop = (max_lane * LANE_H if not devs else y_dev) + 2.6
            self.d += elm.Label().at((xs[-1] / 2, ytop)).label(
                self.circ.title, fontsize=self.t.fontsize + 3, color=self.t.text,
                halign="center")
        return self.d

    # -- output ------------------------------------------------------------
    def save(self, path, dpi=150, show=False):
        fig = self.d.draw(show=False)
        ax, f = fig.ax, fig.getfig()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        per_unit = f.get_size_inches()[0] / (x1 - x0)        # inches per drawing unit

        # exact-position text (schemdraw's own Label adds a hidden offset)
        if self._texts:
            kw = {}
            font = _pick_font(self.t.fonts)
            if font:
                kw["family"] = font
            size = self.t.fontsize
            cap = 0.716 * size / 72 / per_unit               # cap height in drawing units
            arts = [ax.text(t["x"], t["y"] - cap / 2, t["text"], ha=t["ha"], va="baseline",
                            fontsize=size, color=t["color"], **kw) for t in self._texts]
            f.canvas.draw()
            inv, rend = ax.transData.inverted(), f.canvas.get_renderer()
            for a in arts:                                   # make the canvas cover the text
                bb = a.get_window_extent(rend)
                (bx0, by0), (bx1, by1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
                x0, x1 = min(x0, bx0), max(x1, bx1)
                y0, y1 = min(y0, by0), max(y1, by1)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            f.set_size_inches((x1 - x0) * per_unit, (y1 - y0) * per_unit)

        pad = self.t.grid_pitch                             # margin around the circuit
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        f.set_size_inches((x1 - x0) * per_unit, (y1 - y0) * per_unit)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        if self.t.grid:
            import numpy as np
            p = self.t.grid_pitch
            gx = np.arange(math.ceil(x0 / p), math.floor(x1 / p) + 1) * p
            gy = np.arange(math.ceil(y0 / p), math.floor(y1 / p) + 1) * p
            X, Y = np.meshgrid(gx, gy)
            ax.plot(X.ravel(), Y.ravel(), linestyle="none", marker=",",
                    color=self.t.grid, zorder=0)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        f.subplots_adjust(0, 0, 1, 1)
        f.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi, transparent=False,
                  facecolor="white", bbox_extra_artists=ax.get_default_bbox_extra_artists())
        if show:
            import matplotlib.pyplot as plt
            plt.show()

    # -- node ordering -----------------------------------------------------
    @staticmethod
    def _pick_start(comps):
        """Prefer the signal source (SIN/PULSE/AC...) so the circuit reads left->right."""
        sources = [c for c in comps if c.kind in "VI"]
        sig = re.compile(r"\b(ac|sin|pulse|pwl|exp|sffm|am)\b", re.I)
        named = re.compile(r"^[VI](in|i|s|sig|ac|g|src|signal)\d*$", re.I)
        for test in (lambda c: sig.search(c.value), lambda c: named.match(c.name),
                     lambda c: c.kind == "V"):
            for c in sources:
                if test(c):
                    cand = [n.lower() for n in c.nodes if not is_ground(n)]
                    if cand:
                        return cand[0]
        return None

    def _order_nodes(self, comps, two, devs):
        adj = defaultdict(list)
        appear = []
        for c in comps:
            ns = []
            for n in c.nodes:
                nl = n.lower()
                if not is_ground(nl) and nl not in ns:
                    ns.append(nl)
                if not is_ground(nl) and nl not in appear:
                    appear.append(nl)
            for a in ns:
                for b in ns:
                    if a != b and b not in adj[a]:
                        adj[a].append(b)
        first = self._pick_start(comps)
        seeds = ([first] if first else []) + appear
        seen, order = set(), []
        for seed in seeds:
            if seed in seen:
                continue
            q = deque([seed])
            seen.add(seed)
            while q:
                n = q.popleft()
                order.append(n)
                for m in adj[n]:
                    if m not in seen:
                        seen.add(m)
                        q.append(m)
        return self._optimise_order(order, two, devs)

    @staticmethod
    def _optimise_order(order, two, devs):
        """Hill-climb on node order (first node stays put): fewer/shorter arcs and
        tighter device pin spread give cleaner drawings."""
        pairs = [(c.nodes[0].lower(), c.nodes[1].lower()) for c in two
                 if not is_ground(c.nodes[0]) and not is_ground(c.nodes[1])]
        dsets = [[n.lower() for n in c.nodes if not is_ground(n)] for c in devs]
        if len(order) < 3 or len(order) > 40:
            return order

        def cost(o):
            pos = {n: i for i, n in enumerate(o)}
            total, seen = 0.0, defaultdict(int)
            for a, b in pairs:
                i, j = sorted((pos[a], pos[b]))
                seen[(i, j)] += 1
                total += 0 if j - i == 1 else 2 * (j - i - 1) + 1.5
                if seen[(i, j)] > 1:
                    total += 1.5
            for ns in dsets:
                ix = [pos[n] for n in ns]
                if len(ix) > 1:
                    total += max(ix) - min(ix)
            return total

        best = cost(order)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(order)):
                for j in range(i + 1, len(order)):
                    order[i], order[j] = order[j], order[i]
                    c = cost(order)
                    if c < best - 1e-9:
                        best, improved = c, True
                    else:
                        order[i], order[j] = order[j], order[i]
        return order

    # -- two-terminal drawing ---------------------------------------------
    def _draw_between(self, c, pa, pb, s, e, y, left_node, idx, order):
        """Part c between rail/lane points pa (left) and pb (right); the symbol is
        centred in the slot s..e at height y."""
        self._draw_part(c, pa, pb, left_node, centre=((s + e) / 2, y))

    def _draw_chain(self, chain, x, yg):
        """Series parts stacked vertically below the rail at x, ending at height yg.
        The drop is divided into equal slots, one per part, and each symbol is centred
        in its slot, so single-part legs sit mid-height and a two-part branch is
        balanced about the same mid-line (symmetric about the circuit's centre)."""
        n = len(chain)
        drop = -yg
        for k, (c, top) in enumerate(chain):
            y1 = -drop * k / n
            y2 = -drop * (k + 1) / n
            self._draw_part(c, (x, y1), (x, y2), top)

    @staticmethod
    def _fold_bias_chains(comps, two):
        """Find bias branches to stack vertically. A DC voltage source with one terminal
        on ground, whose other terminal touches exactly one other 2-terminal part, is
        paired with that part: returns [(top_node, [(part, node at its top), (bias, node)])].
        Only done when the circuit also has a time-varying source, so plain DC circuits
        (dividers, bridges) keep their normal layout."""
        sig = re.compile(r"\b(ac|sin|pulse|pwl|exp|sffm|am)\b", re.I)
        if not any(c.kind in "VI" and sig.search(c.value) for c in two):
            return []
        count = defaultdict(int)
        for c in comps:
            for n in {_norm(x) for x in c.nodes}:
                count[n] += 1
        folds, taken = [], set()
        for vb in two:
            if vb.kind != "V" or sig.search(vb.value):
                continue
            a, b = _norm(vb.nodes[0]), _norm(vb.nodes[1])
            if (a == "0") == (b == "0"):
                continue
            nb = b if a == "0" else a
            if count[nb] != 2:
                continue
            others = [c for c in two if c is not vb and nb in (_norm(c.nodes[0]), _norm(c.nodes[1]))]
            if len(others) != 1 or others[0].kind in "VI" or id(others[0]) in taken:
                continue
            e = others[0]
            top = _norm(e.nodes[0]) if _norm(e.nodes[1]) == nb else _norm(e.nodes[1])
            if top == "0" or count[top] < 2:
                continue
            taken |= {id(e), id(vb)}
            folds.append((top, [(e, top), (vb, nb)]))
        return folds

    def _ground_bus(self, legs, yg):
        """Ground symbol(s) for the parts hanging off the rail. With the ground_bus
        theme option all legs are tied together and share one ground symbol."""
        if not legs:
            return
        legs = sorted(legs)
        if self.t.ground_bus and len(legs) > 1:
            self.line((legs[0], yg), (legs[-1], yg))
            # one ground symbol, under the leg nearest the middle of the bus
            mid = (legs[0] + legs[-1]) / 2
            gleg = min(legs, key=lambda x: (abs(x - mid), x))
            for x in legs:                      # T-junctions and the ground tap get dots
                if x == gleg or legs[0] < x < legs[-1]:
                    self._dot((x, yg))
            self.ground((gleg, yg))
        else:
            for x in legs:
                self.ground((x, yg))

    # -- parallel-rows ("bridge") layout -----------------------------------
    def _rows_layout(self, two, devs):
        """Draw circuits that are one source plus >=3 parallel chains of parts between
        two nodes (diode bridge + load, Wheatstone bridge, ...). Returns False, having
        drawn nothing, when the circuit is not of that shape."""
        if devs:
            return False
        sources = [c for c in two if c.kind in "VI"]
        net = [c for c in two if c.kind not in "VI"]
        if len(sources) != 1 or len(net) < 4:
            return False
        src = sources[0]

        inc = defaultdict(list)                       # node -> parts touching it
        for c in net:
            inc[_norm(c.nodes[0])].append(c)
            inc[_norm(c.nodes[1])].append(c)
        terms = [n for n, cs in inc.items() if len(cs) >= 3]
        if len(terms) != 2 or any(len(cs) < 2 for cs in inc.values()):
            return False
        A, B = terms
        if len(inc[A]) != len(inc[B]):
            return False

        # trace every chain from A to B
        paths, used = [], set()
        for first in inc[A]:
            path, cur, comp = [], A, first
            while True:
                if id(comp) in used:
                    return False
                used.add(id(comp))
                a, b = _norm(comp.nodes[0]), _norm(comp.nodes[1])
                nxt = b if a == cur else a
                path.append((comp, cur, nxt))
                if nxt in (A, B):
                    break
                others = [x for x in inc[nxt] if x is not comp]
                if len(others) != 1:
                    return False
                cur, comp = nxt, others[0]
            if nxt != B:
                return False
            paths.append(path)
        if len(used) != len(net):
            return False

        # how the source attaches: its ground terminal (if any) is the bottom one
        gnd = "0"
        s0, s1 = _norm(src.nodes[0]), _norm(src.nodes[1])
        if gnd in inc and gnd not in (s0, s1):
            return False
        t_bot = s0 if s0 == gnd else s1
        t_top = s1 if t_bot == s0 else s0
        if t_top not in inc or (t_bot != gnd and t_bot not in inc):
            return False

        internal_row = {}
        for r, p in enumerate(paths):
            for (_, _, v) in p[:-1]:
                internal_row[v] = r
        top_row, bot_row = internal_row.get(t_top), internal_row.get(t_bot)
        if top_row is not None and top_row == bot_row:
            return False

        # face the diodes left -> right where possible
        fwd_a = sum(1 for p in paths for (c, u, v) in p
                    if c.kind == "D" and _norm(c.nodes[0]) == u)
        fwd_b = sum(1 for p in paths for (c, u, v) in p
                    if c.kind == "D" and _norm(c.nodes[0]) == v)
        if fwd_b > fwd_a:
            A, B = B, A
            paths = [[(c, v, u) for (c, u, v) in reversed(p)] for p in paths]

        ordered = [] if top_row is None else [top_row]
        ordered += [r for r in range(len(paths)) if r not in (top_row, bot_row)]
        if bot_row is not None:
            ordered.append(bot_row)

        # ---- geometry ----
        ROW_H, XA = 3.0, 5.0
        W = max(7.0, 3.5 * max(len(p) for p in paths))
        XB = XA + W
        ys = {r: -ROW_H * i for i, r in enumerate(ordered)}
        y_first, y_last = 0.0, -ROW_H * (len(ordered) - 1)
        y_tw, y_bw = 1.0, y_last - 2.0                 # outer wires above / below
        dots = []
        pos = {}                                       # node -> junction point

        for r in ordered:
            p, y = paths[r], ys[r]
            m = len(p)
            xs_ = [XA + i * W / m for i in range(m + 1)]
            for i, (c, u, v) in enumerate(p):
                self._draw_part(c, (xs_[i], y), (xs_[i + 1], y), u)
                if i + 1 < m:
                    pos[v] = (xs_[i + 1], y)
        self.line((XA, y_first), (XA, y_last))
        self.line((XB, y_first), (XB, y_last))
        for r in ordered[1:-1]:
            dots += [(XA, ys[r]), (XB, ys[r])]
        mid_y = ys[ordered[len(ordered) // 2]]
        pos[A], pos[B] = (XA, mid_y), (XB, mid_y)

        def tap_point(n, bottom):
            if n == A or n == B:
                x = XA if n == A else XB
                return (x, y_last if bottom else y_first)
            return pos[n]

        pt = tap_point(t_top, False)
        self.polyline([pt, (pt[0], y_tw), (0, y_tw)])
        dots.append(pt)
        if t_bot in inc:
            pb_ = tap_point(t_bot, True)
            self.polyline([pb_, (pb_[0], y_bw), (0, y_bw)])
            dots.append(pb_)
        if t_bot == gnd:
            self.ground((0, y_bw))
            if t_bot in inc:
                dots.append((0, y_bw))

        self._draw_part(src, (0, y_tw), (0, y_bw), t_top)

        for p_ in dict.fromkeys(dots):
            self._dot(p_)
        if self.t.node_labels:
            for n, (x, y) in pos.items():
                if n != gnd:
                    self.d += elm.Label().at((x - 0.15, y + 0.25)).label(
                        self.disp.get(n, n), fontsize=self.t.fontsize - 2,
                        color=self.t.node_text, halign="right", valign="bottom")
        if self.circ.title and self.t.show_title:
            self.d += elm.Label().at(((XA + XB) / 2, y_tw + 1.6)).label(
                self.circ.title, fontsize=self.t.fontsize + 3, color=self.t.text,
                halign="center")
        return True

    # -- multi-terminal devices -------------------------------------------
    def _device_spec(self, c):
        """Return (factory, [(anchor, node)], caption)."""
        models = self.circ.models
        mtype = models.get(c.model.lower(), "")
        if c.kind == "Q":
            pnp = "PNP" in mtype or (not mtype and "pnp" in c.model.lower())
            cls = elm.BjtPnp if pnp else elm.BjtNpn
            pins = [("collector", c.nodes[0]), ("base", c.nodes[1]), ("emitter", c.nodes[2])]
            return (lambda: cls(circle=True).right()), pins, f"{c.name}\n{c.model}"
        if c.kind == "M":
            pmos = "PMOS" in mtype or (not mtype and re.search(r"pmos|pch|pfet", c.model, re.I))
            cls = elm.PFet if pmos else elm.NFet
            pins = [("drain", c.nodes[0]), ("gate", c.nodes[1]), ("source", c.nodes[2])]
            return (lambda: cls().right().reverse()), pins, f"{c.name}\n{c.model}"
        # subcircuit instance
        low = c.model.lower()
        if any(h in low for h in OPAMP_HINTS) and len(c.nodes) in (3, 5):
            ns = c.nodes
            pins = [("in2", ns[0]), ("in1", ns[1])]
            if len(ns) == 5:
                pins += [("vd", ns[2]), ("vs", ns[3]), ("out", ns[4])]
            else:
                pins += [("out", ns[2])]
            return (lambda: elm.Opamp(sign=True).right()), pins, f"{c.name}\n{c.model}"
        ns = c.nodes
        half = (len(ns) + 1) // 2
        icpins = []
        pins = []
        for k, n in enumerate(ns):
            side = "left" if k < half else "right"
            icpins.append(elm.IcPin(name=self.disp.get(n.lower(), n), side=side,
                                    anchorname=f"p{k}", lblsize=9))
            pins.append((f"p{k}", n))
        return ((lambda: elm.Ic(pins=icpins, pinspacing=1.0, edgepadW=0.6, edgepadH=0.6)
                 .right()), pins, f"{c.name}\n{c.model}")

    def _draw_devices(self, devs, xs, idx, y_dev):
        specs = []
        for c in devs:
            make, pins, caption = self._device_spec(c)
            probe = schemdraw.Drawing(show=False)
            ref = make().at((0, 0))
            probe += ref
            anchors = {k: (v[0], v[1]) for k, v in ref.absanchors.items()}
            cx = (min(anchors[a][0] for a, _ in pins) + max(anchors[a][0] for a, _ in pins)) / 2
            cy = (min(anchors[a][1] for a, _ in pins) + max(anchors[a][1] for a, _ in pins)) / 2
            width = max(anchors[a][0] for a, _ in pins) - min(anchors[a][0] for a, _ in pins)
            wanted = [xs[idx[n.lower()]] for _, n in pins if n.lower() in idx]
            want = sum(wanted) / len(wanted) if wanted else 0.0
            specs.append(dict(c=c, make=make, pins=pins, caption=caption, anchors=anchors,
                              cx=cx, cy=cy, width=width, want=want))
        specs.sort(key=lambda s: s["want"])
        cursor = -1e9
        for sp in specs:
            half = sp["width"] / 2 + 2.5
            x = round(max(sp["want"], cursor + half) * 2) / 2      # snap to half-grid
            cursor = x + half
            sp["x"] = x

        for sp in specs:
            ox, oy = sp["x"] - sp["cx"], y_dev - sp["cy"]
            e = sp["make"]().at((ox, oy)).color(self.t.body)
            self.d += e
            pos = {k: (v[0], v[1]) for k, v in e.absanchors.items()}
            centre = (sp["x"], y_dev)
            all_pins = [pos[a] for a, _ in sp["pins"]]
            ymin = min(p[1] for p in all_pins)
            right_edge = max(p[0] for p in all_pins)
            self.d += elm.Label().at((right_edge + 1.2, y_dev)).label(
                sp["caption"], fontsize=self.t.fontsize - 1, color=self.t.text, halign="left")

            # Classify the exit side of every pin, then give each pin on the same side
            # its own vertical lane (lowest pin innermost) so different nets never overlap.
            sides = defaultdict(list)
            for anchor, node in sp["pins"]:
                p = pos[anchor]
                dx, dy = p[0] - centre[0], p[1] - centre[1]
                if abs(dy) >= abs(dx) * 0.6 and abs(dy) > 0.1:
                    side = "up" if dy > 0 else "down"
                else:
                    side = "right" if dx > 0 else "left"
                sides[side].append((p, node))
            for side, items in sides.items():
                items.sort(key=lambda it: it[0][1])
                for rank, (p, node) in enumerate(items):
                    self._route_pin(p, side, rank, node, ymin, idx, xs)

    def _route_pin(self, pin, side, rank, node, ymin, idx, xs):
        nl = node.lower()
        lane = STUB + 0.6 * rank
        if side == "down":
            pts = [pin, (pin[0], pin[1] - STUB)]
        elif side == "up":
            jog = pin[0] + 1.0 + 0.6 * rank
            pts = [pin, (pin[0], pin[1] + STUB), (jog, pin[1] + STUB)]
        elif side == "left":
            pts = [pin, (pin[0] - lane, pin[1])]
        else:
            pts = [pin, (pin[0] + lane, pin[1])]

        if is_ground(nl):
            if side == "down":
                end = (pin[0], pin[1] - 1.0)
                self.line(pin, end)
                self.ground(end)
                return
            pts = self._nudge(pts, side, None, xs)
            end = (pts[-1][0], ymin - 1.5 - 0.6 * rank)
            self.polyline(pts + [end])
            self.ground(end)
            return

        target_x = xs[idx[nl]]
        pts = self._nudge(pts, side, target_x, xs)
        self.pin_counter += 1
        ym = ymin - 1.0 - 0.6 * self.pin_counter
        x = pts[-1][0]
        pts += [(x, ym), (target_x, ym), (target_x, 0)]
        self.polyline(pts)
        self.degree[nl] += 1

    @staticmethod
    def _nudge(pts, side, target_x, xs):
        """Shift a pin's vertical drop sideways if it would run along (and so look
        connected to) the vertical riser of an unrelated rail node."""
        def clash(v):
            return any(abs(v - nx) < 0.55 for nx in xs
                       if target_x is None or abs(nx - target_x) > 1e-6)
        x, y = pts[-1]
        if not clash(x):
            return pts
        step = -0.4 if side == "left" else 0.4
        for _ in range(25):
            x += step
            if not clash(x):
                break
        if side == "down":
            return pts + [(x, y)]
        return pts[:-1] + [(x, y)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw a circuit diagram from a SPICE netlist.")
    ap.add_argument("netlist", help="netlist file ('-' for stdin)")
    ap.add_argument("-o", "--output", help="output image (png/svg/pdf). Default: <netlist>.png")
    ap.add_argument("--dpi", type=int, default=None,
                    help="raster resolution (default: 150, or 176 for the capture theme)")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    ap.add_argument("--theme", choices=sorted(THEMES), default="default",
                    help="colour/style theme (default: default)")
    ap.add_argument("--no-grid", action="store_true", help="hide the theme's background grid")
    ap.add_argument("--node-labels", action="store_true", help="show node name labels")
    ap.add_argument("--no-node-labels", action="store_true", help="hide node name labels")
    ap.add_argument("--title", help="override the diagram title")
    ap.add_argument("--first-line", choices=["auto", "title", "element"], default="auto",
                    help="how to treat the first line: SPICE title or a circuit element "
                         "(default: auto-detect)")
    args = ap.parse_args(argv)

    if args.netlist == "-":
        text, base = sys.stdin.read(), "netlist"
    else:
        with open(args.netlist, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        base = os.path.splitext(args.netlist)[0]

    circ = parse_netlist(text, first_line=args.first_line)
    theme = THEMES[args.theme]
    if args.title is not None:
        circ.title = args.title
        theme = replace(theme, show_title=True)
    if not circ.components:
        sys.exit("No supported circuit elements found in the netlist.")

    if args.no_grid:
        theme = replace(theme, grid=None)
    if args.node_labels:
        theme = replace(theme, node_labels=True)
    if args.no_node_labels:
        theme = replace(theme, node_labels=False)
    builder = Builder(circ, theme=theme)
    builder.build()

    out = args.output or base + ".svg"
    builder.save(out, dpi=args.dpi or theme.dpi, show=args.show)
    print(f"Wrote {out}  ({len(circ.components)} components, theme: {theme.name})")
    for w in circ.warnings:
        print("warning:", w, file=sys.stderr)


if __name__ == "__main__":
    main()
