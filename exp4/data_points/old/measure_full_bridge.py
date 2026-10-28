#!/usr/bin/env python3
"""
Steady-state measurements for the exp4 full-wave bridge rectifier reports.

Reads the three LTspice transient exports in this folder and prints, for a
one-output-period steady-state window (the last 10 ms of each trace; the
output period is 10 ms in every case), every characteristic parameter of the
task-sheet tables, in the same row order and units as the report tables, so
the simulated column can be transcribed row by row:

  * without filter capacitor: peak input voltage Vm, total diode voltage drop
    2VK, peak output voltage Vo(pk) and current Io(pk), DC and RMS output
    voltage and load current, DC and AC load power, rectification efficiency,
    ripple factor, form factor, peak inverse voltage per diode (the largest
    magnitude of the diode-voltage trace V(D2:1,D2:2) over the last input
    period; that trace is present only in this export), and output frequency
    f0 = 2f;
  * with filter capacitor: peak capacitor voltage Vc(pk), ripple frequency
    fr, peak-to-peak and RMS ripple voltage, DC output voltage, load current,
    ripple factor and percentage ripple.

Theoretical values use the task-sheet formulas with two conducting silicon
diodes (2VK = 1.4 V) at each circuit's source amplitude (100 V for all three
circuits). For the filtered circuits the sheet's approximations
Vr(pp) = IL/(2fC), Vdc = Vc(pk) - Vr(pp)/2 and IL = Vdc/R are solved
simultaneously, giving Vdc = Vc(pk)/(1 + 1/(4fCR)) and Vr(pp) = Vdc/(2fCR),
which reproduces every formula of the with-capacitor table consistently
(the ripple factor then reduces to r = 1/(4*sqrt(3)*f*C*R)).
Columns are located by header name, so exports with and without the diode
trace both load.

Currents are printed in mA and powers in mW, matching the report tables.
Diagnostic checks follow each table: the load resistance implied by the
waveform data itself and an exact integration of the D1N914 diode equation
for the unfiltered circuit, the output valley for the filtered ones, and the
DC level over the two preceding periods for every circuit (steady-state
check).
"""

import math
import re
from pathlib import Path

import numpy as np

FILES = (
    ("full_bridge.txt", "without filter capacitor (task 1)", 0.0, 1e6, 100.0),
    ("full_bridge_with_capacitor.txt", "with C = 100 uF", 100e-6, 1e6, 100.0),
    ("full_bridge_with_capacitor_changed.txt", "with C = 10 uF (modified)",
     10e-6, 1e6, 100.0),
)

VD = 0.7          # per-diode drop used by the task-sheet formulas, V
F_IN = 100.0       # input frequency, Hz
VOUT, VIN, IR = "V(R1:1,R1:2)", "V(1)", "I(R1)"
VDIODE = "V(D2:1,D2:2)"   # present only in the task-1 export (for the PIV)

# D1N914 model card used by the simulations (LTspice default TNOM = 27 C)
D_IS, D_N, D_RS = 168.1e-21, 1.0, 0.1
VT = 1.380649e-23 * 300.15 / 1.602176634e-19   # kT/q at 27 C = 25.85 mV

# Output rows in task-sheet table order: (key, label, unit scale, format).
# Currents are scaled to mA and powers to mW to match the report tables.
UNFILTERED_ROWS = (
    ("Vm",   "Vm",     1,   ".4f"),
    ("2VD",  "2VK",    1,   ".4f"),
    ("Vpk",  "Vo(pk)", 1,   ".4f"),
    ("Ipk",  "Io(pk)", 1e3, ".4f"),
    ("Vdc",  "VDC",    1,   ".4f"),
    ("Idc",  "IDC",    1e3, ".4f"),
    ("Vrms", "VRMS",   1,   ".4f"),
    ("Irms", "IRMS",   1e3, ".4f"),
    ("Pdc",  "PDC",    1e3, ".4f"),
    ("Pac",  "PAC",    1e3, ".4f"),
    ("eta",  "eta",    1,   ".4f"),
    ("r",    "r",      1,   ".4f"),
    ("FF",   "FF",     1,   ".4f"),
    ("PIV",  "PIV",    1,   ".4f"),
    ("f0",   "f0",     1,   ".2f"),
)

FILTERED_ROWS = (
    ("Vc_pk",  "Vc(pk)",  1,   ".4f"),
    ("fr",     "fr",      1,   ".2f"),
    ("Vr_pp",  "Vr(pp)",  1,   ".4f"),
    ("Vr_rms", "Vr(rms)", 1,   ".4f"),
    ("Vdc",    "VDC",     1,   ".4f"),
    ("IL",     "IL",      1e3, ".4f"),
    ("r",      "r",       1,   ".6f"),
    ("pct_r",  "%r",      1,   ".4f"),
)


def load(path):
    with open(path) as fh:
        header = [h.strip().lower() for h in re.split(r"\s+", fh.readline())]
    data = np.genfromtxt(path, delimiter="\t", skip_header=1)
    data = data[~np.isnan(data).any(axis=1)]   # drop truncated trailing rows
    t = data[:, 0]
    col = lambda name: data[:, header.index(name.lower())]
    vout, vin, i_r = col(VOUT), col(VIN), col(IR)
    v_d2 = col(VDIODE) if VDIODE.lower() in header else None
    t_end = t[-1]
    win = (t >= t_end - 10e-3) & (t <= t_end)  # last full output period
    return t, vout, vin, i_r, v_d2, win, t_end


def freq_from_peaks(t, v, win):
    """Output/ripple frequency from well-separated peaks inside the window."""
    tt, vv = t[win], v[win]
    cand = np.where((vv[1:-1] > vv[:-2]) & (vv[1:-1] >= vv[2:]))[0] + 1
    if not len(cand):
        return float("nan")
    kept = []                       # tallest peak wins, 7 ms exclusion zone
    for i in cand[np.argsort(vv[cand])[::-1]]:
        if all(abs(tt[i] - tt[j]) > 3e-3 for j in kept):
            kept.append(i)
    kept.sort()
    if len(kept) < 2:
        return float("nan")
    return 1.0 / np.mean(np.diff(tt[kept]))


def measured(data, cap, R):
    t, vout, vin, i_r, v_d2, win, t_end = data
    tt, dt = t[win], t[win][-1] - t[win][0]
    # Probe cursor average/RMS integrate over time; the exports are adaptively
    # sampled, so trapezoidal integration is required for a faithful average
    w, ii = vout[win], i_r[win]
    vdc = np.trapezoid(w, tt) / dt
    vrms = math.sqrt(np.trapezoid(w ** 2, tt) / dt)
    idc = np.trapezoid(ii, tt) / dt
    irms = math.sqrt(np.trapezoid(ii ** 2, tt) / dt)
    # output/ripple frequency from peak spacing over the last 30 ms (3 peaks)
    wide = (t >= t_end - 30e-3) & (t <= t_end)
    f_pk = freq_from_peaks(t, vout, wide)
    m = {
        "Vm": vin[(t >= t_end - 40e-3) & (t <= t_end)].max(),
        "Vpk": w.max(),
        "Vvalley": w.min(),
        "Vdc": vdc,
        "Vrms": vrms,
        "Idc": idc,
        "Irms": irms,
        "f0": f_pk,
    }
    if cap == 0:
        m["2VD"] = m["Vm"] - m["Vpk"]   # total diode drop seen by the output
        m["Ipk"] = m["Vpk"] / R
        m["Pdc"] = idc ** 2 * R
        m["Pac"] = irms ** 2 * R
        m["eta"] = m["Pdc"] / m["Pac"] * 100
        m["FF"] = vrms / vdc
        m["r"] = math.sqrt(m["FF"] ** 2 - 1)
        if v_d2 is not None:
            # PIV: the largest-magnitude diode voltage over the last input
            # period; the reverse excursion alternates with the forward one,
            # so a single output period does not contain it
            win_in = (t >= t_end - 20e-3) & (t <= t_end)
            m["PIV"] = float(np.max(np.abs(v_d2[win_in])))
    else:
        m["Vc_pk"] = m["Vpk"]
        m["fr"] = f_pk
        m["Vr_pp"] = m["Vpk"] - m["Vvalley"]
        m["Vr_rms"] = math.sqrt(max(vrms ** 2 - vdc ** 2, 0.0))
        m["IL"] = idc
        m["r"] = m["Vr_rms"] / vdc
        m["pct_r"] = m["r"] * 100
    return m


def theoretical(cap, R, vm):
    vk2 = 2 * VD
    vp = vm - vk2
    if cap == 0:
        t = {
            "Vm": vm,
            "2VD": vk2,
            "Vpk": vp,
            "Vdc": 2 * vp / math.pi,
            "Vrms": vp / math.sqrt(2),
            "PIV": vm,
            "f0": 2 * F_IN,
        }
        t["Ipk"] = t["Vpk"] / R
        t["Idc"] = t["Vdc"] / R
        t["Irms"] = t["Vrms"] / R
        t["Pdc"] = t["Idc"] ** 2 * R
        t["Pac"] = t["Irms"] ** 2 * R
        t["eta"] = t["Pdc"] / t["Pac"] * 100
        t["FF"] = t["Vrms"] / t["Vdc"]
        t["r"] = math.sqrt(t["FF"] ** 2 - 1)
        return t
    # task-sheet approximations solved simultaneously:
    # Vr(pp) = IL/(2fC), Vdc = Vc(pk) - Vr(pp)/2, IL = Vdc/R
    #   =>  Vdc = Vc(pk)/(1 + 1/(4fCR)),  Vr(pp) = Vdc/(2fCR)
    two_f_cr = 2 * F_IN * cap * R
    t = {
        "Vc_pk": vp,
        "fr": 2 * F_IN,
        "Vdc": vp / (1.0 + 1.0 / (2 * two_f_cr)),
    }
    t["Vr_pp"] = t["Vdc"] / two_f_cr
    t["Vr_rms"] = t["Vr_pp"] / (2 * math.sqrt(3))
    t["IL"] = t["Vdc"] / R
    t["r"] = t["Vr_rms"] / t["Vdc"]
    t["pct_r"] = t["r"] * 100
    return t


def diode_model_vout(t, vin, win, R):
    """Bridge output from the exact D1N914 diode equation, sampled at the
    export's own time points: |vin| = 2*vd(i) + i*R, solved for i (bisection)."""
    v = np.abs(vin[win])
    out = np.empty_like(v)
    for k, vk in enumerate(v):
        lo, hi = 0.0, vk / R
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            vd_pair = 2.0 * (D_N * VT * math.log1p(mid / D_IS) + mid * D_RS)
            if vd_pair + mid * R < vk:
                lo = mid
            else:
                hi = mid
        out[k] = 0.5 * (lo + hi) * R
    return out


def print_table(m, th, rows):
    print(f"{'quantity':>9} | {'simulated':>12} | {'theoretical':>12} | {'% err':>7}")
    print("-" * 50)
    for key, label, scale, fmt in rows:
        if key not in m:
            continue
        s, tv = m[key] * scale, th[key] * scale
        err = abs(tv - s) / abs(tv) * 100
        print(f"{label:>9} | {format(s, fmt):>12} | {format(tv, fmt):>12} |"
              f" {err:6.2f}%")


for fname, label, cap, R, vm in FILES:
    data = load(Path(__file__).resolve().parent / fname)
    m = measured(data, cap, R)
    th = theoretical(cap, R, vm)
    rows = UNFILTERED_ROWS if cap == 0 else FILTERED_ROWS
    print(f"\n=== {label}  ({fname}, Vm = {vm:g} V) ===")
    print("(voltages in V, currents in mA, powers in mW, frequencies in Hz)")
    print_table(m, th, rows)
    t, vout, vin, i_r, v_d2, win, t_end = data
    if cap == 0:
        w_v, w_i = vout[win], i_r[win]
        k = int(np.argmax(w_v))
        print(f"  check: R implied by the waveforms, Vpk/Ipk = "
              f"{w_v[k] / abs(w_i[k]):.4g} ohm (nominal {R:.4g})")
        model = diode_model_vout(t, vin, win, R)
        dt = t[win][-1] - t[win][0]
        vdc_m = np.trapezoid(model, t[win]) / dt
        i_crest = model.max() / R
        vd_crest = D_N * VT * math.log1p(i_crest / D_IS) + i_crest * D_RS
        print(f"  check: D1N914 exact model Vdc = {vdc_m:.4f} V, "
              f"Vpk = {model.max():.4f} V (measured {m['Vdc']:.4f} / "
              f"{m['Vpk']:.4f}); per-diode drop at crest = {vd_crest:.4f} V")
    else:
        print(f"  check: output valley (capacitor discharge minimum) = "
              f"{m['Vvalley']:.4f} V")
    # steady-state check: DC level (trapezoidal) over the two preceding periods
    for lo in (t_end - 30e-3, t_end - 20e-3):
        w2 = (t >= lo) & (t <= lo + 10e-3)
        dt2 = t[w2][-1] - t[w2][0]
        print(f"  check: Vdc over [{lo * 1e3:.0f},{(lo + 10e-3) * 1e3:.0f}] ms = "
              f"{np.trapezoid(vout[w2], t[w2]) / dt2:.3f} V")
