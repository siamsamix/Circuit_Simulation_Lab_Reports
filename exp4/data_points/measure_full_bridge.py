#!/usr/bin/env python3
"""
Steady-state measurements for the exp4 full-wave bridge rectifier reports.

Reads the three LTspice transient exports in this folder and prints, for a
one-output-period steady-state window (the last 10 ms of each trace; the
output period is 10 ms in every case), the quantities a student would read
off the Probe waveform viewer: input peak, output peak/valley, peak-to-peak
ripple, DC and RMS output, DC/RMS load current, form factor, ripple factor,
rectification efficiency, output/ripple frequency, and — for the unfiltered
circuit, whose export also carries the diode-voltage trace V(D2:1,D2:2) —
the peak inverse voltage as the largest-magnitude value of that trace (the
reverse-biased excursion; the sign convention of the export's diode pin
order is irrelevant for the magnitude).

Theoretical values use the task-sheet formulas with two conducting silicon
diodes (2Vd = 1.4 V) at each circuit's source amplitude (100 V for all three
circuits), so the percentage errors can be transcribed straight into the
report tables. Columns are located by header name, so exports with and
without the diode trace both load.

For the unfiltered circuit two extra checks are printed: the load resistance
implied by the waveform data itself (peak V(R1:1,R1:2) over peak I(R1)), and
the average/peak output predicted by an exact integration of the D1N914 diode
equation (Shockley: Is = 168.1e-21 A, N = 1, Rs = 0.1 ohm, TNOM = 27 C),
which isolates how much of the simulated-vs-theoretical gap comes from the
real voltage-dependent diode drop rather than the constant-0.7 V idealization.
"""

import math
import re
from pathlib import Path

import numpy as np

FILES = (
    ("full_bridge.txt", "without filter capacitor (task 1)", 0.0, 1e5, 100.0),
    ("full_bridge_with_capacitor.txt", "with C = 100 uF", 100e-6, 1e5, 100.0),
    ("full_bridge_with_capacitor_changed.txt", "with C = 10 uF (modified)",
     10e-6, 1e5, 100.0),
)

VD = 0.7          # per-diode drop used by the task-sheet formulas, V
F_IN = 50.0       # input frequency, Hz
VOUT, VIN, IR = "V(R1:1,R1:2)", "V(1)", "I(R1)"
VDIODE = "V(D2:1,D2:2)"   # present only in the task-1 export (for the PIV)

# D1N914 model card used by the simulations (LTspice default TNOM = 27 C)
D_IS, D_N, D_RS = 168.1e-21, 1.0, 0.1
VT = 1.380649e-23 * 300.15 / 1.602176634e-19   # kT/q at 27 C = 25.85 mV


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
        if all(abs(tt[i] - tt[j]) > 7e-3 for j in kept):
            kept.append(i)
    kept.sort()
    if len(kept) < 2:
        return float("nan")
    return 1.0 / np.mean(np.diff(tt[kept]))


def measured(path):
    t, vout, vin, i_r, v_d2, win, t_end = load(path)
    tt, dt = t[win], t[win][-1] - t[win][0]
    # Probe cursor average/RMS integrate over time; the exports are adaptively
    # sampled, so trapezoidal integration is required for a faithful average
    w, ii = vout[win], i_r[win]
    vdc = np.trapezoid(w, tt) / dt
    vrms = math.sqrt(np.trapezoid(w ** 2, tt) / dt)
    idc = np.trapezoid(ii, tt) / dt
    irms = math.sqrt(np.trapezoid(ii ** 2, tt) / dt)
    ff = vrms / vdc
    vr_ac = math.sqrt(max(vrms ** 2 - vdc ** 2, 0.0))   # RMS of the AC part
    # output/ripple frequency from peak spacing over the last 30 ms (3 peaks)
    wide = (t >= t_end - 30e-3) & (t <= t_end)
    f_pk = freq_from_peaks(t, vout, wide)
    m = {
        "Vin_peak": vin[(t >= t_end - 40e-3) & (t <= t_end)].max(),
        "Vpk": w.max(),
        "Vvalley": w.min(),
        "Vr_pp": w.max() - w.min(),
        "Vdc": vdc,
        "Vrms": vrms,
        "Idc": idc,
        "Irms": irms,
        "FF": ff,
        "Vr_ac": vr_ac,
        "r": math.sqrt(ff ** 2 - 1),
        "eta": (idc ** 2 * R) / (irms ** 2 * R) * 100,
        "f_out": f_pk,
    }
    if v_d2 is not None:
        # PIV: the largest-magnitude diode voltage over the last input period;
        # the reverse excursion alternates with the forward one, so a single
        # output period does not contain it
        win_in = (t >= t_end - 20e-3) & (t <= t_end)
        m["PIV"] = float(np.max(np.abs(v_d2[win_in])))
    return m


def theoretical(cap, R, vm):
    vd2 = 2 * VD
    vp = vm - vd2
    t = {
        "Vin_peak": vm,
        "Vpk": vp,
        "Vdc": 2 * vp / math.pi,
        "Vrms": vp / math.sqrt(2),
        "PIV": vm,
        "f_out": 2 * F_IN,
    }
    t["Ipk"] = t["Vpk"] / R
    t["Idc"] = t["Vdc"] / R
    t["Irms"] = t["Vrms"] / R
    t["Pdc"] = t["Idc"] ** 2 * R
    t["Pac"] = t["Irms"] ** 2 * R
    t["eta"] = t["Pdc"] / t["Pac"] * 100
    t["FF"] = t["Vrms"] / t["Vdc"]
    t["r"] = math.sqrt(t["FF"] ** 2 - 1)
    if cap > 0:
        # task-sheet approximation, solved self-consistently:
        # Vr = Vdc/(2 f C R), Vdc = Vp - Vr/2  =>  Vdc (1 + 1/(4 f C R)) = Vp
        two_f_cr = 2 * F_IN * cap * R
        t["Vdc"] = vp / (1.0 + 1.0 / (2 * two_f_cr))
        t["Vr_pp"] = t["Vdc"] / two_f_cr
        t["Vr_ac"] = t["Vr_pp"] / (2 * math.sqrt(3))
        t["r"] = t["Vr_ac"] / t["Vdc"]
        t["Idc"] = t["Vdc"] / R
        t["f_out"] = 2 * F_IN
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


ORDER = ["Vin_peak", "Vpk", "Vvalley", "Vr_pp", "Vdc", "Vrms", "Idc", "Irms",
         "Pdc", "Pac", "eta", "FF", "Vr_ac", "r", "PIV", "f_out"]

for fname, label, cap, R, vm in FILES:
    m = measured(Path(__file__).resolve().parent / fname)
    th = theoretical(cap, R, vm)
    print(f"\n=== {label}  ({fname}, Vm = {vm:g} V) ===")
    print(f"{'quantity':>10} | {'simulated':>12} | {'theoretical':>12} | {'% err':>7}")
    print("-" * 52)
    for k in ORDER:
        if k in m:
            s, tv = m[k], th.get(k, float("nan"))
            if tv != 0 and not math.isnan(tv):
                err = abs(tv - s) / abs(tv) * 100
                print(f"{k:>10} | {s:12.4f} | {tv:12.4f} | {err:6.2f}%")
            else:
                print(f"{k:>10} | {s:12.4f} | {'-':>12} |    -")
    if cap == 0:
        print(f"{'Ipk':>10} | {m['Vpk']/R*1e3:12.4f} | {th['Ipk']*1e3:12.4f} |"
              f" {abs(th['Ipk']-m['Vpk']/R)/th['Ipk']*100:6.2f}%")
        s2vd = m["Vin_peak"] - m["Vpk"]
        print(f"{'2VD':>10} | {s2vd:12.4f} | {2*VD:12.4f} |"
              f" {abs(2*VD-s2vd)/(2*VD)*100:6.2f}%")
        # data checks: implied load resistance and exact-diode-model output
        t, vout, vin, i_r, v_d2, win, t_end = load(
            Path(__file__).resolve().parent / fname)
        w_v, w_i = vout[win], i_r[win]
        k = int(np.argmax(w_v))
        print(f"{'R_check':>10} | Vpk/Ipk = {w_v[k]/abs(w_i[k]):.4g} ohm"
              f"  (nominal {R:.4g})")
        model = diode_model_vout(t, vin, win, R)
        dt = t[win][-1] - t[win][0]
        vdc_m = np.trapezoid(model, t[win]) / dt
        i_crest = model.max() / R
        vd_crest = D_N * VT * math.log1p(i_crest / D_IS) + i_crest * D_RS
        print(f"{'D1N914 model':>12} | Vdc = {vdc_m:.4f} V, Vpk = {model.max():.4f} V"
              f"  (measured {m['Vdc']:.4f} / {m['Vpk']:.4f});"
              f" per-diode drop at crest = {vd_crest:.4f} V")
    else:
        print(f"{'Idc_check':>10} | {m['Vdc']/R:12.6f} | {th['Idc']:12.6f} | "
              f"(Vdc/R vs mean I(R1) above)")
    # steady-state check: DC level (trapezoidal) over the two preceding periods
    t, vout, _, _, _, _, t_end = load(Path(__file__).resolve().parent / fname)
    for lo in (t_end - 30e-3, t_end - 20e-3):
        w2 = (t >= lo) & (t <= lo + 10e-3)
        dt2 = t[w2][-1] - t[w2][0]
        print(f"    Vdc over [{lo*1e3:.0f},{(lo+10e-3)*1e3:.0f}] ms = "
              f"{np.trapezoid(vout[w2], t[w2])/dt2:.3f} V")
