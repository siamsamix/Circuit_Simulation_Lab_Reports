#!/usr/bin/env python3
"""Experiment 1 — quantitative analysis for the lab report.

Theoretical values: derived symbolically (SymPy), evaluated numerically at
the end (python-math skill: exact first, floats last).

Simulated values: read from the SPICE text exports in Data_points/ (the
primary data source per the instructor's instructions) and measured from the
waveform arrays by interpolation and least-squares fits — never by eye.

Circuits (from netlists/):
  Task 1/2  : 10 V DC, S1 closed 0-6 ms, charging R = 1 kOhm, C = 1 uF;
              S2 closes at 7 ms, discharge through (1k + 1.8k) = 2.8 kOhm.
  Task 3   : PULSE(0 5 0 1n 1n 5m 10m), R = 1 kOhm, C = 1 uF.
  Task 4   : 10 sin(2pi*100 t - 70 deg) V, series R = 1 kOhm, C = 1 uF
              (0.5 kOhm shunt across the ideal source -> no effect on v_C).
  Task 5   : task 4 with .ic V(N002) = -8.2748 V.
  Task 6a  : 20 V, R_ch = 3 k, C = 5 uF, S1 opens 75 ms; discharge via
              (3k + 2k) = 5 kOhm after S2 closes at 76 ms.
  Task 6b  : PULSE(0 5 0 1n 1n 25m 50m), R = 0.1 k, C = 50 uF.
  Task 6c/d: 50 sin(2pi*200 t - 50 deg) V, series R = 0.1 k (1 k shunt
              across source), C = 50 uF; (d) with .ic V(N002) = -5.935 V.
"""
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parent / "Data_points"
E = np.exp(1.0)


# ------------------------------------------------------------------ helpers
def load(relname):
    arr = np.genfromtxt(DATA / relname, delimiter="\t", skip_header=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[~np.isnan(arr).any(axis=1)]          # drop trailing bad rows


def interp_at(t, y, x):
    return float(np.interp(x, t, y))


def cross_after(t, y, t0, level):
    """First time >= t0 at which y crosses `level` (linear interpolation)."""
    i0 = int(np.searchsorted(t, t0))
    for i in range(i0, len(t) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 - level) * (y2 - level) < 0:
            return float(t[i] + (level - y1) * (t[i + 1] - t[i]) / (y2 - y1))
    raise ValueError(f"no crossing of {level} after {t0}")


def exp_fit(t, y, mask):
    """Fit y = A exp(-t/tau) on masked samples (t already shifted)."""
    tt, yy = t[mask], np.abs(y[mask])
    slope, inter = np.polyfit(tt, np.log(yy), 1)
    return float(np.exp(inter)), float(-1.0 / slope)


def exp_fit_free_vf(t, y, mask, p0):
    """Fit y = Vf + A exp(-t/tau) with free final value Vf (t shifted)."""
    tt, yy = t[mask], y[mask]
    f = lambda x, Vf, A, tau: Vf + A * np.exp(-x / tau)
    (Vf, A, tau), _ = curve_fit(f, tt, yy, p0=p0, maxfev=20000)
    return float(Vf), float(A), float(tau)


def fit_sinus(t, y, freq):
    """Least-squares fit y ~ A sin(w t + phi); returns (A, phi deg)."""
    w = 2 * np.pi * freq
    S = np.column_stack([np.sin(w * t), np.cos(w * t)])
    (a, b), *_ = np.linalg.lstsq(S, y, rcond=None)
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)))


def pct_err(th, sim):
    return (th - sim) / th * 100.0


def show(label, theory, sim, unit="", digits=4):
    te = pct_err(theory, sim) if theory != 0 else float("nan")
    print(f"  {label:<42s} theory={theory:.{digits}f}{unit}  "
          f"sim={sim:.{digits}f}{unit}  err={te:+.4f}%")


print("=" * 78)

# ---------------------------------------------------------------- task 1/2
print("TASK 1/2 — switching RC (charge 0-6 ms via 1k, discharge via 2.8k)")
d = load("task1,2.txt")
t, vdiff, vc = d[:, 0], d[:, 1], d[:, 2]
vc6, vc7 = interp_at(t, vc, 6e-3), interp_at(t, vc, 7e-3)
tau_c_cross = cross_after(t, vc, 0.0, 10.0 * (1 - 1 / E))
tau_d_cross = cross_after(t, vc, 7e-3, vc7 / E) - 7e-3
m = (t > 7.02e-3) & (t < 1.3e-2) & (vc > 0.05) & (vc < 9.0)
A_d, tau_d_fit = exp_fit(t - 7e-3, vc, m)
Vf_d, A_d2, tau_d_vf = exp_fit_free_vf(t - 7e-3, vc, m, (0.0, 9.9, 2.8e-3))
tau_c_th, tau_d_th = 1e3 * 1e-6, 2.8e3 * 1e-6
vc6_th = 10 * (1 - np.exp(-6))
show("tau_charge (63.2% crossing, ms)", 1.0, tau_c_cross * 1e3)
show("tau_discharge (36.8% crossing, ms)", 2.8, tau_d_cross * 1e3)
show("tau_discharge (Vf=0 log fit, ms)", 2.8, tau_d_fit * 1e3)
print(f"  tau_discharge free-Vf fit: Vf={Vf_d*1e3:.3f} mV, "
      f"A={A_d2:.4f} V, tau={tau_d_vf*1e3:.4f} ms")
show("v_C(6 ms), V", vc6_th, vc6)
show("v_C(7 ms, end of hold), V", vc6_th, vc7)
print(f"  hold droop 6->7 ms: {(vc6 - vc7)*1e3:.3f} mV")
print(f"  v_C(20 ms) = {interp_at(t, vc, 20e-3):.4f} V")

# ---------------------------------------------------------------- task 3
print("=" * 78)
print("TASK 3 — pulse input, v_C at t = 1..10 ms (tau = 1 ms)")
d = load("task3.txt")
t, vr, vsrc, vc = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
tau_p = cross_after(t, vc, 0.0, 5.0 * (1 - 1 / E))
show("tau from charge curve (ms)", 1.0, tau_p * 1e3)
v12 = interp_at(t, vc, 12e-3)               # is the 10 ms-period pulse repeating?
print(f"  v_C(12 ms) = {v12:.4f} V  -> pulse {'REPEATS' if v12 > 1 else 'single-shot'}")
V5 = 5 * (1 - np.exp(-5))
print("   t(ms)   theory(V)   sim(V)    err(%)")
for i in range(1, 11):
    th = 5 * (1 - np.exp(-i)) if i <= 5 else V5 * np.exp(-(i - 5))
    sim = interp_at(t, vc, i * 1e-3)
    print(f"   {i:4d}    {th:9.4f}  {sim:8.4f}   {pct_err(th, sim):+8.4f}")

# ---------------------------------------------------------------- task 4
print("=" * 78)
print("TASK 4 — sinusoid 10 V/100 Hz/-70 deg, transient present")
d4 = load("task4.txt")
t4, vsrc4, vc4 = d4[:, 0], d4[:, 1], d4[:, 2]
w = t4 >= 10e-3                                   # 10 tau -> transient dead
A_src4, ph_src4 = fit_sinus(t4[w], vsrc4[w], 100.0)
A_cap4, ph_cap4 = fit_sinus(t4[w], vc4[w], 100.0)

wRC = sp.pi / 5                                   # 2*pi*100*1e3*1e-6
A4 = sp.N(10 / sp.sqrt(1 + wRC**2), 10)           # 8.467331...
lag4 = sp.N(sp.deg(sp.atan(wRC)), 10)             # 32.1419...
SS0_4 = sp.N(-(10 / sp.sqrt(1 + wRC**2)) * sp.sin(sp.rad(70) + sp.atan(wRC)), 10)
A4f, lag4f, SS0_4f = float(A4), float(lag4), float(SS0_4)
print(f"  theory: A={A4f:.6f} V  lag={lag4f:.6f} deg  "
      f"phi_VC(0deg-ref)={-102 + 0:.0f}.. -> v_C phase = {ph_cap4:.4f} deg")
print(f"  theory v_C,SS(0) = V_IC = {SS0_4f:.6f} V")
show("steady-state amplitude A (V)", A4f, A_cap4)
show("phase of v_C (deg)", -70.0 - lag4f, ph_cap4)
show("phase lag (deg)", lag4f, (ph_src4 - ph_cap4) % 360.0)

ss_th = lambda tt: A4f * np.sin(2 * np.pi * 100 * tt - np.radians(70 + lag4f))
res4 = vc4 - ss_th(t4)
m = (t4 < 6e-3) & (np.abs(res4) > 0.02) & (np.abs(res4) < 6.0)
A_tr4, tau_tr4 = exp_fit(t4, res4, m)
show("transient coefficient A_tr (V)", -SS0_4f, A_tr4)
show("transient tau (ms)", 1.0, tau_tr4 * 1e3)
print(f"  v_C(20 ms) = {interp_at(t4, vc4, 20e-3):.6f} V "
      f"(SS_theory {ss_th(20e-3):.6f} V)")
print(f"  source fit check: A={A_src4:.4f} V (10), phi={ph_src4:.4f} deg (-70)")

# ---------------------------------------------------------------- task 5
print("=" * 78)
print("TASK 5 — transient eliminated (.ic V(N002) = -8.2748 V)")
d5 = load("task5.txt")
t5, vsrc5, vc5 = d5[:, 0], d5[:, 1], d5[:, 2]
i1 = int(np.argmax(t5 > 0))                        # skip t=0 export artifact
print(f"  v_C(first sample, t={t5[i1]:.2e} s) = {vc5[i1]:.4f} V")
res5 = vc5 - ss_th(t5)
res5 = res5[t5 > 0]
print(f"  max |v_C - SS_theory| (t>0) = {np.max(np.abs(res5))*1e3:.4f} mV")
show("required V_IC (V)", SS0_4f, vc5[i1])
show("required Q0 = C*V_IC (uCou)", SS0_4f * 1.0, vc5[i1] * 1.0)
print(f"  v_C(20 ms) = {interp_at(t5, vc5, 20e-3):.6f} V "
      f"(SS_theory {ss_th(20e-3):.6f} V)")

# ---------------------------------------------------------------- task 6a
print("=" * 78)
print("TASK 6a — switching, 20 V, 3k/2k, C = 5 uF, switches 75/76 ms")
d = load("task6/switch_values_changed.txt")
t, vdiff, vc = d[:, 0], d[:, 1], d[:, 2]
v75, v76 = interp_at(t, vc, 75e-3), interp_at(t, vc, 76e-3)
tau_c_cross = cross_after(t, vc, 0.0, 20.0 * (1 - 1 / E))
tau_d_cross = cross_after(t, vc, 76e-3, v76 / E) - 76e-3
tau_d_2x = (cross_after(t, vc, 76e-3, v76 / E**2) - 76e-3) / 2
m = (t > 7.8e-2) & (t < 1.6e-1) & (vc > 0.1) & (vc < 19.0)
A_d, tau_d_fit = exp_fit(t - 76e-3, vc, m)
Vf_d, A_d2, tau_d_vf = exp_fit_free_vf(t - 76e-3, vc, m, (0.04, 19.8, 25e-3))
show("tau_charge (63.2% crossing, ms)", 15.0, tau_c_cross * 1e3)
show("tau_discharge (36.8% crossing, ms)", 25.0, tau_d_cross * 1e3)
show("tau_discharge (2-tau crossing / 2, ms)", 25.0, tau_d_2x * 1e3)
show("tau_discharge (Vf=0 log fit, ms)", 25.0, tau_d_fit * 1e3)
print(f"  tau_discharge free-Vf fit: Vf={Vf_d*1e3:.3f} mV, "
      f"A={A_d2:.4f} V, tau={tau_d_vf*1e3:.4f} ms")
show("v_C(75 ms), V", 20 * (1 - np.exp(-5)), v75)
print(f"  hold droop 75->76 ms: {(v75 - v76)*1e3:.3f} mV")

# ---------------------------------------------------------------- task 6b
print("=" * 78)
print("TASK 6b — pulse 0-5 V, width 25 ms, period 50 ms, R = 0.1k, C = 50 uF")
d = load("task6/pulse_values_changed.txt")
t, vr, vsrc, vc = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
tau_b = cross_after(t, vc, 0.0, 5.0 * (1 - 1 / E))
show("tau (ms)", 5.0, tau_b * 1e3)
v25, v50 = 5 * (1 - np.exp(-5)), 5 * (1 - np.exp(-5)) * np.exp(-5)
v75b = 5 - (5 - v50) * np.exp(-5)
v100b = v75b * np.exp(-5)
for x, th in [(25, v25), (50, v50), (75, v75b), (100, v100b)]:
    show(f"v_C({x} ms), V", th, interp_at(t, vc, x * 1e-3))

# ---------------------------------------------------------------- task 6c/d
print("=" * 78)
print("TASK 6c/d — sinusoid 50 V/200 Hz/-50 deg, 0.1k, 50 uF (tau = 5 ms)")
d6c = load("task6/sinusoidal_transient.txt")
t6, vsrc6, vc6c = d6c[:, 0], d6c[:, 1], d6c[:, 2]
d6d = load("task6/sinusoidal_transient_eliminated.txt")
t6d, vsrc6d, vc6d = d6d[:, 0], d6d[:, 1], d6d[:, 2]

wRC6 = 2 * sp.pi                                   # 2*pi*200*100*50e-6
A6 = sp.N(50 / sp.sqrt(1 + wRC6**2), 10)
lag6 = sp.N(sp.deg(sp.atan(wRC6)), 10)
SS0_6 = sp.N(-(50 / sp.sqrt(1 + wRC6**2)) * sp.sin(sp.rad(50) + sp.atan(wRC6)), 10)
A6f, lag6f, SS0_6f = float(A6), float(lag6), float(SS0_6)
print(f"  theory: A={A6f:.6f} V  lag={lag6f:.6f} deg  V_IC={SS0_6f:.6f} V  "
      f"(netlist .ic = -5.935)")

ss_th6 = lambda tt: A6f * np.sin(2 * np.pi * 200 * tt - np.radians(50 + lag6f))
# (c): v_C(0)=0 -> transient present
res6c = vc6c - ss_th6(t6)
m = (np.abs(res6c) > 0.05) & (np.abs(res6c) < 6.0)
A_tr6, tau_tr6 = exp_fit(t6, res6c, m)
show("transient coefficient A_tr (V)", -SS0_6f, A_tr6)
show("transient tau (ms)", 5.0, tau_tr6 * 1e3)
# steady state measured on the pure-SS run (d)
w = t6d >= 10e-3
A_cap6, ph_cap6 = fit_sinus(t6d[w], vc6d[w], 200.0)
A_src6, ph_src6 = fit_sinus(t6d[w], vsrc6d[w], 200.0)
show("steady-state amplitude A (V)", A6f, A_cap6)
show("phase of v_C (deg)", -50.0 - lag6f, ph_cap6)
show("phase lag (deg)", lag6f, (ph_src6 - ph_cap6) % 360.0)
# (d): transient eliminated
i1 = int(np.argmax(t6d > 0))
res6d = (vc6d - ss_th6(t6d))[t6d > 0]
print(f"  v_C(first sample) = {vc6d[i1]:.6f} V;  "
      f"max |v_C - SS_theory| (t>0) = {np.max(np.abs(res6d))*1e6:.3f} uV")
show("required V_IC (V)", SS0_6f, vc6d[i1])
show("required Q0 = C*V_IC (uCou)", SS0_6f * 50.0, vc6d[i1] * 50.0)

# ------------------------------------------------------- sanity cross-checks
print("=" * 78)
print("CROSS-CHECKS")
import math
assert math.isclose(float(sp.N(wRC)), math.pi / 5, rel_tol=1e-12)
assert math.isclose(float(sp.N(wRC6)), 2 * math.pi, rel_tol=1e-12)
assert math.isclose(20 * (1 - np.exp(-5)), 19.8652, abs_tol=5e-5)
assert math.isclose(5 * (1 - np.exp(-5)), 4.9663, abs_tol=5e-5)
# task4 vs task5: v5 = SS + (IC - SS0) e^-t/tau; diff of runs = SS0-IC decay
diff45 = (vc4 - vc5)[t5 > 0]
print(f"  task4-task5 difference at t=0.5 ms: "
      f"{interp_at(t5[t5 > 0], diff45, 0.5e-3):.4f} V "
      f"(expect (SS0-IC)e^-0.5 = {(-SS0_4f - (-8.2748)) * np.exp(-0.5):.4f} V)")
print(f"  task6c source check: A={A_src6:.4f} V (50), phi={ph_src6:.4f} deg (-50)")
print("ALL CHECKS DONE")
