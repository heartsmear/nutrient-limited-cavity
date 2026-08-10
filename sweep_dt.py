"""
sweep_dt.py -- is the solution dt-converged, at a dx that actually resolves the front?

WHY THIS IS NOT ALREADY KNOWN.  src/ established dt-convergence (F1 = 0.2251/0.2247/0.2244 at
dt = 0.1/0.05/0.025) for the DESIGNED pair (n* = 1.5) at n0 = 2.8-4.2 and dx = 10.  None of that
transfers here: different biology, different regime, and 5-20x coarser grid.  The Lie splitting
error between {reaction+nutrient}, {collective diffusion} and {transport} is O(dt) with a
coefficient set by the GRADIENTS, and those sharpen as dx refines -- so a fine grid is where dt
is most at risk, not least.  `growth_space_factor` is also evaluated explicitly at rho_old, an
O(dt) term that has never been tested (it is inert whenever q == 1).

dx is held FIXED within each n0 and chosen to resolve the front (dx/lambda is reported, and must
be <~ 0.6 -- calibrated in sweep_dx_n0-1000_T500), so this isolates dt.

    py -3.11 src2/sweep_dt.py
    py -3.11 src2/sweep_dt.py --n0s 15,100 --dxs 5,2 --dts 0.1,0.05,0.025,0.0125 --T 150
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                   # noqa: E402
import solver                                   # noqa: E402
import diagnostics as D                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n0s", default="15,100")
    ap.add_argument("--dxs", default="5,2", help="one dx per n0, chosen to resolve that front")
    ap.add_argument("--dts", default="0.1,0.05,0.025,0.0125")
    ap.add_argument("--T", type=float, default=150.0)
    ap.add_argument("--outdir", default=None)
    config.add_model_args(ap)
    a = ap.parse_args()

    n0s = [float(v) for v in a.n0s.split(",")]
    dxs = [float(v) for v in a.dxs.split(",")]
    dts = sorted([float(v) for v in a.dts.split(",")], reverse=True)
    if len(dxs) != len(n0s):
        sys.exit("--dxs must give one dx per n0")
    outdir = a.outdir or os.path.join(HERE, f"sweep_dt_T{a.T:g}")
    os.makedirs(outdir, exist_ok=True)

    print(f"T={a.T:g} h   n*={config.NSTAR:.4f} uM   dt in {dts}", flush=True)
    print(f"{'n0':>8}{'dx':>6}{'N':>6}{'dt':>8}{'rho_peak':>10}{'q_min':>8}{'F1_final':>12}"
          f"{'df_max':>9}{'x_n*':>8}{'dx/lam':>8}{'s':>7}", flush=True)

    rows = []
    for n0, dx in zip(n0s, dxs):
        for dt in dts:
            p = config.params_from_args(a, n0)
            num = config.numerics(T_end=a.T, dt=dt, dx=dx, n_save=151)
            t0 = time.time()
            res = solver.simulate(config.L, p, num, config.initial_conditions())
            el = time.time() - t0
            v = D.validate(res, p, verbose=False)
            rows.append(dict(n0=n0, dx=dx, N=res["N"], dt=dt,
                             rho_peak=v["rho_peak"], q_min=v["q_min"],
                             F1_final=v["F1_final"], delta_f_max=v["delta_f_max"],
                             x_nstar=v["x_nstar"], dx_over_lambda=v["dx_over_lambda"],
                             lambda_front=v["lambda_front"],
                             admissible=v["admissible"], wall_s=round(el, 1)))
            print(f"{n0:>8g}{dx:>6g}{res['N']:>6d}{dt:>8g}{v['rho_peak']:>10.5f}"
                  f"{v['q_min']:>8.4f}{v['F1_final']:>12.6f}{v['delta_f_max']:>9.4f}"
                  f"{v['x_nstar']:>8.2f}{v['dx_over_lambda']:>8.3f}{el:>7.0f}", flush=True)

    path = os.path.join(outdir, "sweep_dt.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {path}", flush=True)

    # --- verdict: drift relative to the FINEST dt, which is the reference ---
    print("\n=== dt convergence (drift vs the finest dt) ===", flush=True)
    for n0, dx in zip(n0s, dxs):
        sub = sorted([r for r in rows if r["n0"] == n0], key=lambda r: -r["dt"])
        ref = sub[-1]
        print(f"  n0={n0:g} uM, dx={dx:g} (dx/lambda={ref['dx_over_lambda']:.3f}"
              f"{'  UNDER-RESOLVED' if ref['dx_over_lambda'] > 0.6 else ''}):", flush=True)
        for r in sub:
            for k in ("F1_final", "delta_f_max", "rho_peak", "x_nstar"):
                pass
            dF = abs(r["F1_final"] - ref["F1_final"]) / max(abs(ref["F1_final"]), 1e-300)
            dR = abs(r["rho_peak"] - ref["rho_peak"]) / max(abs(ref["rho_peak"]), 1e-300)
            dX = abs(r["x_nstar"] - ref["x_nstar"]) / max(abs(ref["x_nstar"]), 1e-300)
            tag = "  <- reference" if r is ref else ""
            print(f"     dt={r['dt']:<7g} F1={r['F1_final']:.6f} ({100*dF:6.2f}%)   "
                  f"rho={r['rho_peak']:.5f} ({100*dR:5.2f}%)   "
                  f"x_n*={r['x_nstar']:7.2f} ({100*dX:5.2f}%){tag}", flush=True)
        coarse = sub[0]
        worst = max(abs(coarse["F1_final"] - ref["F1_final"]) / max(abs(ref["F1_final"]), 1e-300),
                    abs(coarse["rho_peak"] - ref["rho_peak"]) / max(abs(ref["rho_peak"]), 1e-300))
        second = sub[-2]
        near = abs(second["F1_final"] - ref["F1_final"]) / max(abs(ref["F1_final"]), 1e-300)
        print(f"     -> dt={dts[0]:g} drifts {100*worst:.2f}% from dt={dts[-1]:g}; "
              f"the two finest differ by {100*near:.2f}%  "
              f"({'CONVERGED' if near < 0.01 else 'not converged at 1%'})", flush=True)


if __name__ == "__main__":
    main()
