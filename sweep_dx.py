"""
sweep_dx.py -- does the fully jammed cavity converge in dx?

THE CLAIM UNDER TEST.  Without a cell-scale smoothing length the continuum model has no intrinsic
scale, so dx becomes the smallest scale by default and dx SETS the physics: at n0=1000 the same
model gave F1 = 0.49 at dx=10 and 6e-7 at dx=0.5.  Adding the finite cell size (closures.
smoothing_matrix, ell = 1.5 d_cell read off the IBM's 3x3-of-half-diameter density averaging)
gives the PDE a physical smallest scale, and the solution should then converge as dx falls below
ell/3.

So this runs the same jammed cavity at four resolutions in two branches:

    d_cell = 0     point cells, no intrinsic scale   -> expected NOT to converge
    d_cell = 3 um  ell = 4.5 um                      -> expected to converge

The falsifiable part is the second column.  If it does not converge either, the diagnosis was
wrong and the smoothing is not what was missing.

    py -3.11 src2/sweep_dx.py
    py -3.11 src2/sweep_dx.py --n0 50000 --T 300 --dxs 4,2,1,0.5 --dcells 0,3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                   # noqa: E402
import solver                                   # noqa: E402
import closures as C                            # noqa: E402
import diagnostics as D                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n0", type=float, default=5.0e4, help="uM; 5e4 = 50 mM, a rich medium")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--dxs", default="4,2,1,0.5")
    ap.add_argument("--dcells", default="0,3")
    ap.add_argument("--outdir", default=None)
    config.add_model_args(ap)
    a = ap.parse_args()

    dxs = [float(v) for v in a.dxs.split(",")]
    dcells = [float(v) for v in a.dcells.split(",")]
    outdir = a.outdir or os.path.join(HERE, f"sweep_dx_n0-{a.n0:g}_T{a.T:g}")
    os.makedirs(outdir, exist_ok=True)

    print(f"n0={a.n0:g} uM   T={a.T:g} h   dt={a.dt:g}   n*={config.NSTAR:.4f} uM", flush=True)
    print(f"{'d_cell':>7}{'ell':>7}{'dx':>7}{'N':>6}{'rho_peak':>10}{'q_min':>8}"
          f"{'F1_final':>12}{'df_max':>9}{'lam':>8}{'dx/lam':>8}{'x_n*':>8}{'s':>7}", flush=True)

    rows = []
    for dc in dcells:
        for dx in dxs:
            p = replace(config.params_from_args(a, a.n0), d_cell=dc)
            num = config.numerics(T_end=a.T, dt=a.dt, dx=dx, n_save=61)
            t0 = time.time()
            res = solver.simulate(config.L, p, num, config.initial_conditions())
            el = time.time() - t0
            v = D.validate(res, p, verbose=False)
            row = dict(d_cell=dc, ell=dc * p.ell_factor, dx=dx, N=res["N"],
                       rho_peak=v["rho_peak"], q_min=v["q_min"],
                       F1_final=v["F1_final"], delta_f_max=v["delta_f_max"],
                       lambda_front=v["lambda_front"], dx_over_lambda=v["dx_over_lambda"],
                       x_nstar=v["x_nstar"], admissible=v["admissible"],
                       success=res["success"], wall_s=round(el, 1))
            rows.append(row)
            print(f"{dc:>7g}{dc*p.ell_factor:>7.2f}{dx:>7g}{res['N']:>6d}"
                  f"{v['rho_peak']:>10.5f}{v['q_min']:>8.4f}{v['F1_final']:>12.4e}"
                  f"{v['delta_f_max']:>9.4f}{v['lambda_front']:>8.2f}"
                  f"{v['dx_over_lambda']:>8.3f}{v['x_nstar']:>8.2f}{el:>7.0f}", flush=True)

    path = os.path.join(outdir, "sweep_dx.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {path}", flush=True)

    # --- the verdict ---
    # A RELATIVE-spread test alone is wrong here: it can never pass when the limit is zero, and
    # "F1 -> 0 like dx^2" is convergence, not failure.  So test both readings and say which fits:
    #   (a) converged to a finite value  -> small relative change between the two finest grids
    #   (b) converged to zero            -> clean power law F1 ~ C dx^p with p > ~0.5
    print("\n=== convergence in dx ===", flush=True)
    for dc in dcells:
        sub = sorted([r for r in rows if r["d_cell"] == dc], key=lambda r: -r["dx"])
        X = np.array([r["dx"] for r in sub])
        F = np.array([r["F1_final"] for r in sub])
        print(f"  d_cell={dc:g}:  dx={list(X)}", flush=True)
        print(f"            F1_final = {', '.join(f'{v:.4e}' for v in F)}", flush=True)
        rel = abs(F[-1] - F[-2]) / max(abs(F[-2:]).max(), 1e-300)
        pos = F > 0
        if pos.sum() >= 3:
            pw = np.polyfit(np.log(X[pos]), np.log(F[pos]), 1)
            p_exp = float(pw[0])
            pred = np.exp(np.polyval(pw, np.log(X[pos])))
            fit = float(np.max(np.abs(pred / F[pos] - 1.0)))
        else:
            p_exp, fit = np.nan, np.nan
        if rel < 0.05:
            verdict = f"CONVERGED to a finite value ({F[-1]:.4e})"
        elif np.isfinite(p_exp) and p_exp > 0.5 and fit < 0.25:
            verdict = (f"CONVERGED TO ZERO: clean power law F1 ~ dx^{p_exp:.2f} "
                       f"(fit within {100*fit:.0f}%) -- the finite-dx value is a discretisation "
                       f"residue, the limit is 0")
        else:
            verdict = "NOT converged (neither a finite limit nor a clean power law)"
        print(f"            two finest differ by {100*rel:.1f}%;  exponent p={p_exp:.2f}", flush=True)
        print(f"            -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
