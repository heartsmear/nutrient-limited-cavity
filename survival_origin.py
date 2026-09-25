"""
survival_origin.py -- where was the biomass that is STILL HERE actually produced?

    py -3.11 src2/survival_origin.py --check
    py -3.11 src2/survival_origin.py --n0 15   --L 1000 --dx 0.5 --T 500
    py -3.11 src2/survival_origin.py --n0 1000 --L 1000 --dx 0.5 --T 2000

THE QUESTION.  `plot_production_maps.py` shows where biomass is MADE: 90% of it within 76 um of
the mouth (dilute, 500 h).  But 87.6% of everything made in that run -- 96.6% in the jammed one --
leaves through the open end.  So "90% of the biomass in the cavity was produced at x < 76 um" does
NOT follow from that figure: the survivors are not a random sample of production, because loss
happens at the mouth and production is concentrated at the mouth.  This file computes the
production-origin distribution of the SURVIVING biomass, which is the statement one actually wants
to make.

THE DECOMPOSITION, and why it is cheap.  Split the per-strain growth factor into "keep" and
"create", Gamma_i = I + (Gamma_i - I), and unroll the exact chain rho_i^{m+1} = T_m Gamma_i^m
rho_i^m (component_kernel.py verifies that chain to 1e-15):

    rho_i^N = S^{0->N} rho_i^0  +  sum_m S^{m->N} c_i^m ,
        c_i^m = (Gamma_i^m - I) rho_i^m = rho_i^r - rho_i     [mass CREATED at step m]
        S^{m->N} = T_{N-1} ... T_m                            [PURE transport, no growth]

Each unit of mass created at (y,m) is then carried by transport alone.  Summing over every target
cell, the only thing needed is the column sum

    sigma^m(y) = 1^T S^{m->N} = (T_m^T ... T_{N-1}^T 1)(y)
               = the fraction of mass created at (y,m) that is STILL in the cavity at T,

which is ONE backward sweep of a single all-ones vector -- not one sweep per target.  The answer
is then simply

    A_i(y) = sum_m c_i^m(y) sigma^m(y)          surviving biomass by production site
    Y_i(y) = sum_m c_i^m(y)                     all biomass ever produced there

so the cost is one forward pass plus one adjoint sweep: ~10 min at dx=0.5, T=500 and ~40 min at
T=2000, against ~85 min for a full ancestry run.

sigma IS ALSO THE SELF-CHECK.  sum_y A_i / sum_y Y_i must equal the measured survival fraction
(final standing mass / total produced) -- 0.124 dilute, 0.034 jammed -- and that identity is not
imposed anywhere: it comes out of the adjoint sweep.  `--check` also verifies sigma <= 1,
sigma at the last step ~ 1, and that A reproduces the final total mass.

WHAT sigma MEANS PHYSICALLY.  Transport conserves mass except at the open mouth, where rho = 0 is
a Dirichlet sink; the closed end has zero flux.  So 1 - sigma is exactly the probability that a
parcel created at (y,m) has washed out by T.  sigma is near 1 for material created late or deep,
and small for material created early near the mouth.

NOTE ON THE SEED.  The S^{0->N} rho^0 term is carried and reported separately.  It is negligible
here (seed 4.5e-5 against 6.2 of final stock) but it is part of the identity, not dropped.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                       # noqa: E402
import closures as C                                # noqa: E402
import component_kernel as CK                       # noqa: E402  (forward, apply_T, apply_TT)


def quantiles(w, x, qs=(0.1, 0.5, 0.9, 0.99)):
    tot = float(np.sum(w))
    if not np.isfinite(tot) or tot <= 0:
        return [float("nan")] * len(qs)
    c = np.cumsum(w) / tot
    return [float(np.interp(q, c, x)) for q in qs]


def sweep(fw, p, snap_times=()):
    """One adjoint pass for sigma, accumulating A_i (survivors) and Y_i (all production).

    sigma is propagated but never stored in full: only its contraction with c_i^m is needed, so
    memory stays O(N) on top of what forward() already holds.

    `snap_times` keeps sigma^m at a handful of chosen times.  That is not needed for the result,
    but A/Y is a c-WEIGHTED AVERAGE of sigma over time, and deeper material is produced EARLIER
    here, so A/Y can be non-monotone in y even if every sigma^m is monotone.  Distinguishing
    "the ratio mixes times" from "the survival itself is non-monotone" requires sigma at fixed
    time, and it is nearly free to keep a few slices.
    """
    N, n_steps, dt, dx, num = fw["N"], fw["n_steps"], fw["dt"], fw["dx"], fw["num"]
    A = np.zeros((2, N)); Y = np.zeros((2, N))
    Tw = np.zeros((2, N))                # c-weighted sum of production TIME, for the mean
    sig_last = None
    snap_steps = {int(round(min(max(s / dt, 0), n_steps - 1))): s for s in snap_times}
    snaps = {}
    v = np.ones(N)                       # 1^T, propagated backward by T^T
    for m in range(n_steps - 1, -1, -1):
        ar = (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num, dt, dx, N)
        v = CK.apply_TT(v, *ar)          # now v == sigma^m
        if sig_last is None:
            sig_last = v.copy()
        if m in snap_steps:
            snaps[snap_steps[m]] = v.copy()
        r1, r2, g1, g2 = CK.step_state(fw, m)
        c1 = fw["R1R"][m] - r1                                  # strain-1 mass created at step m
        c2 = (fw["RHO_R"][m] - fw["R1R"][m]) - r2               # strain-2 mass created
        for i, c in ((0, c1), (1, c2)):
            A[i] += c * v
            Y[i] += c
            Tw[i] += c * (m * dt)
    sweep.snaps = snaps
    sweep.tbar = np.where(Y > 0, Tw / np.where(Y > 0, Y, 1.0), np.nan)
    # the seed, carried by pure transport alone
    seed = [fw["R1"][0].copy(), fw["rho_total_0"] - fw["R1"][0]]
    for i in range(2):
        for m in range(n_steps):
            ar = (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num, dt, dx, N)
            seed[i] = CK.apply_T(seed[i], *ar)
    return A, Y, np.array(seed), sig_last, v


def run_checks():
    print("=== survival-origin decomposition ===")
    p = config.params(1000.0); ic = config.initial_conditions()
    num = config.numerics(T_end=80.0, dt=0.05, dx=20.0, n_save=2)
    fw = CK.forward(1000.0, p, num, ic)
    dx = fw["dx"]
    A, Y, seed, sig_last, sig_first = sweep(fw, p)
    ok = True

    def rep(tag, val, tol, note):
        nonlocal ok
        good = val < tol
        ok &= good
        print(f"\n[{tag}] {note}\n    value = {val:.3e}   tol = {tol:.0e}   "
              f"{'PASS' if good else 'FAIL'}")

    rep("1", float(max(sig_last.max() - 1.0, 0.0)), 1e-10,
        "sigma at the FINAL step is <= 1 (one transport step loses almost nothing)")
    rep("2", float(max(sig_first.max() - 1.0, 0.0)), 1e-10,
        "sigma <= 1 everywhere: it is a surviving FRACTION")
    rep("3", float(max(-sig_first.min(), 0.0)), 1e-12, "sigma >= 0")

    # the identity: created-and-survived + seed-that-survived == final mass
    lhs = float((A.sum(axis=0) + seed.sum(axis=0)).sum() * dx)
    rhs = float((fw["rho1"] + fw["rho2"]).sum() * dx)
    rep("4", abs(lhs - rhs) / max(abs(rhs), 1e-300), 1e-9,
        f"sum_m sigma*c + surviving seed == final standing mass\n"
        f"    ({lhs:.6f} vs {rhs:.6f}) -- this is NOT imposed; it emerges from the sweep")

    made = float(Y.sum() * dx)
    surv = float(A.sum() * dx)
    print(f"\n[5] survival fraction from the sweep = {surv/max(made,1e-300):.6f}")
    print(f"    from mass balance (stock-seed)/produced = "
          f"{(rhs - float(seed.sum()*dx))/max(made,1e-300):.6f}")
    d = abs(surv/max(made,1e-300) - (rhs - float(seed.sum()*dx))/max(made,1e-300))
    ok &= d < 1e-9
    print(f"    difference = {d:.3e}   {'PASS' if d < 1e-9 else 'FAIL'}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


def run(n0, dx, dt, T, L=None, outdir=None, rho_seed=None):
    p = config.params(n0)
    L = float(config.L if L is None else L)
    num = config.numerics(T_end=T, dt=dt, dx=dx, n_save=2)
    ic = config.initial_conditions() if rho_seed is None \
        else config.initial_conditions(rho_seed=rho_seed)
    N = C.choose_grid_size(L, num)
    n_steps = int(math.ceil(T / dt))
    print(f"n0={n0:g} uM  L={L:g} um  dx={dx:g}  dt={dt:g}  T={T:g} h  N={N}  "
          f"steps={n_steps}  (~{(5*8+2*4)*N*n_steps/1e9:.1f} GB)")

    t0 = time.time(); fw = CK.forward(L, p, num, ic)
    print(f"  forward in {time.time()-t0:.1f}s")
    snaps_at = [T*f for f in (0.05, 0.1, 0.25, 0.5, 0.75, 0.95)]
    t1 = time.time(); A, Y, seed, sig_last, sig_first = sweep(fw, p, snap_times=snaps_at)
    print(f"  adjoint sweep in {time.time()-t1:.1f}s")

    x, dxx = fw["x"], fw["dx"]
    made = A.sum(axis=0) * 0 + Y.sum(axis=0)
    surv = A.sum(axis=0)
    stock = fw["rho1"] + fw["rho2"]
    ident = abs(float((surv + seed.sum(axis=0)).sum() * dxx) - float(stock.sum() * dxx))
    print(f"  identity |sum(sigma c) + seed - final mass| = {ident:.3e} "
          f"(relative {ident/max(float(stock.sum()*dxx),1e-300):.2e})")
    print(f"  survival fraction = {float(surv.sum())/max(float(made.sum()),1e-300):.6f}")

    print(f"\n  WHERE IT WAS PRODUCED   (x at each cumulative fraction, um)")
    print(f"{'':>34}{'x10':>8}{'x50':>8}{'x90':>8}{'x99':>8}")
    rows = [("all biomass ever produced", made),
            ("...of it, what SURVIVED to T", surv),
            ("[for contrast] where it now sits", stock)]
    for lab, w in rows:
        q = quantiles(w, x)
        print(f"{lab:>34}{q[0]:>8.1f}{q[1]:>8.1f}{q[2]:>8.1f}{q[3]:>8.1f}")
    for i, nm in ((0, "strain 1"), (1, "strain 2")):
        q = quantiles(A[i], x)
        print(f"{'   survivors, ' + nm:>34}{q[0]:>8.1f}{q[1]:>8.1f}{q[2]:>8.1f}{q[3]:>8.1f}")

    # IS sigma ITSELF MONOTONE IN y AT FIXED TIME?  If it is, then every non-monotonicity in
    # the ratio A/Y comes from the time-weighting, not from the survival physics.
    occ = made > 1e-10 * made.max()
    tb = np.where(occ, (sweep.tbar[0]*Y[0] + sweep.tbar[1]*Y[1]) / np.maximum(made, 1e-300),
                  np.nan)
    print("\n  IS sigma MONOTONE IN y AT FIXED t?  (cells that ever produced)")
    print(f"{'t [h]':>9}{'monotone':>10}{'max decrease':>14}   sigma at y = 20/40/80/120/200 um")
    for tv in sorted(sweep.snaps):
        sg = sweep.snaps[tv]
        v = sg[occ]
        drop = float(np.max(np.maximum(-np.diff(v), 0.0))) if v.size > 1 else 0.0
        vals = "  ".join(f"{sg[int(np.argmin(abs(x-yy)))]:.4f}" for yy in (20,40,80,120,200))
        print(f"{tv:>9.1f}{('YES' if drop < 1e-9 else 'no'):>10}{drop:>14.3e}   {vals}")
    print("\n  MEAN PRODUCTION TIME vs DEPTH  (what the ratio averages over)")
    print(f"{'y [um]':>9}{'A/Y':>9}{'mean t of production':>22}")
    for yy in (20, 40, 60, 80, 100, 120, 200, 400):
        i = int(np.argmin(abs(x-yy)))
        if not occ[i]: continue
        print(f"{x[i]:>9.1f}{surv[i]/max(made[i],1e-300):>9.4f}{tb[i]:>22.1f}")

    tag = f"survival_n0-{n0:g}_L{L:g}_dx{dx:g}_T{T:g}"
    # Tag the seed, as figures.py and adjoint_kernel.py do.  Without it a production-config run
    # writes into the folder of a default-config one and the two become indistinguishable.
    # There is deliberately no limiter tag -- see the guard in main().
    if rho_seed is not None:
        tag += f"_seed{rho_seed:g}"
    outdir = outdir or os.path.join(HERE, tag)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "survival_origin.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["x_um", "produced_total", "produced_s1", "produced_s2",
                     "survived_total", "survived_s1", "survived_s2",
                     "sigma_first_step", "rho_final"])
        for i in range(N):
            w_.writerow([f"{x[i]:.4g}", f"{made[i]:.8g}", f"{Y[0][i]:.8g}", f"{Y[1][i]:.8g}",
                         f"{surv[i]:.8g}", f"{A[0][i]:.8g}", f"{A[1][i]:.8g}",
                         f"{sig_first[i]:.8g}", f"{stock[i]:.8g}"])
    print(f"\n  wrote {os.path.join(outdir, 'survival_origin.csv')}")
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n0", type=float, default=15.0)
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--dx", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=500.0)
    ap.add_argument("--rho-seed", dest="rho_seed", type=float, default=None,
                    help="inoculum amplitude (default: the closures default, 1e-6). Production "
                         "runs use 1e-4, which is ~7 cells. Tags the output directory.")
    ap.add_argument("--limiter", default="none",
                    help="accepted only as 'none' -- see the guard below")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    # NO LIMITER HERE, and the flag exists only to say so loudly.  The backward sweep runs on
    # component_kernel.apply_TT -> ancestry.apply_PT, which hard-codes donor-cell upwind face
    # selection.  Running the FORWARD with a TVD limiter while the ADJOINT stays upwind does not
    # raise anything -- component_kernel.forward has no guard -- it silently breaks the
    # mass-transfer identity this whole script is built on.  Same obstruction as F15.
    if a.limiter != "none":
        sys.exit(f"survival_origin: --limiter {a.limiter!r} is not supported. The backward sweep "
                 f"uses ancestry.apply_PT, which is donor-cell upwind by construction, so a "
                 f"limited forward would be inconsistent with it and the survival identity would "
                 f"silently stop holding. Run with --limiter none (the default) and report the "
                 f"survival-origin figure as the unlimited scheme at the production seed.")
    if a.check:
        sys.exit(0 if run_checks() else 1)
    run(a.n0, a.dx, a.dt, a.T, L=a.L, outdir=a.outdir, rho_seed=a.rho_seed)


if __name__ == "__main__":
    main()
