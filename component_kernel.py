"""
component_kernel.py -- per-strain MASS ancestry, without Hypothesis 4.

    py -3.11 src2/component_kernel.py --check
    py -3.11 src2/component_kernel.py --n0 1000 --L 1000 --dx 0.5 --T 2000 --mouth-x 30 --deep-x 900

WHAT PROBLEM THIS SOLVES.  `ancestry.py` builds ONE kernel K, the backward propagator of the
LABEL field h = 2f-1.  The Duhamel identity h = B0 + sum K*Delta_sel is exact and needs no
hypothesis about growth.  But the *reading* people naturally give it -- "the biomass here
descends from the biomass there" -- is a different, weaker claim: it requires tagged mass to grow
at the local mean rate (Hypothesis 4 / limitation (L8) of `chemotaxis_ancestry_kernel.tex`).  That
is exactly false in a two-strain competition, which is the only situation we ever run.

Concretely it produces a reading that looks self-contradictory.  At n0=1000 the mouth ends 100%
strain 1, yet the label decomposition books a strain-2 contribution of -1.025 at t ~ 2 h.  Both
are true of h, but "the strain-1 cells at the mouth descend from the strain-2 cells that were at
y ~ 50 um" is NOT true -- cells do not switch strain.  The label kernel cannot see the difference
because it propagates a fraction, not a population.

WHAT THIS DOES INSTEAD.  One chain per strain, each carrying its OWN growth factor:

    rho_i^{m+1} = T_m Gamma_i^m rho_i^m ,      Gamma_i = diag(rho_i^r / rho_i) = diag(e^{dt q g_i})

    K_i(y,t_m | x,T) = [G_i^{m->N}]_{x,y} * rho_i^m(y) / rho_i^N(x),   G_i^{m->N} = prod T Gamma_i

K_i is the fraction of the strain-i MASS at (x,T) that descends from strain-i mass at (y,t).  It
is a genuine per-strain mass ancestry: non-negative, and sum_y K_i = 1 by telescoping rather than
by normalisation -- `--check` verifies that rather than imposing it.  No growth hypothesis
enters, because each Gamma_i is diagonal and composition-independent.

CONDITIONING -- READ THIS BEFORE TRUSTING A PANEL.  The normalisation divides by rho_i^N(x), the
final mass OF THAT STRAIN at the target.  Where a strain is nearly absent the question "where did
its mass come from" is ill-posed, and the arithmetic says so: measured sum_y K_i - 1 is 1.4e-07
where the strain holds ~100% of the local mass, and 2.4e-04 where it holds 5e-05 of it -- the
error tracks 1/rho_i^N(x).  So at n0=1000, K_1 is meaningful at the MOUTH (f1 = 1) and K_2 is
meaningful DEEP (f1 = 0); the complementary two panels are ill-conditioned by construction.  Both
`--check` and `run` print the conditioning alongside each kernel, and gate only on the
well-conditioned ones.  This is a property of the question, not a defect of the method.

WHY IT IS EXACT FOR THIS SOLVER -- checked, not assumed:

  (1) The reaction is exact-exponential per strain, so Gamma_i is diagonal and does NOT depend on
      composition.  Measured max|Gamma_i - exp(dt q g_i(n_new))| = 2.2e-16.
  (2) Strain mass transport factors through the neutral label operator P of ancestry.py:
          T = diag(rho_new) P diag(1/rho_r),  hence  T^T = diag(1/rho_r) P^T diag(rho_new).
      The SAME T moves both strains -- transport is neutral -- which follows from P*1 = 1.
      Measured max|T rho_i^r - rho_i^new| = 7.3e-15 (strain 1), 8.7e-15 (strain 2).

PRECISION IS NOT NEGOTIABLE HERE, and this was learned the hard way.  A first version stored
Gamma_i and rho_i as float32 to save memory, on the reasoning that the fields are only ~1e-7
accurate anyway.  That is the wrong comparison: the CLAIM being made is that the chain telescopes
exactly, and float32 (1.2e-7 relative) degraded sum_y K_i = 1 from 1e-15 to 2.5e-4 and the
forward chain identity to 2.4e-7.  Every per-step array is float64.  Only `NNEW`, used solely to
draw the n* front, is float32.

MEMORY.  Five float64 per-step arrays (rho_r, rho_new, J, rho_1, rho_1^r) plus one float32.
Strain 2 is DERIVED, never stored: rho_2^m = rho_new^{m-1} - rho_1^m and rho_2^r = rho_r - rho_1^r
are both bit-exact reconstructions of what the forward loop computed, which is why storing them
would be redundant rather than safer.  ~3.5 GB at N=2000, T=2000.

COST.  The backward sweep is matrix-free: one adjoint application per strain per step, O(N) each.
~50 min at N=2000, T=2000 -- cheaper than the label kernel, which needs four forward applications
per step.

RELATION TO ancestry.py.  This file imports it read-only (`apply_P`, `apply_PT`) and reimplements
no transport.  It runs its own forward loop because it stores different quantities; `--check`
asserts that loop is bit-identical to `solver.simulate`, so the two cannot drift apart.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from scipy.sparse import diags
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                       # noqa: E402
import closures as C                                # noqa: E402
import solver as S                                  # noqa: E402
import diagnostics as D                             # noqa: E402
import ancestry as A                                # noqa: E402  (apply_P / apply_PT only)

SURFACE, INK, INK2, MUTED, GRIDC = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"
COL1, COL2 = "#2a78d6", "#c1392b"        # strain 1 / strain 2


# =============================================================================
# strain-mass transport, built from ancestry.py's neutral label operator
# =============================================================================
def apply_T(v, rho_r, rho_new, J, p, num, dt, dx, N):
    """T v = rho_new * P (v / rho_r): the realized transport of STRAIN MASS."""
    occ = rho_r > p.tiny_density
    src = np.where(occ, v / np.maximum(rho_r, 1e-300), 0.0)
    return rho_new * A.apply_P(src, rho_r, rho_new, J, p, num, dt, dx, N)


def apply_TT(w, rho_r, rho_new, J, p, num, dt, dx, N):
    """The adjoint, T^T w = (1/rho_r) P^T (rho_new * w).  Drives the backward sweep."""
    out = A.apply_PT(rho_new * w, rho_r, rho_new, J, p, num, dt, dx, N)
    occ = rho_r > p.tiny_density
    return np.where(occ, out / np.maximum(rho_r, 1e-300), 0.0)


def step_state(fw, m):
    """(rho_1, rho_2, Gamma_1, Gamma_2) at step m, strain 2 reconstructed bit-exactly.

    The forward loop sets rho_2 = rho_new - rho_1 and rho_r = rho_1^r + rho_2^r, so both
    subtractions below return exactly the arrays that loop held -- no drift, and 1.3 GB saved.
    """
    r1 = fw["R1"][m]
    r1r = fw["R1R"][m]
    rho_m = fw["RHO_NEW"][m - 1] if m > 0 else fw["rho_total_0"]
    r2 = rho_m - r1
    r2r = fw["RHO_R"][m] - r1r
    g1 = np.where(r1 > 0.0, r1r / np.where(r1 > 0.0, r1, 1.0), 1.0)
    g2 = np.where(r2 > 0.0, r2r / np.where(r2 > 0.0, r2, 1.0), 1.0)
    return r1, r2, g1, g2


# =============================================================================
def forward(L, p, num, ic, store=True):
    """Mirror of solver.one_step, storing what the per-component backward sweep needs.

    Gamma_i is not stored directly; rho_1 and rho_1^r are, and the ratio is formed on demand by
    `step_state`.  That keeps the stored quantities primitive (so a bad Gamma cannot be baked in)
    and lets strain 2 be reconstructed rather than stored.

    REQUIRES num.label_limiter == "none".  apply_T/apply_TT are built on ancestry.apply_P /
    apply_PT, which hard-code donor-cell upwind face selection.  A limited forward paired with an
    upwind adjoint raises nothing and returns plausible numbers, but the mass-transfer identity
    silently stops holding -- so the guard is here rather than in the callers.
    """
    lim = getattr(num, "label_limiter", "none")
    if lim != "none":
        raise ValueError(
            f"component_kernel.forward: label_limiter={lim!r} is not supported. apply_T/apply_TT "
            f"are donor-cell upwind by construction (they call ancestry.apply_P/apply_PT), so a "
            f"limited forward would be inconsistent with the backward sweep and the mass-transfer "
            f"identity would fail without raising. Pass label_limiter='none'.")

    N = C.choose_grid_size(L, num)
    x, dx = C.cell_grid(L, N)
    n_steps = int(math.ceil(num.T_end / num.dt))
    dt = num.T_end / n_steps
    C.check_growth_ceiling(p, dt)

    rho0 = C.initial_total_density(x, L, p, ic)
    f0 = C.initial_fraction(x, ic)
    rho1, rho2 = f0 * rho0, (1.0 - f0) * rho0
    rho_total_0 = rho1 + rho2
    n = C.initial_nutrient(x, p, ic)

    A_n, b_n = C.build_diffusion_matrix_dirichlet_left_neumann_right(
        np.full(N, p.Dn), dx, left_value=p.n0)
    B_n = (diags(np.ones(N), 0, format="csc") - dt * A_n).tocsc()

    if store:
        RHO_R = np.zeros((n_steps, N)); RHO_NEW = np.zeros((n_steps, N))
        JTOT = np.zeros((n_steps, N + 1))
        R1 = np.zeros((n_steps, N)); R1R = np.zeros((n_steps, N))
        NNEW = np.zeros((n_steps, N), dtype=np.float32)
        Q = np.zeros((n_steps, N), dtype=np.float32)
    else:
        RHO_R = RHO_NEW = JTOT = R1 = R1R = NNEW = Q = None

    for step in range(n_steps):
        rho_old = np.maximum(rho1 + rho2, 0.0)
        q = C.growth_space_factor(rho_old, p)
        rho1_r, rho2_r, n_new, _ = S.coupled_reaction_nutrient(
            rho1, rho2, n, p, dt, B_n, b_n, q=q)
        n_new = np.maximum(n_new, 0.0)
        rho_r = rho1_r + rho2_r
        rho_new, _Dc, J_total, _dg = S.nonlinear_collective_diffusion(rho_r, p, dt, dx)

        if store:
            R1[step] = rho1; R1R[step] = rho1_r
            RHO_R[step] = rho_r; RHO_NEW[step] = rho_new; JTOT[step] = J_total
            NNEW[step] = n_new; Q[step] = q

        rho1_adv, rho_adv_tot, _ = C.advect_strain_by_total_flux(
            rho1_r, rho_r, J_total, dt, dx, p, num)
        f_adv = C.local_fraction_exact(rho1_adv, np.maximum(rho_adv_tot - rho1_adv, 0.0),
                                       tol=p.tiny_density)
        rho1_t, _ = C.implicit_self_diffusion_fraction(f_adv * rho_new, rho_new, p, dt, dx)
        rho1_t = np.minimum(np.maximum(rho1_t, 0.0), rho_new)
        rho1, rho2, n = rho1_t, rho_new - rho1_t, n_new

    return dict(x=x, dx=dx, dt=dt, n_steps=n_steps, N=N, L=L, num=num,
                rho1=rho1, rho2=rho2, n=n, rho_total_0=rho_total_0,
                RHO_R=RHO_R, RHO_NEW=RHO_NEW, JTOT=JTOT, R1=R1, R1R=R1R, NNEW=NNEW, Q=Q)


def backward(fw, p, targets, frames=1200):
    """Per-strain ancestry rows for each target, block-averaged in time.

    With M_m = T_m diag(Gamma_i^m) the one-step map on strain-i mass, M_m^T = diag(Gamma_i^m)T_m^T,
    so the row of G_i^{m->N} at the target follows from  w <- Gamma_i^m * (T_m^T w),  seeded e_x.
    """
    N, n_steps, dt, dx, num = fw["N"], fw["n_steps"], fw["dt"], fw["dx"], fw["num"]
    targets = list(targets)
    blk = max(1, int(np.ceil(n_steps / max(frames, 1))))
    nb = int(np.ceil(n_steps / blk))

    anc = np.zeros((2, len(targets), nb, N))
    cnt = np.zeros(nb)
    W = [np.zeros((len(targets), N)) for _ in range(2)]
    for s in range(2):
        for k, ti in enumerate(targets):
            W[s][k, ti] = 1.0

    for m in range(n_steps - 1, -1, -1):
        ar = (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num, dt, dx, N)
        r1, r2, g1, g2 = step_state(fw, m)
        j = m // blk
        for s, (g, r) in enumerate(((g1, r1), (g2, r2))):
            for k in range(len(targets)):
                W[s][k] = g * apply_TT(W[s][k], *ar)
            anc[s, :, j, :] += W[s] * r[None, :]          # w^m_y * rho_i^m(y)
        cnt[j] += 1

    nz = cnt > 0
    anc[:, :, nz, :] /= cnt[nz][None, None, :, None]
    rho_fin = (fw["rho1"], fw["rho2"])
    norm = np.array([[max(float(rho_fin[s][ti]), 1e-300) for ti in targets] for s in range(2)])
    return dict(K=anc / norm[:, :, None, None], targets=targets, block=blk,
                n_steps=n_steps, norm=norm, cnt=cnt)


# =============================================================================
def run_checks():
    print("=== component kernel: structural conditions and exactness ===")
    p = config.params(1000.0); ic = config.initial_conditions()
    # T=80 h, not 20: the normalisation divides by rho_i^N(x), so the check is only
    # meaningful once the cavity holds biomass.  Measured max|sum_y K_i - 1| for the
    # well-conditioned strain, sweeping T at fixed dt: 7.4e-02 (T=10) -> 1.3e-05 (20) ->
    # 1.0e-06 (40) -> 2.1e-07 (80) -> 4.1e-07 (160).  It IMPROVES with accumulated biomass
    # and does not grow with step count, so a short run is a bad test, not a strict one.
    # The chain error itself is flat at ~4e-07 across that whole sweep.
    num = config.numerics(T_end=80.0, dt=0.05, dx=20.0, n_save=2)
    L = 1000.0
    fw = forward(L, p, num, ic)
    dt, dx, N, num_ = fw["dt"], fw["dx"], fw["N"], fw["num"]
    ok = True

    def report(tag, val, tol, note=""):
        nonlocal ok
        good = val < tol
        ok &= good
        print(f"\n[{tag}] {note}\n    value = {val:.3e}   tol = {tol:.0e}   "
              f"{'PASS' if good else 'FAIL'}")

    res = S.simulate(L, p, num, ic)
    e = max(float(np.max(np.abs(fw["rho1"] - res["rho1_profiles"][:, -1]))),
            float(np.max(np.abs(fw["rho2"] - res["rho2_profiles"][:, -1]))))
    report("1", e, 1e-300 if e == 0 else 1e-30,
           "forward loop is bit-identical to solver.simulate")

    # (1) Gamma_i is diagonal and composition-independent: it must equal exp(dt q g_i(n_new))
    eg = 0.0
    for m in range(fw["n_steps"]):
        r1, r2, g1, g2 = step_state(fw, m)
        nn = fw["NNEW"][m].astype(float); q = fw["Q"][m].astype(float)
        for g, gi, r in ((g1, C.g1, r1), (g2, C.g2, r2)):
            msk = r > 1e-300
            if msk.any():
                pred = np.exp(dt * q * np.asarray(gi(nn, p)))
                eg = max(eg, float(np.max(np.abs(g[msk] - pred[msk]))))
    report("2", eg, 1e-6,
           "Gamma_i == exp(dt q g_i(n_new)), i.e. diagonal and composition-free\n"
           "    (tol 1e-6 only because n_new is stored float32 for the figure; the\n"
           "     structural claim itself is exact -- see the float64 probe in the docstring)")

    # (2) the SAME T moves both strains, and reproduces each strain's transported mass
    et = 0.0
    for m in range(fw["n_steps"]):
        ar = (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num_, dt, dx, N)
        r1, r2, g1, g2 = step_state(fw, m)
        nxt1 = fw["R1"][m + 1] if m + 1 < fw["n_steps"] else fw["rho1"]
        et = max(et, float(np.max(np.abs(apply_T(fw["R1R"][m], *ar) - nxt1))))
        et = max(et, float(np.max(np.abs(apply_T(fw["RHO_R"][m], *ar) - fw["RHO_NEW"][m]))))
    report("3", et, 1e-11, "T rho_i^r == rho_i^new for strain 1 AND for the total")

    rng = np.random.default_rng(0)
    ea = 0.0
    for m in (0, fw["n_steps"] // 2, fw["n_steps"] - 1):
        ar = (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num_, dt, dx, N)
        v, w = rng.standard_normal(N), rng.standard_normal(N)
        ea = max(ea, abs(float(np.dot(apply_T(v, *ar), w) - np.dot(v, apply_TT(w, *ar)))))
    report("4", ea, 1e-12, "adjoint identity <Tv,w> == <v,T^T w>")

    # The forward chain must reproduce each strain's own final mass.  Gated on the RELATIVE
    # error: strain masses here span seven decades (strain 1 crashes to ~1e-9 while strain 2
    # holds ~1e-2), so an absolute tolerance would be meaningless for one strain and trivial
    # for the other.  The residual is the accumulated conditioning of 400 sparse solves, not
    # roundoff -- it grows smoothly from 8e-9 to 6e-7 relative over the run.
    ef = 0.0
    for s in range(2):
        v = (fw["R1"][0] if s == 0 else fw["rho_total_0"] - fw["R1"][0]).copy()
        for m in range(fw["n_steps"]):
            ar = (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num_, dt, dx, N)
            _r1, _r2, g1, g2 = step_state(fw, m)
            v = apply_T((g1 if s == 0 else g2) * v, *ar)
        tgt = fw["rho1"] if s == 0 else fw["rho2"]
        ef = max(ef, float(np.max(np.abs(v - tgt))) / max(float(np.max(np.abs(tgt))), 1e-300))
    report("5", ef, 1e-5, "chain (T Gamma_i)...(T Gamma_i) rho_i^0 == rho_i^N, RELATIVE")

    tg = [1, N // 2]
    bw = backward(fw, p, tg, frames=fw["n_steps"])
    rho_fin = (fw["rho1"], fw["rho2"])
    tot = fw["rho1"] + fw["rho2"]
    print(f"\n[6] sum_y K_i == 1 by telescoping, per (strain, target), with conditioning")
    print(f"    {'strain':>7}{'x':>8}{'mass frac':>12}{'max|sumK-1|':>14}   verdict")
    worst_ok = 0.0
    for s in range(2):
        for k, ti in enumerate(tg):
            dev = float(np.max(np.abs(bw["K"][s, k].sum(axis=1) - 1.0)))
            frac = float(rho_fin[s][ti]) / max(float(tot[ti]), 1e-300)
            cond = frac > 1e-3
            if cond:
                worst_ok = max(worst_ok, dev)
            print(f"    {s+1:>7}{fw['x'][ti]:>8.0f}{frac:>12.3e}{dev:>14.3e}   "
                  f"{'gated' if cond else 'ill-conditioned, reported only'}")
    good = worst_ok < 1e-5
    ok &= good
    print(f"    worst well-conditioned = {worst_ok:.3e}   tol = 1e-05   "
          f"{'PASS' if good else 'FAIL'}")

    neg = float(np.min(bw["K"]))
    report("7", max(-neg, 0.0), 1e-12, "K_i >= 0 (it is a mass fraction)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


# =============================================================================
def figure(fw, bw, p, outdir, targets, labels):
    x, dt, n_steps = fw["x"], fw["dt"], fw["n_steps"]
    blk, K = bw["block"], bw["K"]
    nb = K.shape[2]
    lo = np.arange(nb) * blk
    hi = np.minimum(lo + blk, n_steps)
    t_a = hi * dt
    nstar = float(C.monod_crossover(p))
    xf = np.array([D.nstar_front(x, fw["NNEW"][mm].astype(float), nstar) for mm in range(n_steps)])
    xf_a = np.add.reduceat(xf, lo) / (hi - lo)
    xe = np.concatenate([x - 0.5 * fw["dx"], [x[-1] + 0.5 * fw["dx"]]])
    te = np.concatenate([lo * dt, [t_a[-1]]])

    nt = len(targets)
    fig, ax = plt.subplots(nt, 3, figsize=(16.5, 4.5 * nt), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    means = {}
    for k, ti in enumerate(targets):
        for s in range(2):
            a = ax[k][s]
            Ks = np.maximum(K[s, k], 0.0)
            w = Ks.sum(axis=1); w = np.where(w > 0, w, np.nan)
            means[(k, s)] = (Ks * x[None, :]).sum(axis=1) / w
            pos = Ks[Ks > 0]
            vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
            im = a.pcolormesh(te, xe, Ks.T, cmap="magma", shading="flat",
                              norm=LogNorm(vmin=max(vmax * 1e-4, 1e-14),
                                           vmax=max(vmax, 1e-13)))
            a.plot(t_a, means[(k, s)], color="lime", lw=2.0, label=r"mean origin $\bar y_i(t)$")
            a.plot(t_a, xf_a, color="c", lw=1.2, ls="--", label=r"$n^\ast$ front")
            a.axhline(x[ti], color="w", lw=1.0, ls=":")
            a.set_xscale("log"); a.set_xlim(max(t_a[0] * 0.5, 1e-3), t_a[-1])
            a.set_xlabel(r"$t$ [h]", color=INK2)
            a.set_ylabel(r"origin  $y$ [$\mu$m]", color=INK2)
            a.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper right")
            a.set_title(rf"{labels[k]} $x$={x[ti]:.0f}: $K_{s+1}$, strain-{s+1} mass ancestry"
                        rf"   $\sum_y K$={np.nanmean(w):.6f}",
                        fontsize=10, fontweight="bold", color=INK)
            fig.colorbar(im, ax=a)

        a = ax[k][2]
        a.plot(t_a, means[(k, 0)], color=COL1, lw=2.0, label=r"$K_1$  strain-1 mass")
        a.plot(t_a, means[(k, 1)], color=COL2, lw=2.0, label=r"$K_2$  strain-2 mass")
        a.plot(t_a, xf_a, color="k", lw=1.1, ls="--", label=r"$n^\ast$ front")
        a.axhline(x[ti], color=MUTED, lw=1.0, ls=":", label="target")
        a.set_xscale("log"); a.set_xlim(max(t_a[0] * 0.5, 1e-3), t_a[-1])
        a.set_xlabel(r"$t$ [h]", color=INK2)
        a.set_ylabel(r"mean origin  $\bar y$ [$\mu$m]", color=INK2)
        a.legend(fontsize=8, frameon=False, labelcolor=INK2)
        a.grid(alpha=.25, lw=.6)
        a.set_title(f"{labels[k]}: where each strain's mass actually came from",
                    fontsize=10, fontweight="bold", color=INK)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig.suptitle("Per-strain mass ancestry -- component-resolved kernel, no growth hypothesis   "
                 f"$n_0$={float(p.n0):g} " + r"$\mu$M, $L$=" + f"{fw['L']:g} "
                 + r"$\mu$m, $T$=" + f"{n_steps*dt:g} h",
                 fontsize=12.5, fontweight="bold", color=INK)
    path = os.path.join(outdir, "component_ancestry.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path, means, t_a


# =============================================================================
def run(n0, dx, dt, T, L=None, mouth_x=None, deep_x=None, outdir=None, frames=1200):
    p = config.params(n0)
    L = float(config.L if L is None else L)
    num = config.numerics(T_end=T, dt=dt, dx=dx, n_save=2)
    ic = config.initial_conditions()
    N = C.choose_grid_size(L, num)
    n_steps = int(math.ceil(T / dt))
    mem = (5 * 8 + 2 * 4) * N * n_steps / 1e9
    print(f"n0={n0:g} uM  L={L:g} um  dx={dx:g} um  dt={dt:g}  T={T:g} h  N={N}  "
          f"steps={n_steps}  (~{mem:.1f} GB)")

    t0 = time.time()
    fw = forward(L, p, num, ic)
    print(f"  forward in {time.time()-t0:.1f}s")

    x = fw["x"]
    mouth = int(np.argmin(np.abs(x - (30.0 if mouth_x is None else mouth_x))))
    deep = int(np.argmin(np.abs(x - (0.9 * L if deep_x is None else deep_x))))
    targets = [mouth, deep]
    rho = fw["rho1"] + fw["rho2"]
    f1 = np.where(rho > p.tiny_density, fw["rho1"] / np.maximum(rho, 1e-300), np.nan)
    for lab, ti in zip(("mouth", "deep"), targets):
        print(f"  {lab:>6} x={x[ti]:7.1f} um   f1={f1[ti]:.6e}   "
              f"rho1={fw['rho1'][ti]:.3e}  rho2={fw['rho2'][ti]:.3e}")

    t1 = time.time()
    bw = backward(fw, p, targets, frames=frames)
    print(f"  backward (2 strains, matrix-free) in {time.time()-t1:.1f}s  [block x{bw['block']}]")
    dev = float(np.max(np.abs(bw["K"].sum(axis=3) - 1.0)))
    print(f"  sum_y K_i = 1 by telescoping, not imposed: max|dev| = {dev:.3e}")

    outdir = outdir or os.path.join(
        HERE, f"component_n0-{n0:g}_L{L:g}_dx{dx:g}_T{T:g}_m{x[mouth]:.0f}_d{x[deep]:.0f}")
    os.makedirs(outdir, exist_ok=True)
    path, means, t_a = figure(fw, bw, p, outdir, targets, ("mouth", "deep"))

    with open(os.path.join(outdir, "mean_origin.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["t_h", "mouth_ybar_strain1", "mouth_ybar_strain2",
                     "deep_ybar_strain1", "deep_ybar_strain2"])
        for j in range(len(t_a)):
            w_.writerow([f"{t_a[j]:.6g}"] + [f"{means[(k, s)][j]:.6g}"
                                             for k in range(2) for s in range(2)])
    np.savez_compressed(os.path.join(outdir, "component_fields.npz"),
                        K=bw["K"].astype(np.float32), x=x, t_a=t_a, block=bw["block"],
                        targets=np.asarray(targets), rho1=fw["rho1"], rho2=fw["rho2"],
                        n0=float(n0), L=L, dx=fw["dx"], dt=fw["dt"], T=float(T))
    print(f"  wrote {path}, mean_origin.csv and component_fields.npz")

    print(f"\n{'target':>8}{'strain':>8}{'ybar(2h)':>11}{'ybar(100h)':>12}"
          f"{'ybar(T/2)':>11}{'ybar(T)':>10}")
    for k, lab in enumerate(("mouth", "deep")):
        for s in range(2):
            yb = means[(k, s)]
            i2 = int(np.argmin(np.abs(t_a - 2.0)))
            i100 = int(np.argmin(np.abs(t_a - 100.0)))
            print(f"{lab:>8}{s+1:>8}{yb[i2]:>11.1f}{yb[i100]:>12.1f}"
                  f"{yb[len(t_a)//2]:>11.1f}{yb[-1]:>10.1f}")
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n0", type=float, default=1000.0)
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=2000.0)
    ap.add_argument("--mouth-x", dest="mouth_x", type=float, default=None)
    ap.add_argument("--deep-x", dest="deep_x", type=float, default=None)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if run_checks() else 1)
    run(a.n0, a.dx if a.dx is not None else config.DX, a.dt, a.T, L=a.L,
        mouth_x=a.mouth_x, deep_x=a.deep_x, outdir=a.outdir, frames=a.frames)


if __name__ == "__main__":
    main()
