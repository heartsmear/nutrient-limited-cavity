"""
ancestry.py -- the exact deterministic ancestry kernel for the chemotaxis-free cavity.

Port of `src/grounded_ancestry.py` with the motility channel removed.

THE DECOMPOSITION.  The step ordering is reaction, then transport, so the label obeys

    h^{m+1} = P_m ( h^m + Delta_sel^m )

and unrolling gives the exact discrete Duhamel identity

    h^N  =  G^0 h^0        +   sum_m G^m Delta_sel^m
            \___ B0 ___/       \____ A+ - A- ____/          G^m = P_{N-1} ... P_m

with the sign criterion   sigma_T h = B0 + (A+ - A-),   inverse sorting  <=>  A- > A+ + B0.

Two simplifications relative to src/, both from having no drift:
  * There is no D channel, and no G^{m+1}-vs-G^m index trap -- with one source applied BEFORE
    transport, every increment is carried by the same G^m.  (In src/ the drift was applied AFTER
    P_m and had to be propagated by G^{m+1}; getting that wrong silently broke the identity.)
  * B0 is KEPT.  It is exactly 0 for seed_f1 = 0.5 (h^0 == 0), which is the default, but it is one
    term and carrying it is what makes the criterion honest for any other seed.

COST -- read before choosing dx.  P_m is recovered by eps-probing the realized transport with
H = eps*I: that is O(N^2) storage per step and the composition W = W @ P is O(N^3).  At N=100 it is
seconds; at N=1500 (the dx needed to resolve a jammed run) it is impractical.  **Ancestry and
properly-resolved jammed runs are mutually exclusive.**  `--max-n` refuses rather than hanging.

    py -3.11 src2/ancestry.py --check
    py -3.11 src2/ancestry.py --n0 15 --dx 10 --T 400
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
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.colors import TwoSlopeNorm, LogNorm         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                               # noqa: E402
import closures as C                                        # noqa: E402
import solver as S                                          # noqa: E402
import diagnostics as D                                     # noqa: E402

EPS_PROBE = 1.0e-7
C1, C2, CN = "#1f77b4", "#d62728", "#2ca02c"
MAX_N_DEFAULT = 250


def _h(r1, r2, p):
    rho = r1 + r2
    f = C.local_fraction_exact(r1, np.maximum(rho - r1, 0.0), tol=p.tiny_density)
    return 2.0 * f - 1.0


# =============================================================================
# Forward pass: exactly solver.simulate, recording the trajectory
# =============================================================================
def forward(L, p, num, ic):
    """Mirror of solver.one_step in a loop, storing what the backward pass needs.

    Storing every P_m would be N^2 per step; instead we store the TRAJECTORY (rho_r, rho_new,
    J_total, the selection increment) and REBUILD P_m during the backward sweep.  Same arithmetic,
    far less memory.  `--check` asserts this loop reproduces solver.simulate bit-for-bit.
    """
    N = C.choose_grid_size(L, num)
    x, dx = C.cell_grid(L, N)
    n_steps = int(math.ceil(num.T_end / num.dt))
    dt = num.T_end / n_steps
    C.check_growth_ceiling(p, dt)

    rho0 = C.initial_total_density(x, L, p, ic)
    f0 = C.initial_fraction(x, ic)
    rho1, rho2 = f0 * rho0, (1.0 - f0) * rho0
    n = C.initial_nutrient(x, p, ic)
    h0 = _h(rho1, rho2, p)

    A_n, b_n = C.build_diffusion_matrix_dirichlet_left_neumann_right(
        np.full(N, p.Dn), dx, left_value=p.n0)
    B_n = (diags(np.ones(N), 0, format="csc") - dt * A_n).tocsc()

    RHO_R = np.zeros((n_steps, N)); RHO_NEW = np.zeros((n_steps, N))
    JTOT = np.zeros((n_steps, N + 1)); D_SEL = np.zeros((n_steps, N))
    NPRE = np.zeros((n_steps, N))

    for step in range(n_steps):
        NPRE[step] = n
        h_before = _h(rho1, rho2, p)
        rho_old = np.maximum(rho1 + rho2, 0.0)
        q = C.growth_space_factor(rho_old, p)

        rho1_r, rho2_r, n_new, _ = S.coupled_reaction_nutrient(
            rho1, rho2, n, p, dt, B_n, b_n, q=q)
        n_new = np.maximum(n_new, 0.0)
        rho_r = rho1_r + rho2_r
        D_SEL[step] = _h(rho1_r, rho2_r, p) - h_before          # the selection increment

        rho_new, _Dc, J_total, _dg = S.nonlinear_collective_diffusion(rho_r, p, dt, dx)
        RHO_R[step] = rho_r; RHO_NEW[step] = rho_new; JTOT[step] = J_total

        rho1_adv, rho_adv_tot, _ = C.advect_strain_by_total_flux(
            rho1_r, rho_r, J_total, dt, dx, p, num)
        f_adv = C.local_fraction_exact(rho1_adv, np.maximum(rho_adv_tot - rho1_adv, 0.0),
                                       tol=p.tiny_density)
        rho1_t, _ = C.implicit_self_diffusion_fraction(f_adv * rho_new, rho_new, p, dt, dx)
        rho1_t = np.minimum(np.maximum(rho1_t, 0.0), rho_new)

        rho1, rho2, n = rho1_t, rho_new - rho1_t, n_new

    return dict(x=x, dx=dx, dt=dt, n_steps=n_steps, N=N, h0=h0, L=L, num=num,
                rho1=rho1, rho2=rho2, n=n, h_sim=_h(rho1, rho2, p),
                RHO_R=RHO_R, RHO_NEW=RHO_NEW, JTOT=JTOT, D_SEL=D_SEL, NPRE=NPRE)


# =============================================================================
# The realized neutral transport matrix P_m
# =============================================================================
def transport_matrix(rho_r, rho_new, J_total, p, num, dt, dx, N):
    """P_m by eps-probing the REALIZED transport (upwind advection + implicit label diffusion).

    Probing is done in f, not h: the recursion h^{m+1} = P_m h^{r,m} holds iff P_m 1 = 1, so the
    operator must be recovered from the affine map on fractions.  H = eps*I is pushed through the
    exact same code path the forward step took.
    """
    nsub = C.compute_advection_substeps(rho_r, J_total, dt, dx, p, num)
    hsub = dt / nsub
    H = EPS_PROBE * np.eye(N)
    rho_path = rho_r.copy()
    left_pos = J_total[1:N] >= 0.0
    for _ in range(nsub):
        F = 0.5 * (1.0 + H)
        Fface = np.empty((N + 1, N))
        Fface[0, :] = F[0, :]
        Fface[1:N, :] = np.where(left_pos[:, None], F[:-1, :], F[1:, :])
        Fface[N, :] = F[-1, :]
        M1 = rho_path[:, None] * F
        J1 = J_total[:, None] * Fface
        M1n = M1 - hsub * (J1[1:, :] - J1[:-1, :]) / dx
        rho_n = np.maximum(rho_path - hsub * (J_total[1:] - J_total[:-1]) / dx, 0.0)
        Fn = np.full_like(M1n, 0.5)
        occ = rho_n > p.tiny_density
        if occ.any():
            Fn[occ, :] = M1n[occ, :] / rho_n[occ, None]
        H = 2.0 * np.clip(Fn, 0.0, 1.0) - 1.0
        rho_path = rho_n

    if (not p.use_self_diffusion) or np.all(rho_new <= p.tiny_density):
        P = H / EPS_PROBE
    else:
        rho = np.maximum(rho_new, 0.0)
        B = np.asarray(C.self_diffusivity(rho, p)) * rho
        A = C.build_diffusion_matrix_neumann_neumann(B, dx)
        M = diags(np.maximum(rho, p.tiny_density), 0, format="csc") - dt * A
        P = spsolve(M, rho[:, None] * H) / EPS_PROBE
        if P.ndim == 1:
            P = P.reshape((N, 1))
    P = np.asarray(P, dtype=float)
    P[np.abs(P) < 1.0e-12] = 0.0
    return P


# =============================================================================
# Backward pass
# =============================================================================
def pick_targets(fw, p):
    """(mouth, deep, deep_is_inverse).

    mouth: the n* front cell -- where mouth-type inverse sorting lives (strain 1 at n > n*).
    deep : the most massive DEEP-INVERSE cell (n < n* yet h < 0) -- the graveyard.  Chosen by
           biomass so the kernel is computed where the signal is, not at an arbitrary coordinate.
    """
    nstar = float(C.monod_crossover(p))
    x, n, h = fw["x"], fw["n"], fw["h_sim"]
    rho = fw["rho1"] + fw["rho2"]
    k = np.where(np.diff(np.sign(n - nstar)) != 0)[0]
    mouth = int(k[0]) if k.size else 2
    live = rho > 1e-6 * max(rho.max(), 1e-30)
    deep_inv = (n < nstar) & (h < 0.0) & live
    if deep_inv.any():
        return mouth, int(np.argmax(rho * deep_inv)), True
    deep_side = (n < nstar) & live
    fallback = int(np.argmax(rho * deep_side)) if deep_side.any() else int(0.6 * len(x))
    return mouth, fallback, False


def backward(fw, p, targets):
    """Compose G^m backward, accumulating A+/A-/B0 and each target's ancestry row."""
    N, n_steps, dt, dx, num = fw["N"], fw["n_steps"], fw["dt"], fw["dx"], fw["num"]
    targets = list(targets)
    s_final = np.asarray(C.g1(fw["n"], p)) - np.asarray(C.g2(fw["n"], p))
    sig = np.sign(s_final); sig[sig == 0] = 1.0

    W = np.eye(N)
    h_sel = np.zeros(N)
    A_plus = np.zeros(N); A_minus = np.zeros(N)
    anc = np.zeros((len(targets), n_steps, N))
    Csel = np.zeros((len(targets), n_steps, N))

    for mm in range(n_steps - 1, -1, -1):
        P = transport_matrix(fw["RHO_R"][mm], fw["RHO_NEW"][mm], fw["JTOT"][mm],
                             p, num, dt, dx, N)
        W = W @ P                                   # now W = G^mm = P_{N-1} ... P_mm
        d = fw["D_SEL"][mm]
        h_sel += W @ d
        for k, ti in enumerate(targets):
            anc[k, mm] = W[ti, :]
            Csel[k, mm] = W[ti, :] * d
        absd = np.abs(d); pos = d > 0.0; neg = d < 0.0
        Wp = W[:, pos] @ absd[pos] if pos.any() else np.zeros(N)
        Wn = W[:, neg] @ absd[neg] if neg.any() else np.zeros(N)
        A_plus += np.where(sig > 0, Wp, Wn)
        A_minus += np.where(sig > 0, Wn, Wp)

    B0 = W @ fw["h0"]                               # W = G^0 here
    return dict(sigma=sig, A_plus=A_plus, A_minus=A_minus, B0=sig * B0,
                h_recon=B0 + h_sel, h_sel=h_sel, anc=anc, Csel=Csel, targets=targets)


# =============================================================================
# Figure
# =============================================================================
def _decimate(A, target=1500):
    """Block-average along time before imshow: a (30000, N) array aliases into ~800 px and the
    subsampling beats against slow structure to give stripes that are not in the data."""
    n = A.shape[0]
    if n <= target:
        return A
    blk = int(np.ceil(n / target))
    return A[:(n // blk) * blk].reshape(-1, blk, A.shape[1]).mean(axis=1)


def figures(fw, bw, p, outdir, labels):
    x, dt, n_steps = fw["x"], fw["dt"], fw["n_steps"]
    t = (np.arange(n_steps) + 1) * dt
    nstar = float(C.monod_crossover(p))
    xf = np.array([D.nstar_front(x, fw["NPRE"][mm], nstar) for mm in range(n_steps)])
    targets = bw["targets"]

    fig, ax = plt.subplots(3, 2, figsize=(14, 14))
    tone = (C1, "#8c564b")

    for k, (ti, lab) in enumerate(zip(targets, labels)):
        anc, Csel = bw["anc"][k], bw["Csel"][k]
        wsum = anc.sum(axis=1); wsum[wsum == 0] = np.nan
        ybar = (anc * x[None, :]).sum(axis=1) / wsum
        ysd = np.sqrt(np.maximum((anc * x[None, :] ** 2).sum(axis=1) / wsum - ybar ** 2, 0.0))

        # K spans decades -- a near-delta at t->T and spread earlier.  A linear scale is set by
        # the final spike and renders everything else black.
        Kp = _decimate(np.maximum(anc, 0.0))
        vmax = float(np.percentile(Kp[Kp > 0], 99)) if np.any(Kp > 0) else 1.0
        a = ax[k, 0]
        im = a.imshow(Kp.T, origin="lower", aspect="auto", cmap="magma",
                      norm=LogNorm(vmin=max(vmax * 1e-4, 1e-9), vmax=max(vmax, 1e-8)),
                      extent=[t[0], t[-1], x[0], x[-1]])
        a.fill_between(t, ybar - ysd, ybar + ysd, color="lime", alpha=.18, lw=0)
        a.plot(t, ybar, color="lime", lw=2.0, label=r"mean ancestry $\bar y(t)\pm\sigma$")
        a.plot(t, xf, color="c", lw=1.2, ls="--", label=r"$n^\ast$ front")
        a.axhline(x[ti], color="w", lw=1.0, ls=":")
        occ_y = np.where(Kp.max(axis=0) > vmax * 1e-4)[0]
        if occ_y.size:
            a.set_ylim(0, min(x[-1], x[occ_y[-1]] * 1.25 + 20))
        a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"$y$ [$\mu$m]"); a.legend(fontsize=8)
        a.set_title(rf"({'AC'[k]}) {lab}: $K(y,t\,|\,x={x[ti]:.0f}\,\mu$m$,T)$   "
                    rf"$h$={fw['h_sim'][ti]:+.3f}   "
                    rf"$\sum_y K$={wsum[np.isfinite(wsum)].mean():.6f}",
                    fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=a).set_label(r"$K(y,t\,|\,x,T)$  (log)")

        a = ax[k, 1]
        Cd = _decimate(Csel)
        v = float(np.nanpercentile(np.abs(Cd), 99.5)) or 1.0
        im = a.imshow(np.clip(Cd, -v, v).T, origin="lower", aspect="auto", cmap="coolwarm_r",
                      norm=TwoSlopeNorm(0.0, -v, v), extent=[t[0], t[-1], x[0], x[-1]])
        a.plot(t, xf, color="k", lw=1.0, ls="--")
        a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"$y$ [$\mu$m]")
        a.set_title(rf"({'BD'[k]}) {lab}: contribution $C=K\,\Delta_{{\rm sel}}$ "
                    r"(blue $=$ added strain 1)", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=a)

    a = ax[2, 0]
    for k, lab in enumerate(labels):
        a.plot(t, np.cumsum(bw["Csel"][k].sum(axis=1)), color=tone[k], lw=1.8, label=lab)
    a.axhline(0, color="0.7", lw=.7)
    a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"cumulative contribution to $h$")
    a.legend(fontsize=8)
    a.set_title("(E) when each target's composition was set", fontsize=10, fontweight="bold")
    a.grid(alpha=.25)

    a = ax[2, 1]
    a.plot(x, bw["A_plus"], color=C2, lw=1.6, label=r"$A_+$ (reinforcing)")
    a.plot(x, bw["A_minus"], color=C1, lw=1.6, label=r"$A_-$ (opposing)")
    a.plot(x, bw["B0"], color="0.6", lw=1.2, ls=":", label=r"$B_0$ (inherited)")
    for k, ti in enumerate(bw["targets"]):
        a.axvline(x[ti], color=tone[k], lw=1.2, ls=":")
    a.axhline(0, color="0.7", lw=.7)
    inv = bw["A_minus"] > (bw["A_plus"] + bw["B0"])
    a.fill_between(x, 0, 1, where=inv, transform=a.get_xaxis_transform(),
                   color=C1, alpha=.10, label=r"$A_->A_++B_0$")
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel("exposure")
    a.legend(fontsize=7)
    a.set_title(r"(F) inverse-sorting criterion", fontsize=10, fontweight="bold")
    a.grid(alpha=.25)

    fig.suptitle(rf"Ancestry kernel — $n_0$={p.n0:g} $\mu$M, $n^\ast$={nstar:.3f} $\mu$M, "
                 rf"targets {labels[0]} $x$={x[targets[0]]:.0f}, "
                 rf"{labels[1]} $x$={x[targets[1]]:.0f} $\mu$m",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    path = os.path.join(outdir, "ancestry.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


# =============================================================================
def run(n0, dx, dt, T, outdir=None, target_x=None, max_n=MAX_N_DEFAULT, p=None):
    p = p if p is not None else config.params(n0)
    num = config.numerics(T_end=T, dt=dt, dx=dx, n_save=2)
    ic = config.initial_conditions()
    N = C.choose_grid_size(config.L, num)
    if N > max_n:
        sys.exit(f"N={N} exceeds --max-n={max_n}. The eps-probe is O(N^2) in storage and the "
                 f"composition O(N^3) per step, so ancestry and a properly-resolved jammed run "
                 f"are mutually exclusive. Use a coarser dx and state that it is under-resolved, "
                 f"or raise --max-n deliberately.")

    outdir = outdir or os.path.join(HERE, f"ancestry_n0-{n0:g}_dx{dx:g}_T{T:g}")
    os.makedirs(outdir, exist_ok=True)
    print(f"n0={n0:g} uM  dx={dx:g} um  dt={dt:g}  T={T:g} h  N={N}  n*={config.NSTAR:.4f} uM")

    t0 = time.time()
    fw = forward(config.L, p, num, ic)
    print(f"  forward ({fw['n_steps']} steps) in {time.time()-t0:.1f}s")

    mouth, deep, deep_inv = pick_targets(fw, p)
    if target_x is not None:
        mouth = int(np.argmin(np.abs(fw["x"] - target_x)))
    targets = [mouth, deep]
    dlab = "deep" if deep_inv else "deep (NO inverse cells: normal-sorted)"
    print(f"  targets: mouth x={fw['x'][mouth]:.1f} (h={fw['h_sim'][mouth]:+.4f}), "
          f"{dlab} x={fw['x'][deep]:.1f} (h={fw['h_sim'][deep]:+.4f})")
    if not deep_inv:
        print("    !! no cell has n<n* AND h<0 at T -- the graveyard is gone; row 2 is NOT a "
              "graveyard target")

    t1 = time.time()
    bw = backward(fw, p, targets)
    print(f"  backward + kernel in {time.time()-t1:.1f}s")

    gm = np.abs(fw["h_sim"]) > 1e-9
    err = float(np.max(np.abs(bw["h_recon"] - fw["h_sim"])))
    corr = (float(np.corrcoef(bw["h_recon"][gm], fw["h_sim"][gm])[0, 1]) if gm.sum() > 2
            else float("nan"))
    ident = float(np.max(np.abs(bw["sigma"] * fw["h_sim"]
                                - (bw["B0"] + bw["A_plus"] - bw["A_minus"]))))
    rho = fw["rho1"] + fw["rho2"]
    w = rho / max(rho.sum(), 1e-300)
    print(f"  Duhamel   : max|h_recon - h_sim| = {err:.3e}   corr = {corr:.6f}   "
          f"biomass-weighted = {float((w*np.abs(bw['h_recon']-fw['h_sim'])).sum()):.3e}")
    print(f"  identity  : sigma_T h = B0 + (A+ - A-)   max err = {ident:.3e}")
    rs = bw["anc"].sum(axis=2)
    print(f"  row sums  : sum_y K in [{rs.min():.8f}, {rs.max():.8f}]  (must be 1)")
    if abs(rs.min() - 1) > 1e-4 or abs(rs.max() - 1) > 1e-4:
        print("    !! WARNING: kernel not row-stochastic -- do not trust K or ybar")
    print(f"  max|B0| = {np.max(np.abs(bw['B0'])):.3e}  (exactly 0 for seed_f1=0.5)")
    for ti, lab in zip(targets, ("mouth", "deep")):
        print(f"    {lab:6s} x={fw['x'][ti]:7.1f}: A+={bw['A_plus'][ti]:.4g}  "
              f"A-={bw['A_minus'][ti]:.4g}  B0={bw['B0'][ti]:.3g}  "
              f"inverse={bool(bw['A_minus'][ti] > bw['A_plus'][ti] + bw['B0'][ti])}")

    path = figures(fw, bw, p, outdir, labels=("mouth", "deep" if deep_inv else "deep (normal)"))
    with open(os.path.join(outdir, "exposure.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["x_um", "h_sim", "h_recon", "sigma_T", "A_plus", "A_minus", "B0", "inverse"])
        for i in range(fw["N"]):
            inv = bw["A_minus"][i] > bw["A_plus"][i] + bw["B0"][i]
            w_.writerow([f"{fw['x'][i]:.4g}", f"{fw['h_sim'][i]:.8g}", f"{bw['h_recon'][i]:.8g}",
                         f"{bw['sigma'][i]:.0f}", f"{bw['A_plus'][i]:.8g}",
                         f"{bw['A_minus'][i]:.8g}", f"{bw['B0'][i]:.8g}", int(inv)])
    print(f"  wrote {path} and exposure.csv")
    return dict(err=err, corr=corr, ident=ident)


def run_checks():
    """The forward pass must equal solver.simulate exactly, and the identities must hold."""
    ok = True
    p = config.params(15.0)
    num = config.numerics(T_end=20.0, dt=0.05, dx=20.0, n_save=2)
    ic = config.initial_conditions()

    print("=== forward pass reproduces solver.simulate bit-for-bit ===")
    fw = forward(config.L, p, num, ic)
    ref = S.simulate(config.L, p, num, ic)
    d1 = float(np.max(np.abs(fw["rho1"] - ref["rho1_final"])))
    d2 = float(np.max(np.abs(fw["rho2"] - ref["rho2_final"])))
    dn = float(np.max(np.abs(fw["n"] - ref["n_final"])))
    print(f"  max|d rho1|={d1:.3e}  max|d rho2|={d2:.3e}  max|d n|={dn:.3e}")
    ok &= (d1 == 0.0 and d2 == 0.0 and dn == 0.0)
    print(f"  {'PASS' if d1 == d2 == dn == 0.0 else 'FAIL'}")

    print("\n=== Duhamel identity and the two channels ===")
    print("  NOTE: the raw max-norm is meaningless off the support -- there rho -> 0, f is set by")
    print("  the empty-cell convention f=1/2 and the label transport is undefined, so the residual")
    print("  is O(1) BY CONSTRUCTION.  Gate on the biomass-weighted residual and occupied cells.")
    bw = backward(fw, p, [3, 20])
    e = np.abs(bw["h_recon"] - fw["h_sim"])
    i = np.abs(bw["sigma"] * fw["h_sim"] - (bw["B0"] + bw["A_plus"] - bw["A_minus"]))
    rho = fw["rho1"] + fw["rho2"]
    w = rho / max(rho.sum(), 1e-300)
    occ = rho > 1e-2 * max(rho.max(), 1e-300)
    ew, iw = float((w * e).sum()), float((w * i).sum())
    print(f"  Duhamel  weighted={ew:.3e}  occupied-max={e[occ].max():.3e}  (raw={e.max():.3e})")
    print(f"  identity weighted={iw:.3e}  occupied-max={i[occ].max():.3e}")
    b0 = float(np.max(np.abs(bw["B0"])))
    print(f"  max|B0| = {b0:.3e}   (must be exactly 0 for seed_f1=0.5 -> h^0 == 0)")
    rs = bw["anc"].sum(axis=2)
    print(f"  row sums sum_y K in [{rs.min():.8f}, {rs.max():.8f}]")
    ok &= (ew < 1e-5 and iw < 1e-5 and e[occ].max() < 1e-3 and b0 == 0.0
           and abs(rs.min() - 1) < 1e-4 and abs(rs.max() - 1) < 1e-4)

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n0", type=float, default=15.0)
    ap.add_argument("--dx", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=400.0)
    ap.add_argument("--target-x", dest="target_x", type=float, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-n", dest="max_n", type=int, default=MAX_N_DEFAULT,
                    help=f"refuse above this N (default {MAX_N_DEFAULT}); the probe is O(N^2)/O(N^3)")
    config.add_model_args(ap)
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if run_checks() else 1)
    run(a.n0, a.dx, a.dt, a.T, outdir=a.outdir, target_x=a.target_x, max_n=a.max_n,
        p=config.params_from_args(a, a.n0))


if __name__ == "__main__":
    main()
