"""Second-order, NON-NEGATIVE ancestry kernel by TVD solution of the adjoint equation.

WHY THIS EXISTS
---------------
ancestry.py recovers P_m by eps-probing, which forces the label transport to be LINEAR, which
caps it at first-order upwind (Godunov), which is why the jammed trajectory fails its own
convergence gate (purity 2.6x, biomass-weighted |dh| 2.2%).  The limiter fixes that but is
nonlinear, so the probe silently returns the upwind matrix instead.

Duhamel never needs an operator -- it needs K.  Solving for K DIRECTLY as the adjoint field lifts
the linearity requirement entirely: the scheme transporting K may be nonlinear, and a TVD scheme
is nonlinear, second-order and positivity-preserving at once.  Godunov never applied; it was
imported by choosing to probe.

THE ADJOINT (derived, and confirmed to 1e-16 against apply_PT by adjoint_kernel_check.py)
-----------------------------------------------------------------------------------------
With  v = w / rho  and  B = D_self * rho:

    d_tau w = d_x(J v) + d_x(B d_x v)

BOTH terms act on v -- which is why apply_PT opens with  g = u/rho_n.  Two details are load
bearing, each verified against a control that fails without it:

  * the 1/rho INSIDE the diffusion term (control without it: 21% wrong);
  * GHOST VALUES OF ZERO outside the domain -- the exit state, i.e. ancestry lost through the
    mouth (control without it: 33% wrong, and conservation collapses).

The advective half is conservative in flux form.  Substituting rho- = rho+ + c*(J_{j+1} - J_j)
into the scatter form collapses it to

    out_j = rho+_j v_j + c ( F_{j+1} - F_j ),     F_e = J_e * vhat_e

with vhat_e the DOWNWIND-in-J cell -- upwind for the backward transport.  Conservation is then a
duality: sum(out) = sum(w) holds precisely because the forward preserves constants (row sums of
T equal rho+).  This is the form a limiter can act on, and the limiter goes on **v**, not on w.

Imports the stack read-only; defines no physics.

    py -3.11 src2/adjoint_kernel.py --check
"""
import argparse
import csv
import os
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import ancestry as A
import closures as C
import config

HERE = os.path.dirname(os.path.abspath(__file__))
OCC_REL = 1.0e-6            # relative-occupancy floor for every error metric (see check())


# =============================================================================
# Face reconstruction of v.  limiter="none" reproduces apply_PT bit-for-bit.
# =============================================================================
def _vhat(v, J, N, kind):
    """Face values of v for the flux F = J*vhat.

    For J_e >= 0 the backward transport carries information leftward, so the donor is the RIGHT
    cell e and the next cell upstream is e+1; for J_e < 0 the donor is e-1 and upstream is e-2.
    Outside the domain v is zero (exit state).
    """
    pos = J >= 0.0
    e = np.arange(N + 1)
    D = np.where(pos, e, e - 1)               # donor cell index
    U = np.where(pos, e + 1, e - 2)           # one further upstream
    Acc = np.where(pos, e - 1, e)             # acceptor (downstream of the donor)

    def get(idx):
        """v at an arbitrary cell index, zero outside the domain."""
        inside = (idx >= 0) & (idx <= N - 1)
        return np.where(inside, v[np.clip(idx, 0, N - 1)], 0.0)

    vD = get(D)
    if kind == "none":
        return vD
    a = vD - get(U)                            # upstream difference
    b = get(Acc) - vD                          # downstream difference
    ok = (D >= 0) & (D <= N - 1) & (U >= 0) & (U <= N - 1) & (Acc >= 0) & (Acc <= N - 1)
    corr = np.zeros(N + 1)
    corr[ok] = 0.5 * C._limited_slope(a[ok], b[ok], kind)
    return vD + corr


def _zalesak_theta(wL, Aflux, c, N):
    """Face-wise FCT factors theta_e in [0,1] enforcing  wL + c*div(theta*A) >= 0.

    A SLOPE limiter cannot do this job here: the scheme conserves w but reconstructs v = w/rho,
    so TVD bounds v while positivity is needed for w, and rho spans orders of magnitude across
    the front.  FCT limits the ANTIDIFFUSIVE FLUX against bounds taken from w itself, so
    positivity holds by construction rather than by inheritance.

    theta is a FACE field, so theta*A is still a flux and the divergence still telescopes --
    conservation is untouched.  Only the lower bound is enforced; that is what K >= 0 needs.
    """
    # face e adds +c*A_e to cell e-1 and -c*A_e to cell e
    cL = c * Aflux[1:N]        # -> cells 0..N-2
    cR = -c * Aflux[1:N]       # -> cells 1..N-1
    Pm = np.zeros(N)
    np.add.at(Pm, np.arange(0, N - 1), np.maximum(-cL, 0.0))
    np.add.at(Pm, np.arange(1, N), np.maximum(-cR, 0.0))
    Qm = np.maximum(wL, 0.0)
    Rm = np.where(Pm > 1e-300, np.minimum(1.0, Qm / np.maximum(Pm, 1e-300)), 1.0)
    th = np.ones(N + 1)
    th[1:N] = np.minimum(np.where(cL < 0.0, Rm[:N - 1], 1.0),
                         np.where(cR < 0.0, Rm[1:N], 1.0))
    return np.clip(th, 0.0, 1.0)


def adjoint_advect(w, rho_r, J_total, p, num, dt, dx, N, kind="none", fct=True, stats=None):
    """Adjoint advection substeps, conservative flux form.  kind='none' == apply_PT.

    With fct=True the high-order flux is FCT-limited so w stays non-negative.  `stats` collects
    the retained antidiffusive fraction -- if that runs small the scheme is first order wearing
    a costume, so it must be reported, not assumed.
    """
    hsub, _, states = A._adv_states(rho_r, J_total, dt, dx, p, num, N)
    c = hsub / dx
    u = np.asarray(w, dtype=float).copy()
    for _rho_m, rho_p in reversed(states):
        v = np.where(rho_p > p.tiny_density, u / np.maximum(rho_p, 1e-300), 0.0)
        base = rho_p * v
        if kind == "none":
            F = J_total * _vhat(v, J_total, N, "none")
            u = base + c * (F[1:] - F[:-1])
            continue
        FL = J_total * _vhat(v, J_total, N, "none")
        FH = J_total * _vhat(v, J_total, N, kind)
        Aflux = FH - FL
        if fct:
            wL = base + c * (FL[1:] - FL[:-1])
            th = _zalesak_theta(wL, Aflux, c, N)
            if stats is not None:
                tot = float(np.abs(Aflux).sum())
                stats.append(1.0 if tot <= 1e-300
                             else float((th * np.abs(Aflux)).sum()) / tot)
            Aflux = th * Aflux
        F = FL + Aflux
        u = base + c * (F[1:] - F[:-1])
    return u


def adjoint_diffuse(w, rho_new, p, dt, dx):
    """Adjoint of the implicit self-diffusion:  z - dt*A(z/rho) = w.  Exact, linear."""
    rho = np.maximum(rho_new, 0.0)
    B = np.asarray(C.self_diffusivity(rho, p)) * rho
    Ad = C.build_diffusion_matrix_neumann_neumann(B, dx)
    M = diags(np.maximum(rho, p.tiny_density), 0, format="csc") - dt * Ad
    return rho * spsolve(M.T.tocsc(), w)


def adjoint_step(w, rho_r, rho_new, J_total, p, num, dt, dx, N, kind="none",
                 fct=True, stats=None):
    """One backward step of K.  P = D A, so P^T = A^T D^T -- diffusion first, then advection."""
    u = np.asarray(w, dtype=float).copy()
    if p.use_self_diffusion and not np.all(rho_new <= p.tiny_density):
        u = adjoint_diffuse(u, rho_new, p, dt, dx)
    return adjoint_advect(u, rho_r, J_total, p, num, dt, dx, N, kind, fct, stats)


# =============================================================================
def check(kinds=("none", "minmod", "vanleer", "superbee")):
    """Regression (kind='none' == apply_PT), non-negativity, and conservation."""
    ok = True
    p = config.params(15.0)
    num = config.numerics(T_end=6.0, dt=0.05, dx=10.0, n_save=2)
    ic = config.initial_conditions()
    fw = A.forward(1000.0, p, num, ic, store=True)
    N, dt, dx = fw["N"], fw["dt"], fw["dx"]
    steps = [m for m in range(0, fw["n_steps"], max(1, fw["n_steps"] // 12))]
    rng = np.random.default_rng(0)

    print("=== R1  kind='none' reproduces ancestry.apply_PT (N=%d) ===" % N)
    worst = 0.0
    for m in steps:
        w = rng.uniform(0.0, 1.0, N)
        ref = A.apply_PT(w, fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m],
                         p, num, dt, dx, N)
        got = adjoint_step(w, fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m],
                           p, num, dt, dx, N, "none")
        rho = np.maximum(fw["RHO_R"][m], 0.0)
        occ = rho > OCC_REL * max(float(rho.max()), 1e-30)
        if occ.any():
            s = max(float(np.abs(ref[occ]).max()), 1e-30)
            worst = max(worst, float(np.abs(got - ref)[occ].max()) / s)
    r1 = worst < 1e-12
    ok = ok and r1
    print("    max relative difference = %.3e   %s" % (worst, "PASS" if r1 else "FAIL"))

    print()
    print("=== R2  K >= 0 and sum(K) conserved, per kind ===")
    print("    %-10s %5s %14s %16s %12s" %
          ("limiter", "FCT", "min(K)", "max|sum-1| int.", "antidiff kept"))
    for kind in kinds:
        for fct in ((False, True) if kind != "none" else (False,)):
            kmin, cons, stats = 0.0, 0.0, []
            for m in steps:
                P = np.empty((N, N))
                e = np.zeros(N)
                for j in range(N):
                    e[:] = 0.0
                    e[j] = 1.0
                    P[:, j] = adjoint_step(e, fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m],
                                           p, num, dt, dx, N, kind, fct, stats)
                kmin = min(kmin, float(P.min()))
                # columns are K rows; conservation is on the column sums, off the boundary
                cs = P.sum(axis=0)
                rho = np.maximum(fw["RHO_NEW"][m], 0.0)
                occ = rho > 1e-3 * max(float(rho.max()), 1e-30)
                occ[:3] = False                  # the mouth legitimately loses ancestry
                if occ.any():
                    cons = max(cons, float(np.max(np.abs(cs[occ] - 1.0))))
            good = (kmin >= -1e-14) and (cons < 1e-8)
            ok = ok and (good or not fct)        # only the FCT rows are required to pass
            kept = ("%11.1f%%" % (100.0 * np.mean(stats))) if stats else "          -"
            print("    %-10s %5s %14.3e %16.3e %12s   %s"
                  % (kind, "on" if fct else "off", kmin, cons, kept,
                     "OK" if good else "FAIL"))

    print()
    print("    A slope limiter cannot protect K: the scheme conserves w but reconstructs")
    print("    v = w/rho, so TVD bounds v while positivity is needed for w, and rho spans")
    print("    orders of magnitude across the front.  FCT limits the antidiffusive FLUX")
    print("    against bounds taken from w itself -- positivity by construction.")
    print("    'antidiff kept' is the fraction of second-order correction that survives; if it")
    print("    runs small the scheme is first order in disguise and buys nothing over upwind.")
    return ok


def sweep(fw, p, num, target, kind="none", fct=True, frames=241):
    """Backward sweep for one target: Duhamel prediction, exposure split, ancestral mean.

    Row x of G^{m->N} obeys  w <- P_m^T w  from w = e_x at m = N, and the Duhamel term for step m
    is <w^m, Delta^m>.  So the kernel and the accumulation are the SAME sweep -- K is never
    stored in full.

        h(x,T) = <G^{0->N} row, h^0>  +  sum_m <w^m, Delta^m>
    """
    n_steps, N, dt, dx = fw["n_steps"], fw["N"], fw["dt"], fw["dx"]
    x = fw["x"]
    s_fin = np.asarray(C.g1(fw["n"], p)) - np.asarray(C.g2(fw["n"], p))
    sig = 1.0 if s_fin[target] >= 0 else -1.0

    blk = max(1, int(np.ceil(n_steps / max(frames, 1))))
    nb = int(np.ceil(n_steps / blk))
    anc = np.zeros((nb, N))
    cnt = np.zeros(nb)

    # Weak-selection source, draft2.tex "Weak selection approximation": the exact per-step
    # increment D_SEL carries the realized (1-h^2) factor, and setting 1-h^2 -> 1 leaves
    # 0.5*dt*s(n).  Accumulated through the SAME adjoint weights, so the two predictions differ
    # only in the source and the gap is the cost of the |h|<<1 assumption -- which this run
    # violates outright (|h| -> 1 at the bottleneck).  NPRE is the per-step pre-reaction
    # nutrient; absent only when an old cache without it is reused.
    NP = fw.get("NPRE")
    w = np.zeros(N)
    w[target] = 1.0
    acc = Ap = Am = acc_weak = 0.0
    ybar = np.full(n_steps, np.nan)
    mass = np.empty(n_steps)
    kmin = 0.0
    stats = []
    for m in range(n_steps - 1, -1, -1):
        w = adjoint_step(w, fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m],
                         p, num, dt, dx, N, kind, fct, stats)
        kmin = min(kmin, float(w.min()))
        d = fw["D_SEL"][m]
        acc += float(np.dot(w, d))
        if NP is not None:
            nm = np.asarray(NP[m])
            acc_weak += 0.5 * dt * float(np.dot(w, np.asarray(C.g1(nm, p))
                                                - np.asarray(C.g2(nm, p))))
        sd = sig * d
        Ap += float(np.dot(w, np.where(sd > 0.0, np.abs(d), 0.0)))
        Am += float(np.dot(w, np.where(sd < 0.0, np.abs(d), 0.0)))
        tot = float(w.sum())
        mass[m] = tot
        if tot > 1e-300:
            ybar[m] = float(np.dot(w, x)) / tot
        b = m // blk
        anc[b] += w
        cnt[b] += 1.0
    anc /= np.maximum(cnt, 1.0)[:, None]
    B0 = sig * float(np.dot(w, fw["h0"]))
    inherited = float(np.dot(w, fw["h0"]))
    return dict(h_pred=acc + inherited, A_plus=Ap, A_minus=Am, B0=B0,
                h_weak=(acc_weak + inherited if NP is not None else float("nan")),
                sigma=sig, ybar=ybar, mass=mass, min_K=kmin, anc=anc, blk=blk,
                t_blk=(np.arange(nb) + 0.5) * blk * dt,
                kept=(float(np.mean(stats)) if stats else 1.0))


#: keys of the forward trajectory that the sweeps and profiles actually consume.  NPRE is in
#: fw but unused here, so it is not cached -- that alone saves 640 MB at N=2000, 40k steps.
_CACHE_KEYS = ("x", "h0", "h_sim", "n", "rho1", "rho2",
               "RHO_R", "RHO_NEW", "JTOT", "D_SEL")
_CACHE_SCALARS = ("n_steps", "N", "dt", "dx", "L")


def load_or_run_forward(path, L, p, num, ic):
    """Forward trajectory, cached to disk.

    fields.npz cannot serve here: it stores K block-averaged to 241 frames, which is an OUTPUT.
    The profile passes need RHO_R/RHO_NEW/JTOT/D_SEL at every one of the 40k steps (~2.6 GB in
    float64).  Kept UNCOMPRESSED and float64 on purpose -- compression would cost minutes on a
    file this size, and float32 would perturb the coefficients at ~1e-7 and break the bit-exact
    kind='none' regression against apply_PT.
    """
    if path and os.path.exists(path):
        print("  loading cached forward from %s (%.1f GB)"
              % (path, os.path.getsize(path) / 1e9))
        d = np.load(path, mmap_mode="r")
        fw = {k: d[k] for k in _CACHE_KEYS}
        for k in _CACHE_SCALARS:
            fw[k] = d[k].item()
        fw["num"] = num
        return fw
    fw = A.forward(L, p, num, ic, store=True, allow_limiter=True)
    if path:
        t0 = time.time()
        np.savez(path, **{k: fw[k] for k in _CACHE_KEYS},
                 **{k: np.asarray(fw[k]) for k in _CACHE_SCALARS})
        print("  cached forward -> %s (%.1f GB, %.0fs)"
              % (path, os.path.getsize(path) / 1e9, time.time() - t0))
    return fw


def profiles(fw, p, num):
    """h_theory(x), A+(x), A-(x) for EVERY x, by the forward recursion u <- P_m(u + v^m).

    The per-target adjoint sweep gives one x per pass, so a profile would cost N sweeps.  The
    same sums S = sum_m G^m v^m obey a FORWARD recursion instead, which delivers all x in one
    pass -- the identity ancestry.py uses for its own A+/A-.

    This uses ancestry.apply_P, i.e. the FIRST-ORDER (upwind) kernel.  That is deliberate and it
    is not a tautology here: the trajectory was integrated with the LIMITER, so reproducing
    h_sim with an upwind kernel is a genuine test of the transport, and the residual is the
    measured gap.  The second-order vanleer+FCT values are the per-target rows of summary.csv.

    Only sgn(s_T) = +-1 enters the exposure split, so two sign-resolved sums suffice for all x.
    """
    n_steps, N, dt, dx = fw["n_steps"], fw["N"], fw["dt"], fw["dx"]

    def accumulate(V, u0):
        u = np.array(u0, dtype=float)
        for m in range(n_steps):
            u = A.apply_P(u + V[m], fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m],
                          p, num, dt, dx, N)
        return u

    D = fw["D_SEL"]
    h_th = accumulate(D, fw["h0"])
    Sp = accumulate(np.maximum(D, 0.0), np.zeros(N))
    Sm = accumulate(np.abs(np.minimum(D, 0.0)), np.zeros(N))

    # Same recursion, weak-selection source 0.5*dt*s(n) in place of the exact increment.
    h_weak = np.full(N, np.nan)
    if fw.get("NPRE") is not None:
        NP = np.asarray(fw["NPRE"])
        h_weak = accumulate(0.5 * dt * (np.asarray(C.g1(NP, p))
                                        - np.asarray(C.g2(NP, p))), fw["h0"])

    s_fin = np.asarray(C.g1(fw["n"], p)) - np.asarray(C.g2(fw["n"], p))
    sig = np.where(s_fin >= 0.0, 1.0, -1.0)
    Ap = np.where(sig > 0, Sp, Sm)
    Am = np.where(sig > 0, Sm, Sp)
    return dict(h_theory=h_th, h_weak=h_weak, A_plus=Ap, A_minus=Am,
                sigma=sig, s_final=s_fin)


def _profile_figure(fw, p, pr, tg, saved, outdir, a):
    """h_sim vs h_theory with inverse-sorted bands, and the A+/A- diagnosis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, h_sim, h_th = fw["x"], fw["h_sim"], pr["h_theory"]
    Ap, Am, sig = pr["A_plus"], pr["A_minus"], pr["sigma"]
    rho = fw["rho1"] + fw["rho2"]
    occ = rho > 1e-6 * max(float(rho.max()), 1e-30)
    inv = (sig * h_sim < 0.0) & occ                 # the definition
    diag = (Am > Ap) & occ                          # the A+/A- criterion
    nstar = float(C.monod_crossover(p))
    xn = x[np.argmax(fw["n"] < nstar)] if np.any(fw["n"] < nstar) else np.nan

    fig, ax = plt.subplots(3, 1, figsize=(9.5, 9.6), sharex=True)

    def bands(axis, mask, label):
        d = np.diff(mask.astype(int))
        starts = list(x[1:][d == 1])
        ends = list(x[1:][d == -1])
        if mask[0]:
            starts = [x[0]] + starts
        if mask[-1]:
            ends = ends + [x[-1]]
        for i, (s0, s1) in enumerate(zip(starts, ends)):
            axis.axvspan(s0, s1, color="#5b8ff9", alpha=0.22,
                         label=label if i == 0 else None)

    bands(ax[0], inv, "inverse sorted ($\\sigma_T h<0$)")
    ax[0].plot(x[occ], h_sim[occ], color="k", lw=2.4, label="$h_{\\rm sim}$")
    ax[0].plot(x[occ], h_th[occ], color="#d62728", lw=1.2, ls="--", label="$h_{\\rm theory}$")
    hw = pr.get("h_weak")
    if hw is not None and np.isfinite(hw).any():
        ax[0].plot(x[occ], np.clip(hw[occ], -1.6, 1.6), color="#1f77b4", lw=1.2, ls="-.",
                   label="$h_{\\rm weak}=\\frac{1}{2}A_K$  (clipped)")
    ax[0].axhline(1.0, color="0.75", lw=0.7, ls=":")
    ax[0].axhline(-1.0, color="0.75", lw=0.7, ls=":")
    ax[0].axhline(0, color="0.6", lw=0.8)
    if np.isfinite(xn):
        ax[0].axvline(xn, color="g", ls=":", lw=1.4, label="$n^*$ front")
    for name, ti in tg:
        ax[0].plot(fw["x"][ti], h_sim[ti], "o", ms=7, mfc="none", mec="k")
        ax[0].annotate(name, (fw["x"][ti], h_sim[ti]), textcoords="offset points",
                       xytext=(6, 8), fontsize=8)
    ax[0].set_ylabel("$h = 2f_1-1$")
    ax[0].legend(loc="lower left", fontsize=8, ncol=2)
    ax[0].set_title("composition: theory vs simulation, with inverse-sorted regions shaded",
                    fontsize=10)

    bands(ax[1], diag, "$A_->A_+$ (criterion)")
    ax[1].plot(x[occ], Ap[occ], color="#2ca02c", lw=1.8, label="$A_+$ reinforcing")
    ax[1].plot(x[occ], Am[occ], color="#d62728", lw=1.8, label="$A_-$ opposing")
    if np.isfinite(xn):
        ax[1].axvline(xn, color="g", ls=":", lw=1.4)
    ax[1].set_ylabel("exposure")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title("exposure decomposition: inverse sorting $\\Leftrightarrow A_->A_+$", fontsize=10)

    ax[2].semilogy(x[occ], np.abs(h_th - h_sim)[occ], color="0.25", lw=1.2)
    for name, ti in tg:
        r = saved.get("%s_%s" % (name, a.kinds.split(",")[-1]))
        if r is not None:
            ax[2].plot(fw["x"][ti], abs(r["h_pred"] - h_sim[ti]), "*", ms=13,
                       color="#d62728",
                       label="vanleer+FCT (per target)" if name == "mouth" else None)
    ax[2].set_ylabel("$|h_{\\rm theory}-h_{\\rm sim}|$")
    ax[2].set_xlabel("$x$ [$\\mu$m]")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_title("residual: profile is the first-order (upwind) kernel; stars are "
                    "second-order per-target", fontsize=10)

    agree = int(np.sum(inv[occ] == diag[occ]))
    fig.suptitle("adjoint ancestry kernel   n$_0$=%g, T=%g h, trajectory=%s   "
                 "criterion agrees with definition on %d/%d occupied cells"
                 % (a.n0, a.T, a.limiter, agree, int(occ.sum())), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, os.path.join(outdir, "profiles.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n0", type=float, default=15.0)
    ap.add_argument("--dx", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=94.0)
    ap.add_argument("--L", type=float, default=1000.0)
    ap.add_argument("--rho-seed", dest="rho_seed", type=float, default=1.0e-4)
    ap.add_argument("--limiter", default="vanleer",
                    help="trajectory scheme (the physics run)")
    ap.add_argument("--kinds", default="none,vanleer",
                    help="comma-separated adjoint schemes to compare")
    ap.add_argument("--mouth-x", dest="mouth_x", type=float, default=30.0)
    ap.add_argument("--deep-x", dest="deep_x", type=float, default=900.0)
    ap.add_argument("--frames", type=int, default=241)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache", default=None,
                    help="path to a .npz forward-trajectory cache (~2.6 GB); "
                         "reused if it exists, written if not")
    ap.add_argument("--only", default="both",
                    choices=("both", "targets", "profiles"),
                    help="run only one stage; 'profiles' leaves summary.csv, "
                         "trajectory.csv, fields.npz and adjoint.png untouched")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if check() else 1)

    from dataclasses import replace as _replace
    p = config.params(a.n0)
    num = config.numerics(T_end=a.T, dt=a.dt, dx=a.dx, n_save=2, label_limiter=a.limiter)
    ic = _replace(config.initial_conditions(), rho_seed=a.rho_seed)
    print("n0=%g L=%g dx=%g dt=%g T=%g rho_seed=%g   trajectory=%s"
          % (a.n0, a.L, a.dx, a.dt, a.T, ic.rho_seed, a.limiter))
    fw = load_or_run_forward(a.cache, a.L, p, num, ic)
    N = fw["N"]
    print("  N=%d  n_steps=%d" % (N, fw["n_steps"]))

    tg = [("mouth", int(np.argmin(np.abs(fw["x"] - a.mouth_x)))),
          ("deep", int(np.argmin(np.abs(fw["x"] - a.deep_x))))]
    tag = ("adjoint_n0-%g_L%g_dx%g_T%g_m%g_d%g"
           % (a.n0, a.L, a.dx, a.T, a.mouth_x, a.deep_x))
    if a.rho_seed is not None:
        tag += "_seed%g" % a.rho_seed
    if a.limiter != "none":
        tag += "_traj-%s" % a.limiter
    outdir = a.outdir or os.path.join(HERE, tag)
    os.makedirs(outdir, exist_ok=True)

    rows, saved = [], {}
    if a.only in ("both", "targets"):
        print()
        print("  %-7s %-9s %8s %10s %10s %10s %10s %9s %9s %9s %8s %7s"
              % ("target", "adjoint", "x[um]", "h_pred", "h_weak", "h_sim", "residual",
                 "res_weak", "A+", "A-", "min(K)", "kept"))
    for name, ti in (tg if a.only in ("both", "targets") else []):
        for kind in a.kinds.split(","):
            r = sweep(fw, p, num, ti, kind, fct=True, frames=a.frames)
            res = abs(r["h_pred"] - fw["h_sim"][ti])
            inv = bool(r["sigma"] * fw["h_sim"][ti] < 0.0)
            res_w = abs(r["h_weak"] - fw["h_sim"][ti])
            print("  %-7s %-9s %8.1f %10.5f %10.5f %10.5f %10.2e %9.2e %9.4f %9.4f %8.1e %6.1f%%"
                  % (name, kind, fw["x"][ti], r["h_pred"], r["h_weak"], fw["h_sim"][ti],
                     res, res_w, r["A_plus"], r["A_minus"], r["min_K"], 100 * r["kept"]))
            rows.append(dict(target=name, adjoint=kind, x_um=fw["x"][ti],
                             h_pred=r["h_pred"], h_weak=r["h_weak"],
                             residual_weak=res_w, h_sim=fw["h_sim"][ti], residual=res,
                             A_plus=r["A_plus"], A_minus=r["A_minus"], B0=r["B0"],
                             sigma_T=r["sigma"], inverse=int(inv),
                             identity_err=abs(r["sigma"] * fw["h_sim"][ti]
                                              - (r["B0"] + r["A_plus"] - r["A_minus"])),
                             min_K=r["min_K"], antidiff_kept=r["kept"]))
            saved["%s_%s" % (name, kind)] = r

    if a.only == "profiles":
        print("  --only profiles: reusing the target-stage outputs already in %s"
              % os.path.basename(outdir))
        # recover the per-target second-order points so panel (c) keeps its stars
        sp = os.path.join(outdir, "summary.csv")
        if os.path.exists(sp):
            with open(sp, newline="") as fh:
                for q in csv.DictReader(fh):
                    saved["%s_%s" % (q["target"], q["adjoint"])] = {
                        "h_pred": float(q["h_pred"])}
            print("     recovered %d target rows from summary.csv" % len(saved))
    if rows:
      with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
          wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
          wcsv.writeheader()
          wcsv.writerows(rows)

      ref = saved["mouth_%s" % a.kinds.split(",")[-1]]
      tt = np.arange(fw["n_steps"]) * fw["dt"]
      with open(os.path.join(outdir, "trajectory.csv"), "w", newline="") as fh:
          wcsv = csv.writer(fh)
          keys = sorted(saved)
          wcsv.writerow(["t_h"] + ["ybar_" + k for k in keys] + ["mass_" + k for k in keys])
          for i in range(fw["n_steps"]):
              wcsv.writerow([tt[i]] + [saved[k]["ybar"][i] for k in keys]
                            + [saved[k]["mass"][i] for k in keys])

      np.savez_compressed(os.path.join(outdir, "fields.npz"),
                          x=fw["x"], h_sim=fw["h_sim"], n_final=fw["n"],
                          nstar=float(C.monod_crossover(p)), L=a.L, dx=fw["dx"], dt=fw["dt"],
                          T=a.T, targets=np.array([ti for _, ti in tg]),
                          t_blk=ref["t_blk"],
                          **{("anc_" + k): saved[k]["anc"].astype(np.float32)
                             for k in saved})

      _figure(fw, p, saved, tg, outdir, a)

    if a.only == "targets":
        print()
        print("  wrote %s" % outdir)
        return
    print()
    print("  profiles over all x (forward recursion, first-order kernel)...")
    pr = profiles(fw, p, num)
    rho_t = fw["rho1"] + fw["rho2"]
    occ = rho_t > 1e-6 * max(float(rho_t.max()), 1e-30)
    inv = (pr["sigma"] * fw["h_sim"] < 0.0) & occ
    diag = (pr["A_minus"] > pr["A_plus"]) & occ
    with open(os.path.join(outdir, "profiles.csv"), "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["x_um", "h_sim", "h_theory", "h_weak", "A_plus", "A_minus",
                       "sigma_T", "s_final", "inverse", "criterion", "occupied"])
        for i in range(fw["N"]):
            wcsv.writerow([fw["x"][i], fw["h_sim"][i], pr["h_theory"][i], pr["h_weak"][i],
                           pr["A_plus"][i], pr["A_minus"][i], pr["sigma"][i],
                           pr["s_final"][i], int(inv[i]), int(diag[i]), int(occ[i])])
    _profile_figure(fw, p, pr, tg, saved, outdir, a)
    res = np.abs(pr["h_theory"] - fw["h_sim"])[occ]
    w = (fw["rho1"] + fw["rho2"])[occ]
    w = w / max(w.sum(), 1e-300)
    print("    residual over occupied cells: max=%.2e  biomass-weighted=%.2e"
          % (res.max(), float((w * res).sum())))
    print("    inverse-sorted (definition sigma_T*h<0): %d/%d occupied cells"
          % (int(inv[occ].sum()), int(occ.sum())))
    print("    criterion A- > A+ agrees with it on %d/%d"
          % (int(np.sum(inv[occ] == diag[occ])), int(occ.sum())))
    print()
    print("  wrote %s" % outdir)
    print("        summary.csv, trajectory.csv, profiles.csv, fields.npz,")
    print("        adjoint.png (kernel heatmaps), profiles.png (h + A+/A-)")


def _savefig(fig, path, tries=5):
    """Save a figure, retrying past transient Windows file locks.

    A viewer or scanner holding the PNG raises OSError 22 and would otherwise discard the
    figures at the very end of a multi-hour sweep.  Falls back to a timestamped name.
    """
    import time
    for i in range(tries):
        try:
            fig.savefig(path, dpi=150)
            return path
        except OSError as exc:
            if i == tries - 1:
                alt = "%s.%d.png" % (os.path.splitext(path)[0], int(time.time()))
                fig.savefig(alt, dpi=150)
                print("    !! %s locked (%s); wrote %s" % (os.path.basename(path), exc,
                                                           os.path.basename(alt)))
                return alt
            time.sleep(1.5)


def _figure(fw, p, saved, tg, outdir, a):
    """K(y,t) heatmaps with the ancestral mean overlaid on the n* front."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kind = a.kinds.split(",")[-1]
    nstar = float(C.monod_crossover(p))
    tt = np.arange(fw["n_steps"]) * fw["dt"]
    fig, axes = plt.subplots(1, len(tg), figsize=(6.2 * len(tg), 4.4), squeeze=False)
    for ax, (name, ti) in zip(axes[0], tg):
        r = saved["%s_%s" % (name, kind)]
        K = r["anc"]
        pos = K > 0
        vmax = float(K.max()) if K.size else 1.0
        im = ax.pcolormesh(fw["x"], r["t_blk"], np.where(pos, K, np.nan),
                           cmap="viridis", vmin=0.0, vmax=vmax, shading="auto")
        ax.plot(r["ybar"], tt, color="w", lw=1.6, label=r"$\bar y(t)$ ancestral mean")
        ax.set_xlabel(r"$y$ [$\mu$m]")
        ax.set_ylabel("$t$ [h]")
        ax.set_title("%s target  $x=%.0f\\,\\mu$m   $h_{\\rm pred}=%.4f$ vs $h_{\\rm sim}=%.4f$"
                     % (name, fw["x"][ti], r["h_pred"], fw["h_sim"][ti]), fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.colorbar(im, ax=ax, label=r"$K(y,t\mid x,T)$")
    fig.suptitle("adjoint ancestry kernel, %s + FCT   (min K = %.1e, antidiffusive flux kept "
                 "%.1f%%)   n$_0$=%g, T=%g h, trajectory=%s"
                 % (kind, min(saved[k]["min_K"] for k in saved),
                    100 * min(saved[k]["kept"] for k in saved), a.n0, a.T, a.limiter),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, os.path.join(outdir, "adjoint.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
