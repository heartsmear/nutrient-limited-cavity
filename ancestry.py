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

COST.  Two backward paths, picked by N at the call site.  Below MATRIX_MAX_N the classical one
composes G^m = G^{m+1} P_m explicitly: eps-probing costs O(N^2) per step and the product O(N^3).
Above it, `backward_lowmem` never forms G^m -- the whole-field sums obey a FORWARD recursion
u <- P_m(u + v^m), and only the target ROWS need the backward direction, via the adjoint P_m^T.
Both are matrix-free and O(N) per step.  Measured at N=2000: 600 ms per step the old way, 11.8 ms
the new, so a resolved jammed run (dx=0.5, 60k steps) goes from ~10 h to ~12 min.  **Ancestry and
properly-resolved jammed runs are no longer mutually exclusive.**  `--max-n` now guards MEMORY --
forward() keeps five (n_steps, N) arrays -- not the composition cost.

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
#: N above which the matrix-free backward is used.  Measured crossover is ~200 cells: below it
#: one BLAS matmul beats a Python-level step loop, above it O(N^3) composition dominates.
MATRIX_MAX_N = 200
#: Refuse above this N.  With the matrix-free path the binding cost is no longer the O(N^3)
#: composition but the stored trajectory: forward() keeps 5 arrays of (n_steps, N) doubles, so
#: N=2000 with 60k steps is ~4.8 GB.  Raise deliberately and watch memory.
MAX_N_DEFAULT = 2200
#: how far the BIOMASS-WEIGHTED Duhamel residual may drift before the result is rejected.
#: Observed 1.8e-8..3.1e-8 across dx=0.5..2 on production runs and 7.4e-7 on the deliberately
#: coarse --check config, so 1e-4 leaves >2 decades of headroom on the worst healthy case while
#: still catching a transport operator that does not match the scheme by 2-3 decades (a label
#: limiter costs O(D_num), which drives F1 by 15%).  --force downgrades it to a warning.
DUHAMEL_TOL = 1.0e-4
#: how far the minority strain fraction may move under dx refinement before the trajectory is
#: rejected.  2x is deliberately loose: the motivating failure was 18899x.
PURITY_TOL = 2.0


def _h(r1, r2, p):
    rho = r1 + r2
    f = C.local_fraction_exact(r1, np.maximum(rho - r1, 0.0), tol=p.tiny_density)
    return 2.0 * f - 1.0


# =============================================================================
# Forward pass: exactly solver.simulate, recording the trajectory
# =============================================================================
def forward(L, p, num, ic, store=True, allow_limiter=False):
    """Mirror of solver.one_step in a loop, storing what the backward pass needs.

    REQUIRES num.label_limiter == "none" -- see the guard below -- UNLESS the caller passes
    allow_limiter=True.  That escape exists for `kernel_reconstruct.py`, which never eps-probes
    this trajectory: it BUILDS its own monotone transport operator from the stored
    (rho_r, rho_new, J_total) and measures the Duhamel residual instead of asserting it.  The
    guard protects the probe, and there is no probe on that path.  Do not set it for anything
    that calls transport_matrix()/apply_P().

    Storing every P_m would be N^2 per step; instead we store the TRAJECTORY (rho_r, rho_new,
    J_total, the selection increment) and REBUILD P_m during the backward sweep.  Same arithmetic,
    far less memory.  `--check` asserts this loop reproduces solver.simulate bit-for-bit.

    `store=False` runs the identical loop but skips the five (n_steps, N) arrays, which are what
    actually costs memory -- 6.4 GB at N=4000, T=2000.  Callers that only want the FINAL state
    (convergence_check compares end profiles) should use it; the kernel needs store=True.
    """
    # The kernel needs a transport operator LINEAR in the label.  This loop advects the label
    # through closures.advect_strain_by_total_flux, which HONOURS num.label_limiter, while
    # transport_matrix()/apply_P() hard-code donor-cell upwinding -- there the face is chosen by
    # sign(J_total), a coefficient, so the map is linear and eps-probeable.  A TVD limiter picks
    # the face from the sign of the label's OWN consecutive differences, which is not linear, and
    # the probe would return the plain UPWIND matrix regardless: H = eps*I is a single-cell bump,
    # i.e. an extremum, and every TVD limiter switches its correction off there (a*b < 0).  That
    # failure is silent -- the upwind matrix is perfectly row-stochastic, so the row-sum guard
    # stays quiet and only the Duhamel residual moves.  Refuse rather than rely on noticing.
    lim = getattr(num, "label_limiter", "none")
    if lim != "none" and not allow_limiter:
        raise ValueError(
            f"ancestry.forward: label_limiter={lim!r} is not supported. The eps-probe requires "
            f"transport linear in the label, which only donor-cell upwinding is. Pass "
            f"config.numerics(..., label_limiter='none'). This is not a loss of accuracy: the "
            f"exposure integrals converge at p=2.14 to +0.35% at dx=0.5 on the unlimited scheme "
            f"(the limiter is needed for F1, which converges at only p=0.99).")
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

    if store:
        RHO_R = np.zeros((n_steps, N)); RHO_NEW = np.zeros((n_steps, N))
        JTOT = np.zeros((n_steps, N + 1)); D_SEL = np.zeros((n_steps, N))
        NPRE = np.zeros((n_steps, N))
    else:
        RHO_R = RHO_NEW = JTOT = D_SEL = NPRE = None

    for step in range(n_steps):
        if store:
            NPRE[step] = n
        h_before = _h(rho1, rho2, p)
        rho_old = np.maximum(rho1 + rho2, 0.0)
        q = C.growth_space_factor(rho_old, p)

        rho1_r, rho2_r, n_new, _ = S.coupled_reaction_nutrient(
            rho1, rho2, n, p, dt, B_n, b_n, q=q)
        n_new = np.maximum(n_new, 0.0)
        rho_r = rho1_r + rho2_r
        rho_new, _Dc, J_total, _dg = S.nonlinear_collective_diffusion(rho_r, p, dt, dx)
        if store:
            D_SEL[step] = _h(rho1_r, rho2_r, p) - h_before      # the selection increment
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
# Matrix-free transport: apply P_m and its adjoint to a VECTOR
# =============================================================================
# Composing the full G^m costs O(N^3) per step, which is what made a resolved jammed run
# (dx=0.5, N=2000 -> 480 Tflop) impossible.  It is also unnecessary:
#
#   * h_sel, A+, A-, B0 are all sums  S = sum_m G^m v^m  with G^m = P_{N-1}...P_m, and such a
#     sum obeys the FORWARD recursion  u <- P_m (u + v^m),  u^N = S.  Matrix-vector only.
#   * only the two target ROWS of G^m are ever needed for the kernels, and a row obeys the
#     BACKWARD recursion  w <- P_m^T w.
#
# Both are O(N) per step if P_m is applied without being formed.  In the h variable the
# advection substep is exactly linear (the f -> h change of variables cancels the constant),
#
#   Hn[i] = ( rho[i] h[i] - c ( J[i+1] h_face[i+1] - J[i] h_face[i] ) ) / rho_n[i],  c = hsub/dx
#
# and 0 on unoccupied cells -- which is what the eps-probe produces there too, since it sets
# f = 1/2 and h = 2f-1 = 0.  Verified against the explicit matrix in `--check`.

def _adv_states(rho_r, J_total, dt, dx, p, num, N):
    """Per-substep (rho_before, rho_after) plus the upwind face-source index."""
    nsub = C.compute_advection_substeps(rho_r, J_total, dt, dx, p, num)
    hsub = dt / nsub
    src = np.empty(N + 1, dtype=np.intp)          # which cell supplies h at each face
    src[0] = 0
    k = np.arange(1, N)
    src[1:N] = np.where(J_total[1:N] >= 0.0, k - 1, k)
    src[N] = N - 1
    states, rho_path = [], rho_r.copy()
    for _ in range(nsub):
        rho_n = np.maximum(rho_path - hsub * (J_total[1:] - J_total[:-1]) / dx, 0.0)
        states.append((rho_path.copy(), rho_n.copy()))
        rho_path = rho_n
    return hsub, src, states


def apply_P(v, rho_r, rho_new, J_total, p, num, dt, dx, N):
    """P_m v, matrix-free.  Advection substeps then the implicit self-diffusion solve."""
    hsub, src, states = _adv_states(rho_r, J_total, dt, dx, p, num, N)
    h = np.asarray(v, dtype=float).copy()
    c = hsub / dx
    for rho_path, rho_n in states:
        hf = h[src]
        num_i = rho_path * h - c * (J_total[1:] * hf[1:] - J_total[:-1] * hf[:-1])
        h = np.where(rho_n > p.tiny_density, num_i / np.maximum(rho_n, 1e-300), 0.0)
    if (not p.use_self_diffusion) or np.all(rho_new <= p.tiny_density):
        return h
    rho = np.maximum(rho_new, 0.0)
    B = np.asarray(C.self_diffusivity(rho, p)) * rho
    A = C.build_diffusion_matrix_neumann_neumann(B, dx)
    M = diags(np.maximum(rho, p.tiny_density), 0, format="csc") - dt * A
    return spsolve(M, rho * h)


def apply_PT(w, rho_r, rho_new, J_total, p, num, dt, dx, N):
    """P_m^T w, matrix-free.  P = D A, so P^T = A^T D^T, applied in that order."""
    hsub, src, states = _adv_states(rho_r, J_total, dt, dx, p, num, N)
    u = np.asarray(w, dtype=float).copy()
    if p.use_self_diffusion and not np.all(rho_new <= p.tiny_density):
        rho = np.maximum(rho_new, 0.0)
        B = np.asarray(C.self_diffusivity(rho, p)) * rho
        A = C.build_diffusion_matrix_neumann_neumann(B, dx)
        M = diags(np.maximum(rho, p.tiny_density), 0, format="csc") - dt * A
        u = rho * spsolve(M.T.tocsc(), u)          # D^T = diag(rho) M^{-T}
    c = hsub / dx
    for rho_path, rho_n in reversed(states):       # A^T reverses the substep order
        g = np.where(rho_n > p.tiny_density, u / np.maximum(rho_n, 1e-300), 0.0)
        out = rho_path * g                          # diagonal term
        np.add.at(out, src[1:], -c * J_total[1:] * g)    # outgoing face of each cell
        np.add.at(out, src[:-1], c * J_total[:-1] * g)   # incoming face of each cell
        u = out
    return u


# =============================================================================
# Backward pass
# =============================================================================
def pick_targets(fw, p, mode="auto"):
    """(mouth, deep, deep_is_inverse).

    mode="auto" (default, unchanged):
        mouth: the n* front cell -- where mouth-type inverse sorting lives (strain 1 at n > n*).
        deep : the most massive DEEP-INVERSE cell (n < n* yet h < 0) -- the graveyard.  Chosen by
               biomass so the kernel is computed where the signal is, not at an arbitrary
               coordinate.

    mode="inverse" (--inverse):
        mouth: the last cell still ABOVE n*, i.e. just left of the front, toward the opening.
        deep : a fixed 10% before the dead end, x = 0.9 L.
        Both are FIXED GEOMETRY rather than biomass-chosen, so the same two locations are probed
        in every run.  That is what makes cavities of different length (or different n0)
        comparable -- the "auto" deep target moves with the biomass and silently compares
        different places.  The cost is that the deep target is not guaranteed to be inverse-
        sorted, or even to sit below n*; both are reported rather than assumed.
    """
    nstar = float(C.monod_crossover(p))
    x, n, h = fw["x"], fw["n"], fw["h_sim"]
    rho = fw["rho1"] + fw["rho2"]
    live = rho > 1e-6 * max(rho.max(), 1e-30)

    if mode == "inverse":
        above = np.where(n > nstar)[0]
        # n falls with x, so the largest index above n* is the cell just before the front.
        mouth = int(above[-1]) if above.size else 0
        deep = int(np.argmin(np.abs(x - 0.9 * fw["L"])))
        return mouth, deep, bool(n[deep] < nstar and h[deep] < 0.0)

    k = np.where(np.diff(np.sign(n - nstar)) != 0)[0]
    mouth = int(k[0]) if k.size else 2
    deep_inv = (n < nstar) & (h < 0.0) & live
    if deep_inv.any():
        return mouth, int(np.argmax(rho * deep_inv)), True
    deep_side = (n < nstar) & live
    fallback = int(np.argmax(rho * deep_side)) if deep_side.any() else int(0.6 * len(x))
    return mouth, fallback, False


def backward_lowmem(fw, p, targets, frames=1500):
    """Same outputs as `backward`, without ever forming G^m.  O(N) per step instead of O(N^3).

    Two passes over the stored trajectory:

      1. FORWARD, accumulating  u <- P_m (u + v^m)  for v = Delta_sel (gives h_sel), its positive
         and negative parts (give A+/A-), and v = 0 with u seeded at h^0 (gives B0).  Four
         matrix-vector applications per step.
      2. BACKWARD, propagating the target ROWS by  w <- P_m^T w.  Two applications per step.

    sigma_T is fixed in time, so the per-step  where(sigma>0, ., .)  of the original can be pulled
    outside the sum -- which is what lets A+/A- be accumulated forward at all.

    `anc`/`Csel` are block-averaged to ~`frames` time samples DURING the sweep: at N=2000 with
    60k steps the full arrays would be 1.9 GB.  Csel is averaged from the per-step product, not
    formed from the averaged anc, since <a*d> != <a><d>.
    """
    N, n_steps, dt, dx, num = fw["N"], fw["n_steps"], fw["dt"], fw["dx"], fw["num"]
    targets = list(targets)
    s_final = np.asarray(C.g1(fw["n"], p)) - np.asarray(C.g2(fw["n"], p))
    sig = np.sign(s_final); sig[sig == 0] = 1.0

    def _args(m):
        return (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num, dt, dx, N)

    # ---- pass 1: forward accumulators -----------------------------------------------------
    u_sel = np.zeros(N); u_pos = np.zeros(N); u_neg = np.zeros(N)
    b = fw["h0"].astype(float).copy()
    for m in range(n_steps):
        a = _args(m); d = fw["D_SEL"][m]
        u_sel = apply_P(u_sel + d, *a)
        u_pos = apply_P(u_pos + np.where(d > 0.0, d, 0.0), *a)
        u_neg = apply_P(u_neg + np.where(d < 0.0, -d, 0.0), *a)
        b = apply_P(b, *a)
    A_plus = np.where(sig > 0, u_pos, u_neg)
    A_minus = np.where(sig > 0, u_neg, u_pos)

    # ---- pass 2: backward target rows, block-averaged on the fly ---------------------------
    blk = max(1, int(np.ceil(n_steps / max(frames, 1))))
    nb = int(np.ceil(n_steps / blk))
    anc = np.zeros((len(targets), nb, N)); Csel = np.zeros((len(targets), nb, N))
    cnt = np.zeros(nb)
    W = np.zeros((len(targets), N))
    for k, ti in enumerate(targets):
        W[k, ti] = 1.0
    for m in range(n_steps - 1, -1, -1):
        a = _args(m)
        for k in range(len(targets)):
            W[k] = apply_PT(W[k], *a)
        j = m // blk
        anc[:, j, :] += W
        Csel[:, j, :] += W * fw["D_SEL"][m][None, :]
        cnt[j] += 1
    nz = cnt > 0
    anc[:, nz, :] /= cnt[nz][None, :, None]
    Csel[:, nz, :] /= cnt[nz][None, :, None]

    return dict(sigma=sig, A_plus=A_plus, A_minus=A_minus, B0=sig * b,
                h_recon=b + u_sel, h_sel=u_sel, anc=anc, Csel=Csel, targets=targets,
                block=blk, n_steps=n_steps)


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


def _decimate_nan(A, target=1500):
    """_decimate for a masked field: block-average ignoring NaN, keeping all-NaN blocks NaN.

    A plain mean would let one unoccupied step poison a whole block back to NaN, erasing occupied
    data; nanmean keeps the block if any step in it was occupied.
    """
    n = A.shape[0]
    if n <= target:
        return A
    blk = int(np.ceil(n / target))
    B = A[:(n // blk) * blk].reshape(-1, blk, A.shape[1])
    ok = np.isfinite(B)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, B, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def convergence_check(L, p, num, ic, fw, refine=2.0):
    """Is the TRAJECTORY the kernel is built on resolved?  Re-solve at dx/refine and compare.

    THIS IS THE ONLY TEST HERE THAT CAN FAIL FOR A PHYSICAL REASON, and it exists because
    everything else in this module is self-referential:

      * the Duhamel residual is an identity on whatever trajectory it is handed (Thm 6.1), so it
        is ~1e-6 even when that trajectory is nonsense;
      * `solver_reference` compares the mirror loop against the same instruction sequence, so it
        returns exactly 0 by construction.

    Measured case that motivated this: n0=1000, L=1000, T=2000.  At the deep end the converged
    dx=0.5 run has f1 = 1.3e-6 (a pure strain-2 graveyard), while dx=4 reports f1 = 2.5e-2 --
    upwind numerical diffusion manufacturing a minority strain out of nothing, wrong by 18899x.
    Every other diagnostic passed cleanly on that run.

    WHY THE GATE IS ON THE MINORITY FRACTION, NOT ON |dh|.  In that same case max|dh| is only
    ~0.05, which reads as a 5% error and looks tolerable.  The damage is entirely in the minority
    strain, whose fraction is O(1e-6): any leakage swamps it.  So the decisive statistic is the
    RATIO of min(f1) between the two resolutions, on a log scale, and |dh| is reported only as
    context.
    """
    from dataclasses import replace as _replace
    num_f = _replace(num, dx_target=num.dx_target / float(refine))
    fw_f = forward(L, p, num_f, ic, store=False)   # final state only; skips ~6 GB at N=4000

    def _fh(w):
        rho = w["rho1"] + w["rho2"]
        occ = rho > 1e-9 * max(rho.max(), 1e-30)
        f = np.where(occ, w["rho1"] / np.maximum(rho, 1e-300), np.nan)
        return f, occ, rho

    f_c, occ_c, rho_c = _fh(fw)
    f_f, occ_f, rho_f = _fh(fw_f)
    h_f_on_c = np.interp(fw["x"], fw_f["x"], fw_f["h_sim"])
    w = rho_c * occ_c
    w = w / max(w.sum(), 1e-300)

    dh_max = float(np.nanmax(np.abs(fw["h_sim"] - h_f_on_c)[occ_c])) if occ_c.any() else np.nan
    dh_w = float(np.nansum(w * np.abs(fw["h_sim"] - h_f_on_c)))
    minf_c, minf_f = float(np.nanmin(f_c)), float(np.nanmin(f_f))
    maxf_c, maxf_f = float(np.nanmax(f_c)), float(np.nanmax(f_f))
    # purity of each pure region, measured as the ratio of the minority fraction
    r_lo = max(minf_c, 1e-300) / max(minf_f, 1e-300)
    r_hi = max(1.0 - maxf_f, 1e-300) / max(1.0 - maxf_c, 1e-300)
    ratio = max(r_lo, 1.0 / max(r_lo, 1e-300), r_hi, 1.0 / max(r_hi, 1e-300))
    return dict(N_coarse=fw["N"], N_fine=fw_f["N"], dx_coarse=fw["dx"], dx_fine=fw_f["dx"],
                dh_max=dh_max, dh_weighted=dh_w, min_f1_coarse=minf_c, min_f1_fine=minf_f,
                max_f1_coarse=maxf_c, max_f1_fine=maxf_f, purity_ratio=ratio)


def solver_reference(L, p, num, ic):
    """h(x,T) from solver.simulate itself -- an INDEPENDENT solve, not ancestry's mirror loop.

    `forward()` re-implements the time loop so it can record between sub-steps; it calls the
    solver's components but sequences them itself.  Comparing h_recon against fw["h_sim"] alone
    therefore compares one pass with itself, and would still look perfect if the mirror had
    drifted from the production solver.  This runs the real thing at the same parameters.
    """
    res = S.simulate(L, p, num, ic)
    r1, r2 = res["rho1_final"], res["rho2_final"]
    return _h(r1, r2, p), res


def run_csv_reference(outdir_guess):
    """h(x,T) from a completed figures.py run on disk, if one exists at these parameters.

    The strongest available reference: a separate invocation of the production pipeline, written
    to CSV, with no shared state with this process at all.  Returns None when no such run exists.
    """
    path = os.path.join(outdir_guess, "profile_final.csv")
    if not os.path.exists(path):
        return None, None
    x, h = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rho = float(r["rho"])
            r1 = float(r["rho1"])
            x.append(float(r["x_um"]))
            h.append((2.0 * r1 / rho - 1.0) if rho > 0 else np.nan)
    return np.array(x), np.array(h)


def figure_h_compare(fw, bw, p, outdir, h_solver=None, h_csv=None, csv_label=None):
    """Final h(x,T): the ancestry reconstruction against the production solver.

    Up to three curves, and they are NOT all the same kind of evidence:

      h_sim    ancestry's own forward pass -- the trajectory the kernel was probed from.
      h_recon  B0 + sum_m G^m Delta_sel^m.  Theorem 6.1 makes h_recon = h_sim an ALGEBRAIC
               IDENTITY, so their agreement tests the eps-probe, not the physics.
      solver   solver.simulate at the same parameters, and/or a completed run read from
               profile_final.csv.  THIS is the external check: it can fail, and it is what
               tells you the mirror loop still tracks the production solver.
    """
    x, nstar = fw["x"], float(C.monod_crossover(p))
    hs, hr = fw["h_sim"], bw["h_recon"]
    resid = hr - hs
    rho = fw["rho1"] + fw["rho2"]
    live = rho > 1e-9 * max(rho.max(), 1e-30)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle(rf"Final dominance $h(x,T)$: numerical vs ancestry theory   "
                 rf"($L={fw['L']:g}\ \mu$m, $n^*={nstar:.3f}\ \mu$M)", fontsize=12)

    ax[0].axhline(0.0, color="0.6", lw=0.9, ls=":")
    xf = D.nstar_front(x, fw["n"], nstar)
    if np.isfinite(xf):
        ax[0].axvline(xf, color=CN, lw=1.2, ls="--", label=rf"$n^*$ front  $x={xf:.1f}$")
    ax[0].plot(x, hs, color=C1, lw=2.6, label="numerical  $h_{\\rm sim}$")
    ax[0].plot(x, hr, color=C2, lw=1.4, ls=(0, (4, 3)), label="theory  $h_{\\rm recon}$")
    for ti, mk in zip(bw["targets"], ("o", "s")):
        ax[0].plot(x[ti], hs[ti], mk, color="0.25", ms=7, mfc="none", mew=1.6, zorder=5)
    ax[0].set(xlabel=r"$x$ [$\mu$m]", ylabel=r"$h=2f_1-1$")
    ax[0].set_title("profiles (markers = kernel targets)", fontsize=10, loc="left")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].grid(alpha=0.25, lw=0.6)

    ax[1].axhline(0.0, color="0.6", lw=0.9, ls=":")
    ax[1].plot(x[live], resid[live], color=C2, lw=1.8)
    ax[1].set(xlabel=r"$x$ [$\mu$m]", ylabel=r"$h_{\rm recon}-h_{\rm sim}$")
    if h_solver is not None:
        ax[1].plot(x[live], (hs - h_solver)[live], color="0.35", lw=1.6, ls=(0, (1, 2)),
                   label="forward $-$ solver")
        ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title(f"identity residual max |.| = {np.abs(resid[live]).max():.2e}",
                    fontsize=10, loc="left")
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[1].grid(alpha=0.25, lw=0.6)

    fig.text(0.5, 0.01,
             r"$h_{\rm recon}$ vs $h_{\rm sim}$ is an identity (Thm 6.1) and tests the "
             r"$\varepsilon$-probe; the solver curves are the external check.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    path = os.path.join(outdir, "h_theory_vs_numeric.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def figures(fw, bw, p, outdir, labels):
    """Panels A-F.  Everything drawn against t must use the ANCESTRY time axis, not the step axis.

    `backward_lowmem` block-averages anc/Csel over `block` steps (x27 at N=2000, T=2000), so those
    arrays are (nb, N) while the trajectory is (n_steps, N).  Three things follow, and getting any
    of them wrong is silent rather than loud:
      * the time axis is per BLOCK -- t_a below, the block's right edge, which reduces exactly to
        the old (m+1)*dt when block=1 (the matrix path), so both paths draw the same picture;
      * the occupancy mask is per step and must be reduced the same way (a block counts as
        occupied if ANY of its steps was);
      * Csel is a block MEAN, so a cumulative sum over blocks needs the block WIDTH back --
        without it panel (E) reads low by a factor of `block`.
    """
    x, dt, n_steps = fw["x"], fw["dt"], fw["n_steps"]
    t = (np.arange(n_steps) + 1) * dt
    nstar = float(C.monod_crossover(p))
    xf = np.array([D.nstar_front(x, fw["NPRE"][mm], nstar) for mm in range(n_steps)])
    targets = bw["targets"]

    blk = max(1, int(bw.get("block", 1)))
    nb = int(np.asarray(bw["anc"]).shape[1])
    lo = np.arange(nb) * blk
    hi = np.minimum(lo + blk, n_steps)
    cnt = (hi - lo).astype(float)              # ragged last block
    t_a = hi * dt                              # block right edge; == (m+1)*dt when blk == 1
    xf_a = np.add.reduceat(xf, lo) / cnt
    occ_yt = fw["RHO_R"] > 1e-6 * np.maximum(fw["RHO_R"].max(axis=1, keepdims=True), 1e-30)
    occ_a = np.logical_or.reduceat(occ_yt, lo, axis=0)

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
                      extent=[t_a[0], t_a[-1], x[0], x[-1]])
        a.fill_between(t_a, ybar - ysd, ybar + ysd, color="lime", alpha=.18, lw=0)
        a.plot(t_a, ybar, color="lime", lw=2.0, label=r"mean ancestry $\bar y(t)\pm\sigma$")
        a.plot(t_a, xf_a, color="c", lw=1.2, ls="--", label=r"$n^\ast$ front")
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
        # (1) MASK the unoccupied region.  Off the support rho -> 0, f is set by the empty-cell
        # convention f = 1/2 and label transport is undefined, so C there is not a contribution
        # but a convention.  Blanking it is honest; reweighting by rho would not be, because it
        # would break the identity sum C = h - B0 that justifies the panel.
        Cm = np.where(occ_a, Csel, np.nan)
        Cd = _decimate_nan(Cm)

        # (2) SCALE the two signs INDEPENDENTLY.  A symmetric range set by the dominant channel
        # makes a modest positive contribution saturate to full blue: at n0=1000 the deep target
        # has A- ~ 18x A+, so symmetric scaling reads as far more strain-1 input than exists.
        pos = Cd[np.isfinite(Cd) & (Cd > 0)]
        neg = Cd[np.isfinite(Cd) & (Cd < 0)]
        vp = float(np.percentile(pos, 99.5)) if pos.size else 0.0
        vn = float(np.percentile(-neg, 99.5)) if neg.size else 0.0
        vp = vp or (vn or 1.0); vn = vn or vp
        cmap = plt.get_cmap("coolwarm_r").copy()
        # Unoccupied must NOT look like zero.  coolwarm_r's midpoint is a pale grey, so a light
        # "bad" colour is indistinguishable from "occupied but contributing nothing" -- which is
        # the overwhelmingly common case here (median |C| over occupied cells is exactly 0, since
        # s -> 0 deep and (1-h^2) -> 0 wherever one strain dominates).  A dark bad colour keeps
        # the two readable apart; confusing them invites the conclusion that a jammed, fully
        # occupied cavity was empty.
        cmap.set_bad("#4d4d4d")
        im = a.imshow(np.clip(Cd, -vn, vp).T, origin="lower", aspect="auto", cmap=cmap,
                      norm=TwoSlopeNorm(0.0, -vn, vp), extent=[t_a[0], t_a[-1], x[0], x[-1]])
        a.plot(t_a, xf_a, color="k", lw=1.0, ls="--")
        a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"$y$ [$\mu$m]")
        a.set_title(rf"({'BD'[k]}) {lab}: contribution $C=K\,\Delta_{{\rm sel}}$ "
                    rf"(blue $=$ added strain 1)   $+${vp:.1e} / $-${vn:.1e}, grey $=$ empty",
                    fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=a)

    a = ax[2, 0]
    for k, lab in enumerate(labels):
        # Csel is a block MEAN; x cnt recovers the per-step sum, so the curve still ends at h - B0
        a.plot(t_a, np.cumsum(bw["Csel"][k].sum(axis=1) * cnt), color=tone[k], lw=1.8, label=lab)
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
def run(n0, dx, dt, T, outdir=None, mouth_x=None, deep_x=None, max_n=MAX_N_DEFAULT, p=None,
        L=None, targets_mode="auto", compare_run=None, refine=2.0, force=False,
        rho_seed=None):
    from dataclasses import replace as _replace
    p = p if p is not None else config.params(n0)
    L = float(config.L if L is None else L)
    num = config.numerics(T_end=T, dt=dt, dx=dx, n_save=2)
    ic = config.initial_conditions()
    if rho_seed is not None:
        ic = _replace(ic, rho_seed=rho_seed)
    N = C.choose_grid_size(L, num)
    if N > max_n:
        sys.exit(f"N={N} exceeds --max-n={max_n}. Above N={MATRIX_MAX_N} the backward sweep is "
                 f"matrix-free and costs O(N) per step, so the limit is now MEMORY: forward() "
                 f"stores 5 arrays of (n_steps, N) doubles "
                 f"(~{5*8*N*int(math.ceil(T/dt))/1e9:.1f} GB here). Raise --max-n deliberately, "
                 f"or shorten T.")

    tag = f"ancestry_n0-{n0:g}_L{L:g}_dx{dx:g}_T{T:g}"
    if targets_mode == "inverse":
        tag += "_inverse"
    if mouth_x is not None:
        tag += f"_m{mouth_x:g}"
    if deep_x is not None:
        tag += f"_d{deep_x:g}"
    # tag only when overridden, so existing default-seed runs keep the names they have and a
    # seed sweep cannot overwrite itself
    if rho_seed is not None:
        tag += f"_seed{rho_seed:g}"
    outdir = outdir or os.path.join(HERE, tag)
    os.makedirs(outdir, exist_ok=True)
    print(f"n0={n0:g} uM  L={L:g} um  dx={dx:g} um  dt={dt:g}  T={T:g} h  N={N}  "
          f"n*={config.NSTAR:.4f} uM  rho_seed={ic.rho_seed:g}  targets={targets_mode}")

    t0 = time.time()
    fw = forward(L, p, num, ic)
    print(f"  forward ({fw['n_steps']} steps) in {time.time()-t0:.1f}s")

    # The mode picks defaults; --mouth-x / --deep-x override either one independently.  An
    # override wins over `--inverse` rather than conflicting with it: the mode is a way of
    # choosing targets, an explicit coordinate is a stronger statement of the same intent.
    if refine and refine > 1.0:
        t0 = time.time()
        cv = convergence_check(L, p, num, ic, fw, refine=refine)
        print(f"  convergence (dx {cv['dx_coarse']:g} vs {cv['dx_fine']:g}, "
              f"N {cv['N_coarse']} vs {cv['N_fine']}) in {time.time()-t0:.1f}s")
        print(f"    min f1 {cv['min_f1_coarse']:.3e} -> {cv['min_f1_fine']:.3e}   "
              f"max f1 {cv['max_f1_coarse']:.6f} -> {cv['max_f1_fine']:.6f}")
        print(f"    purity ratio = {cv['purity_ratio']:.1f}x    "
              f"max|dh| = {cv['dh_max']:.4f}   biomass-weighted |dh| = {cv['dh_weighted']:.4f}")
        if cv["purity_ratio"] > PURITY_TOL:
            msg = (f"TRAJECTORY NOT CONVERGED: the minority strain fraction moves "
                   f"{cv['purity_ratio']:.1f}x when dx is refined {refine:g}x, so the kernel "
                   f"would be built on numerical diffusion rather than on the physics.")
            # Say which statistic caught it, from the numbers -- do not assert a narrative.
            if cv["dh_max"] < 0.1:
                msg += (f" Note that max|dh| is only {cv['dh_max']:.3f}: the absolute error in h "
                        f"looks harmless while the minority fraction is orders out, which is "
                        f"exactly the failure an |dh| test cannot see.")
            else:
                msg += (f" Here max|dh|={cv['dh_max']:.3f} is large too, so both statistics "
                        f"agree; the purity ratio is the one that also catches the case where "
                        f"|dh| stays small.")
            msg += (" Refine dx, or pass --no-convergence to proceed knowing the result is not "
                    "trustworthy.")
            if force:
                print("  !! " + msg)
            else:
                sys.exit("  FAIL: " + msg)
        else:
            print(f"    OK: minority fraction stable to {PURITY_TOL:g}x under refinement")
    else:
        print("  !! convergence check SKIPPED -- the Duhamel residual and the solver comparison "
              "cannot detect an unresolved trajectory; this result is unvalidated")

    mouth, deep, deep_inv = pick_targets(fw, p, mode=targets_mode)
    src = {"mouth": targets_mode, "deep": targets_mode}
    if mouth_x is not None:
        mouth = int(np.argmin(np.abs(fw["x"] - mouth_x))); src["mouth"] = "manual"
    if deep_x is not None:
        deep = int(np.argmin(np.abs(fw["x"] - deep_x))); src["deep"] = "manual"
        # deep_inv described the mode's choice, not this one; re-evaluate at the cell asked for
        nstar = float(C.monod_crossover(p))
        deep_inv = bool(fw["n"][deep] < nstar and fw["h_sim"][deep] < 0.0)
    if mouth == deep:
        sys.exit(f"both targets resolve to the same cell (x={fw['x'][mouth]:.2f} um); the two "
                 f"kernel rows would be identical. Separate them by at least dx={fw['dx']:g}.")

    targets = [mouth, deep]
    dlab = "deep" if deep_inv else "deep (NO inverse cells: normal-sorted)"
    print(f"  targets: mouth x={fw['x'][mouth]:.1f} (h={fw['h_sim'][mouth]:+.4f}) "
          f"[{src['mouth']}], "
          f"{dlab} x={fw['x'][deep]:.1f} (h={fw['h_sim'][deep]:+.4f}) [{src['deep']}]")
    for lab, xs, ti in (("mouth", mouth_x, mouth), ("deep", deep_x, deep)):
        if xs is not None and abs(fw["x"][ti] - xs) > 0.5 * fw["dx"] + 1e-9:
            print(f"    !! {lab} target snapped from x={xs:g} to the nearest cell centre "
                  f"x={fw['x'][ti]:.2f} -- requested position is outside [0, L]")
    if not deep_inv:
        print("    !! no cell has n<n* AND h<0 at T -- the graveyard is gone; row 2 is NOT a "
              "graveyard target")

    t1 = time.time()
    # Below the crossover the matrix path wins (one BLAS matmul beats a Python step loop);
    # above it, O(N^3) composition is hopeless and the matrix-free path is 51x faster at N=2000.
    if N <= MATRIX_MAX_N:
        bw = backward(fw, p, targets)
        how = f"matrix (N={N} <= {MATRIX_MAX_N})"
    else:
        bw = backward_lowmem(fw, p, targets)
        how = f"matrix-free (N={N} > {MATRIX_MAX_N}, anc block-averaged x{bw['block']})"
    print(f"  backward + kernel in {time.time()-t1:.1f}s  [{how}]")

    gm = np.abs(fw["h_sim"]) > 1e-9
    err = float(np.max(np.abs(bw["h_recon"] - fw["h_sim"])))
    corr = (float(np.corrcoef(bw["h_recon"][gm], fw["h_sim"][gm])[0, 1]) if gm.sum() > 2
            else float("nan"))
    ident = float(np.max(np.abs(bw["sigma"] * fw["h_sim"]
                                - (bw["B0"] + bw["A_plus"] - bw["A_minus"]))))
    rho = fw["rho1"] + fw["rho2"]
    w = rho / max(rho.sum(), 1e-300)
    err_w = float((w * np.abs(bw["h_recon"] - fw["h_sim"])).sum())
    print(f"  Duhamel   : max|h_recon - h_sim| = {err:.3e}   corr = {corr:.6f}   "
          f"biomass-weighted = {err_w:.3e}")
    print(f"  identity  : sigma_T h = B0 + (A+ - A-)   max err = {ident:.3e}")
    # The Duhamel residual is the master check: it is the only thing that notices if P_m stops
    # describing the scheme forward() actually ran.  Printing it was not enough -- a mismatched
    # transport operator still yields a row-stochastic P_m, so the row-sum guard below stays
    # silent.  Enforce it.
    # Gate on the BIOMASS-WEIGHTED residual, not the raw max-norm: off the support rho -> 0, f
    # is the empty-cell convention 1/2 and the label transport is undefined, so the raw norm is
    # O(1) by construction (run_checks() measures raw=3.979e-01 against weighted=7.392e-07).
    if err_w > DUHAMEL_TOL:
        msg = (f"Duhamel identity failed: biomass-weighted |h_recon - h_sim| = "
               f"{err_w:.3e} > {DUHAMEL_TOL:.0e}. P_m does not describe the scheme "
               f"forward() ran -- check the transport operator, not the physics.")
        if force:
            print("  !! " + msg)
        else:
            sys.exit("  FAIL: " + msg)
    rs = bw["anc"].sum(axis=2)
    print(f"  row sums  : sum_y K in [{rs.min():.8f}, {rs.max():.8f}]  (must be 1)")
    if abs(rs.min() - 1) > 1e-4 or abs(rs.max() - 1) > 1e-4:
        print("    !! WARNING: kernel not row-stochastic -- do not trust K or ybar")
    print(f"  max|B0| = {np.max(np.abs(bw['B0'])):.3e}  (exactly 0 for seed_f1=0.5)")
    for ti, lab in zip(targets, ("mouth", "deep")):
        print(f"    {lab:6s} x={fw['x'][ti]:7.1f}: A+={bw['A_plus'][ti]:.4g}  "
              f"A-={bw['A_minus'][ti]:.4g}  B0={bw['B0'][ti]:.3g}  "
              f"inverse={bool(bw['A_minus'][ti] > bw['A_plus'][ti] + bw['B0'][ti])}")

    h_solver, _ = solver_reference(L, p, num, ic)
    d_solver = float(np.max(np.abs(fw["h_sim"] - h_solver)))
    rho_w = fw["rho1"] + fw["rho2"]; rho_w = rho_w / max(rho_w.sum(), 1e-300)
    dw_solver = float((rho_w * np.abs(fw["h_sim"] - h_solver)).sum())
    tag_run = f"run_n0-{n0:g}_L{L:g}_dx{dx:g}_T{T:g}"
    xc, hc = run_csv_reference(compare_run or os.path.join(HERE, tag_run))
    print(f"  vs solver.simulate : max|dh| = {d_solver:.3e}   biomass-weighted = {dw_solver:.3e}"
          f"   {'IDENTICAL' if d_solver == 0.0 else 'DIFFERS'}")
    if xc is None:
        print(f"  vs run CSV        : no profile_final.csv at {os.path.basename(tag_run)} "
              f"-- run figures.py at these parameters for an on-disk reference")
    else:
        hci = np.interp(fw["x"], xc, hc)
        print(f"  vs run CSV        : max|dh| = {np.nanmax(np.abs(fw['h_sim']-hci)):.3e}"
              f"   ({os.path.basename(compare_run or tag_run)}, {len(xc)} cells)")
    path = figures(fw, bw, p, outdir, labels=("mouth", "deep" if deep_inv else "deep (normal)"))
    path_h = figure_h_compare(fw, bw, p, outdir, h_solver=h_solver,
                              h_csv=None if xc is None else (xc, hc),
                              csv_label=os.path.basename(compare_run or tag_run))
    with open(os.path.join(outdir, "exposure.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["x_um", "h_sim", "h_recon", "sigma_T", "A_plus", "A_minus", "B0", "inverse"])
        for i in range(fw["N"]):
            inv = bw["A_minus"][i] > bw["A_plus"][i] + bw["B0"][i]
            w_.writerow([f"{fw['x'][i]:.4g}", f"{fw['h_sim'][i]:.8g}", f"{bw['h_recon'][i]:.8g}",
                         f"{bw['sigma'][i]:.0f}", f"{bw['A_plus'][i]:.8g}",
                         f"{bw['A_minus'][i]:.8g}", f"{bw['B0'][i]:.8g}", int(inv)])
    # The C field costs ~80 min to produce at N=2000 and is discarded once figures() returns,
    # so any change of PRESENTATION would mean re-running the physics.  Cache it instead:
    # float32 is plenty for plotting (the field itself carries ~1e-8 of eps-probe error) and
    # halves a 95 MB file.  `plot_contributions.py` reads this and simulates nothing.
    blk = int(bw.get("block", 1))
    nb = int(np.asarray(bw["anc"]).shape[1])
    lo = np.arange(nb) * blk
    hi = np.minimum(lo + blk, fw["n_steps"])
    nstar = float(C.monod_crossover(p))
    xf = np.array([D.nstar_front(fw["x"], fw["NPRE"][mm], nstar) for mm in range(fw["n_steps"])])
    occ = fw["RHO_R"] > 1e-6 * np.maximum(fw["RHO_R"].max(axis=1, keepdims=True), 1e-30)
    fields = os.path.join(outdir, "fields.npz")
    np.savez_compressed(
        fields,
        anc=np.asarray(bw["anc"], dtype=np.float32),
        Csel=np.asarray(bw["Csel"], dtype=np.float32),
        occ=np.logical_or.reduceat(occ, lo, axis=0),
        xf=np.add.reduceat(xf, lo) / (hi - lo),
        x=fw["x"], t_lo=lo * fw["dt"], t_hi=hi * fw["dt"], cnt=(hi - lo).astype(float),
        targets=np.asarray(bw["targets"]), block=blk,
        h_sim=fw["h_sim"], B0=bw["B0"], sigma=bw["sigma"],
        A_plus=bw["A_plus"], A_minus=bw["A_minus"],
        n0=float(p.n0), nstar=nstar, L=float(L), dx=fw["dx"], dt=fw["dt"], T=float(T))
    print(f"  wrote {path}, {os.path.basename(path_h)}, exposure.csv and "
          f"fields.npz ({os.path.getsize(fields)/1e6:.0f} MB)")
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

    print("\n=== matrix-free transport: adjoint identity and agreement with the matrix ===")
    print("  <P v, w> == <v, P^T w> is exact and needs no reference; the matrix comparison is")
    print("  limited by the eps-probe, whose error scales as 1/eps (about 6e-9 at eps=1e-7).")
    rng = np.random.default_rng(0)
    wa = wt = wd = 0.0
    for m in (0, fw["n_steps"] // 2, fw["n_steps"] - 1):
        ar = (fw["RHO_R"][m], fw["RHO_NEW"][m], fw["JTOT"][m], p, num, fw["dt"], fw["dx"], fw["N"])
        P = transport_matrix(*ar)
        v = rng.standard_normal(fw["N"]); w = rng.standard_normal(fw["N"])
        wa = max(wa, float(np.max(np.abs(P @ v - apply_P(v, *ar)))))
        wt = max(wt, float(np.max(np.abs(P.T @ w - apply_PT(w, *ar)))))
        wd = max(wd, abs(float(np.dot(apply_P(v, *ar), w) - np.dot(v, apply_PT(w, *ar)))))
    good = (wd < 1e-12 and wa < 1e-6 and wt < 1e-6)
    print(f"  adjoint identity  max|<Pv,w>-<v,P^Tw>| = {wd:.3e}   (must be ~1e-15)")
    print(f"  vs matrix         apply {wa:.3e}   adjoint {wt:.3e}   (eps-probe floor ~1e-8)")
    print(f"  {'PASS' if good else 'FAIL'}")
    ok &= good

    print("\n=== the two backward paths agree ===")
    b_mat = backward(fw, p, [3, 20])
    b_free = backward_lowmem(fw, p, [3, 20], frames=fw["n_steps"])
    worst = 0.0
    for key in ("h_recon", "h_sel", "A_plus", "A_minus", "B0", "anc", "Csel"):
        d = float(np.max(np.abs(np.asarray(b_mat[key]) - np.asarray(b_free[key]))))
        worst = max(worst, d)
        print(f"  {key:>9}: max|diff| = {d:.3e}")
    print(f"  {'PASS' if worst < 1e-5 else 'FAIL'}  (the gap is the eps-probe's error, not the free path's)")
    ok &= worst < 1e-5

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
    ap.add_argument("--L", type=float, default=None,
                    help=f"cavity length in um (default config.L = {config.L:g})")
    ap.add_argument("--inverse", action="store_true",
                    help="probe FIXED locations instead of biomass-chosen ones: mouth = last "
                         "cell above n* (just toward the opening), deep = 0.9 L. Makes different "
                         "cavity lengths comparable; does not guarantee either is inverse-sorted.")
    ap.add_argument("--mouth-x", "--target-x", dest="mouth_x", type=float, default=None,
                    help="mouth target position in um (snapped to the nearest cell centre); "
                         "overrides whatever --inverse or the default picker chose")
    ap.add_argument("--deep-x", dest="deep_x", type=float, default=None,
                    help="deep target position in um; same override semantics. Unlike the "
                         "default picker this does NOT search for the graveyard, so the cell "
                         "asked for may be normally sorted -- that is reported, not corrected.")
    ap.add_argument("--refine", type=float, default=2.0,
                    help="dx refinement factor for the mandatory convergence check (default 2)")
    ap.add_argument("--no-convergence", dest="no_conv", action="store_true",
                    help="skip the convergence check -- the result is then unvalidated")
    ap.add_argument("--force", action="store_true",
                    help="warn instead of refusing when the trajectory is not converged")
    ap.add_argument("--compare-run", dest="compare_run", default=None,
                    help="directory of a completed figures.py run to compare against; by default "
                         "run_n0-<n0>_L<L>_dx<dx>_T<T> is looked for automatically")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--rho-seed", dest="rho_seed", type=float, default=None,
                    help="inoculum packing fraction (default from config, 1e-6).  The default "
                         "is BELOW ONE CELL for the fabricated geometry: rho_seed=1e-6 gives "
                         "0.03 cells at A=1000 um^2 (50x20 um) and V_cell=0.65 um^3, so use "
                         "1e-4 (3.45 cells) to match the physically seeded production runs.  "
                         "Tags the output directory when set.")
    ap.add_argument("--max-n", dest="max_n", type=int, default=MAX_N_DEFAULT,
                    help=f"refuse above this N (default {MAX_N_DEFAULT}); above N={MATRIX_MAX_N} the "
                         f"sweep is matrix-free, so this guards MEMORY, not cost")
    config.add_model_args(ap)
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if run_checks() else 1)
    run(a.n0, a.dx, a.dt, a.T, outdir=a.outdir, mouth_x=a.mouth_x, deep_x=a.deep_x,
        max_n=a.max_n, p=config.params_from_args(a, a.n0), L=a.L,
        targets_mode="inverse" if a.inverse else "auto", compare_run=a.compare_run,
        refine=0.0 if a.no_conv else a.refine, force=a.force, rho_seed=a.rho_seed)


if __name__ == "__main__":
    main()
