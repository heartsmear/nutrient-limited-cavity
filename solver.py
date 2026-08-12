"""
solver.py -- the ONLY time-stepping loop for the E. coli / glucose cavity.

Clean re-implementation of `src/solver.py` + `src/chemotaxis_cavity.simulate_chemo`, which
duplicated each other.  That duplication is why guards added to one never protected the other:
`simulate_chemo` was the path every grounded run took, and it called `strain_diagnostics` zero
times and checked no invariant at all.

One step, in order (first-order Lie splitting):
  1+2. reaction + nutrient diffusion, JOINTLY IMPLICIT (damped Newton).  Splitting these two was
       an O(dt) error large enough to have produced a spurious published result.
  3.   total-biomass collective diffusion: backward Euler, Newton + Armijo, D_col at the NEW
       density.  Jamming is carried by D_col alone -- no pressure projection, no rho_jam clip.
  4.   strain advection by the realized total flux: explicit donor-cell upwind, CFL-subcycled.
  5.   label diffusion of the fraction: backward Euler, direct solve.

Removed relative to src/: chemotaxis entirely, the pressure projection, the Picard
collective-diffusion fallback (verified identical to Newton to 8e-9), and the USE_NEWTON_DIFFUSION
switch.

Guards that are now invariants rather than hopes:
  * rho <= rho_max, enforced by closures.growth_space_factor and ASSERTED here.
  * dt admissible for the growth ceiling (check_growth_ceiling).
  * dx resolves the nutrient boundary layer -- reported, and part of `admissible`.
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from closures import (
    ModelParams, Numerics, InitialConditions,
    g1, g2, growth_space_factor, check_growth_ceiling, local_fraction_exact,
    collective_diffusivity, collective_diffusivity_deriv, self_diffusivity,
    choose_grid_size, cell_grid,
    build_diffusion_matrix_dirichlet_left_neumann_right,
    diffusive_flux_dirichlet_left_neumann_right,
    advect_strain_by_total_flux, implicit_self_diffusion_fraction,
    initial_total_density, initial_fraction, initial_nutrient, smoothing_matrix,
    validate_model_params,
)


# =============================================================================
# 1+2. Coupled reaction + nutrient diffusion
# =============================================================================
def coupled_reaction_nutrient(rho1, rho2, n, p: ModelParams, dt, B_n, b_n,
                              q=None, S=None, iters: int = 60, tol: float = 1e-12,
                              tol_relative: bool = True):
    """Solve local Monod growth and nutrient diffusion SIMULTANEOUSLY over one step.

        rho_i^new = rho_i exp(dt * q * g_i(n^new))
        (I - dt A_n) n^new = n^old + dt b_n - (1/Y)(rho^new - rho^old)

    Root:  F(n) = B_n n - n_old - dt b_n + (R(n) - rho_old)/Y,  R(n) = sum rho_i e^{dt q g_i(n)}
    Jacobian J = B_n + diag(R'(n)/Y);  damped Newton with n >= 0 and residual backtracking.

    `q` is the growth-space factor evaluated EXPLICITLY at rho^old, so it is a per-cell constant
    here and the Jacobian structure is untouched.  q is exactly 1.0 below rho_c, so sub-rho_c runs
    are bit-identical to a solver without this term.

    Growth is exact-exponential (the closed form at frozen n, unconditionally positive) and
    consumption is yield-exact (Delta rho / Y, not dt*(g1 rho1 + g2 rho2)/Y): the two agree as
    dt -> 0 but only the former conserves n + rho/Y to round-off at finite dt.
    """
    r1o = np.maximum(rho1, 0.0)
    r2o = np.maximum(rho2, 0.0)
    n_old = np.maximum(n, 0.0)
    rho_old = r1o + r2o
    if q is None:
        q = 1.0

    # Finite cell size: growth is evaluated at the nutrient a cell SENSES over its own body,
    # ntilde = S n, and the nutrient it consumes is drawn from that same body, S^T (drho)/Y.
    # S's rows sum to 1, so S^T conserves total consumption exactly.  S=None -> identity, and
    # everything below reduces to the point-cell arithmetic bit-for-bit.
    def sense(v):
        return v if S is None else S @ v

    def spread(v):
        return v if S is None else S.T @ v

    def R_and_dR(nc):
        nt = sense(nc)
        g1v = q * np.asarray(g1(nt, p))
        g2v = q * np.asarray(g2(nt, p))
        e1 = r1o * np.exp(dt * g1v)
        e2 = r2o * np.exp(dt * g2v)
        dg1 = q * p.mu1 * p.Ks1 / (p.Ks1 + nt) ** 2
        dg2 = q * p.mu2 * p.Ks2 / (p.Ks2 + nt) ** 2
        return e1, e2, e1 * dt * dg1 + e2 * dt * dg2      # dR/d(ntilde), diagonal in ntilde

    # The residual F has the units of the nutrient budget, whose natural magnitude is
    # |n| + max(rho)/Y.  At jamming rho/Y ~ 3.4e6, so an ABSOLUTE tol of 1e-12 is three orders
    # below double precision and can never be met: the solve reaches round-off in ~5 iterations
    # and then burns the remaining 55 every step.  (Measured: iters pinned at the 60 cap on 88%
    # of steps while the residual sat at 1.1e-18 RELATIVE.)  So tol is relative to that scale.
    scale = max(1.0, float(np.max(np.abs(n_old))) + float(np.max(rho_old)) / p.Y)
    tol_eff = tol * scale if tol_relative else tol

    nc = n_old.copy()
    it = 0
    Fprev = np.inf
    for it in range(1, iters + 1):
        e1, e2, dR = R_and_dR(nc)
        F = B_n @ nc - n_old - dt * b_n + spread(e1 + e2 - rho_old) / p.Y
        Fn0 = float(np.max(np.abs(F)))
        if Fn0 < tol_eff:
            break
        if tol_relative and it > 3 and Fn0 > 0.99 * Fprev:
            break                              # stagnated at round-off; more iterations are noise
        Fprev = Fn0
        Dd = diags(dR / p.Y, 0, format="csc")
        J = (B_n + (Dd if S is None else (S.T @ Dd @ S))).tocsc()   # chain rule through S
        delta = spsolve(J, -F)
        step = 1.0
        for _ls in range(40):                       # backtracking: keep n >= 0 and reduce |F|
            ntry = np.maximum(nc + step * delta, 0.0)
            e1t, e2t, _ = R_and_dR(ntry)
            Ft = B_n @ ntry - n_old - dt * b_n + spread(e1t + e2t - rho_old) / p.Y
            if float(np.max(np.abs(Ft))) <= Fn0:
                break
            step *= 0.5
        nc = np.maximum(nc + step * delta, 0.0)

    e1, e2, _ = R_and_dR(nc)
    budget = float(np.max(np.abs(B_n @ nc - n_old - dt * b_n
                                 + spread(e1 + e2 - rho_old) / p.Y)))
    return e1, e2, nc, {
        "reaction_budget_error_linf": budget,
        "reaction_newton_iters": float(it),
        "reaction_biomass_produced": float(np.sum((e1 + e2) - rho_old)),
    }


# =============================================================================
# 3. Collective diffusion of total biomass
# =============================================================================
def _dcol_coefficient_jacobian(rho, p: ModelParams, dx):
    """Tridiagonal B = d/drho of ( A(D_col(rho)) @ rho ) through the rho-dependent face
    diffusivities -- the product-rule partner of the frozen operator A.

    Interior face i+1/2 has D_f = (g_i + g_{i+1})/2 with g_k = D_col(rho_k), so contributions are
    local and B is tridiagonal.  The left boundary uses the Dirichlet ghost (D_face[0] = g_0
    DIRECTLY, factor 2 for the half cell -- not a 1/2 average); the right is zero-flux.
    """
    N = len(rho)
    inv2 = 1.0 / dx**2
    gp = np.asarray(collective_diffusivity_deriv(rho, p))
    Djr = rho[1:] - rho[:-1]
    lo = np.zeros(N); di = np.zeros(N); up = np.zeros(N)
    lo[1:] = -inv2 * 0.5 * gp[:-1] * Djr
    up[:-1] = inv2 * 0.5 * gp[1:] * Djr
    if N > 2:
        di[1:-1] = inv2 * 0.5 * gp[1:-1] * (Djr[1:] - Djr[:-1])
    di[0] = -inv2 * 2.0 * gp[0] * rho[0]
    if N > 1:
        di[0] += inv2 * 0.5 * gp[0] * Djr[0]
        di[-1] = -inv2 * 0.5 * gp[-1] * Djr[-1]
    return diags([lo[1:], di, up[:-1]], [-1, 0, 1], format="csc")


def nonlinear_collective_diffusion(rho_r, p: ModelParams, dt, dx,
                                   iters: int = 40, tol: float = 1e-10, lam_min: float = 1 / 64):
    """Backward Euler  F(rho) = rho - rho_r - dt A(D_col(rho)) rho = 0, by Newton + Armijo.

    D_col is iterated to the NEW density rather than frozen at rho_r: freezing is only first order
    in dt, harmless sub-jamming but not in the jammed phase where the post-jamming branch has
    slope ~1000.  The Armijo line search globalizes across the D_col kink at rho_jam.

    Returns (rho_new, D_col(rho_new), J_total, diag).  At convergence rho_new satisfies the
    implicit relation for that D_col, so rho_new - rho_r = -dt div J_total holds to tol -- which
    the strain advection downstream depends on.  Convergence is REPORTED, never assumed.
    """
    N = len(rho_r)
    I = diags(np.ones(N), 0, format="csc")

    def residual(r):
        Dc = np.asarray(collective_diffusivity(r, p))
        A, b = build_diffusion_matrix_dirichlet_left_neumann_right(Dc, dx, left_value=0.0)
        return r - rho_r - dt * (A @ r + b), A

    rho = np.maximum(rho_r, 0.0)
    F, A = residual(rho)
    nF = float(np.max(np.abs(F)))
    lam = 1.0
    it = 0
    for it in range(1, iters + 1):
        if nF < tol:
            break
        J = (I - dt * (A + _dcol_coefficient_jacobian(rho, p, dx))).tocsc()
        drho = spsolve(J, -F)
        lam = 1.0
        while lam >= lam_min:
            trial = np.maximum(rho + lam * drho, 0.0)
            Ft, At = residual(trial)
            nFt = float(np.max(np.abs(Ft)))
            if nFt < (1.0 - 1.0e-4 * lam) * nF:
                break
            lam *= 0.5
        rho, F, A, nF = trial, Ft, At, nFt

    Dc = np.asarray(collective_diffusivity(rho, p))
    J_total = diffusive_flux_dirichlet_left_neumann_right(rho, Dc, dx, left_value=0.0)
    return rho, Dc, J_total, {
        "diffusion_newton_iters": float(it),
        "diffusion_newton_resid": nF,
        "diffusion_converged": 1.0 if nF < tol else 0.0,
        "diffusion_newton_lambda": lam,
    }


# =============================================================================
# One step
# =============================================================================
def one_step(rho1, rho2, n, p: ModelParams, num: Numerics, dx, dt, B_n, b_n, S=None):
    """Advance the coupled PDE system by one dt.  Returns (rho1, rho2, n, J_total, diag)."""
    rho_old = np.maximum(rho1 + rho2, 0.0)
    total_in = float(np.sum(rho_old))
    pure1_in = total_in > p.tiny_density and np.max(np.maximum(rho2, 0.0)) <= p.tiny_density
    pure2_in = total_in > p.tiny_density and np.max(np.maximum(rho1, 0.0)) <= p.tiny_density
    resource_old = float(np.sum(n + rho_old / p.Y) * dx)

    # 1+2. growth cannot proceed into occupied space; q is EXACTLY 1 below rho_c
    q = growth_space_factor(rho_old, p)
    rho1_r, rho2_r, n_new, diag_react = coupled_reaction_nutrient(
        rho1, rho2, n, p, dt, B_n, b_n, q=q, S=S,
        tol_relative=num.reaction_tol_relative)
    n_new = np.maximum(n_new, 0.0)
    rho_r = rho1_r + rho2_r

    # 3. total biomass
    rho_new, Dc, J_total, diag_diff = nonlinear_collective_diffusion(rho_r, p, dt, dx)

    # 4/5. partition by the realized flux, then implicit relative diffusion
    rho1_adv, rho_adv_total, diag_adv = advect_strain_by_total_flux(
        rho1_r, rho_r, J_total, dt, dx, p, num)
    f_adv = local_fraction_exact(
        rho1_adv, np.maximum(rho_adv_total - rho1_adv, 0.0), tol=p.tiny_density)
    rho1_new, diag_self = implicit_self_diffusion_fraction(
        f_adv * rho_new, rho_new, p, dt, dx)

    if pure1_in:
        rho1_new = rho_new.copy()
    elif pure2_in:
        rho1_new = np.zeros_like(rho_new)
    rho1_new = np.minimum(np.maximum(rho1_new, 0.0), rho_new)
    rho2_new = rho_new - rho1_new

    diag: Dict[str, float] = {}
    diag.update(diag_react); diag.update(diag_diff)
    diag.update(diag_adv); diag.update(diag_self)
    diag.update({
        "resource_old": resource_old,
        "resource_new": float(np.sum(n_new + (rho1_new + rho2_new) / p.Y) * dx),
        "max_rho": float(np.max(rho_new)),
        "min_q": float(np.min(q)) if np.ndim(q) else float(q),
        "opening_biomass_flux": float(J_total[0]),
        "min_n": float(np.min(n_new)),
        "max_n": float(np.max(n_new)),
    })
    return rho1_new, rho2_new, n_new, J_total, diag


# =============================================================================
# Simulate
# =============================================================================
def simulate(L: float, p: ModelParams, num: Numerics, ic: InitialConditions,
             strict: bool = False, progress: bool = False) -> Dict[str, object]:
    """Advance to num.T_end, saving num.n_save frames.

    `strict=True` raises on a rho > rho_max violation instead of recording it.  With
    growth_space_factor active that violation should be unreachable, which makes it a genuine
    invariant test rather than a warning.
    """
    validate_model_params(p)
    N = choose_grid_size(L, num)
    x, dx = cell_grid(L, N)
    n_steps = int(math.ceil(num.T_end / num.dt))
    dt = num.T_end / n_steps
    dt_risky, rho_danger = check_growth_ceiling(p, dt)

    save_steps = np.unique(np.linspace(0, n_steps, num.n_save, dtype=int))
    save_index = {int(s): i for i, s in enumerate(save_steps)}
    n_saved = len(save_steps)

    rho0 = initial_total_density(x, L, p, ic)
    f0 = initial_fraction(x, ic)
    rho1, rho2 = f0 * rho0, (1.0 - f0) * rho0
    n = initial_nutrient(x, p, ic)

    A_n, b_n = build_diffusion_matrix_dirichlet_left_neumann_right(
        np.full(N, p.Dn), dx, left_value=p.n0)
    B_n = (diags(np.ones(N), 0, format="csc") - dt * A_n).tocsc()
    S = smoothing_matrix(N, dx, p)            # None when d_cell = 0 or ell <= dx

    R1 = np.zeros((N, n_saved)); R2 = np.zeros((N, n_saved))
    NN = np.zeros((N, n_saved)); tt = np.zeros(n_saved)
    scalar_hist: List[Dict[str, float]] = []

    rho_peak = float(np.max(rho1 + rho2))
    q_min = 1.0
    f_min, f_max = 1.0, 0.0            # positivity of the FRACTION: must stay in [0,1]
    adv_substeps_max = 1
    adv_capped = 0
    resource_drift = 0.0
    diffusion_failures = 0
    success, message = True, "completed"

    def save(step, t, d=None):
        i = save_index[step]
        R1[:, i] = rho1; R2[:, i] = rho2; NN[:, i] = n; tt[i] = t
        row = {"t_h": t, "rho_max": float(np.max(rho1 + rho2))}
        if d:
            row.update({k: float(v) for k, v in d.items()
                        if isinstance(v, (int, float, np.floating))})
        scalar_hist.append(row)

    save(0, 0.0)
    for step in range(1, n_steps + 1):
        t = step * dt
        try:
            rho1, rho2, n, J_total, diag = one_step(rho1, rho2, n, p, num, dx, dt, B_n, b_n, S=S)
        except Exception as exc:
            success, message = False, f"failed at step {step}, t={t:g} h: {exc}"
            break

        if not (np.all(np.isfinite(rho1)) and np.all(np.isfinite(rho2)) and np.all(np.isfinite(n))):
            success, message = False, f"non-finite value at step {step}, t={t:g} h"
            break

        rho_peak = max(rho_peak, float(diag["max_rho"]))
        # f in [0,1] is an invariant of the scheme; src/ tracked it and src2 had lost it.
        occ = (rho1 + rho2) > p.tiny_density
        if occ.any():
            fo = rho1[occ] / (rho1[occ] + rho2[occ])
            f_min = min(f_min, float(fo.min())); f_max = max(f_max, float(fo.max()))
        ns = int(diag.get("advection_substeps", 1))
        adv_substeps_max = max(adv_substeps_max, ns)
        adv_capped += int(ns >= num.max_advection_substeps)
        # Per-step version of the growth-ceiling condition: only a cell that STARTS in the danger
        # band can jump the ramp, so police membership of the band rather than refusing dt up front.
        if dt_risky and rho_peak > rho_danger and success:
            success, message = False, (
                f"dt={dt:g} cannot resolve the growth ramp at this density: rho={rho_peak:.5f} "
                f"exceeds rho_danger={rho_danger:.5f} at t={t:g} h, so a cell could step past "
                f"rho_max={p.rho_max:g}. Lower dt or rho_c.")
        q_min = min(q_min, float(diag["min_q"]))
        resource_drift = max(resource_drift,
                             abs(diag["resource_new"] - diag["resource_old"]))
        diffusion_failures += int(diag["diffusion_converged"] < 1.0)

        if rho_peak > p.rho_max * (1.0 + 1e-12):
            msg = (f"rho={rho_peak:.6f} exceeds rho_max={p.rho_max:g} at t={t:g} h -- the "
                   f"growth-space factor should make this unreachable")
            if strict:
                raise RuntimeError(msg)
            if success:
                success, message = False, msg

        if step in save_index:
            save(step, t, diag)
        if progress and step % max(1, n_steps // 20) == 0:
            # \r only on a terminal; piped/redirected output would otherwise be one long line
            end = "\r" if sys.stdout.isatty() else "\n"
            print(f"    t={t:8.2f} h  rho_max={rho_peak:.4f}  q_min={q_min:.4f}", end=end,
                  flush=True)

    if progress and sys.stdout.isatty():
        print()

    return {
        "x": x, "dx": dx, "t": tt, "N": N, "dt": dt, "L": L,
        "rho1_profiles": R1, "rho2_profiles": R2, "n_profiles": NN,
        "rho1_final": rho1, "rho2_final": rho2, "n_final": n,
        "history_rows": scalar_hist,
        # IC provenance, so summary.csv records WHICH run this was, not just its parameters.
        "rho_mode": ic.rho_mode, "fraction_mode": ic.fraction_mode,
        "nutrient_mode": ic.nutrient_mode, "seed_f1": ic.seed_f1,
        "reaction_tol_relative": num.reaction_tol_relative,
        "rho_peak": rho_peak, "q_min": q_min,
        "f_min": f_min, "f_max": f_max,
        "adv_substeps_max": adv_substeps_max, "adv_capped_steps": adv_capped,
        "max_resource_drift": resource_drift,
        "diffusion_failures": diffusion_failures,
        "success": success, "message": message,
        "ell": float(p.ell_factor * p.d_cell), "smoothing_cells": (0 if S is None
                                                                   else int(S.getnnz() / max(N, 1))),
        "params": p, "numerics": num, "initial_conditions": ic,
    }
