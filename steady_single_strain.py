"""
steady_single_strain.py -- direct Newton solve of the SINGLE-STRAIN STEADY cavity.

Read-only against the stack (imports closures/config, reimplements no closure physics).
Run with py -3.11.

    py -3.11 src2/steady_single_strain.py --regime dilute
    py -3.11 src2/steady_single_strain.py --regime jammed
    py -3.11 src2/steady_single_strain.py --regime dilute --continuation
    py -3.11 src2/steady_single_strain.py --budget          # jamming criterion A_eff > B
    py -3.11 src2/steady_single_strain.py --verify-lstar    # both L* formulas
    py -3.11 src2/steady_single_strain.py --fold            # the jamming fold + stability

WHAT IS SOLVED
--------------
The steady problem of reviewthis.tex (eqs steady_rho_new / steady_n_new / steady_bcs_new),
one strain, default strain 2 -- the "glutton" (mu = 0.78 1/h, Ks = 15 uM):

    d_x( D_col(rho) rho_x ) + g(n) rho = 0
    D_n n_xx                           = g(n) rho / Y
    rho(0) = 0,  n(0) = n0,  J_rho(L) = 0,  n_x(L) = 0

WHY THIS FILE EXISTS.  The chapter prescribes no numerics at all, and nothing in src2 solves
these equations directly: check_fold_*.py / check_Lstar_arccosh.py attack the steady state by
QUADRATURE after eliminating rho through the invariant, and solver.py is the two-strain
TIME-DEPENDENT code whose steady state costs T = 50000 h to reach.  This is the direct solve,
and it is independent of both.

THE DISCRETISATION, AND THE ONE REASON TO PREFER IT
---------------------------------------------------
Kirchhoff form: the biomass equation is discretised as the Laplacian of the potential
Psi(rho) = int_0^rho D_col, NOT as a face-averaged D_col gradient.  Both operators come from
the existing builder closures.build_diffusion_matrix_dirichlet_left_neumann_right, so the
boundary treatment is the stack's, not a private reimplementation:

    A_lap, _   = build(ones(N),      dx, left_value=0)     # d_xx on Psi;  Psi(0)=0 as rho(0)=0
    A_n,   b_n = build(full(N, Dn),  dx, left_value=n0)     # Dn d_xx on n

    F_rho = A_lap @ Psi(rho)        + q(rho) g(n) rho
    F_n   = Y * (A_n @ n + b_n)     - q(rho) g(n) rho       # scaled by Y: same units as F_rho

Because every face coefficient is constant, A_n = Dn * A_lap exactly, and A_lap @ (c*1) is
-2c/dx^2 * e_0, which is exactly -b_n/Y at c = Y*Dn*n0.  Therefore

    F_rho + F_n  =  A_lap @ ( Psi(rho) + Y*Dn*n - Y*Dn*n0 )

and A_lap (Dirichlet left, no-flux right) is nonsingular, so F = 0 FORCES the chapter's
integrated invariant Psi(rho) + Y*Dn*n = Y*Dn*n0 at every cell to linear-solve precision --
~1e-13, not merely to truncation error.  The central theorem becomes a free correctness check,
and it holds on either side of rho_jam.

Second payoff: dPsi/drho = D_col exactly, so the Jacobian needs only collective_diffusivity and
never its derivative.  The D_col kink at rho_jam -- the thing solver.py's Armijo search exists to
survive -- is integrated away inside Psi and never enters the Jacobian.

TRAPS -- all of these were hit while building this, none are hypothetical
-------------------------------------------------------------------------
  * The TRIVIAL state rho == 0, n == n0 solves the system at EVERY L.  Newton falls onto it from
    a weak guess.  Guarded: the solve reports branch="trivial" and fails rather than returning
    zeros, and refuses outright below L_onset, where the trivial state is the only one.
  * AN UNCONVERGED ITERATE LOOKS LIKE AN ANSWER.  Every seed is built from the invariant, so it
    already carries rho(L) = Psi^-1(A) and a machine-precision invariant residual.  A failed
    solve therefore reports the right closed-end density and passes checks 1, 4, 7 and 12.  The
    n0 = 1000 run did exactly that before the march existed.  main() now refuses to validate a
    non-converged result and says so in block capitals.
  * THE QUADRATURE SATURATES.  L(p*) diverges logarithmically as n(L) -> 0, and the log's
    argument is a difference of nearly equal tabulated values of S, so float64 floors it: L
    stops responding to p* beyond ~425 um (n0=15) / ~89 um (n0=1000).  Cancellation, not
    resolution -- refining the table does not move it.  check_fold_*.py is unaffected because
    its endpoint is JAMMING, where g(n_jam) > 0 and the sqrt singularity is genuine.  Any
    cross-check against the quadrature must sit at moderate p*/A (this file uses A/2), never at
    the production L and never at p* -> A.
  * THE RESIDUAL HAS A ROUND-OFF FLOOR THAT GRIDS MAKE WORSE.  F differences terms of size
    Y*Dn*n0/dx^2, so refining dx RAISES the smallest attainable residual.  At dx = 0.019 um the
    floor is ~1e-9 while the reaction scale is 0.5; a fixed relative tolerance of 1e-9 is then
    unreachable and a perfectly good solve reports failure.  Steady.tolerance() floors the
    threshold accordingly.
  * MOST OF THE NUTRIENT PROFILE IS NUMERICAL NOISE.  The continuum model reports n = 8e-68 uM
    without complaint.  Anything below NOISE_FLOOR_UM = 1e-12 uM is treated as noise and drawn
    as such: the log axes clip and shade there, `x_below_noise_floor_um` and
    `cavity_fraction_below_noise_floor` are reported, and check 5 is labelled a POSITIVITY test
    (the solver could have returned zero or negative and broken the maximum principle) rather
    than a physical confirmation of the tex's "n(L) > 0 at finite L".  Nothing reported depends
    on that region -- rho has saturated at Psi^-1(A) long before it.  --noise-floor to change.
    NB the floor is deliberately a NUMERICAL statement, not a molecule count: converting uM to
    molecules needs a chamber cross-section, which the 1D model does not have.
  * A GOOD R^2 DOES NOT MEAN THE RIGHT EXPONENT.  The fold window vanishes at onset as
    (n0-n0*)^(3/2) and the turning points separate as (n0-n0*)^(1/2) -- the cusp normal form.
    Fitting a single power law to the width over n0-n0* in [1, 40] returns alpha = 1.31 with
    R^2 = 0.998, which looks like a measurement and is not one: the exponent only emerges within
    ~1 uM of onset (1.487 at 0.13 uM, 1.498 at 0.33 uM, already 1.543 by 3.8 uM).  What decided
    the answer was the SAMPLING RANGE, not the fit quality.  fold_cusp() therefore reports LOCAL
    exponents and samples close in.
  * TURNING POINTS MUST BE ROOT-FOUND, NOT READ OFF SCAN NODES.  Taking the extremum of a scan
    locates it only to the node spacing, which leaves the window width wobbling in its fourth
    decimal -- enough to answer "does it fold", useless for measuring how the width vanishes.
    fold_window_refined() bounded-minimises +-L over the bracketing nodes and pins the window to
    ~1e-10: identical to 10 decimals across an 8x change in scan resolution.
  * NEWTON HAS NO OPINION ABOUT STABILITY.  Above n0 ~ 900 the branch FOLDS, and inside the
    fold window three steady states share one L.  Newton converges to whichever one the seed is
    nearest -- including the UNSTABLE middle branch -- so sweeping L up and then down and looking
    for a hysteresis loop demonstrates nothing: it is not a dynamical simulation, and the loop it
    draws is an artifact of the seeding.  What is legitimate is (i) parametrise the branch by
    RHO, where the quadrature supplies both the matching L and an exact seed, so each point of
    the S-curve is pinned deliberately, and (ii) classify each point by the rightmost eigenvalue
    of the linearisation.  `leading_eigenvalue` must use SHIFT-INVERT about 0: the spectrum is
    dominated by diffusive modes of order -D_n/dx^2, ~1e10 after the 1/Y rescaling, and ARPACK
    asked for "largest real part" simply fails to converge.
  * THE FOLD SCAN MUST BE UNIFORM IN RHO, NOT IN p* (or theta).  The turning points sit at
    rho ~ rho_jam, i.e. p* ~ B.  Once p_cap >> B they occupy a vanishing slice of the p*-range:
    at n0 = 3e4 the fold lives at theta = 0.974, so a scan over theta in [0.02, 0.97] never
    reaches it and reports "no fold" for a branch that plainly folds.  This produced a spurious
    "the fold disappears above n0 ~ 1e4" before it was caught.  check_fold_bistability.py records
    the same lesson for the same reason.
  * CONTINUATION MUST STEP IN (L - L_onset), NOT IN L.  Near onset delta_L ~ (L - L_onset), so
    a modest-looking 13% geometric step in L is an 8x jump in amplitude and Newton drops onto
    the trivial branch on step 2.  walk_branch() steps geometrically in (L - L_onset) and
    bisects on failure.
  * solver.py's Armijo (solver.py:199-206) assigns the trial OUTSIDE the while, so an exhausted
    line search silently accepts a rejected step and the residual can rise.  Not copied -- the
    accept here is inside the loop and exhaustion is reported as non-convergence.
  * D_self is COMPUTED AND PLOTTED BUT IS NOT A TERM IN THESE EQUATIONS.  It is the relative-label
    diffusivity and has no meaning for one strain; it is shown only as a diagnostic of the
    closure at the realised densities.
  * growth_space_factor q(rho) is NOT in reviewthis.tex, and is inert here anyway: q == 1 unless
    n0 > 53018 uM (= Psi(0.95)/(Y*Dn)).  Off by default, --growth-space to enable.
  * The master-curve check needs n << Ks as well as dilute.  The src2 regimes have n0/Ks2 = 1.0
    (n0=15) and 66.7 (n0=1000), so it is run at its OWN small n0 (--mc-n0) where the
    approximation it tests is actually valid.  Reporting it at n0=15 would be testing nothing.
    The same applies to the transcritical slope law, which additionally assumes D_col ~ D0:
    at n0 = 15, D_col is 3.0% above D0 at the realised density and the measured slope comes out
    2.9% low to match, while at n0 = 0.15 (D_col/D0 - 1 = 3e-4) it lands within 0.06%.  So the
    verdict is taken at --onset-n0 and the production-n0 value is recorded as check10c, a
    MEASUREMENT of the dilute approximation's error rather than a failure of the law.

WHAT THE VALIDATION ESTABLISHES
-------------------------------
The load-bearing evidence is not the invariant (which the discretisation makes exact by
construction and which every seed satisfies too) but the three genuinely independent
comparisons: solve_bvp on the invariant-reduced scalar ODE by adaptive collocation
(agrees to 5e-16 / 7e-15), the pressure-plane quadrature at p* = A/2 (8e-7 / 5e-6), and the
two-strain TIME-DEPENDENT solver marched to steady state (7.6e-4 at n0=15 with T=6000 h,
1.0e-6 at n0=1000 with T=50000 h).  Grid convergence is second order (measured 2.005) and
L_onset matches (pi/2)sqrt(D0/g(n0)) to 2.6e-6 with Da_crit = 2.4673884 against pi^2/4.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from scipy.integrate import cumulative_trapezoid, quad, solve_bvp      # noqa: E402
from scipy.integrate import IntegrationWarning                         # noqa: E402
from scipy.interpolate import PchipInterpolator                        # noqa: E402
from scipy.linalg import eigh_tridiagonal                              # noqa: E402
from scipy.optimize import brentq, minimize_scalar                     # noqa: E402
from scipy.sparse import bmat, diags                                   # noqa: E402
from scipy.sparse.linalg import eigs, spsolve                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import closures as C  # noqa: E402  (read-only import of the live stack)
import config         # noqa: E402

C1, C2, CN = "#1f77b4", "#d62728", "#2ca02c"

#: Figure suptitles duplicate the LaTeX caption in the thesis, so the Paper/ copies are rendered
#: without them.  --no-suptitle reproduces that instead of hand-editing the calls out.
SUPTITLE = True


def _suptitle(fig, *args, **kw):
    # was `_suptitle(fig, ...)` -- the fig.suptitle -> _suptitle replace also caught this line,
    # so every figure-writing run died in infinite recursion unless --no-suptitle was passed
    if SUPTITLE:
        fig.suptitle(*args, **kw)

TINY = 1.0e-300
RHO_HI = 1.0 - 1.0e-9          # Psi table domain is [0,1]; never step past it
ATANH_INV_SQRT3 = float(np.arctanh(1.0 / np.sqrt(3.0)))   # 0.658479, the sech^2 seed phase
NOISE_FLOOR_UM = 1.0e-12   # below this, n is numerical noise -- see the docstring note


# =============================================================================
# Strain selection and growth
# =============================================================================
def strain_kinetics(p: C.ModelParams, which: int = 2):
    """(mu, Ks, label) for the chosen strain.  Default 2 = the glutton, as requested."""
    if which == 1:
        return p.mu1, p.Ks1, "1 (scavenger)"
    if which == 2:
        return p.mu2, p.Ks2, "2 (glutton)"
    raise ValueError(f"strain must be 1 or 2, got {which}")


def g_of(n, mu: float, Ks: float):
    """Monod growth -- delegated to the stack so this file owns no kinetics."""
    return C.monod(n, mu, Ks)


def dg_of(n, mu: float, Ks: float):
    """g'(n) = mu Ks/(Ks+n)^2.  Same analytic form solver.py:94-96 uses."""
    nn = np.maximum(np.asarray(n, dtype=float), 0.0)
    return mu * Ks / (Ks + nn) ** 2


def q_and_dq(rho, p: C.ModelParams, enabled: bool):
    """growth_space_factor and its derivative.  The library has no derivative (closures.py:137)."""
    arr = np.asarray(rho, dtype=float)
    if not enabled:
        return np.ones_like(arr), np.zeros_like(arr)
    q = np.asarray(C.growth_space_factor(arr, p), dtype=float)
    span = p.rho_max - p.rho_c
    dq = np.where((arr > p.rho_c) & (arr < p.rho_max), -1.0 / span, 0.0)
    return q, dq


# =============================================================================
# Psi(rho), Psi^-1 and the budget constants
# =============================================================================
class Closure:
    """Psi, Psi^-1 and the A/B budget constants for one parameter set.

    Same construction as check_Lstar_arccosh.psi_tables / check_fold_bistability.Cavity --
    that table is currently duplicated four times across the check_* scripts and lives in
    closures.py nowhere, so it is copied rather than promoted, keeping this file read-only.
    """

    def __init__(self, p: C.ModelParams, n_grid: int = 200_001):
        self.p = p
        self.YDn = p.Y * p.Dn
        rho = np.linspace(0.0, 1.0, n_grid)
        psi = np.concatenate(([0.0], cumulative_trapezoid(C.collective_diffusivity(rho, p), rho)))
        self.psi = PchipInterpolator(rho, psi)
        self.psi_inv = PchipInterpolator(psi, rho)
        self.psi_max = float(psi[-1])                    # Psi(1)
        self.B = float(self.psi(p.rho_jam))              # endpoint pressure cost of jamming
        self.n0_jam = self.B / self.YDn                  # minimum feed that can reach rho_jam

    def A(self, n0: float) -> float:
        return self.YDn * n0

    def p_cap(self, n0: float) -> float:
        """Largest admissible p*: n cannot go negative and rho cannot exceed 1."""
        return min(self.A(n0), self.psi_max) * (1.0 - 1.0e-9)

    def rho_ceiling(self, n0: float) -> float:
        """Psi^-1(A) -- the tex's hard bound on rho for the whole branch."""
        return float(self.psi_inv(min(self.A(n0), self.psi_max)))


# =============================================================================
# The steady residual and its analytic Jacobian
# =============================================================================
class Steady:
    """Residual/Jacobian of the coupled steady system on a cell-centred grid."""

    def __init__(self, L: float, N: int, p: C.ModelParams, cl: Closure,
                 mu: float, Ks: float, use_q: bool = False):
        self.L, self.N, self.p, self.cl = float(L), int(N), p, cl
        self.mu, self.Ks, self.use_q = mu, Ks, use_q
        self.x, self.dx = C.cell_grid(L, N)
        # Psi(0) = 0 because rho(0) = 0, so the Dirichlet load of the biomass block vanishes.
        self.A_lap, _ = C.build_diffusion_matrix_dirichlet_left_neumann_right(
            np.ones(N), self.dx, left_value=0.0)
        self.A_n, self.b_n = C.build_diffusion_matrix_dirichlet_left_neumann_right(
            np.full(N, p.Dn), self.dx, left_value=p.n0)

    # ---- packing -----------------------------------------------------------
    def pack(self, rho, n):
        return np.concatenate([np.asarray(rho, float), np.asarray(n, float)])

    def unpack(self, u):
        return u[:self.N], u[self.N:]

    # ---- reaction term and its two derivatives -----------------------------
    def _reaction(self, rho, n):
        q, dq = q_and_dq(rho, self.p, self.use_q)
        g = np.asarray(g_of(n, self.mu, self.Ks), dtype=float)
        dg = dg_of(n, self.mu, self.Ks)
        R = q * g * rho
        dR_drho = g * (q + rho * dq)
        dR_dn = q * dg * rho
        return R, dR_drho, dR_dn

    # ---- residual ----------------------------------------------------------
    def residual(self, u):
        rho, n = self.unpack(u)
        R, _, _ = self._reaction(rho, n)
        F_rho = self.A_lap @ self.cl.psi(rho) + R
        F_n = self.p.Y * (self.A_n @ n + self.b_n) - R
        return np.concatenate([F_rho, F_n])

    # ---- Jacobian ----------------------------------------------------------
    def jacobian(self, u):
        rho, n = self.unpack(u)
        _, dR_drho, dR_dn = self._reaction(rho, n)
        Dcol = np.asarray(C.collective_diffusivity(rho, self.p), dtype=float)
        # d(A_lap @ Psi(rho))/drho = A_lap @ diag(Psi'(rho)) = A_lap @ diag(D_col) exactly.
        J11 = (self.A_lap @ diags(Dcol, 0, format="csc")) + diags(dR_drho, 0, format="csc")
        J12 = diags(dR_dn, 0, format="csc")
        J21 = diags(-dR_drho, 0, format="csc")
        J22 = self.p.Y * self.A_n - diags(dR_dn, 0, format="csc")
        return bmat([[J11, J12], [J21, J22]], format="csc")

    # ---- the residual scale the tolerance is measured against --------------
    def scale(self, u):
        """Physical size of the residual: the reaction term it has to balance."""
        rho, n = self.unpack(u)
        R, _, _ = self._reaction(rho, n)
        return max(float(np.max(np.abs(R))), self.mu * 1.0e-12)

    def tolerance(self, u, tol_rel: float):
        """Convergence threshold, floored by what float64 can actually deliver.

        F is a difference of terms of size |A_lap @ Psi| ~ max(Psi)/dx^2, so the residual cannot
        be driven below the round-off of THOSE terms however good the iterate is.  At dx = 0.019
        um that floor is ~1e-10 while the reaction scale is 0.5, so a fixed relative tolerance of
        1e-9 is unreachable and the solve would report failure at a perfectly converged state.
        Refining the grid tightens nothing here -- it raises the floor.
        """
        # Both differenced quantities -- Psi and Y*Dn*n -- are bounded by Y*Dn*n0 (that IS the
        # invariant), so the round-off of the second difference is eps * Y*Dn*n0 / dx^2.
        floor = 10.0 * np.finfo(float).eps * self.cl.YDn * self.p.n0 / self.dx ** 2
        return max(tol_rel * self.scale(u), floor)


# =============================================================================
# Damped Newton
# =============================================================================
def newton(st: Steady, u0, tol: float = 1.0e-9, iters: int = 200,
           tau: float = 0.99, lam_min: float = 1.0e-7, verbose: bool = False):
    """Newton with fraction-to-boundary then Armijo backtracking.  L-inf norm throughout.

    Fraction-to-boundary keeps rho in [0, RHO_HI] and n >= 0 by shortening the step rather than
    clamping the iterate, so the search direction survives intact (solver.py clamps, which can
    destroy descent).  Armijo accepts INSIDE the loop; exhaustion is reported, not swallowed.
    """
    u = np.array(u0, dtype=float)
    u[:st.N] = np.clip(u[:st.N], 0.0, RHO_HI)
    u[st.N:] = np.maximum(u[st.N:], 0.0)

    F = st.residual(u)
    nF = float(np.max(np.abs(F)))
    hist = [dict(iter=0, residual=nF, lam=np.nan, step=np.nan)]
    converged, message = False, "max iterations"

    for it in range(1, iters + 1):
        tol_eff = st.tolerance(u, tol)
        if nF <= tol_eff:
            converged, message = True, "converged"
            break

        du = spsolve(st.jacobian(u), -F)
        if not np.all(np.isfinite(du)):
            message = "singular Jacobian"
            break

        # --- fraction to the boundary: largest lam keeping rho in [0,RHO_HI], n >= 0 -------
        lam = 1.0
        drho, dn = du[:st.N], du[st.N:]
        rho, n = u[:st.N], u[st.N:]
        neg = drho < 0
        if np.any(neg):
            lam = min(lam, tau * float(np.min(rho[neg] / (-drho[neg]))))
        pos = drho > 0
        if np.any(pos):
            lam = min(lam, tau * float(np.min((RHO_HI - rho[pos]) / drho[pos])))
        negn = dn < 0
        if np.any(negn):
            lam = min(lam, tau * float(np.min(n[negn] / (-dn[negn]))))
        lam = max(lam, 0.0)

        # --- Armijo ------------------------------------------------------------------------
        accepted = False
        while lam >= lam_min:
            trial = u + lam * du
            Ft = st.residual(trial)
            nFt = float(np.max(np.abs(Ft)))
            if nFt < (1.0 - 1.0e-4 * lam) * nF:
                u, F, nF, accepted = trial, Ft, nFt, True
                break
            lam *= 0.5
        if not accepted:
            message = f"line search exhausted at iter {it} (resid {nF:.3e})"
            break

        hist.append(dict(iter=it, residual=nF, lam=lam,
                         step=float(np.max(np.abs(lam * du)))))
        if verbose:
            print(f"    newton {it:3d}  |F|={nF:.6e}  lam={lam:.4g}")
    else:
        message = "max iterations"

    if not converged and nF <= st.tolerance(u, tol):
        converged, message = True, "converged"
    return u, dict(converged=converged, message=message, iters=len(hist) - 1,
                   residual=nF, tol_eff=st.tolerance(u, tol), history=hist)


# =============================================================================
# Analytic scales from the chapter
# =============================================================================
def onset_length(p: C.ModelParams, mu: float, Ks: float) -> float:
    """L_onset = (pi/2) sqrt(D0/g(n0))  -- tex eq:L_onset_expanded."""
    g0 = float(g_of(p.n0, mu, Ks))
    return 0.5 * np.pi * np.sqrt(p.D0 / g0)


def onset_amplitude_slope(p: C.ModelParams, mu: float, Ks: float) -> float:
    """d(delta_L)/dL at onset = 3 g0 kappa0 / (2 g1)  -- tex eq:deltaL_linear_general."""
    g0 = float(g_of(p.n0, mu, Ks))
    g1 = float(dg_of(p.n0, mu, Ks))
    return 3.0 * g0 * np.sqrt(g0 / p.D0) / (2.0 * g1)


def dilute_seed_nu(xi, xi_L: float):
    """nu(xi) for nu'' = nu(1-nu), nu(0)=1, on a half-infinite cavity, plus its image.

    The infinite-domain solution is exactly nu = (3/2) sech^2(xi/2 + artanh(1/sqrt3)) -- check
    nu(0) = 1.5/cosh^2(0.658479) = 1.  Adding the mirror image about xi_L imposes nu'(xi_L) = 0
    approximately and supplies the factor 2 that makes the tail 3.21 e^-xi rather than 1.61 e^-xi.
    Used only as an initial guess, so approximate is fine.
    """
    def f(z):
        return 1.5 / np.cosh(0.5 * np.asarray(z, float) + ATANH_INV_SQRT3) ** 2
    return np.minimum(f(xi) + f(2.0 * xi_L - xi), 1.0)


# =============================================================================
# Initial guesses -- three tiers
# =============================================================================
def guess_dilute(st: Steady):
    """Tier 1: analytic sech^2 profile of the dilute, low-nutrient master curve."""
    p, cl = st.p, st.cl
    kappa0 = np.sqrt(float(g_of(p.n0, st.mu, st.Ks)) / p.D0)
    nu = dilute_seed_nu(kappa0 * st.x, kappa0 * st.L)
    n = p.n0 * nu
    rho = np.asarray(cl.psi_inv(np.clip(cl.YDn * (p.n0 - n), 0.0, cl.psi_max)), float)
    return st.pack(np.clip(rho, 0.0, RHO_HI), np.maximum(n, 0.0)), "dilute-analytic"


def _source_table(cl: Closure, n0: float, mu: float, Ks: float, p_hi: float, N: int):
    """S(p) = int_0^p g(n(r)) Psi^-1(r) dr.  Same construction as check_fold_bistability."""
    r = np.linspace(0.0, p_hi, N)
    rho_r = cl.psi_inv(r)
    g = C.monod(np.maximum(n0 - r / cl.YDn, 0.0), mu, Ks)
    S = np.concatenate(([0.0], cumulative_trapezoid(g * rho_r, r)))
    return PchipInterpolator(r, S)


def quad_length(S, p_star: float, limit: int = 300) -> float:
    """L(p*) = int_0^{p*} dp/q, q^2 = 2[S(p*)-S(p)].  s = p*-p = u^2 kills the singularity."""
    Sstar = float(S(p_star))

    def integrand(u):
        q2 = 2.0 * (Sstar - float(S(p_star - u * u)))
        return 0.0 if q2 <= 0.0 else 2.0 * u / np.sqrt(q2)

    # Near total depletion S'(p*) -> 0 and this integral diverges logarithmically; quad then
    # reports max-subdivisions.  That regime is measured and reported by check 6, not hidden,
    # so the warning itself is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(integrand, 0.0, np.sqrt(p_star), limit=limit)
    return float(val)


def quad_pstar_for_length(cl: Closure, n0: float, mu: float, Ks: float, L: float,
                          N: int = 400_001):
    """Invert L(p*) = L.  Returns (p_star, S).  L(p*) -> L_onset as p*->0 and -> inf at p_cap."""
    p_hi = cl.p_cap(n0)
    S = _source_table(cl, n0, mu, Ks, p_hi, N)
    lo, hi = p_hi * 1.0e-9, p_hi * (1.0 - 1.0e-12)
    f = lambda v: quad_length(S, v) - L      # noqa: E731
    if f(lo) > 0.0:
        return None, S                        # L below onset: no positive branch
    for _ in range(60):
        if f(hi) > 0.0:
            break
        hi = 0.5 * (hi + p_hi)
    else:
        return None, S
    return float(brentq(f, lo, hi, xtol=1.0e-14, rtol=1.0e-12)), S


def quad_profile(cl: Closure, n0: float, L: float, p_star: float, S, x, npts: int = 20_001):
    """Parametric steady profile from the pressure plane, evaluated on the grid x.

    Integrating from the CLOSED end outward in s = p*-p = u^2 keeps the integrand regular:
        distance-from-closed-end(p) = int_0^{sqrt(p*-p)} 2u du / sqrt(2[S(p*) - S(p*-u^2)])

    The curve is then anchored at the MOUTH, not at the closed end, and flat-extended at the
    ceiling beyond its own length.  That matters for seeding: once the nutrient is exhausted,
    lengthening the cavity only adds more material sitting at rho = Psi^-1(p*), so the mouth
    structure is what a longer cavity shares with this one.  Anchoring at the closed end instead
    would slide that structure inward and hand Newton a badly wrong guess.
    """
    Sstar = float(S(p_star))
    u = np.linspace(0.0, np.sqrt(p_star), npts)
    pp = p_star - u * u                       # p descending from p* (closed end) to 0 (mouth)
    q2 = np.maximum(2.0 * (Sstar - S(pp)), 0.0)
    integ = np.where(q2 > 0.0, 2.0 * u / np.sqrt(np.maximum(q2, TINY)), 0.0)
    dist = np.concatenate(([0.0], cumulative_trapezoid(integ, u)))     # from the closed end
    L_curve = float(dist[-1])
    s_mouth = L_curve - dist                  # distance from the mouth
    order = np.argsort(s_mouth)
    sm, pm = s_mouth[order], pp[order]
    # dist PLATEAUS wherever the integrand vanishes -- which it does over the depleted tail, so
    # s_mouth is only non-decreasing, and PCHIP demands strictly increasing.  Drop the repeats.
    keep = np.concatenate(([True], np.diff(sm) > 0.0))
    sm, pm = sm[keep], pm[keep]
    if sm.size < 2:
        rho0 = np.full_like(np.asarray(x, float), float(cl.psi_inv(p_star)))
        return np.clip(rho0, 0.0, RHO_HI), np.full_like(rho0, max(n0 - p_star / cl.YDn, 0.0))
    p_of_s = PchipInterpolator(sm, pm, extrapolate=False)
    xq = np.clip(np.asarray(x, float), float(sm[0]), float(sm[-1]))
    p_grid = np.clip(np.nan_to_num(p_of_s(xq), nan=p_star), 0.0, p_star)
    rho = np.asarray(cl.psi_inv(p_grid), float)
    nn = n0 - p_grid / cl.YDn
    return np.clip(rho, 0.0, RHO_HI), np.maximum(nn, 0.0)


def guess_quadrature(st: Steady, N_table: int = 400_001):
    """Tier 2: the pressure-plane parametric profile at the p* that gives this L.

    When L exceeds what the quadrature can represent -- it saturates near total depletion, see
    check 6 -- brentq finds no p*.  That is not a dead end: past saturation the nutrient IS
    exhausted, so the right seed is simply the most-depleted profile available, flat-extended.
    Without this fall-back the deep-jammed corner (large L, large n0) has no working seed at
    all, because the march cannot get there either: at high n0 the branch FOLDS, and stepping L
    upward has nowhere to go once it passes the turning point.
    """
    p_star, S = quad_pstar_for_length(st.cl, st.p.n0, st.mu, st.Ks, st.L, N_table)
    label = "quadrature"
    if p_star is None:
        p_star = st.cl.p_cap(st.p.n0) * (1.0 - 1.0e-12)
        label = "quadrature-saturated"
    rho, n = quad_profile(st.cl, st.p.n0, st.L, p_star, S, st.x)
    return st.pack(rho, n), label


def guess_onset(st: Steady, L_on: float):
    """Tier 3 seed: the transcritical near-onset mode rho ~ (Y Dn/D0) delta_L sin(kappa0 x)."""
    p, cl = st.p, st.cl
    kappa0 = np.sqrt(float(g_of(p.n0, st.mu, st.Ks)) / p.D0)
    delta_L = max(onset_amplitude_slope(p, st.mu, st.Ks) * (st.L - L_on), 1.0e-9 * p.n0)
    delta_L = min(delta_L, 0.9 * p.n0)
    delta = delta_L * np.sin(np.clip(kappa0 * st.x, 0.0, 0.5 * np.pi))
    n = p.n0 - delta
    rho = np.asarray(cl.psi_inv(np.clip(cl.YDn * delta, 0.0, cl.psi_max)), float)
    return st.pack(np.clip(rho, 0.0, RHO_HI), np.maximum(n, 0.0)), "near-onset"


# =============================================================================
# Solve
# =============================================================================
def _seed_from(prev, L: float, N: int):
    """Stretch a converged profile onto the grid of a longer cavity.

    The profile is anchored at the mouth and grows into the interior, so holding it fixed in x
    and flat-extending the closed-end value is a far better predictor than rescaling x.
    """
    xs_old, rho_old, n_old, L_old = prev
    x_new, _ = C.cell_grid(L, N)
    xq = np.minimum(x_new, L_old)
    return np.concatenate([
        np.clip(PchipInterpolator(xs_old, rho_old, extrapolate=True)(xq), 0.0, RHO_HI),
        np.maximum(PchipInterpolator(xs_old, n_old, extrapolate=True)(xq), 0.0)])


def walk_branch(N: int, p: C.ModelParams, cl: Closure, mu: float, Ks: float, L_target: float,
                use_q: bool = False, tol: float = 1.0e-9, steps: int = 40,
                L_min_factor: float = 1.02, max_refine: int = 6, verbose: bool = False):
    """Continuation along the colonised branch from just above onset up to L_target.

    Stepping is geometric in (L - L_onset), NOT in L.  That matters: the amplitude obeys
    delta_L ~ (L - L_onset) near onset, so geometric-in-L steps that look modest (13% in L)
    are an 8x jump in amplitude just above onset, and Newton falls straight back onto the
    trivial state.  Geometric in (L - L_onset) keeps the amplitude change per step bounded.

    On a failed step the interval to the last good L is bisected and retried, so a hard
    stretch costs extra solves rather than the whole walk.  Returns the converged states.
    """
    L_on = onset_length(p, mu, Ks)
    s_end = L_target - L_on
    if s_end <= 0:
        return []
    targets = L_on + np.geomspace(max((L_min_factor - 1.0) * L_on, s_end * 1e-6), s_end, steps)

    def attempt(Lk, prev):
        st = Steady(Lk, N, p, cl, mu, Ks, use_q)
        u0 = _seed_from(prev, Lk, N) if prev is not None else guess_onset(st, L_on)[0]
        u, info = newton(st, u0, tol=tol)
        rho, n = st.unpack(u)
        good = info["converged"] and float(np.max(rho)) > 1.0e-10 * cl.rho_ceiling(p.n0)
        return good, (st.x.copy(), rho.copy(), n.copy(), float(Lk)), info

    prev, rows = None, []
    for Lk in targets:
        good, state, info = attempt(Lk, prev)
        if not good and prev is not None:
            lo = prev[3]
            for _ in range(max_refine):                       # bisect toward the last good L
                mid = 0.5 * (lo + Lk)
                gmid, smid, _ = attempt(mid, prev)
                if gmid:
                    prev, lo = smid, mid
                    good, state, info = attempt(Lk, prev)
                    if good:
                        break
                else:
                    Lk_try = mid
                    if abs(Lk_try - lo) / max(lo, 1e-30) < 1e-6:
                        break
                    Lk = Lk_try
        if good:
            prev = state
        rows.append(dict(L=float(state[3]), rho_L=float(state[1][-1]), n_L=float(state[2][-1]),
                         rho_max=float(state[1].max()), p_star=float(cl.psi(state[1][-1])),
                         iters=int(info["iters"]), residual=float(info["residual"]),
                         converged=bool(good)))
        if verbose:
            print(f"  L={state[3]:9.3f}  rho(L)={rows[-1]['rho_L']:.6g}  "
                  f"iters={rows[-1]['iters']:3d}  {'ok' if good else 'FAILED'}")
    return rows, prev


def _march_seed(L: float, N: int, p: C.ModelParams, cl: Closure, mu: float, Ks: float,
                use_q: bool, tol: float, L_on: float, steps: int = 30, N_march: int = 400):
    """Tier 3: WALK L up from just above onset, reseeding each step, and return the profile.

    Needed whenever the cheap seeds miss.  At n0 = 1000 both do: the dilute profile assumes
    rho << rho_jam when rho reaches 0.66, and the quadrature seed does not exist at all because
    L(p*) saturates near total depletion (see check 6).  Marching is what always works, since
    each step is a small perturbation of the last.

    The march runs on a COARSE grid (N_march) and the result is interpolated onto the target
    grid -- the shape is what is wanted, and 30 solves at N=2000 would be wasted effort.
    """
    _, prev = walk_branch(min(N, N_march), p, cl, mu, Ks, L, use_q=use_q, tol=tol, steps=steps)
    if prev is None or prev[3] < 0.999 * L:
        return None
    return _seed_from(prev, L, N)


def solve_steady(L: float, N: int, p: C.ModelParams, cl: Closure, mu: float, Ks: float,
                 use_q: bool = False, tol: float = 1.0e-9, u0=None, seed_label: str = "",
                 verbose: bool = False, allow_march: bool = True):
    """One steady solve.  Tries the guess tiers in order and reports which one landed."""
    st = Steady(L, N, p, cl, mu, Ks, use_q)
    L_on = onset_length(p, mu, Ks)

    if u0 is not None:
        tiers = [(u0, seed_label or "supplied")]
    else:
        # Lazy: guess_quadrature builds a 400k-point table, so it is only paid for when the
        # cheap analytic seed has actually failed.
        tiers = [lambda: guess_dilute(st),
                 lambda: guess_quadrature(st),
                 lambda: guess_onset(st, L_on)]

    best = None
    for _tier in tiers:
        u_guess, label = _tier() if callable(_tier) else _tier
        if u_guess is None:
            continue
        u, info = newton(st, u_guess, tol=tol, verbose=verbose)
        rho, n = st.unpack(u)
        info["seed"] = label
        trivial = float(np.max(rho)) <= 1.0e-10 * max(cl.rho_ceiling(p.n0), 1.0e-30)
        if info["converged"] and not trivial:
            best = (u, info)
            break
        if best is None:
            best = (u, dict(info, branch="trivial" if trivial else "unconverged"))
    else:
        if allow_march and u0 is None and L > L_on:
            if verbose:
                print("    cheap seeds missed; marching from onset")
            u_m = _march_seed(L, N, p, cl, mu, Ks, use_q, tol, L_on)
            if u_m is not None:
                u, info = newton(st, u_m, tol=tol, verbose=verbose)
                info["seed"] = "march-from-onset"
                if info["converged"] and float(np.max(st.unpack(u)[0])) > 0.0:
                    best = (u, info)

    u, info = best
    rho, n = st.unpack(u)
    trivial = float(np.max(rho)) <= 1.0e-10 * max(cl.rho_ceiling(p.n0), 1.0e-30)
    info["branch"] = "trivial" if trivial else "colonised"
    info["below_onset"] = bool(L < L_on)
    return dict(st=st, x=st.x, dx=st.dx, N=N, L=L, rho=rho, n=n, p=p, cl=cl,
                mu=mu, Ks=Ks, use_q=use_q, L_onset=L_on, info=info)


def continuation(N: int, p: C.ModelParams, cl: Closure, mu: float, Ks: float, L_target: float,
                 L_min_factor: float = 1.02, steps: int = 40, use_q: bool = False,
                 tol: float = 1.0e-9, verbose: bool = False):
    """The colonised branch from just above L_onset up to L_target, for branch.csv/the figure."""
    rows, _ = walk_branch(N, p, cl, mu, Ks, L_target, use_q=use_q, tol=tol, steps=steps,
                          L_min_factor=L_min_factor, verbose=verbose)
    return rows


# =============================================================================
# Derived fields
# =============================================================================
def derived(res):
    """Everything plotted or written, reconstructed from (rho, n)."""
    p, cl, st = res["p"], res["cl"], res["st"]
    rho, n, dx = res["rho"], res["n"], res["dx"]
    Psi = np.asarray(cl.psi(rho), float)
    out = dict(
        Psi=Psi,
        Dcol=np.asarray(C.collective_diffusivity(rho, p), float),
        Dself=np.asarray(C.self_diffusivity(rho, p), float),
        g=np.asarray(g_of(n, res["mu"], res["Ks"]), float),
        q=q_and_dq(rho, p, res["use_q"])[0],
        # J_rho = -Psi_x and J_n = -Dn n_x on the N+1 faces, from the stack's flux helper so the
        # discretisation matches the operator exactly.
        J_rho=C.diffusive_flux_dirichlet_left_neumann_right(Psi, np.ones(st.N), dx, 0.0),
        J_n=C.diffusive_flux_dirichlet_left_neumann_right(n, np.full(st.N, p.Dn), dx, p.n0),
        invariant=Psi + cl.YDn * n - cl.YDn * p.n0,
    )
    out["biomass"] = float(np.sum(rho) * dx)
    out["noise_floor"] = float(res.get("noise_floor", NOISE_FLOOR_UM))
    below = np.flatnonzero(n < out["noise_floor"])
    out["x_noise"] = float(res["x"][below[0]]) if below.size else float("nan")
    out["frac_noise"] = float(below.size) / len(n)
    return out


def front_resolution(x, n, dx):
    """dx/lambda at the steepest nutrient decay -- the same convention as diagnostics.py."""
    nn = np.maximum(np.asarray(n, float), TINY)
    ratio = nn[:-1] / np.maximum(nn[1:], TINY)
    k = int(np.argmax(ratio))
    r = float(ratio[k])
    if not np.isfinite(r) or r <= 1.0:
        return float("nan"), float("nan"), r
    lam = dx / np.log(r)
    return lam, dx / lam, r


# =============================================================================
# Validation
# =============================================================================
def check_onset_eigenvalue(p: C.ModelParams, mu: float, Ks: float, N: int):
    """L_onset from the DISCRETE linearisation about the trivial state.

    The linearised operator is D0 d_xx + g(n0) with r(0)=0, r_x(L)=0 -- exactly A_lap of the
    solver.  Its eigenvalues scale as 1/dx^2 = N^2/L^2, so lambda_1 = 0 has the closed form
    L = N sqrt(-D0 e1 / g0) with e1 the largest eigenvalue of A_lap at dx = 1.  No bisection.
    """
    A, _ = C.build_diffusion_matrix_dirichlet_left_neumann_right(np.ones(N), 1.0, 0.0)
    d = A.diagonal(0).copy()
    e = A.diagonal(1).copy()
    e1 = float(eigh_tridiagonal(d, e, select="i", select_range=(N - 1, N - 1))[0][0])
    g0 = float(g_of(p.n0, mu, Ks))
    L_disc = N * np.sqrt(-p.D0 * e1 / g0)
    return L_disc, g0 * L_disc ** 2 / p.D0


def measure_onset_slope(p: C.ModelParams, cl: Closure, mu: float, Ks: float, N: int,
                        eps=(0.01, 0.005, 0.0025, 0.00125), use_q: bool = False):
    """d(delta_L)/dL at onset, extrapolated to L -> L_onset.

    delta_L/(L-L_onset) still drifts by several percent at eps = 0.01 because the law is only
    leading order, and the drift is close to linear in eps -- so the ratio is fitted against eps
    and the INTERCEPT is reported.  Returns (slope_extrapolated, slope_predicted, (dL, delta)).
    """
    L_on = onset_length(p, mu, Ks)
    s_pred = onset_amplitude_slope(p, mu, Ks)
    eps = np.asarray(eps, dtype=float)
    delta = []
    for e in eps:
        r = solve_steady(L_on * (1.0 + e), N, p, cl, mu, Ks, use_q=use_q)
        delta.append(p.n0 - float(r["n"][-1])
                     if r["info"]["branch"] == "colonised" else np.nan)
    delta = np.array(delta)
    m = np.isfinite(delta)
    s_ext = float("nan")
    if m.sum() >= 3:
        ratio = delta[m] / (L_on * eps[m])
        s_ext = float(np.polyfit(eps[m], ratio, 1)[1])      # intercept = the eps -> 0 limit
    return s_ext, s_pred, (L_on * eps, delta)


def check_bvp(p: C.ModelParams, cl: Closure, mu: float, Ks: float, L: float):
    """Independent solve_bvp of the invariant-reduced scalar ODE.  Returns rho(L).

    n_xx = g(n) Psi^-1(Y Dn (n0-n)) / (Y Dn),  n(0)=n0, n'(L)=0.  This is the same physics by a
    completely different route (rho eliminated, adaptive collocation, no Newton on a fixed grid),
    seeded from a plain analytic profile rather than from this solver's answer.
    """
    YDn, n0 = cl.YDn, p.n0

    def rhs(x, y):
        n = np.clip(y[0], 0.0, n0)
        rho = cl.psi_inv(np.clip(YDn * (n0 - n), 0.0, cl.psi_max))
        return np.vstack([y[1], C.monod(n, mu, Ks) * rho / YDn])

    def bc(ya, yb):
        return np.array([ya[0] - n0, yb[1]])

    kappa0 = np.sqrt(float(g_of(n0, mu, Ks)) / p.D0)
    x = np.linspace(0.0, L, 20_000)
    nu = dilute_seed_nu(kappa0 * x, kappa0 * L)
    y = np.vstack([n0 * nu, np.gradient(n0 * nu, x)])
    sol = solve_bvp(rhs, bc, x, y, tol=1.0e-8, max_nodes=1_000_000)
    if not sol.success:
        return float("nan"), sol.status
    return float(cl.psi_inv(max(YDn * (n0 - float(sol.sol(L)[0])), 0.0))), 0


def master_curve_nuL(xi_L: float) -> float:
    """nu_L for nu''=nu(1-nu), nu(0)=1, nu'(xi_L)=0, from the energy integral.

    nu'^2 = (nu^2 - nuL^2) - (2/3)(nu^3 - nuL^3), and xi_L = int_{nuL}^1 dv/|nu'|.  The 1/sqrt
    endpoint is removed by v = nuL + u^2, the same trick the length quadrature uses.
    """
    def xi_of(nuL):
        def integrand(u):
            v = nuL + u * u
            E = (v * v - nuL * nuL) - (2.0 / 3.0) * (v ** 3 - nuL ** 3)
            return 0.0 if E <= 0.0 else 2.0 * u / np.sqrt(E)
        val, _ = quad(integrand, 0.0, np.sqrt(max(1.0 - nuL, 0.0)), limit=400)
        return float(val)

    lo, hi = 1.0e-12, 1.0 - 1.0e-12
    if xi_of(hi) > xi_L:
        return float("nan")
    return float(brentq(lambda v: xi_of(v) - xi_L, lo, hi, xtol=1.0e-14, rtol=1.0e-13))


def validate(res, cl: Closure, args, quiet=False):
    """The full suite.  Returns a list of (name, value) rows destined for summary.csv."""
    p, mu, Ks = res["p"], res["mu"], res["Ks"]
    x, rho, n, dx, L, N = res["x"], res["rho"], res["n"], res["dx"], res["L"], res["N"]
    d = derived(res)
    rows, verdicts = [], []

    SKIP = "SKIP"

    def rec(name, value, ok=None, note=""):
        rows.append((name, value))
        if ok is None:
            return
        verdict = SKIP if ok is SKIP else ("PASS" if ok else "FAIL")
        rows.append((name + "_pass", verdict))
        # The caveat belongs in the artifact, not only on the console: whoever reads
        # summary.csv later is exactly the person who needs to know that check5's 1.7e-18 is a
        # positivity test and not a concentration.
        if note:
            rows.append((name + "_note", note))
        verdicts.append((name, verdict, note))

    # 1 -- the discrete invariant, which the Kirchhoff form makes exact
    inv = float(np.max(np.abs(d["invariant"])))
    inv_rel = inv / max(cl.YDn * p.n0, TINY)
    rec("check1_invariant_linf", f"{inv:.6e}")
    rec("check1_invariant_relative", f"{inv_rel:.6e}", inv_rel < 1.0e-10)

    # 2 -- monotonicity: n_x <= 0 and rho_x >= 0 (tex Prop)
    dn = np.diff(n)
    dr = np.diff(rho)
    tol_m = 1.0e-9 * max(float(np.max(np.abs(n))), 1.0)
    ok_n = bool(np.all(dn <= tol_m))
    ok_r = bool(np.all(dr >= -1.0e-9 * max(float(np.max(rho)), 1e-30)))
    rec("check2_max_dn_positive", f"{float(np.max(dn)):.6e}")
    rec("check2_min_drho", f"{float(np.min(dr)):.6e}", ok_n and ok_r)

    # 3 -- A vs B endpoint budget classification (tex Thm)
    A, B = cl.A(p.n0), cl.B
    ratio = A / B
    predicted = "sub-jammed" if ratio < 1.0 else ("critical" if abs(ratio - 1) < 1e-12 else "jams")
    observed = "jams" if float(rho.max()) >= p.rho_jam else "sub-jammed"
    rec("A_budget", f"{A:.8g}")
    rec("B_jamming_cost", f"{B:.8g}")
    rec("A_over_B", f"{ratio:.8g}")
    rec("n0_jam_uM", f"{cl.n0_jam:.8g}")
    # n_jam is only defined in the excess-budget case; writing 0 when A < B would read as the
    # CRITICAL case A = B (jamming exactly at depletion), which is a different regime.
    rec("n_jam_uM", f"{p.n0 - B / cl.YDn:.8g}" if ratio > 1.0 else "not reached (A<B)")
    rec("check3_predicted", predicted)
    rec("check3_observed", observed,
        (predicted == "jams") == (observed == "jams"))

    # 4 -- the density ceiling rho <= Psi^-1(A)
    ceil = cl.rho_ceiling(p.n0)
    rec("rho_ceiling_Psi_inv_A", f"{ceil:.10g}")
    rec("check4_rho_max", f"{float(rho.max()):.10g}", float(rho.max()) <= ceil * (1 + 1e-9))

    # 5 -- n(L) > 0 strictly (tex Prop: depletion is not reached at finite length).
    # What this CAN test is that the solver preserves positivity -- it could have returned 0 or
    # a negative value and violated the maximum principle.  What it CANNOT do is confirm the
    # proposition physically: n(L) = 8e-68 uM is ~60 orders below one molecule in the whole
    # cavity, so its magnitude is continuum fiction and is labelled as such.
    below_floor = float(n[-1]) < d["noise_floor"]
    rec("check5_n_at_L_uM", f"{float(n[-1]):.6e}", float(n[-1]) > 0.0,
        f"positivity only -- n(L) is below the {d['noise_floor']:g} uM noise floor, so its "
        "magnitude carries no information" if below_floor else "")

    # 6 -- independent pressure-plane quadrature.
    #
    # The round trip is only conditioned where the closed end still holds nutrient.  As n(L)->0
    # the source slope S'(p*) = g(n)rho -> 0, so S(p*)-S(p) ~ (p*-p)^2 and L diverges
    # LOGARITHMICALLY (the tex's "depletion is not reached at finite length").  The log's
    # argument is then a difference of nearly equal tabulated values, and double precision
    # floors it: L_quad SATURATES and stops responding to p*.  Measured below.  This is
    # cancellation, not resolution -- no grid refinement moves it.  It does not affect
    # check_fold_*.py, whose endpoint is JAMMING, where g(n_jam) > 0 and the sqrt is genuine.
    #
    # So the real cross-check is run at MODERATE depletion, where both methods are valid, and
    # the production-L round trip is reported as a diagnostic.
    try:
        S = _source_table(cl, p.n0, mu, Ks, cl.p_cap(p.n0), args.n_quad)
        L_sat = quad_length(S, cl.p_cap(p.n0))
        rec("check6_quadrature_saturation_L_um", f"{L_sat:.6g}")

        # Place the cross-check at HALF THE NUTRIENT BUDGET, p* = A/2.  Choosing it by LENGTH
        # instead is a trap: at n0 = 1000 the whole branch p*/A = 1e-4 .. 0.9 spans only
        # L = 34.7 .. 39.6 um, and everything above that is the resolution-limited sliver where
        # L(p*) has not converged -- a test point picked there compares against a number the
        # quadrature cannot compute.  At p* = A/2 both regimes agree to ~1e-6.
        p_test = float(args.quad_frac) * cl.A(p.n0)
        L_test = quad_length(S, p_test)
        h = 1.0e-4 * p_test
        cond = abs((quad_length(S, p_test + h) - quad_length(S, p_test - h))
                   / (2 * h) * p_test / L_test)          # |dlnL/dlnp*|
        r_mod = solve_steady(L_test, N, p, cl, mu, Ks, use_q=res["use_q"])
        ok_mod = r_mod["info"]["converged"] and r_mod["info"]["branch"] == "colonised"
        p_back = float(cl.psi(r_mod["rho"][-1]))          # Newton: L -> p*
        rel_mod = abs(p_back - p_test) / p_test
        rec("check6_L_test_um", f"{L_test:.8g}")
        rec("check6_n_at_L_test_over_n0", f"{float(r_mod['n'][-1]) / p.n0:.4e}")
        rec("check6_condition_dlnL_dlnpstar", f"{cond:.4g}")
        rec("check6_pstar_roundtrip_rel_error", f"{rel_mod:.3e}", ok_mod and rel_mod < 1.0e-2,
            "consistency only -- quadrature also seeded the solve"
            if res["info"].get("seed") == "quadrature" else "independent")

        p_star = float(cl.psi(rho[-1]))
        L_q = quad_length(S, p_star)
        rel = abs(L_q - L) / L
        rec("check6b_L_quadrature_at_production_um", f"{L_q:.8g}")
        rec("check6b_L_rel_error", f"{rel:.3e}",
            SKIP if L > 0.9 * L_sat else (rel < 5.0e-3),
            f"L={L:g} exceeds the quadrature's saturation length {L_sat:.0f} um "
            f"(n(L)/n0={float(n[-1])/p.n0:.1e}); round trip is ill-posed in float64"
            if L > 0.9 * L_sat else "")
    except Exception as exc:                                          # noqa: BLE001
        rec("check6_quadrature", f"failed: {exc}", False)

    # 7 -- independent solve_bvp on the invariant-reduced scalar ODE
    try:
        rho_bvp, status = check_bvp(p, cl, mu, Ks, L)
        rel = abs(rho_bvp - float(rho[-1])) / max(float(rho[-1]), TINY)
        rec("check7_rho_L_solve_bvp", f"{rho_bvp:.10g}")
        rec("check7_rho_L_rel_error", f"{rel:.3e}", np.isfinite(rel) and rel < 1.0e-2)
    except Exception as exc:                                          # noqa: BLE001
        rec("check7_rho_L_solve_bvp", f"failed: {exc}", False)

    # 8 -- dx convergence (order measured, not assumed)
    conv = []
    if args.dx_grids > 1:
        Ns = [N * (2 ** k) for k in range(args.dx_grids)]
        for Nk in Ns:
            r = solve_steady(L, Nk, p, cl, mu, Ks, use_q=res["use_q"])
            if r["info"]["converged"] and r["info"]["branch"] == "colonised":
                conv.append((L / Nk, float(np.sum(r["rho"]) * (L / Nk)), float(r["rho"][-1])))
        for (h, M, rl) in conv:
            rec(f"check8_biomass_dx{h:g}", f"{M:.10g}")
        order = float("nan")
        if len(conv) >= 3:
            M = np.array([c[1] for c in conv])
            num, den = M[0] - M[1], M[1] - M[2]
            if abs(den) > 0 and num / den > 0:
                order = float(np.log2(num / den))
        rec("check8_observed_order", f"{order:.3f}",
            SKIP if not np.isfinite(order) else (0.7 < order < 3.0),
            "" if np.isfinite(order) else "not enough converged grids to fit an order")
    res["_dxconv"] = conv

    # 9 -- Psi-table resolution
    cl2 = Closure(p, n_grid=max(args.n_grid * 2, 100_001))
    r2 = solve_steady(L, N, p, cl2, mu, Ks, use_q=res["use_q"])
    shift = abs(float(r2["rho"][-1]) - float(rho[-1])) / max(float(rho[-1]), TINY)
    rec("check9_ngrid_doubling_rel_shift", f"{shift:.3e}", shift < 1.0e-6)

    # 10 -- L_onset, two independent ways
    L_on = onset_length(p, mu, Ks)
    L_disc, da_c = check_onset_eigenvalue(p, mu, Ks, N)
    rec("L_onset_analytic_um", f"{L_on:.8g}")
    rec("check10a_L_onset_discrete_um", f"{L_disc:.8g}")
    rec("check10a_rel_error", f"{abs(L_disc - L_on) / L_on:.3e}",
        abs(L_disc - L_on) / L_on < 5.0e-3)
    rec("check10a_Da_crit_discrete", f"{da_c:.8g}")
    rec("Da_crit_analytic", f"{np.pi ** 2 / 4:.8g}")
    rec("Da_0_at_L", f"{float(g_of(p.n0, mu, Ks)) * L ** 2 / p.D0:.8g}")

    # 10b -- the transcritical amplitude law delta_L = 3 g0 kappa0/(2 g1) * (L - L_onset).
    #
    # It is a LEADING-ORDER law in (L - L_onset) and it assumes D_col ~ D0.  Both matter:
    #   * the measured ratio drifts with eps, so it is Richardson-extrapolated to eps -> 0;
    #   * D_col is 3.0% above D0 at the realised density when n0 = 15, and the extrapolated
    #     slope comes out 2.9% low to match.  The solver makes no dilute assumption, so the
    #     verdict is taken at --onset-n0, where the assumption the law rests on actually holds
    #     (D_col/D0 - 1 ~ 3e-4).  The production-n0 value is recorded as a measurement of the
    #     dilute approximation's error, not as a failure.
    if args.onset_slope:
        for tag, pp, cc in (("check10b", config.params(args.onset_n0, dself_shape=p.dself_shape,
                                                       Dself_floor=p.Dself_floor), None),
                            ("check10c", p, cl)):
            cc = cc or Closure(pp, n_grid=args.n_grid)
            s_meas, s_pred, fit = measure_onset_slope(pp, cc, mu, Ks, N, use_q=res["use_q"])
            ratio = s_meas / s_pred if np.isfinite(s_meas) else float("nan")
            dcol_excess = float(C.collective_diffusivity(
                np.array([cc.rho_ceiling(pp.n0)]), pp)[0]) / pp.D0
            rec(f"{tag}_n0_uM", f"{pp.n0:g}")
            rec(f"{tag}_Dcol_over_D0_at_ceiling", f"{dcol_excess:.6f}")
            rec(f"{tag}_slope_predicted", f"{s_pred:.8g}")
            rec(f"{tag}_slope_measured_extrapolated", f"{s_meas:.8g}")
            if tag == "check10b":
                res["_onset_fit"] = fit
                res["_onset_slope_pred"] = s_pred
                rec("check10b_slope_ratio", f"{ratio:.6f}",
                    SKIP if not np.isfinite(ratio) else abs(ratio - 1.0) < 0.01,
                    "verdict taken where D_col ~ D0 actually holds")
            else:
                rec("check10c_slope_ratio", f"{ratio:.6f}", None,
                    "diagnostic: the dilute approximation's error at the production n0")
    else:
        rec("check10b_slope_ratio", "nan", SKIP, "not run (--no-onset-slope)")

    # 11 -- dilute master curve, run at its OWN n0 where n << Ks actually holds
    mc = []
    if args.master_curve:
        p_mc = config.params(args.mc_n0, dself_shape=p.dself_shape, Dself_floor=p.Dself_floor)
        cl_mc = Closure(p_mc, n_grid=args.n_grid)
        k0 = np.sqrt(float(g_of(p_mc.n0, mu, Ks)) / p_mc.D0)
        for xi in (2.0, 3.0, 4.0, 5.0):
            Lm = xi / k0
            r = solve_steady(Lm, N, p_mc, cl_mc, mu, Ks, use_q=res["use_q"])
            if r["info"]["branch"] != "colonised":
                continue
            mc.append((xi, 1.0 - float(r["n"][-1]) / p_mc.n0, 1.0 - master_curve_nuL(xi)))
        rec("mc_n0_uM", f"{args.mc_n0:g}")
        rec("mc_n0_over_Ks", f"{args.mc_n0 / Ks:.4g}")
        err = max((abs(a - b) for _, a, b in mc), default=float("nan"))
        rec("check11_master_curve_max_abs_error", f"{err:.3e}",
            SKIP if not np.isfinite(err) else err < 0.02,
            "" if np.isfinite(err) else "no colonised solutions in the xi_L sweep")
    res["_mc"] = mc

    # 12 -- vs the time-dependent stack
    for tag, rundir in (("dilute", "run_n0-15_dx5_T6000"),
                        ("jammed", "run_n0-1000_dx0.5_T50000")):
        path = os.path.join(HERE, rundir, "summary.csv")
        if not os.path.exists(path):
            continue
        vals = {}
        with open(path, newline="") as fh:
            for r in csv.reader(fh):
                if len(r) == 2:
                    vals[r[0]] = r[1]
        try:
            if abs(float(vals.get("n0_uM", "nan")) - p.n0) > 1e-9:
                continue
            rp = float(vals["rho_peak"])
        except (ValueError, KeyError):
            continue
        rel = abs(rp - float(rho.max())) / max(rp, TINY)
        rec(f"check12_{tag}_timedep_rho_peak", f"{rp:.10g}")
        rec(f"check12_{tag}_rel_error", f"{rel:.3e}", rel < 5.0e-3)

    if not quiet:
        print("\n  validation")
        print("  " + "-" * 74)
        for name, verdict, note in verdicts:
            print(f"  {verdict:<4s}  {name:<44s} {note}")
        npass = sum(1 for _, v, _ in verdicts if v == "PASS")
        nfail = sum(1 for _, v, _ in verdicts if v == "FAIL")
        nskip = sum(1 for _, v, _ in verdicts if v == "SKIP")
        print("  " + "-" * 74)
        print(f"  {npass} passed, {nfail} failed, {nskip} skipped")
    return rows, verdicts


# =============================================================================
# CSV output
# =============================================================================
def write_csvs(outdir, res, cl: Closure, val_rows, branch_rows=None):
    p, mu, Ks = res["p"], res["mu"], res["Ks"]
    x, rho, n, d = res["x"], res["rho"], res["n"], derived(res)
    info = res["info"]

    with open(os.path.join(outdir, "profile.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_um", "rho", "n_uM", "Psi", "D_col", "D_self", "g", "q",
                    "J_rho", "J_n", "invariant_residual"])
        # fluxes live on faces; report the face to the LEFT of each cell centre
        for i in range(len(x)):
            w.writerow([f"{x[i]:.6g}", f"{rho[i]:.10g}", f"{n[i]:.10g}", f"{d['Psi'][i]:.10g}",
                        f"{d['Dcol'][i]:.10g}", f"{d['Dself'][i]:.10g}", f"{d['g'][i]:.10g}",
                        f"{d['q'][i]:.10g}", f"{d['J_rho'][i]:.10g}", f"{d['J_n'][i]:.10g}",
                        f"{d['invariant'][i]:.6e}"])

    lam, dxlam, drop = front_resolution(x, n, res["dx"])
    rows = [
        ("model", "single-strain steady, reviewthis.tex"),
        ("strain", res["strain_label"]), ("mu", mu), ("Ks", Ks),
        ("n0_uM", p.n0), ("Y", p.Y), ("Dn", p.Dn), ("D0", p.D0), ("eta", p.eta),
        ("rho_jam", p.rho_jam), ("rho_c", p.rho_c), ("rho_max", p.rho_max),
        ("dself_shape", p.dself_shape), ("Dself_floor", p.Dself_floor),
        ("Dself_jam_slope", p.Dself_jam_slope),
        ("growth_space_factor_enabled", res["use_q"]),
        ("L_um", res["L"]), ("dx_um", res["dx"]), ("N", res["N"]),
        ("discretisation", "Kirchhoff (Laplacian of Psi)"),
        ("newton_seed", info.get("seed", "")),
        ("newton_iters", info["iters"]),
        ("newton_residual_linf", f"{info['residual']:.6e}"),
        ("newton_tol_effective", f"{info['tol_eff']:.6e}"),
        ("newton_converged", info["converged"]),
        ("newton_message", info["message"]),
        ("branch", info["branch"]),
        ("rho_at_L", f"{float(rho[-1]):.10g}"),
        ("rho_max", f"{float(rho.max()):.10g}"),
        ("rho_max_over_jam", f"{float(rho.max()) / p.rho_jam:.10g}"),
        ("n_at_L_uM", f"{float(n[-1]):.6e}"),
        ("biomass_integral_um", f"{d['biomass']:.10g}"),
        ("noise_floor_uM", f"{d['noise_floor']:.3e}"),
        ("x_below_noise_floor_um", f"{d['x_noise']:.6g}"),
        ("cavity_fraction_below_noise_floor", f"{d['frac_noise']:.4f}"),
        ("eps_dilute", f"{cl.A(p.n0) / (p.D0 * p.rho_jam):.6g}"),
        ("L_over_L_onset", f"{res['L'] / res['L_onset']:.6g}"),
        ("lambda_front_um", f"{lam:.6g}"), ("dx_over_lambda", f"{dxlam:.6g}"),
        ("front_drop_factor", f"{drop:.6g}"),
    ] + list(val_rows)
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in rows:
            w.writerow([k, v])

    with open(os.path.join(outdir, "newton_history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iter", "residual_linf", "lambda_armijo", "step_linf"])
        for h in info["history"]:
            w.writerow([h["iter"], f"{h['residual']:.8e}",
                        "" if not np.isfinite(h["lam"]) else f"{h['lam']:.6g}",
                        "" if not np.isfinite(h["step"]) else f"{h['step']:.6e}"])

    if branch_rows:
        with open(os.path.join(outdir, "branch.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["L_um", "rho_L", "n_L_uM", "rho_max", "p_star", "newton_iters",
                        "residual", "converged"])
            for r in branch_rows:
                w.writerow([f"{r['L']:.8g}", f"{r['rho_L']:.10g}", f"{r['n_L']:.6e}",
                            f"{r['rho_max']:.10g}", f"{r['p_star']:.10g}", r["iters"],
                            f"{r['residual']:.3e}", r["converged"]])


# =============================================================================
# Figures
# =============================================================================
def _panel(figsize=(6.4, 4.4)):
    fig, a = plt.subplots(figsize=figsize)
    a.grid(alpha=.2)
    return fig, a


def _finish(fig, path, what, res):
    _suptitle(fig, f"{what}\n{_run_title(res)}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _run_title(res):
    p = res["p"]
    return (f"steady single strain {res['strain_label']} -- $n_0$={p.n0:g} $\mu$M, "
            f"$\mu$={res['mu']:g} 1/h, $K_s$={res['Ks']:g} $\mu$M, "
            f"$L$={res['L']:g} $\mu$m, $N$={res['N']}")


# --- one PNG per quantity -----------------------------------------------------------------
def figure_nutrient(res, cl: Closure, path):
    """n(x).  The log axis is clipped at the noise floor: left to autoscale it spans 60+
    decades at n0=1000, drawing numerical noise at the same weight as data."""
    p, d = res["p"], derived(res)
    x, n = res["x"], res["n"]
    fl = d["noise_floor"]
    fig, a = _panel()
    a.plot(x, np.maximum(n, fl * 1e-1), color=CN, lw=1.8)
    a.set_yscale("log")
    a.set_ylim(fl * 1e-1, float(p.n0) * 3.0)
    a.axhspan(fl * 1e-1, fl, color="grey", alpha=.15)
    a.axhline(fl, color="grey", lw=1, label=rf"numerical noise floor ({fl:g} $\mu$M)")
    if np.isfinite(d["x_noise"]):
        a.axvline(d["x_noise"], color=C2, ls="-.", lw=1,
                  label=rf"$x$={d['x_noise']:.0f} $\mu$m ({100*d['frac_noise']:.0f}% of cavity"
                        r" below)")
    n_jam = p.n0 - cl.B / cl.YDn
    if n_jam > 0:
        a.axhline(n_jam, color="k", ls="--", lw=1, label=rf"$n_{{\rm jam}}$={n_jam:.4g}")
    a.legend(fontsize=7, loc="lower left")
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$n$ [$\mu$M]")
    _finish(fig, path, "nutrient", res)


def figure_density(res, cl: Closure, path):
    p, rho = res["p"], res["rho"]
    fig, a = _panel()
    a.plot(res["x"], rho, color=C1, lw=1.8)
    ceil = cl.rho_ceiling(p.n0)
    a.axhline(ceil, color=C2, ls="--", lw=1, label=r"$\Psi^{-1}(A)$ ceiling")
    if float(rho.max()) > 0.2 * p.rho_jam:
        a.axhline(p.rho_jam, color="k", ls=":", lw=1, label=r"$\rho_{\rm jam}$")
    else:
        a.plot([], [], " ", label=rf"$\rho_{{\rm jam}}={p.rho_jam:g}$ (off scale, "
                                 rf"$\rho_{{\max}}/\rho_{{\rm jam}}$="
                                 rf"{float(rho.max())/p.rho_jam:.3g})")
        a.set_ylim(0.0, max(float(rho.max()), ceil) * 1.15)
    a.legend(fontsize=7)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$\rho$")
    _finish(fig, path, "biomass", res)


def figure_diffusivity_x(res, cl: Closure, path):
    d = derived(res)
    fig, a = _panel()
    a.plot(res["x"], d["Dcol"], color=C1, lw=1.8, label=r"$D_{\rm col}$")
    a.plot(res["x"], d["Dself"], color=C2, lw=1.8, ls="--",
           label=r"$D_{\rm self}$ (not a term in the single-strain model)")
    a.set_yscale("log")
    a.legend(fontsize=7)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"[$\mu$m$^2$/h]")
    _finish(fig, path, "transport closures along the cavity", res)


def figure_growth(res, cl: Closure, path):
    d = derived(res)
    fl = d["noise_floor"]
    gfl = float(g_of(fl, res["mu"], res["Ks"]))
    fig, a = _panel()
    a.plot(res["x"], np.maximum(d["g"], gfl * 1e-1), color=CN, lw=1.8, label=r"$g(n)$")
    a.plot(res["x"], d["q"], color="grey", lw=1.2, ls=":", label=r"$q(\rho)$")
    a.set_yscale("log")
    a.set_ylim(gfl * 1e-1, max(float(np.max(d["g"])), 1.0) * 3.0)
    a.axhspan(gfl * 1e-1, gfl, color="grey", alpha=.15)
    a.axhline(gfl, color="grey", lw=1, label="growth at the noise floor")
    a.legend(fontsize=7)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"[1/h]")
    _finish(fig, path, "growth rate", res)


def figure_flux(res, cl: Closure, path):
    """J_rho = -Psi_x against +Y*Dn*n_x.  The flux invariant makes these identical."""
    p, d = res["p"], derived(res)
    xf = np.linspace(0.0, res["L"], res["N"] + 1)
    fig, a = _panel()
    a.plot(xf, d["J_rho"], color=C1, lw=2.4, label=r"$J_\rho=-\Psi_x$")
    a.plot(xf, -p.Y * d["J_n"], color=C2, lw=1.2, ls="--", label=r"$+YD_n n_x$")
    a.legend(fontsize=7)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel("flux")
    _finish(fig, path, "flux invariant: the two curves must coincide", res)


# --- the closure set, with the invariant, on one page --------------------------------------
def figure_closures(res, cl: Closure, path):
    """D_col, D_self and Psi over the FULL density range, plus the invariant they imply.

    No realised-range shading: these three are properties of the closure, not of this run, and
    the shading invited them to be read as run output.  The fourth panel is the one thing here
    that IS run output -- the integrated invariant Psi(rho(x)) = Y*Dn*(n0-n(x)), whose two
    curves coincide to machine precision because the Kirchhoff discretisation forces it.
    """
    p, d = res["p"], derived(res)
    r = np.linspace(0.0, 1.0, 4001)
    Dc = np.asarray(C.collective_diffusivity(r, p), float)
    Ds = np.asarray(C.self_diffusivity(r, p), float)
    Ps = np.asarray(cl.psi(r), float)
    sub = r <= p.rho_jam
    i_max = int(np.argmax(Dc[sub]))

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
    for a, y, name in ((ax[0, 0], Dc, r"$D_{\rm col}(\rho)$"),
                       (ax[0, 1], Ds, r"$D_{\rm self}(\rho)$"),
                       (ax[1, 0], Ps, r"$\Psi(\rho)=\int_0^\rho D_{\rm col}\,d s$")):
        a.plot(r, y, color=C1, lw=1.8)
        a.axvline(p.rho_jam, color="k", ls=":", lw=1, label=r"$\rho_{\rm jam}$")
        a.set_yscale("log")
        a.set_xlabel(r"$\rho$")
        a.set_title(name, fontsize=10, fontweight="bold")
        a.grid(alpha=.2)
        a.legend(fontsize=7)
    ax[0, 0].axvline(r[i_max], color=C2, ls="--", lw=1)
    ax[0, 0].annotate(rf"max at $\rho$={r[i_max]:.3f}", (r[i_max], Dc[i_max]),
                      textcoords="offset points", xytext=(6, 8), fontsize=7, color=C2)
    ax[0, 1].annotate("discontinuous at jamming\n(relative-label diffusivity;\n"
                      "no role for one strain)", (0.03, 0.06), xycoords="axes fraction",
                      fontsize=7, color="grey")

    a = ax[1, 1]
    a.plot(res["x"], d["Psi"], color=C1, lw=2.4, label=r"$\Psi(\rho(x))$")
    a.plot(res["x"], cl.YDn * (p.n0 - res["n"]), color=C2, lw=1.2, ls="--",
           label=r"$YD_n\,(n_0-n(x))$")
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$\Psi$")
    a.set_title(r"integrated invariant $\Psi(\rho)+YD_n n=YD_n n_0$"
                f"\n(residual {np.max(np.abs(d['invariant']))/max(cl.YDn*p.n0, TINY):.1e}"
                " relative)", fontsize=10, fontweight="bold")
    a.grid(alpha=.2)
    a.legend(fontsize=7)
    _finish(fig, path, "closures over the full density range, and the invariant they imply", res)

    # Every plotted array beside the PNG (rule: plotted arrays are always saved).  The closure
    # curves of (a)-(c) exist nowhere else on disk; (d) repeats profile.csv's Psi for convenience.
    stem = os.path.splitext(path)[0]
    with open(stem + "_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rho", "D_col", "D_self", "Psi", "rho_at_Dcol_max_subjam"])
        for k in range(r.size):
            w.writerow([f"{r[k]:.6g}", f"{Dc[k]:.10g}", f"{Ds[k]:.10g}", f"{Ps[k]:.10g}",
                        f"{r[i_max]:.6g}" if k == 0 else ""])
    with open(stem + "_invariant.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_um", "Psi_of_rho", "YDn_n0_minus_n"])
        rhs = cl.YDn * (p.n0 - res["n"])
        for i in range(len(res["x"])):
            w.writerow([f"{res['x'][i]:.6g}", f"{d['Psi'][i]:.10g}", f"{rhs[i]:.10g}"])


def figure_validation(res, cl: Closure, path):
    d = derived(res)
    info = res["info"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    a = ax[0, 0]
    a.plot(res["x"], np.abs(d["invariant"]) / max(cl.YDn * res["p"].n0, TINY),
           color=C1, lw=1.4)
    a.set_yscale("log")
    a.set_title(r"(a) relative invariant residual $|\Psi+YD_nn-YD_nn_0|/(YD_nn_0)$",
                fontsize=9, fontweight="bold")
    a.set_xlabel(r"$x$ [$\mu$m]")

    a = ax[0, 1]
    it = [h["iter"] for h in info["history"]]
    rr = [h["residual"] for h in info["history"]]
    a.semilogy(it, np.maximum(rr, TINY), "o-", color=C2, lw=1.4, ms=4)
    a.axhline(info["tol_eff"], color="grey", ls="--", lw=1, label="tolerance")
    a.legend(fontsize=7)
    a.set_title(f"(b) Newton convergence -- seed '{info.get('seed','')}'",
                fontsize=9, fontweight="bold")
    a.set_xlabel("iteration"); a.set_ylabel(r"$\|F\|_\infty$")

    a = ax[1, 0]
    conv = res.get("_dxconv") or []
    if len(conv) >= 2:
        h = np.array([c[0] for c in conv])
        M = np.array([c[1] for c in conv])
        a.loglog(h[:-1], np.abs(M[:-1] - M[-1]), "o-", color=C1, lw=1.4, ms=5,
                 label="|error| vs finest")
        a.loglog(h, np.abs(M[0] - M[-1]) * (h / h[0]) ** 2, ":", color="grey",
                 lw=1, label=r"$O(\Delta x^2)$")
        a.legend(fontsize=7)
    else:
        a.text(.5, .5, "dx convergence not run\n(--dx-grids 1)", ha="center", va="center",
               transform=a.transAxes, fontsize=9, color="grey")
    a.set_title("(c) grid convergence of the biomass integral", fontsize=9, fontweight="bold")
    a.set_xlabel(r"$\Delta x$ [$\mu$m]")

    a = ax[1, 1]
    fit = res.get("_onset_fit")
    if fit is not None and np.any(np.isfinite(fit[1])):
        dL, dl = fit
        slope = res.get("_onset_slope_pred", np.nan)
        a.plot(dL, dl / (slope * dL), "o-", color=C1, ms=5, lw=1.2,
               label="measured / predicted")
        a.axhline(1.0, color=C2, ls="--", lw=1.4, label="transcritical law")
        a.set_ylim(0.9, 1.1)
        a.legend(fontsize=7)
    else:
        a.text(.5, .5, "onset slope not run\n(--no-onset-slope)", ha="center", va="center",
               transform=a.transAxes, fontsize=9, color="grey")
    a.set_title("(d) transcritical onset law, in the dilute regime it assumes",
                fontsize=9, fontweight="bold")
    a.set_xlabel(r"$L-L_{\rm onset}$ [$\mu$m]")
    a.set_ylabel(r"$\delta_L\,/\,\frac{3g_0\kappa_0}{2g_1}(L-L_{\rm onset})$")

    for a in ax.ravel():
        a.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Every plotted array beside the PNG.  (b) is newton_history.csv; (a), (c), (d) are here --
    # the onset fit of (d) in particular exists nowhere else on disk.
    stem = os.path.splitext(path)[0]
    with open(stem + "_residual.csv", "w", newline="") as fh:              # (a)
        w = csv.writer(fh)
        w.writerow(["x_um", "invariant_residual_relative"])
        rel = np.abs(d["invariant"]) / max(cl.YDn * res["p"].n0, TINY)
        for i in range(len(res["x"])):
            w.writerow([f"{res['x'][i]:.6g}", f"{rel[i]:.6e}"])
    with open(stem + "_dxconv.csv", "w", newline="") as fh:                # (c)
        w = csv.writer(fh)
        w.writerow(["dx_um", "biomass_integral_um", "rho_L"])
        for c in conv:
            w.writerow([f"{c[0]:.8g}", f"{c[1]:.12g}", f"{c[2]:.12g}" if len(c) > 2 else ""])
    with open(stem + "_onset_fit.csv", "w", newline="") as fh:             # (d)
        w = csv.writer(fh)
        w.writerow(["dL_um", "delta_L_uM", "measured_over_predicted", "slope_predicted"])
        if fit is not None:
            dL, dl = fit
            slope = res.get("_onset_slope_pred", np.nan)
            for k in range(len(dL)):
                w.writerow([f"{dL[k]:.10g}", f"{dl[k]:.10g}",
                            f"{dl[k] / (slope * dL[k]):.10g}", f"{slope:.10g}"])


def figure_branch(rows, res, cl: Closure, path, quad_curve=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ok = [r for r in rows if r["converged"]]
    L = np.array([r["L"] for r in ok])
    rL = np.array([r["rho_L"] for r in ok])

    p = res["p"]
    a = ax[0]
    a.plot(L, rL, "o-", color=C1, lw=1.6, ms=3.5, label="Newton branch")
    if quad_curve is not None:
        a.plot(quad_curve[1], quad_curve[0], "--", color=C2, lw=1.3, label="quadrature")
    a.axvline(res["L_onset"], color="grey", ls=":", lw=1,
              label=rf"$L_{{\rm onset}}$={res['L_onset']:.1f}")
    if float(rL.max()) > 0.2 * p.rho_jam:
        a.axhline(p.rho_jam, color="k", ls=":", lw=1, label=r"$\rho_{\rm jam}$")
    else:
        a.plot([], [], " ", label=rf"$\rho_{{\rm jam}}={p.rho_jam:g}$ (off scale)")
        a.set_ylim(0.0, float(rL.max()) * 1.15)
    a.set_xscale("log")
    a.set_xlabel(r"$L$ [$\mu$m]"); a.set_ylabel(r"$\rho(L)$")
    a.set_title("(a) colonised branch", fontsize=10, fontweight="bold")
    a.legend(fontsize=7)

    # (b) as a RATIO, over the whole branch.  Plotting delta_L against a straight line is
    # misleading once delta_L saturates -- it cannot exceed n0, while the line runs to 9x n0.
    # The ratio shows both that the law is exact at onset and exactly where it stops holding.
    a = ax[1]
    dL = L - res["L_onset"]
    delta = p.n0 - np.array([r["n_L"] for r in ok])
    slope = onset_amplitude_slope(p, res["mu"], res["Ks"])
    m = dL > 0
    a.semilogx(dL[m], delta[m] / (slope * dL[m]), "o-", color=C1, lw=1.4, ms=4,
               label="measured / predicted")
    a.axhline(1.0, color=C2, ls="--", lw=1.4, label="transcritical law")
    a.axhline(0.0, color="grey", lw=0.8)
    a.set_ylim(-0.05, 1.15)
    a.set_xlabel(r"$L-L_{\rm onset}$ [$\mu$m]")
    a.set_ylabel(r"$\delta_L\,/\,\frac{3g_0\kappa_0}{2g_1}(L-L_{\rm onset})$")
    a.set_title(r"(b) the near-onset law, and where it stops holding ($\delta_L\to n_0$)",
                fontsize=10, fontweight="bold")
    a.legend(fontsize=7)

    for a in ax:
        a.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # the branch points are branch.csv; the quadrature overlay exists only here
    if quad_curve is not None:
        with open(os.path.splitext(path)[0] + "_quadrature.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rho_L", "L_um"])
            for rv, lv in zip(quad_curve[0], quad_curve[1]):
                w.writerow([f"{rv:.10g}", f"{lv:.10g}"])


# =============================================================================
# Sweep modes: the budget criterion, and the two L* formulas
# =============================================================================
#
# Psi, and therefore B = Psi(rho_jam), depend only on D0/eta/rho_jam -- NOT on n0.  So one
# Closure is built once and shared across an entire n0 sweep; rebuilding it per n0 would spend
# seconds on an identical PCHIP table each time.  Only A(n0), p_cap(n0) and rho_ceiling(n0) take
# n0, and they take it as an argument.
def _with_n0(p: C.ModelParams, n0: float) -> C.ModelParams:
    from dataclasses import replace
    return replace(p, n0=float(n0))


def _sweep_N(L: float, dx_target: float, lo: int = 60, hi: int = 2000) -> int:
    return int(np.clip(round(L / dx_target), lo, hi))


def _write_rows_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            out = []
            for k in fields:
                v = r.get(k)
                out.append("" if v is None else
                           (f"{v:.10g}" if isinstance(v, float) else v))
            w.writerow(out)


def jam_length(cl: Closure, n0: float, mu: float, Ks: float, n_quad: int = 400_001):
    """L at which rho(L) first reaches rho_jam; +inf when A <= B (the cavity can never jam)."""
    if cl.A(n0) <= cl.B:
        return float("inf")
    S = _source_table(cl, n0, mu, Ks, cl.p_cap(n0), n_quad)
    return quad_length(S, cl.B)


def budget_sweep(p_base: C.ModelParams, cl: Closure, mu: float, Ks: float,
                 n0s, Ls, dx_target: float = 0.5, use_q: bool = False,
                 tol: float = 1.0e-9, verbose: bool = False):
    """Test the tex's jamming criterion A_eff > B against direct numerical solutions.

    A_eff := Y*Dn*(n0 - n(L)) is the budget the colony ACTUALLY spends, as opposed to the
    budget A = Y*Dn*n0 it is offered.  The invariant gives Psi(rho(L)) = A_eff, and rho is
    monotone, so rho_max = rho(L) and

        A_eff >= B   <=>   rho_max >= rho_jam

    should hold at every point.  Sweeping (n0, L) tests that wherever the solver actually goes,
    and -- the point of the sweep -- separates it from the cruder A > B, which is NECESSARY but
    not sufficient: at n0 = 900 the cavity is offered A/B = 1.15 and still stays sub-jammed for
    every L below L_jam.
    """
    rows = []
    for n0 in n0s:
        p = _with_n0(p_base, n0)
        A, B = cl.A(n0), cl.B
        L_on = onset_length(p, mu, Ks)
        prev = None
        for L in np.sort(np.asarray(Ls, float)):      # ascending, so each L can seed the next
            rec = dict(n0=float(n0), L=float(L), A=A, B=B, A_over_B=A / B, L_onset=L_on)
            if L <= L_on:
                rec.update(branch="trivial", rho_max=0.0, n_L=float(n0), A_eff=0.0,
                           A_eff_over_B=0.0, rho_over_jam=0.0, jam_numeric=False,
                           jam_criterion=False, agree=True, converged=True, iters=0)
                rows.append(rec)
                continue
            N = _sweep_N(L, dx_target)
            r = None
            if prev is not None:                      # continuation within the n0 column
                r = solve_steady(L, N, p, cl, mu, Ks, use_q=use_q, tol=tol,
                                 u0=_seed_from(prev, L, N), seed_label="prev-L",
                                 allow_march=False)
                if not (r["info"]["converged"] and r["info"]["branch"] == "colonised"):
                    r = None
            if r is None:
                r = solve_steady(L, N, p, cl, mu, Ks, use_q=use_q, tol=tol)
            ok = r["info"]["converged"] and r["info"]["branch"] == "colonised"
            if ok:
                prev = (r["x"].copy(), r["rho"].copy(), r["n"].copy(), float(L))
            rho_max = float(r["rho"].max())
            n_L = float(r["n"][-1])
            A_eff = cl.YDn * (n0 - n_L)
            jam_num = rho_max >= p.rho_jam
            jam_crit = A_eff >= B
            rec.update(branch=r["info"]["branch"], rho_max=rho_max, n_L=n_L, A_eff=A_eff,
                       A_eff_over_B=A_eff / B, rho_over_jam=rho_max / p.rho_jam,
                       jam_numeric=bool(jam_num), jam_criterion=bool(jam_crit),
                       agree=bool(jam_num == jam_crit), converged=bool(ok),
                       iters=int(r["info"]["iters"]))
            rows.append(rec)
            if verbose:
                print("  n0=%8.4g L=%8.3f  rho/rho_jam=%7.4f  A_eff/B=%7.4f  %s"
                      % (n0, L, rec["rho_over_jam"], rec["A_eff_over_B"],
                         "agree" if rec["agree"] else "DISAGREE"))
    return rows


# --- the two L* formulas -------------------------------------------------------------------
def lstar_exact(cl: Closure, n0: float, mu: float, Ks: float, theta: float, n_quad: int):
    """eq:Lstar_exact_new -- no approximation beyond the steady PDE itself."""
    S = _source_table(cl, n0, mu, Ks, cl.p_cap(n0), n_quad)
    return quad_length(S, cl.YDn * n0 * (1.0 - theta))


def lstar_dilute_monod(cl: Closure, p_par: C.ModelParams, n0: float, mu: float, Ks: float,
                       theta: float, n_quad: int):
    """eq:Lstar_dilute_full_monod_rho0zero -- dilute (Psi ~ D0 rho), full Monod retained.

    Same quadrature as the exact one with rho(p) = p/D0 in place of Psi^-1(p), so differencing
    the two isolates the dilute approximation on its own.
    """
    p_hi = min(cl.YDn * n0, cl.psi_max) * (1.0 - 1.0e-9)
    r = np.linspace(0.0, p_hi, n_quad)
    g = C.monod(np.maximum(n0 - r / cl.YDn, 0.0), mu, Ks)
    S = PchipInterpolator(r, np.concatenate(([0.0],
                                             cumulative_trapezoid(g * (r / p_par.D0), r))))
    return quad_length(S, cl.YDn * n0 * (1.0 - theta))


def lstar_dilute_linear(cl: Closure, p_par: C.ModelParams, n0: float, mu: float, Ks: float,
                        theta: float, limit: int = 400):
    """eq:Lstar_dilute_linear_monod, evaluated from the CLOSED FORM exactly as printed.

    q^2 = (2k/D0)[ (n0/2)(p*^2-p^2) - (1/(3 Y Dn))(p*^3-p^3) ],  k = mu/Ks.
    The bracket vanishes linearly as p -> p* (it behaves as s*p*_*n_*), so the same
    s = p*-p = u^2 substitution removes the endpoint singularity.
    """
    k = mu / Ks
    YDn = cl.YDn
    p_star = YDn * n0 * (1.0 - theta)

    def integrand(u):
        pp = p_star - u * u
        br = (n0 / 2.0) * (p_star ** 2 - pp ** 2) - (p_star ** 3 - pp ** 3) / (3.0 * YDn)
        q2 = (2.0 * k / p_par.D0) * br
        return 0.0 if q2 <= 0.0 else 2.0 * u / np.sqrt(q2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(integrand, 0.0, np.sqrt(p_star), limit=limit)
    return float(val)


def check_printed_bracket(cl: Closure, n0: float, p_star: float) -> float:
    """Verify the ALGEBRA of the printed closed form, not merely its consequence:

        int_p^{p*} (n0 - s/(Y Dn)) s ds  ==  (n0/2)(p*^2-p^2) - (1/(3 Y Dn))(p*^3-p^3)
    """
    err = 0.0
    for frac in (0.1, 0.35, 0.6, 0.85):
        pp = p_star * frac
        num, _ = quad(lambda t: (n0 - t / cl.YDn) * t, pp, p_star, limit=200)
        closed = (n0 / 2.0) * (p_star ** 2 - pp ** 2) - (p_star ** 3 - pp ** 3) / (3.0 * cl.YDn)
        err = max(err, abs(num - closed) / max(abs(num), TINY))
    return err


def fold_window(cl: Closure, n0: float, mu: float, Ks: float, n_quad: int = 400_001,
                npts: int = 240, rho_lo: float = 0.02):
    """The window of L over which the steady branch is MULTI-VALUED, or None if monotone.

    Returns (window, S, rho_scan, L_scan): the branch is returned parametrised by rho, which is
    also what a bifurcation diagram wants on its vertical axis.

    THE SCAN IS UNIFORM IN RHO, NOT IN p* (equivalently, not in theta).  The turning points sit
    near rho ~ rho_jam, i.e. at p* ~ B; once p_cap >> B they occupy a vanishing slice of the
    p*-range, so a theta-uniform scan puts no nodes there at all and reports "no fold" for a
    branch that plainly folds -- at n0 = 3e4 the fold lives at theta = 0.974 while a scan over
    theta in [0.02, 0.97] never reaches it.  check_fold_bistability.py records the same lesson.
    """
    S = _source_table(cl, n0, mu, Ks, cl.p_cap(n0), n_quad)
    rho_hi = float(cl.psi_inv(cl.p_cap(n0)))
    if rho_hi <= rho_lo:
        return None, S, np.array([]), np.array([])
    rho = np.linspace(rho_lo, rho_hi, npts)
    L = np.array([quad_length(S, float(cl.psi(r))) for r in rho])
    flips = np.where(np.diff(np.sign(np.diff(L))) != 0)[0] + 1
    if len(flips) < 2 or not (L[flips[0]] > L[flips[1]]):
        return None, S, rho, L
    return (float(L[flips[1]]), float(L[flips[0]])), S, rho, L


def fold_window_refined(cl: Closure, n0: float, mu: float, Ks: float,
                        n_quad: int = 400_001, npts: int = 300):
    """The fold window with its turning points ROOT-FOUND, not read off scan nodes.

    A scan locates an extremum only to the node spacing, which leaves the window width wobbling
    in its fourth decimal -- fine for "does it fold", useless for measuring how the width VANISHES
    at onset.  Each turning point is therefore refined by bounded minimisation of +-L over the
    bracketing nodes, which costs ~30 extra quadrature calls and pins the window to ~1e-10.
    """
    win, S, rho, L = fold_window(cl, n0, mu, Ks, n_quad=n_quad, npts=npts)
    if win is None:
        return None, S
    flips = np.where(np.diff(np.sign(np.diff(L))) != 0)[0] + 1
    i_hi, i_lo = flips[0], flips[1]                    # local max of L, then local min

    def L_of(r):
        return quad_length(S, float(cl.psi(float(r))))

    out = []
    for i, sign in ((i_hi, -1.0), (i_lo, +1.0)):
        a, b = rho[max(i - 1, 0)], rho[min(i + 1, len(rho) - 1)]
        res = minimize_scalar(lambda r: sign * L_of(r), bounds=(a, b), method="bounded",
                              options=dict(xatol=1.0e-12))
        out.append((float(res.x), float(sign * res.fun)))
    (rho_hi, L_hi), (rho_lo, L_lo) = out
    return dict(L_lo=L_lo, L_hi=L_hi, width=L_hi - L_lo,
                rho_at_L_hi=rho_hi, rho_at_L_lo=rho_lo), S


def fold_cusp(cl: Closure, p_base: C.ModelParams, mu: float, Ks: float,
              n0_guess: float = None, dns=None, n_quad: int = 2_000_001, npts: int = 2400):
    """Measure the fold ONSET and its scaling exponents from an n0 sweep of the window.

    Two things the sweep gives that a single scan cannot.

    (1) The onset.  Bisecting on "is a fold detected" finds where the window drops below the
        scan's own resolution.  The turning points instead SEPARATE as (n0-n0*)^(1/2), so their
        squared separation is LINEAR in n0 and extrapolates to zero at n0* -- a well-conditioned
        measurement that uses the sweep rather than one marginal detection.

    (2) A falsifiable prediction.  At a cusp the window width vanishes as (n0-n0*)^(3/2) and the
        separation as (n0-n0*)^(1/2).  Measuring both tests whether the fold is born in a cusp
        rather than merely asserting that it appears.

    Both exponents are LOCAL limits: they only emerge within ~1 uM of onset.  Fitting a single
    power law out to n0-n0* ~ 40 returns alpha ~ 1.31 and looks respectable (R^2 = 0.998) while
    being wrong -- the sampling range, not the fit quality, is what decides this.
    """
    n0_guess = n0_guess if n0_guess is not None else fold_threshold(p_base, cl, mu, Ks)[1]
    dns = dns if dns is not None else np.array([0.14, 0.34, 0.84, 1.84, 3.84, 7.84, 15.84])
    rows = []
    for dn in dns:
        r, _ = fold_window_refined(cl, float(n0_guess + dn), mu, Ks, n_quad, npts)
        if r is not None and r["width"] > 0.0:
            rows.append(dict(n0=float(n0_guess + dn), width=r["width"],
                             d_rho=r["rho_at_L_lo"] - r["rho_at_L_hi"],
                             L_lo=r["L_lo"], L_hi=r["L_hi"]))
    out = dict(rows=rows, n0_star=float("nan"), alpha_width=float("nan"),
               alpha_drho=float("nan"), n0_guess=float(n0_guess))
    if len(rows) < 3:
        return out
    nv = np.array([r["n0"] for r in rows])
    dv = np.array([r["d_rho"] for r in rows])
    wv = np.array([r["width"] for r in rows])
    c = np.polyfit(nv[:3], dv[:3] ** 2, 1)            # (d_rho)^2 is linear near the cusp
    out["n0_star"] = float(-c[1] / c[0])
    dn = nv - out["n0_star"]
    ok = dn > 0
    if ok.sum() >= 2:
        lw = np.log(wv[ok]); ld = np.log(dv[ok]); ln = np.log(dn[ok])
        out["alpha_width"] = float((lw[1] - lw[0]) / (ln[1] - ln[0]))
        out["alpha_drho"] = float((ld[1] - ld[0]) / (ln[1] - ln[0]))
        out["local"] = [(float(dn[ok][k]),
                         float((lw[k + 1] - lw[k]) / (ln[k + 1] - ln[k])),
                         float((ld[k + 1] - ld[k]) / (ln[k + 1] - ln[k])))
                        for k in range(ok.sum() - 1)]
    return out


def count_states(th, L, L_target: float) -> int:
    """How many steady states share this cavity length, from the scanned branch."""
    d = L - L_target
    return int(np.sum(d[:-1] * d[1:] < 0.0))


def lstar_sweep(p_base: C.ModelParams, cl: Closure, mu: float, Ks: float, n0s, thetas,
                dx_target: float = 0.5, n_quad: int = 400_001, use_q: bool = False,
                tol: float = 1.0e-9, dilute: bool = False, verbose: bool = False):
    """For each (n0, theta): predict L* from the formula, SOLVE the PDE at that L*, and read
    n(L)/n0 back.  If the formula is right the solve returns theta."""
    rows, folds = [], {}
    for n0 in n0s:
        p = _with_n0(p_base, n0)
        L_on = onset_length(p, mu, Ks)
        win, _S, rho_scan, L_scan = fold_window(cl, n0, mu, Ks, n_quad)
        folds[float(n0)] = win
        if win is not None and verbose:
            print("  n0=%g: FOLD, L in [%.4f, %.4f] um is multi-valued"
                  % (n0, win[0], win[1]))
        for th in thetas:
            L_ex = lstar_exact(cl, n0, mu, Ks, th, n_quad)
            in_fold = bool(win is not None and win[0] <= L_ex <= win[1])
            rec = dict(n0=float(n0), theta=float(th), L_exact=L_ex, L_onset=L_on,
                       eps_dilute=cl.A(n0) / (p.D0 * p.rho_jam), n0_over_Ks=n0 / Ks,
                       in_fold=in_fold, n_states=count_states(rho_scan, L_scan, L_ex))
            if dilute:
                rec["L_dilute_monod"] = lstar_dilute_monod(cl, p, n0, mu, Ks, th, n_quad)
                rec["L_dilute_linear"] = lstar_dilute_linear(cl, p, n0, mu, Ks, th)
            if not np.isfinite(L_ex) or L_ex <= L_on:
                rec.update(theta_num=float("nan"), converged=False)
                rows.append(rec)
                continue
            r = solve_steady(L_ex, _sweep_N(L_ex, dx_target), p, cl, mu, Ks,
                             use_q=use_q, tol=tol)
            ok = r["info"]["converged"] and r["info"]["branch"] == "colonised"
            rec.update(theta_num=float(r["n"][-1]) / n0, converged=bool(ok))
            rows.append(rec)
            if verbose:
                print("  n0=%8.4g theta=%5.3f  L*=%9.3f  theta_num=%.6f%s"
                      % (n0, th, L_ex, rec["theta_num"],
                         "   [in fold, %d states]" % rec["n_states"] if in_fold else ""))
    return rows, folds


def leading_eigenvalue(st: Steady, u, k: int = 4):
    """Rightmost eigenvalue of the linearisation about a steady state; > 0 means unstable.

    The residual carries the nutrient block scaled by Y (F_n = Y * dn/dt), so the DYNAMICAL
    Jacobian is that block-row divided by Y again.  ARPACK cannot chase the rightmost eigenvalue
    directly here -- the spectrum is dominated by diffusive modes of order -D_n/dx^2, which after
    the 1/Y rescaling are ~1e10 -- so it is found by SHIFT-INVERT about 0, where the mode that
    actually decides stability lives.  At a turning point the Jacobian is singular and this (like
    Newton itself) is ill-conditioned; such points are reported, not hidden.
    """
    scale = np.concatenate([np.ones(st.N), np.full(st.N, 1.0 / st.p.Y)])
    Jd = (diags(scale, 0, format="csc") @ st.jacobian(u)).tocsc()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam = eigs(Jd, k=k, sigma=0.0, which="LM",
                       return_eigenvectors=False, tol=1.0e-9)
        return float(np.max(lam.real))
    except Exception:                                                  # noqa: BLE001
        return float("nan")


def fold_branch(p_base: C.ModelParams, cl: Closure, mu: float, Ks: float, n0: float,
                npts: int = 90, N: int = 400, n_quad: int = 400_001, tol: float = 1.0e-9,
                rho_lo: float = 0.02, verify: bool = True, stability: bool = True,
                verbose: bool = False):
    """Trace the steady branch parametrised by rho, verifying each point against the PDE.

    Parametrising by RHO rather than by L is what makes the unstable middle segment reachable
    at all: as a function of L the branch is multi-valued, so there is no "the" solution to
    solve for, and seeding by hand lands wherever it happens to land.  Given rho, the quadrature
    supplies both the matching L and an essentially exact profile to seed Newton with, so each
    point of the S-curve -- including the unstable one -- is pinned deliberately.
    """
    p = _with_n0(p_base, n0)
    S = _source_table(cl, n0, mu, Ks, cl.p_cap(n0), n_quad)
    rho_hi = float(cl.psi_inv(cl.p_cap(n0)))
    rows = []
    for rt in np.linspace(rho_lo, rho_hi * (1.0 - 1.0e-9), npts):
        p_star = float(cl.psi(rt))
        L = quad_length(S, p_star)
        rec = dict(n0=float(n0), rho_branch=float(rt), L=float(L),
                   rho_over_jam=float(rt) / p.rho_jam)
        if verify and np.isfinite(L) and L > 0.0:
            st = Steady(L, N, p, cl, mu, Ks, False)
            rho0, nn0 = quad_profile(cl, n0, L, p_star, S, st.x)
            u, info = newton(st, st.pack(rho0, nn0), tol=tol)
            rho, _ = st.unpack(u)
            rec.update(rho_num=float(rho.max()), converged=bool(info["converged"]),
                       resid=float(info["residual"]))
            rec["rho_err"] = abs(rec["rho_num"] - rt)
            if stability:
                lam = leading_eigenvalue(st, u)
                rec["lambda_max"] = lam
                rec["stable"] = bool(lam < 0.0) if np.isfinite(lam) else None
        rows.append(rec)
        if verbose:
            print("   rho=%.4f  L=%.5f  rho_num=%s  lam=%s"
                  % (rt, L, rec.get("rho_num"), rec.get("lambda_max")))
    return rows, S


def fold_threshold(p_base: C.ModelParams, cl: Closure, mu: float, Ks: float,
                   lo: float = None, hi: float = 3.0, n_quad: int = 200_001,
                   npts: int = 400, iters: int = 22):
    """Bracket the smallest n0 whose branch folds, by bisection on 'a fold is detected'.

    The window width vanishes CONTINUOUSLY at the threshold (7.6e-3 um at n0 = 900), so what a
    scan of finite resolution returns is a bracket, not a number: below some width the two
    turning points fall between adjacent nodes and the fold stops being visible.  Reported as a
    bracket for that reason.
    """
    lo = lo if lo is not None else cl.n0_jam * 1.001      # below n0_crit nothing can jam at all
    has = lambda v: fold_window(cl, v, mu, Ks, n_quad=n_quad, npts=npts)[0] is not None  # noqa
    if has(lo):
        return (float("nan"), float(lo))
    if not has(hi * cl.n0_jam):
        return (float(hi * cl.n0_jam), float("nan"))
    a, b = lo, hi * cl.n0_jam
    for _ in range(iters):
        m = 0.5 * (a + b)
        if has(m):
            b = m
        else:
            a = m
    return (float(a), float(b))


def dcol_shape(p: C.ModelParams):
    """Where D_col peaks below rho_jam, and how far it falls from there to rho_jam.

    That fractional drop is the MECHANICAL driver of the fold: with no interior maximum
    (eta <= 4) D_col rises monotonically and no amount of nutrient produces one.
    """
    r = np.linspace(0.0, p.rho_jam, 20001)
    D = np.asarray(C.collective_diffusivity(r, p), float)
    i = int(np.argmax(D))
    return float(r[i]), float((D[i] - D[-1]) / D[i])


def fold_params_study(p_base: C.ModelParams, cl_base: Closure, mu0: float, Ks0: float,
                      n_quad: int = 200_001, npts: int = 300, iters: int = 18,
                      hi: float = 8.0, n_grid: int = 150_001, verbose: bool = False):
    """What sets whether a cavity folds at all.

    Non-dimensionalising the branch with P = p/B gives

        L = sqrt(B/mu_max) * F(P*; nu0, Gamma),   nu0 = A/B,  Gamma = B/(Y Dn Ks) = n0_crit/Ks,

    so mu_max only rescales L and cannot move any threshold, while everything else should enter
    through the two groups plus the SHAPE of Psi.  This measures all three claims:

      mu      -- width * sqrt(mu) invariant, and the turning point in rho unmoved
      collapse-- parameter sets reaching the same Gamma by different routes (Ks, Y, Dn, D0)
                 must give the same (A/B)*
      gamma   -- the law (A/B)* = f(Gamma)
      closure -- eta and rho_jam change the SHAPE of Psi, so they are expected to move the law
                 even at fixed Gamma; ordered here by the D_col drop
    """
    out = {}

    # --- mu: scales out ------------------------------------------------------------------
    rows = []
    for mu in (0.2, mu0, 3.0):
        r, _ = fold_window_refined(cl_base, 1500.0, mu, Ks0, n_quad=800_001, npts=800)
        if r:
            rows.append(dict(mu=mu, L_lo=r["L_lo"], width=r["width"],
                             width_sqrt_mu=r["width"] * np.sqrt(mu),
                             rho_at_L_lo=r["rho_at_L_lo"]))
    out["mu"] = rows

    # --- Gamma collapse: same Gamma, different routes ------------------------------------
    G0 = cl_base.B / (cl_base.YDn * Ks0)
    routes = [("baseline", Ks0, p_base.Y, p_base.Dn, p_base.D0),
              ("Ks x2, Y /2", 2 * Ks0, p_base.Y / 2, p_base.Dn, p_base.D0),
              ("Ks /10, Dn x10", Ks0 / 10, p_base.Y, p_base.Dn * 10, p_base.D0),
              ("Ks x10, Y /10", 10 * Ks0, p_base.Y / 10, p_base.Dn, p_base.D0),
              ("D0 x2, Ks x2", 2 * Ks0, p_base.Y, p_base.Dn, 2 * p_base.D0),
              ("D0 /2, Ks /2", Ks0 / 2, p_base.Y, p_base.Dn, p_base.D0 / 2)]
    rows = []
    for name, Ks, Y, Dn, D0 in routes:
        p = config.params(15.0, Y=Y, Dn=Dn, D0=D0)
        cl = Closure(p, n_grid=n_grid)
        t = fold_threshold(p, cl, mu0, Ks, n_quad=n_quad, npts=npts, iters=iters, hi=hi)
        rows.append(dict(route=name, Ks=Ks, Y=Y, Dn=Dn, D0=D0, B=cl.B,
                         Gamma=cl.B / (cl.YDn * Ks), n0_crit=cl.n0_jam, n0_star=t[1],
                         AB_star=cl.A(t[1]) / cl.B if np.isfinite(t[1]) else float("nan")))
        if verbose:
            print("    %-16s Gamma=%.4f  (A/B)*=%s" % (name, rows[-1]["Gamma"],
                                                       rows[-1]["AB_star"]))
    out["collapse"] = rows

    # --- the law (A/B)* = f(Gamma), swept through Ks --------------------------------------
    # The Ks grid is dense between 2 and 8 uM because that is where the continuous window
    # CLOSES, and the closure is the one feature a coarse grid cannot see.  The bisection floor
    # is also dropped from n0_crit*1.001 to n0_crit*(1+1e-7): at the old floor every Ks below
    # ~4 uM returned (A/B)* = 1.001 -- the floor itself, not a measurement -- which made the
    # window look 0.1% wide where it is actually zero.
    ks_grid = np.unique(np.concatenate([
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5,
                  6.0, 7.0, 8.0]),
        np.array([10.0, Ks0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0])]))
    floor = cl_base.n0_jam * (1.0 + 1.0e-7)
    rows = []
    for Ks in ks_grid:
        t = fold_threshold(p_base, cl_base, mu0, Ks, lo=floor, n_quad=n_quad, npts=npts,
                           iters=max(iters, 26), hi=hi)
        ab = cl_base.A(t[1]) / cl_base.B if np.isfinite(t[1]) else float("nan")
        rows.append(dict(Ks=Ks, Gamma=cl_base.B / (cl_base.YDn * Ks),
                         n0_crit=cl_base.n0_jam, n0_star=t[1], AB_star=ab,
                         window_pct=100.0 * (ab - 1.0) if np.isfinite(ab) else float("nan")))
        if verbose:
            print("    Ks=%-8g Gamma=%9.3f  (A/B)*=%s" % (Ks, rows[-1]["Gamma"], ab))
    out["gamma"] = rows
    out["Ks_closure"] = fold_closure_Ks(rows, p_base, cl_base, mu0,
                                        n_quad=n_quad, npts=npts)

    # --- the closure shape, compared at FIXED Gamma ---------------------------------------
    rows = []
    for D0, eta, rj in ((p_base.D0, 4.0, 0.64), (p_base.D0, 4.5, 0.64),
                        (p_base.D0, 5.8, 0.55), (p_base.D0, 5.8, 0.64),
                        (p_base.D0, 5.8, 0.72), (p_base.D0, 7.0, 0.64)):
        p = config.params(15.0, D0=D0, eta=eta, rho_jam=rj)
        cl = Closure(p, n_grid=n_grid)
        Ks = cl.B / (cl.YDn * G0)                     # hold Gamma at the baseline value
        peak, drop = dcol_shape(p)
        t = fold_threshold(p, cl, mu0, Ks, lo=cl.n0_jam * (1.0 + 1.0e-7), n_quad=n_quad,
                           npts=npts, iters=max(iters, 26), hi=hi)
        rows.append(dict(D0=D0, eta=eta, rho_jam=rj, B=cl.B, Ks_for_fixed_Gamma=Ks,
                         Gamma=G0, rho_peak=peak, dcol_drop=drop, n0_star=t[1],
                         AB_star=cl.A(t[1]) / cl.B if np.isfinite(t[1]) else float("nan")))
        if verbose:
            print("    eta=%-5g rho_jam=%-5g drop=%5.1f%%  (A/B)*=%s"
                  % (eta, rj, 100 * drop, rows[-1]["AB_star"]))
    out["closure"] = rows

    # --- what n0 above n0* buys: the bistable L-window ------------------------------------
    rows = []
    for n0 in (900.0, 1000.0, 1500.0, 3000.0, 10000.0, 100000.0):
        r, _ = fold_window_refined(cl_base, n0, mu0, Ks0, n_quad=800_001, npts=900)
        if r:
            rows.append(dict(n0=n0, L_lo=r["L_lo"], L_hi=r["L_hi"], width=r["width"],
                             width_pct=100.0 * r["width"] / r["L_lo"]))
    out["window"] = rows
    return out


# --- figures for the sweep modes -----------------------------------------------------------
def figure_budget(rows, cl: Closure, p_base: C.ModelParams, mu: float, Ks: float, path,
                  n_quad: int = 400_001):
    n0s = sorted({r["n0"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.4))

    a = ax[0]
    for r in rows:
        if r["branch"] == "trivial":
            a.plot(r["n0"], r["L"], "x", color="0.6", ms=5, mew=1.1)
        elif r["jam_numeric"]:
            a.plot(r["n0"], r["L"], "s", color=C2, ms=5)
        else:
            a.plot(r["n0"], r["L"], "o", color=C1, ms=5, mfc="none", mew=1.1)
        if not r["agree"]:
            a.plot(r["n0"], r["L"], "*", color="k", ms=14)
    nn = np.geomspace(min(n0s), max(n0s), 80)
    a.plot(nn, [onset_length(_with_n0(p_base, v), mu, Ks) for v in nn], "-",
           color="0.35", lw=1.5, label=r"$L_{\rm onset}$ (analytic)")
    lj = np.array([jam_length(cl, v, mu, Ks, n_quad) for v in nn])
    fin = np.isfinite(lj)
    a.plot(nn[fin], lj[fin], "--", color=C2, lw=1.8, label=r"$L_{\rm jam}$ (quadrature)")
    a.axvline(cl.n0_jam, color="k", ls=":", lw=1.3,
              label=r"$n_0^{\rm crit}$=%.1f $\mu$M ($A\!=\!B$)" % cl.n0_jam)
    a.plot([], [], "x", color="0.6", label="empty (trivial state)")
    a.plot([], [], "o", color=C1, mfc="none", label="colonised, sub-jammed")
    a.plot([], [], "s", color=C2, label="jammed")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$n_0$ [$\mu$M]"); a.set_ylabel(r"$L$ [$\mu$m]")
    a.set_title("(a) phase diagram from direct solutions", fontsize=10, fontweight="bold")
    # Below the axes, not on top of them: every corner of this panel carries data.
    a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
             frameon=False)
    a.grid(alpha=.2, which="both")

    a = ax[1]
    live = [r for r in rows if r["branch"] != "trivial"]
    a.plot([r["rho_over_jam"] for r in live], [r["A_eff_over_B"] for r in live],
           "o", color=C1, ms=4.5, mfc="none", mew=1.1)
    a.axhline(1.0, color=C2, ls="--", lw=1.3)
    a.axvline(1.0, color=C2, ls="--", lw=1.3)
    bad = [r for r in live if not r["agree"]]
    if bad:
        a.plot([r["rho_over_jam"] for r in bad], [r["A_eff_over_B"] for r in bad],
               "*", color="k", ms=14, label="criterion disagrees")
        a.legend(fontsize=7)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$\rho_{\max}/\rho_{\rm jam}$   (direct solve)")
    a.set_ylabel(r"$A_{\rm eff}/B$   (criterion)")
    a.set_title("(b) criterion vs numerics: both cross 1 together\n"
                "%d/%d points agree" % (sum(r["agree"] for r in rows), len(rows)),
                fontsize=10, fontweight="bold")
    a.grid(alpha=.2, which="both")

    _suptitle(fig, r"jamming criterion $A_{\rm eff}=YD_n(n_0-n(L))>B$ against direct solutions"
                 "\n" r"$\mu$=%g 1/h, $K_s$=%g $\mu$M, $B$=%.2f" % (mu, Ks, cl.B),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, .90))
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # The points are budget_sweep.csv; the two analytic curves of (a) exist only here.
    stem = os.path.splitext(path)[0]
    L_on_curve = [onset_length(_with_n0(p_base, v), mu, Ks) for v in nn]
    with open(stem + "_analytic.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n0_uM", "L_onset_analytic_um", "L_jam_quadrature_um"])
        for k in range(nn.size):
            w.writerow([f"{nn[k]:.10g}", f"{L_on_curve[k]:.10g}",
                        f"{lj[k]:.10g}" if np.isfinite(lj[k]) else ""])
    with open(stem + "_marks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in (("n0_crit_uM", cl.n0_jam), ("B", cl.B), ("mu_per_h", mu), ("Ks_uM", Ks),
                     ("points_agree", sum(r["agree"] for r in rows)), ("points", len(rows))):
            w.writerow([k, f"{v:.10g}"])


def figure_fold(main_rows, family, widths, cl: Closure, p_base: C.ModelParams,
                mu: float, Ks: float, path, n0_main: float, window, thresh=None, cusp=None):
    """The fold: an S-curve in (L, rho), its stability, and how it grows with n0."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax = ax.ravel()

    # (a) the bifurcation diagram, with stability from the eigenvalue
    a = ax[0]
    L = np.array([r["L"] for r in main_rows])
    rho = np.array([r["rho_branch"] for r in main_rows])
    a.plot(L, rho, "-", color="0.55", lw=1.2, zorder=1)
    st_ = np.array([r.get("stable") is True for r in main_rows])
    un_ = np.array([r.get("stable") is False for r in main_rows])
    Ln = np.array([r.get("rho_num", np.nan) for r in main_rows])
    a.plot(L[st_], Ln[st_], "o", color=C1, ms=4.5, label="stable (PDE solve)", zorder=3)
    a.plot(L[un_], Ln[un_], "x", color=C2, ms=6, mew=1.6,
           label=r"unstable, $\mathrm{Re}\,\lambda>0$", zorder=3)
    if window is not None:
        a.axvspan(window[0], window[1], color=CN, alpha=.16, zorder=0,
                  label="3 steady states")
        for v in window:
            a.axvline(v, color=CN, lw=1, ls="--", zorder=2)
    a.axhline(p_base.rho_jam, color="k", ls=":", lw=1, label=r"$\rho_{\rm jam}$")
    # The branch runs out to L ~ 90 um at the top, which squashes the fold -- 0.65 um wide --
    # into a vertical sliver.  Zoom on the window that is actually the subject of the panel.
    if window is not None:
        w = max(window[1] - window[0], 1.0e-6)
        a.set_xlim(window[0] - 4.0 * w, window[1] + 4.0 * w)
        inside = (L >= window[0]) & (L <= window[1])
        if inside.any():
            lo_r, hi_r = rho[inside].min(), rho[inside].max()
            pad = 0.25 * max(hi_r - lo_r, 1.0e-3)
            a.set_ylim(max(0.0, lo_r - 3 * pad), hi_r + pad)
        # make "three steady states" literal: one L, three intersections
        Lmid = 0.5 * (window[0] + window[1])
        d = L - Lmid
        cross = [i for i in range(len(L) - 1) if d[i] * d[i + 1] < 0.0]
        if cross:
            a.axvline(Lmid, color="k", lw=1.1, ls="-", alpha=.55, zorder=2)
            for i in cross:
                t = d[i] / (d[i] - d[i + 1])
                a.plot(Lmid, rho[i] + t * (rho[i + 1] - rho[i]), "o", color="k",
                       ms=8, mfc="none", mew=1.6, zorder=4)
            a.plot([], [], "ko", mfc="none", mew=1.6,
                   label=r"%d states at $L$=%.2f $\mu$m" % (len(cross), Lmid))
    a.set_xlabel(r"$L$ [$\mu$m]"); a.set_ylabel(r"$\rho(L)$")
    a.set_title(r"(a) fold at $n_0$=%g $\mu$M" % n0_main, fontsize=10, fontweight="bold")
    a.legend(fontsize=7, loc="lower right"); a.grid(alpha=.2)

    # (b) how the fold appears and grows with n0
    a = ax[1]
    n0_star = (cusp or {}).get("n0_star", float("nan"))
    cols = plt.cm.viridis(np.linspace(0, .85, len(family)))
    for c, (v, rr) in zip(cols, sorted(family.items())):
        # the two thresholds are drawn as named curves rather than as numbers in the caption:
        # at n0_crit the branch only just reaches rho_jam, and at n0* its tangent first goes
        # vertical -- the fold being born is visible, not merely asserted.
        if abs(v - cl.n0_jam) / cl.n0_jam < 5.0e-3:
            sty = dict(color="k", lw=2.2, ls="-",
                       label=r"$n_0^{\rm crit}$=%.0f (jamming just possible)" % cl.n0_jam)
        elif np.isfinite(n0_star) and abs(v - n0_star) / n0_star < 5.0e-3:
            sty = dict(color=C2, lw=2.2, ls="--",
                       label=r"$n_0^{*}$=%.0f (fold appears)" % n0_star)
        else:
            sty = dict(color=c, lw=1.5, ls="-", label=r"$n_0$=%g" % v)
        a.plot([r["L"] for r in rr], [r["rho_branch"] for r in rr], **sty)
    a.axhline(p_base.rho_jam, color="k", ls=":", lw=1)
    if window is not None:
        w = max(window[1] - window[0], 1.0e-6)
        a.set_xlim(window[0] - 6.0 * w, window[1] + 10.0 * w)
        a.set_ylim(0.0, 0.80)
    a.set_xlabel(r"$L$ [$\mu$m]"); a.set_ylabel(r"$\rho(L)$")
    a.set_title("(b) the branch as $n_0$ increases", fontsize=10, fontweight="bold")
    a.legend(fontsize=7, loc="lower right"); a.grid(alpha=.2)

    # (c) width of the multi-valued window
    a = ax[2]
    vs = np.array([w[0] for w in widths])
    lo = np.array([w[1] for w in widths])
    hi = np.array([w[2] for w in widths])
    ok = np.isfinite(lo)
    a.fill_between(vs[ok], lo[ok], hi[ok], color=CN, alpha=.3, label="multi-valued $L$")
    a.plot(vs[ok], lo[ok], "-", color=C1, lw=1.4, label=r"lower tipping point")
    a.plot(vs[ok], hi[ok], "-", color=C2, lw=1.4, label=r"upper tipping point")
    a.axvline(cl.n0_jam, color="k", ls=":", lw=1.2,
              label=r"$n_0^{\rm crit}$=%.0f $\mu$M ($A\!=\!B$)" % cl.n0_jam)
    n0s = (cusp or {}).get("n0_star", float("nan"))
    if not np.isfinite(n0s) and thresh is not None:
        n0s = thresh[1]
    if np.isfinite(n0s):
        a.axvline(n0s, color=C2, ls="-.", lw=1.6,
                  label=r"$n_0^{*}$=%.1f $\mu$M (fold appears)" % n0s)
    a.set_xscale("log")
    a.set_xlabel(r"$n_0$ [$\mu$M]"); a.set_ylabel(r"$L$ [$\mu$m]")
    a.set_title("(c) where the branch is multi-valued", fontsize=10, fontweight="bold")
    a.legend(fontsize=7); a.grid(alpha=.2, which="both")

    # (d) the cusp: how the fold is born
    a = ax[3]
    if cusp and cusp.get("rows") and np.isfinite(cusp.get("n0_star", np.nan)):
        nv = np.array([r["n0"] for r in cusp["rows"]])
        dn = nv - cusp["n0_star"]
        wv = np.array([r["width"] for r in cusp["rows"]])
        dv = np.array([r["d_rho"] for r in cusp["rows"]])
        m = dn > 0
        a.loglog(dn[m], wv[m], "o-", color=C1, lw=1.4, ms=5,
                 label=r"window width, $\alpha$=%.3f" % cusp["alpha_width"])
        a.loglog(dn[m], dv[m], "s-", color=C2, lw=1.4, ms=5,
                 label=r"tipping-point separation in $\rho$, $\alpha$=%.3f"
                       % cusp["alpha_drho"])
        d0 = dn[m][0]
        a.loglog(dn[m], wv[m][0] * (dn[m] / d0) ** 1.5, ":", color="0.4", lw=1.2,
                 label=r"cusp: $(n_0-n_0^*)^{3/2}$")
        a.loglog(dn[m], dv[m][0] * (dn[m] / d0) ** 0.5, "--", color="0.4", lw=1.2,
                 label=r"cusp: $(n_0-n_0^*)^{1/2}$")
        a.set_xlabel(r"$n_0-n_0^{*}$ [$\mu$M]")
        a.set_title(r"(d) the fold is born in a cusp at $n_0^{*}$=%.1f $\mu$M"
                    % cusp["n0_star"], fontsize=10, fontweight="bold")
        a.legend(fontsize=7, loc="lower right")
    else:
        a.text(.5, .5, "cusp scaling not measured", ha="center", va="center",
               transform=a.transAxes, fontsize=9, color="grey")
        a.set_title("(d) cusp scaling", fontsize=10, fontweight="bold")
    a.grid(alpha=.2, which="both")

    _suptitle(fig, "the jamming fold: three steady states share one cavity length"
                 "\n" r"$\mu$=%g 1/h, $K_s$=%g $\mu$M   "
                 r"(branch traced by quadrature, every point re-solved from the PDE)"
                 % (mu, Ks), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # (a) is fold_branch.csv, (c) fold_window_vs_n0.csv, (d) fold_cusp.csv.  The family of (b)
    # and the marked values exist only here.
    stem = os.path.splitext(path)[0]
    with open(stem + "_family.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n0_uM", "L_um", "rho_branch"])
        for v, rr in sorted(family.items()):
            for r in rr:
                w.writerow([f"{v:.10g}", f"{r['L']:.10g}", f"{r['rho_branch']:.10g}"])
    with open(stem + "_marks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        marks = [("n0_main_uM", n0_main), ("n0_crit_uM", cl.n0_jam), ("rho_jam", p_base.rho_jam),
                 ("window_L_lo_um", window[0] if window is not None else float("nan")),
                 ("window_L_hi_um", window[1] if window is not None else float("nan")),
                 ("n0_star_cusp_uM", (cusp or {}).get("n0_star", float("nan"))),
                 ("n0_star_bisection_uM", thresh[1] if thresh is not None else float("nan")),
                 ("alpha_width", (cusp or {}).get("alpha_width", float("nan"))),
                 ("alpha_drho", (cusp or {}).get("alpha_drho", float("nan")))]
        for k, v in marks:
            w.writerow([k, f"{v:.10g}"])


def fold_closure_Ks(gamma_rows, p_base: C.ModelParams, cl: Closure, mu0: float,
                    tol_pct=1.0e-3, iters=13, n_quad=200_001, npts=400):
    """Ks at which the continuous-jamming window shuts -- by bisection, not extrapolation.

    Below this Ks the fold reaches down to n0_crit itself, so jamming is a fold at every supply
    that jams at all.  The width vanishes continuously, so "closure" means "width below
    tol_pct"; at the measured slope ~2.1 %/uM that pins Ks to ~1e-4 uM.  A linear extrapolation
    of the width is NOT equivalent and is worse: the width is convex near closure, so fitting
    over Ks in [4.25, 5.0] returns 3.98, below the last node measured shut (4.0).
    """
    op = [r["Ks"] for r in gamma_rows
          if np.isfinite(r["window_pct"]) and r["window_pct"] > tol_pct]
    if not op:
        return float("nan")
    hi = min(op)
    shut = [r["Ks"] for r in gamma_rows
            if r["Ks"] < hi and (not np.isfinite(r["window_pct"])
                                 or r["window_pct"] <= tol_pct)]
    if not shut:
        return float("nan")
    lo = max(shut)
    floor = cl.n0_jam * (1.0 + 1.0e-7)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        t = fold_threshold(p_base, cl, mu0, mid, lo=floor, n_quad=n_quad, npts=npts,
                           iters=26, hi=8.0)
        w = (100.0 * (t[1] / cl.n0_jam - 1.0)) if np.isfinite(t[1]) else 0.0
        if w > tol_pct:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fold_params_closure_Ks(gamma_rows, nfit=4):
    """Ks at which the continuous-jamming window shuts, by extrapolating its width to zero.

    Below this Ks the fold reaches all the way down to n0_crit, so jamming is a fold at every
    supply that jams at all.  Bracketing the closure by "lowest Ks with a measurable window"
    would just report a grid node; the width vanishes continuously, so a linear fit through the
    lowest open points is both sharper and independent of where the sweep put its nodes.
    """
    # A CLOSED row is not window_pct == 0: fold_threshold returns its own lower bracket, so a
    # closed row reads window_pct = 100*(floor-1) = 1e-5 %, not zero.  Filtering at 1e-6 % let
    # those through, the "lowest open points" were four closed ones with identical y, and the
    # fitted slope came out ~0 -- which sent the extrapolated closure to -8e15.  Cut a long way
    # above the floor and far below the smallest real window (0.6 % at Ks=4.25).
    o = sorted((r for r in gamma_rows
                if np.isfinite(r["window_pct"]) and r["window_pct"] > 1.0e-2),
               key=lambda r: r["Ks"])[:nfit]
    if len(o) < 2:
        return float("nan")
    k = np.array([r["Ks"] for r in o])
    w = np.array([r["window_pct"] for r in o])
    sl, ic = np.polyfit(k, w, 1)
    return float(-ic / sl) if sl > 0 else float("nan")


def figure_fold_params(study, cl: Closure, mu0: float, Ks0: float, path):
    """What sets whether a cavity folds: the Gamma law, the mu invariance, the closure shape."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = ax.ravel()

    # (a) the master law
    a = ax[0]
    g = [r for r in study["gamma"] if np.isfinite(r["AB_star"])]
    a.semilogx([r["Gamma"] for r in g], [r["AB_star"] for r in g], "o-", color=C1,
               lw=1.6, ms=6, label=r"$(A/B)^*$ measured")
    a.axhline(1.0, color="k", ls=":", lw=1, label=r"$A=B$ (jamming becomes possible)")
    col = [r for r in study["collapse"] if np.isfinite(r["AB_star"])]
    a.plot([r["Gamma"] for r in col], [r["AB_star"] for r in col], "x", color=C2, ms=11,
           mew=2, label="collapse check: %d routes to one $\\Gamma$" % len(col))
    a.set_xlabel(r"$\Gamma=B/(YD_nK_s)=n_0^{\rm crit}/K_s$")
    a.set_ylabel(r"$(A/B)^*$")
    a.set_title(r"(a) the threshold depends on the parameters only through $\Gamma$",
                fontsize=10, fontweight="bold")
    a.legend(fontsize=7); a.grid(alpha=.2, which="both")

    # (b) mu scales out
    a = ax[1]
    m = study["mu"]
    mv = np.array([r["mu"] for r in m])
    a.loglog(mv, [r["width"] for r in m], "o-", color=C1, lw=1.5, ms=6,
             label=r"window width $\propto\mu^{-1/2}$")
    a.loglog(mv, [r["width_sqrt_mu"] for r in m], "s--", color=C2, lw=1.5, ms=6,
             label=r"width $\times\sqrt{\mu}$  (invariant)")
    a.set_xlabel(r"$\mu_{\max}$ [1/h]"); a.set_ylabel(r"$L$ window [$\mu$m]")
    a.set_title(r"(b) $\mu_{\max}$ only rescales $L$; it cannot make a fold",
                fontsize=10, fontweight="bold")
    a.legend(fontsize=7); a.grid(alpha=.2, which="both")

    # (c) the mechanical driver
    a = ax[2]
    cr = study["closure"]
    for r in cr:
        okv = np.isfinite(r["AB_star"])
        a.plot(100 * r["dcol_drop"], r["AB_star"] if okv else 1.0,
               "o" if okv else "v", color=C1 if okv else "0.5", ms=8,
               mfc=C1 if okv else "none", mew=1.5)
        a.annotate(r"$\eta$=%g, $\rho_{\rm jam}$=%g" % (r["eta"], r["rho_jam"]),
                   (100 * r["dcol_drop"], r["AB_star"] if okv else 1.0),
                   textcoords="offset points", xytext=(7, 5), fontsize=6.5)
    a.plot([], [], "v", color="0.5", mfc="none", mew=1.5, label="no fold found")
    a.axhline(1.0, color="k", ls=":", lw=1)
    a.set_xlabel(r"$D_{\rm col}$ drop from its peak to $\rho_{\rm jam}$  [%]")
    a.set_ylabel(r"$(A/B)^*$")
    a.set_title(r"(c) at fixed $\Gamma$, the closure SHAPE still decides",
                fontsize=10, fontweight="bold")
    a.legend(fontsize=7); a.grid(alpha=.2)

    # (d) the two edges, and what n0 beyond them buys
    a = ax[3]
    a.semilogx([r["Ks"] for r in g], [r["n0_star"] for r in g], "o-", color=C2, lw=1.6,
               ms=6, label=r"$n_0^*$  (jamming becomes a fold)")
    a.axhline(cl.n0_jam, color=C1, lw=1.6,
              label=r"$n_0^{\rm crit}$=%.1f $\mu$M  ($K_s$-independent)" % cl.n0_jam)
    a.fill_between([r["Ks"] for r in g], cl.n0_jam, [r["n0_star"] for r in g],
                   color=CN, alpha=.22, label="jams, but continuously")
    # The closure is where the window WIDTH reaches zero, and the width goes to zero
    # continuously -- so the smallest Ks that happens to be resolvable is a property of
    # the grid, not of the physics.  Extrapolate the width to zero from the lowest open
    # points instead: grid-independent, and it lands between the last closed node and the
    # first open one rather than on whichever node the sweep happened to place there.
    Ks_c = study.get("Ks_closure", float("nan"))
    if not np.isfinite(Ks_c):
        Ks_c = fold_params_closure_Ks(g)
    if np.isfinite(Ks_c):
        a.axvline(Ks_c, color="0.35", ls="--", lw=1.2,
                  label=r"window shuts at $K_s$=%.2f $\mu$M (width$\to$0)" % Ks_c)
    a.set_xlabel(r"$K_s$ [$\mu$M]"); a.set_ylabel(r"$n_0$ [$\mu$M]")
    a.set_title(r"(d) $K_s$ slides the upper edge only", fontsize=10, fontweight="bold")
    a.legend(fontsize=7); a.grid(alpha=.2, which="both")

    _suptitle(fig, "what sets whether a cavity folds"
                 "\n" r"$L=\sqrt{B/\mu_{\max}}\;F(P_*;\,\nu_0=A/B,\;"
                 r"\Gamma=n_0^{\rm crit}/K_s;\;\mathrm{shape\ of}\ \Psi)$", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # The four panels are foldparams_{gamma,collapse,mu,closure}.csv; the closure Ks marked in
    # (d) was only ever printed, so it is recorded here together with how it was obtained.
    with open(os.path.splitext(path)[0] + "_marks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        src = "bisection" if np.isfinite(study.get("Ks_closure", float("nan"))) else \
              "linear extrapolation"
        w.writerow(["n0_crit_uM", f"{cl.n0_jam:.10g}"])
        w.writerow(["Ks_closure_uM", f"{Ks_c:.10g}"])
        w.writerow(["Ks_closure_method", src])


def figure_lstar_general(rows, cl: Closure, mu: float, Ks: float, path):
    n0s = sorted({r["n0"] for r in rows})
    cols = plt.cm.viridis(np.linspace(0, .85, len(n0s)))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    a = ax[0]
    for c, n0 in zip(cols, n0s):
        rr = sorted([r for r in rows if r["n0"] == n0], key=lambda z: z["theta"])
        th = np.array([r["theta"] for r in rr])
        Lq = np.array([r["L_exact"] for r in rr])
        tn = np.array([r["theta_num"] for r in rr])
        fold = np.array([r["in_fold"] for r in rr])
        a.plot(th, Lq, "-", color=c, lw=1.7, label=r"$n_0$=%g $\mu$M" % n0)
        a.plot(tn[~fold], Lq[~fold], "o", color=c, ms=5.5, mfc="none", mew=1.4)
        if fold.any():
            a.plot(tn[fold], Lq[fold], "D", color=c, ms=6.5, mfc="none", mew=1.6)
    a.plot([], [], "ko", mfc="none", label=r"direct solve at that $L^*$")
    a.plot([], [], "kD", mfc="none", label="inside a fold: $L^*$ is multi-valued")
    a.set_xlabel(r"$\theta=n_*/n_0$"); a.set_ylabel(r"$L^*$ [$\mu$m]")
    a.set_yscale("log")
    a.set_title("(a) circles land on the curve iff the formula is right",
                fontsize=10, fontweight="bold")
    a.legend(fontsize=7); a.grid(alpha=.2, which="both")

    a = ax[1]
    for c, n0 in zip(cols, n0s):
        rr = sorted([r for r in rows if r["n0"] == n0 and np.isfinite(r["theta_num"])],
                    key=lambda z: z["theta"])
        if not rr:
            continue
        th = np.array([r["theta"] for r in rr])
        err = np.abs(np.array([r["theta_num"] for r in rr]) - th) / th
        fold = np.array([r["in_fold"] for r in rr])
        a.semilogy(th, np.maximum(err, 1.0e-17), "-", color=c, lw=1.4,
                   label=r"$n_0$=%g" % n0)
        a.semilogy(th[~fold], np.maximum(err[~fold], 1.0e-17), "o", color=c, ms=4.5)
        if fold.any():
            a.semilogy(th[fold], np.maximum(err[fold], 1.0e-17), "D", color=c, ms=6.5,
                       mfc="none", mew=1.6)
    a.axhline(1.0e-2, color="grey", ls=":", lw=1, label="1% error")
    a.plot([], [], "kD", mfc="none", label="inside a fold (not an error)")
    a.set_xlabel(r"$\theta=n_*/n_0$")
    a.set_ylabel(r"$|\theta_{\rm num}-\theta|\,/\,\theta$")
    a.set_title("(b) round-trip error", fontsize=10, fontweight="bold")
    a.legend(fontsize=7); a.grid(alpha=.2, which="both")

    _suptitle(fig, "verification of the general threshold length   "
                 r"$L^*=\int_0^{p_*}\!dp\,/\,\sqrt{2\int_p^{p_*}\!g(n(s))\rho(s)\,ds}$"
                 "\nexact: no approximation beyond the steady PDE", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_lstar_dilute(rows, cl: Closure, mu: float, Ks: float, path, bracket_err=None):
    n0s = sorted({r["n0"] for r in rows})
    cols = plt.cm.viridis(np.linspace(0, .85, len(n0s)))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    a = ax[0]
    for c, n0 in zip(cols, n0s):
        rr = sorted([r for r in rows if r["n0"] == n0], key=lambda z: z["theta"])
        th = np.array([r["theta"] for r in rr])
        a.plot(th, [r["L_exact"] for r in rr], "-", color=c, lw=1.9,
               label=r"$n_0$=%g $\mu$M" % n0)
        a.plot(th, [r["L_dilute_monod"] for r in rr], "--", color=c, lw=1.3)
        a.plot(th, [r["L_dilute_linear"] for r in rr], ":", color=c, lw=1.8)
        tn = np.array([r["theta_num"] for r in rr])
        a.plot(tn, [r["L_exact"] for r in rr], "o", color=c, ms=4.5, mfc="none", mew=1.1)
    a.plot([], [], "k-", label="exact quadrature")
    a.plot([], [], "k--", label=r"dilute $\Psi\approx D_0\rho$, full Monod")
    a.plot([], [], "k:", label=r"dilute $+$ linear Monod $g\approx kn$")
    a.plot([], [], "ko", mfc="none", label="direct solve")
    a.set_xlabel(r"$\theta=n_*/n_0$"); a.set_ylabel(r"$L^*$ [$\mu$m]")
    a.set_yscale("log")
    a.set_title("(a) the two approximations, separated", fontsize=10, fontweight="bold")
    a.legend(fontsize=6.5); a.grid(alpha=.2, which="both")

    a = ax[1]
    ths = sorted({r["theta"] for r in rows})
    th_ref = ths[len(ths) // 2]
    sel = sorted([r for r in rows if r["theta"] == th_ref], key=lambda z: z["n0"])
    nn = np.array([r["n0"] for r in sel])
    a.semilogx(nn, [r["L_dilute_monod"] / r["L_exact"] for r in sel], "s--", color=C1,
               lw=1.5, ms=5.5, label=r"dilute / exact   (isolates $\Psi\approx D_0\rho$)")
    a.semilogx(nn, [r["L_dilute_linear"] / r["L_exact"] for r in sel], "o:", color=C2,
               lw=1.5, ms=5.5, label=r"dilute$+$linear / exact   (adds $g\approx kn$)")
    a.axhline(1.0, color="k", lw=1)
    a.axvline(Ks, color="grey", ls="-.", lw=1.2, label=r"$n_0=K_s$=%g $\mu$M" % Ks)
    a.axvline(cl.n0_jam, color="grey", ls=":", lw=1.2,
              label=r"$n_0=n_0^{\rm crit}$=%.0f $\mu$M" % cl.n0_jam)
    a.set_xlabel(r"$n_0$ [$\mu$M]"); a.set_ylabel(r"$L^*_{\rm approx}/L^*_{\rm exact}$")
    a.set_title(r"(b) where each approximation holds ($\theta$=%g)" % th_ref,
                fontsize=10, fontweight="bold")
    a.legend(fontsize=6.5); a.grid(alpha=.2, which="both")

    extra = "" if bracket_err is None else (
        "   printed closed form vs its defining integral: %.1e" % bracket_err)
    _suptitle(fig, r"verification of the DILUTE threshold length ($\Psi\approx D_0\rho$, "
                 r"$g\approx kn$)" + extra, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # the curves and points are lstar_sweep.csv; these are the scalars the figure marks
    with open(os.path.splitext(path)[0] + "_marks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerow(["theta_ref_panel_b", f"{th_ref:.10g}"])
        w.writerow(["Ks_uM", f"{Ks:.10g}"])
        w.writerow(["n0_crit_uM", f"{cl.n0_jam:.10g}"])
        w.writerow(["closed_form_vs_integral_err",
                    "" if bracket_err is None else f"{bracket_err:.6e}"])


# =============================================================================
# main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=sorted(config.REGIMES), default=None)
    ap.add_argument("--n0", type=float, default=None, help="feed concentration [uM]")
    ap.add_argument("--L", type=float, default=None, help="cavity length [um]")
    ap.add_argument("--dx", type=float, default=None, help="cell size [um]")
    ap.add_argument("--N", type=int, default=None, help="cell count (overrides --dx)")
    ap.add_argument("--strain", type=int, choices=(1, 2), default=2,
                    help="2 = glutton (default), 1 = scavenger")
    ap.add_argument("--mu", type=float, default=None,
                    help="override the strain's mu_max [1/h]")
    ap.add_argument("--K", "--Ks", dest="Ks", type=float, default=None,
                    help="override the strain's half-saturation Ks [uM]")
    ap.add_argument("--noise-floor", type=float, default=NOISE_FLOOR_UM,
                    help="n below this [uM] is treated and drawn as numerical noise")
    ap.add_argument("--growth-space", action="store_true",
                    help="enable q(rho); inert below n0 = 53018 uM")
    ap.add_argument("--tol", type=float, default=1.0e-9, help="relative Newton tolerance")
    ap.add_argument("--n-grid", type=int, default=200_001, help="Psi table resolution")
    ap.add_argument("--n-quad", type=int, default=400_001, help="quadrature source-table size")
    ap.add_argument("--quad-frac", type=float, default=0.5,
                    help="p*/A at which the quadrature cross-check is run; 0.5 keeps it clear "
                         "of onset (p*->0) and of total depletion (p*->A), where the round "
                         "trip is ill-conditioned or resolution-limited")
    ap.add_argument("--continuation", action="store_true", help="solve the branch from onset")
    ap.add_argument("--Lmin-factor", type=float, default=1.02)
    ap.add_argument("--Lsteps", type=int, default=40)
    ap.add_argument("--no-validate", dest="validate", action="store_false", default=True)
    ap.add_argument("--dx-grids", type=int, default=3, help="grids for the dx study (1 = off)")
    ap.add_argument("--no-onset-slope", dest="onset_slope", action="store_false", default=True)
    ap.add_argument("--no-master-curve", dest="master_curve", action="store_false", default=True)
    ap.add_argument("--onset-n0", type=float, default=0.15,
                    help="n0 for the transcritical-slope verdict; must be dilute enough that "
                         "D_col ~ D0, which the law assumes and the solver does not")
    ap.add_argument("--mc-n0", type=float, default=0.15,
                    help="n0 for the master-curve check; must satisfy n0 << Ks")
    ap.add_argument("--budget", action="store_true",
                    help="MODE: sweep L and n0 and test the jamming criterion A_eff > B "
                         "against direct solutions; writes steady_budget.png/.csv")
    ap.add_argument("--verify-lstar", action="store_true",
                    help="MODE: verify the general and the dilute L* formulas against direct "
                         "solutions; writes steady_lstar_general.png and "
                         "steady_lstar_dilute.png plus their CSVs")
    ap.add_argument("--fold", action="store_true",
                    help="MODE: trace the steady branch, verify every point against the PDE, "
                         "classify stability, and plot the fold; writes steady_fold.png/.csv")
    ap.add_argument("--fold-params", action="store_true",
                    help="MODE: measure WHICH parameters decide that a fold exists -- the "
                         "mu invariance, the Gamma collapse, the (A/B)* law and the closure "
                         "shape; writes steady_fold_params.png and four CSVs")
    ap.add_argument("--fold-n0", type=float, default=1500.0,
                    help="n0 [uM] for the detailed fold panel")
    ap.add_argument("--fold-N", type=int, default=400,
                    help="cells for the PDE re-solve at each branch point")
    ap.add_argument("--no-fold-stability", dest="fold_stability",
                    action="store_false", default=True,
                    help="skip the eigenvalue classification (much faster)")
    ap.add_argument("--budget-n0", default=None,
                    help="comma list of n0 [uM] for --budget (default brackets n0_crit)")
    ap.add_argument("--budget-L", default=None,
                    help="comma list of L [um] for --budget (default resolves L_onset..L_jam)")
    ap.add_argument("--lstar-n0", default=None, help="comma list of n0 [uM] for --verify-lstar")
    ap.add_argument("--theta", default=None,
                    help="comma list of theta = n_*/n0 for --verify-lstar")
    ap.add_argument("--sweep-dx", type=float, default=0.5,
                    help="target dx [um] for the solves inside a sweep")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--verbose", action="store_true")
    config.add_model_args(ap)
    ap.add_argument("--no-suptitle", dest="suptitle", action="store_false", default=True,
                    help="omit figure suptitles (the Paper/ copies; the LaTeX caption "
                         "already carries the same text)")
    a = ap.parse_args()
    global SUPTITLE
    SUPTITLE = a.suptitle

    base = config.REGIMES[a.regime] if a.regime else config.REGIMES["dilute"]
    n0 = a.n0 if a.n0 is not None else base["n0"]
    dx = a.dx if a.dx is not None else base["dx"]
    L = a.L if a.L is not None else config.L
    N = a.N if a.N is not None else max(int(round(L / dx)), 8)
    dx = L / N

    p = config.params_from_args(a, n0)
    mu, Ks, label = strain_kinetics(p, a.strain)
    if a.mu is not None or a.Ks is not None:
        mu = a.mu if a.mu is not None else mu
        Ks = a.Ks if a.Ks is not None else Ks
        label = f"custom (mu={mu:g}, Ks={Ks:g})"
    cl = Closure(p, n_grid=a.n_grid)
    L_on = onset_length(p, mu, Ks)

    print("=" * 78)
    print("  single-strain STEADY cavity -- Kirchhoff/Newton  (reviewthis.tex)")
    print("=" * 78)
    print(f"  strain {label}:  mu={mu:g} 1/h   Ks={Ks:g} uM")
    print(f"  n0={n0:g} uM   L={L:g} um   N={N}   dx={dx:g} um")
    print(f"  A=Y*Dn*n0={cl.A(n0):.6g}   B=Psi(rho_jam)={cl.B:.6g}   A/B={cl.A(n0)/cl.B:.6g}"
          f"   (A=B at n0={cl.n0_jam:.1f} uM)")
    print(f"  L_onset={L_on:.4g} um   L/L_onset={L / L_on:.4g}"
          f"   rho ceiling Psi^-1(A)={cl.rho_ceiling(n0):.6g}")
    def _grid(spec, default):
        return np.array([float(v) for v in spec.split(",")]) if spec else default

    # --- sweep MODES.  These replace the single-point run: their subject is the (n0, L) plane,
    # so the --n0/--L of a single run would only be noise.  One Closure serves the whole sweep
    # because Psi, and hence B, do not depend on n0.
    if a.budget or a.verify_lstar or a.fold or a.fold_params:
        outdir = a.outdir or os.path.join(HERE, "steady_sweeps")
        os.makedirs(outdir, exist_ok=True)
        t_sweep = time.time()

        if a.budget:
            # L_onset ~ 34.7 um and L_jam ~ 37.4-38.9 um for this strain, so the sub-jammed
            # band is only a few um wide: the default L grid is dense there and coarse outside,
            # otherwise the whole transition falls between two samples.
            n0s = _grid(a.budget_n0, np.array([200., 400., 600., 782.1, 900., 1200.,
                                               2000., 4000., 8000.]))
            Ls = _grid(a.budget_L, np.unique(np.concatenate([
                np.geomspace(25., 33., 3), np.linspace(34., 45., 12),
                np.geomspace(50., 400., 5)])))
            print("\n  --budget: %d n0 x %d L = %d solves"
                  % (len(n0s), len(Ls), len(n0s) * len(Ls)))
            rows = budget_sweep(p, cl, mu, Ks, n0s, Ls, dx_target=a.sweep_dx,
                                use_q=a.growth_space, tol=a.tol, verbose=a.verbose)
            fields = ["n0", "L", "A", "B", "A_over_B", "A_eff", "A_eff_over_B", "L_onset",
                      "rho_max", "rho_over_jam", "n_L", "branch", "jam_numeric",
                      "jam_criterion", "agree", "converged", "iters"]
            _write_rows_csv(os.path.join(outdir, "budget_sweep.csv"), rows, fields)
            figure_budget(rows, cl, p, mu, Ks,
                          os.path.join(outdir, "steady_budget.png"), a.n_quad)
            # Only converged solves can testify: an unconverged rho_max is not a measurement,
            # so counting it as "agreeing" would inflate the score with points that say nothing.
            usable = [r for r in rows if r["converged"]]
            nbad = len(rows) - len(usable)
            print("  A_eff>B  vs  rho_max>=rho_jam :  %d/%d agree"
                  % (sum(r["agree"] for r in usable), len(usable))
                  + ("   (%d unconverged solves excluded)" % nbad if nbad else ""))
            crude = sum(1 for r in usable if (r["A_over_B"] > 1.0) != r["jam_numeric"])
            print("  A>B alone (the crude form)    :  %d/%d agree   "
                  "(%d points offered A>B that do NOT jam)"
                  % (len(usable) - crude, len(usable), crude))

        if a.fold:
            n0m = float(a.fold_n0)
            win, _S, _r, _L = fold_window(cl, n0m, mu, Ks, n_quad=a.n_quad, npts=300)
            print("\n  --fold: branch at n0=%g, %s"
                  % (n0m, "window L in [%.4f, %.4f] um" % win if win else "NO fold"))
            main_rows, _ = fold_branch(p, cl, mu, Ks, n0m, npts=a.Lsteps, N=a.fold_N,
                                       n_quad=a.n_quad, tol=a.tol,
                                       stability=a.fold_stability, verbose=a.verbose)
            thr_pre = fold_threshold(p, cl, mu, Ks, lo=cl.n0_jam * (1.0 + 1.0e-7),
                                     n_quad=a.n_quad, npts=400, iters=26, hi=8.0)
            cusp = fold_cusp(cl, p, mu, Ks, n0_guess=thr_pre[1])
            cusp_pre = cusp["n0_star"] if np.isfinite(cusp["n0_star"]) else thr_pre[1]
            fam_n0 = [cl.n0_jam * 1.0005, cusp_pre, 1500., 3000., 10000.]
            family = {v: fold_branch(p, cl, mu, Ks, v, npts=70, N=a.fold_N,
                                     n_quad=a.n_quad, verify=False)[0] for v in fam_n0}
            # Dense just above n0*: with a plain geomspace from n0_crit the second node landed
            # at 968 uM, so the shaded band in panel (c) started well to the RIGHT of the n0*
            # line -- a grid artifact that read as an inconsistency between the two.
            widths = []
            wgrid = np.unique(np.concatenate([
                np.geomspace(cl.n0_jam * 1.001, cusp_pre * 0.999, 3),
                cusp_pre * (1.0 + np.geomspace(2.0e-4, 1.5, 12)),
                np.geomspace(cusp_pre * 3.0, 1.0e5, 10)]))
            for v in wgrid:
                w, _, _, _ = fold_window(cl, float(v), mu, Ks, n_quad=200_001, npts=400)
                widths.append((float(v), w[0] if w else float("nan"),
                               w[1] if w else float("nan")))
            thr = thr_pre
            print("  fold onset: detection bisection n0 = %.3f uM;  cusp extrapolation "
                  "n0* = %.3f uM  (A/B = %.4f)"
                  % (thr[1], cusp["n0_star"], cl.A(cusp["n0_star"]) / cl.B))
            print("  cusp exponents as n0 -> n0*:  width %.4f (3/2 = 1.5),  "
                  "separation %.4f (1/2 = 0.5)"
                  % (cusp["alpha_width"], cusp["alpha_drho"]))
            _write_rows_csv(os.path.join(outdir, "fold_cusp.csv"), cusp["rows"],
                            ["n0", "width", "d_rho", "L_lo", "L_hi"])
            fields = ["n0", "rho_branch", "L", "rho_over_jam", "rho_num", "rho_err",
                      "lambda_max", "stable", "converged", "resid"]
            _write_rows_csv(os.path.join(outdir, "fold_branch.csv"), main_rows, fields)
            _write_rows_csv(os.path.join(outdir, "fold_window_vs_n0.csv"),
                            [dict(n0=v, L_lo=lo, L_hi=hi) for v, lo, hi in widths],
                            ["n0", "L_lo", "L_hi"])
            figure_fold(main_rows, family, widths, cl, p, mu, Ks,
                        os.path.join(outdir, "steady_fold.png"), n0m, win, thr, cusp)
            ver = [r for r in main_rows if "rho_err" in r]
            if ver:
                print("  branch verified against the PDE: max |rho_num - rho| = %.2e (%d pts)"
                      % (max(r["rho_err"] for r in ver), len(ver)))
            if a.fold_stability:
                nu = sum(1 for r in main_rows if r.get("stable") is False)
                print("  stability: %d of %d branch points have Re(lambda) > 0"
                      % (nu, len(main_rows)))

        if a.fold_params:
            print("\n  --fold-params: what decides that a fold exists")
            st = fold_params_study(p, cl, mu, Ks, n_quad=a.n_quad, verbose=a.verbose)
            for key, fields in (
                    ("mu", ["mu", "L_lo", "width", "width_sqrt_mu", "rho_at_L_lo"]),
                    ("collapse", ["route", "Ks", "Y", "Dn", "D0", "B", "Gamma",
                                  "n0_crit", "n0_star", "AB_star"]),
                    ("gamma", ["Ks", "Gamma", "n0_crit", "n0_star", "AB_star",
                               "window_pct"]),
                    ("closure", ["D0", "eta", "rho_jam", "B", "Ks_for_fixed_Gamma", "Gamma",
                                 "rho_peak", "dcol_drop", "n0_star", "AB_star"]),
                    ("window", ["n0", "L_lo", "L_hi", "width", "width_pct"])):
                _write_rows_csv(os.path.join(outdir, "foldparams_%s.csv" % key),
                                st[key], fields)
            figure_fold_params(st, cl, mu, Ks,
                               os.path.join(outdir, "steady_fold_params.png"))

            w = st["mu"]
            print("\n  mu_max scales out:  width*sqrt(mu) = "
                  + ", ".join("%.8f" % r["width_sqrt_mu"] for r in w)
                  + "   (mu = " + ", ".join("%g" % r["mu"] for r in w) + ")")
            ab = [r["AB_star"] for r in st["collapse"] if np.isfinite(r["AB_star"])]
            print("  Gamma collapse:     (A/B)* = "
                  + ", ".join("%.6f" % v for v in ab)
                  + "   over %d routes (Ks, Y, Dn, D0 varied)" % len(ab))
            print("\n  %-9s %10s %11s %11s %10s" % ("Ks", "Gamma", "n0_crit", "n0*",
                                                    "window"))
            for r in st["gamma"]:
                print("  %-9g %10.3f %11.2f %11s %9s"
                      % (r["Ks"], r["Gamma"], r["n0_crit"],
                         "%.2f" % r["n0_star"] if np.isfinite(r["n0_star"]) else "none",
                         "%.1f%%" % r["window_pct"] if np.isfinite(r["window_pct"]) else "-"))
            Ksc = st.get("Ks_closure", float("nan"))
            print("  window shuts at Ks = %.4f uM (bisected); below it, jamming is a "
                  "fold at every n0 that jams at all" % Ksc)
            print("    linear extrapolation of the width gives %.4f uM, below the last "
                  "node measured shut -- the width is convex near closure"
                  % fold_params_closure_Ks(st["gamma"]))
            print("\n  %-6s %-8s %11s %13s %11s" % ("eta", "rho_jam", "rho_peak",
                                                    "D_col drop", "(A/B)*"))
            for r in st["closure"]:
                print("  %-6g %-8g %11.4f %12.1f%% %11s"
                      % (r["eta"], r["rho_jam"], r["rho_peak"], 100 * r["dcol_drop"],
                         "%.4f" % r["AB_star"] if np.isfinite(r["AB_star"]) else "none"))

        if a.verify_lstar:
            n0s = _grid(a.lstar_n0, np.array([0.15, 1.5, 15., 150., 1500.]))
            ths = _grid(a.theta, np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]))
            print("\n  --verify-lstar: %d n0 x %d theta" % (len(n0s), len(ths)))
            rows, folds = lstar_sweep(p, cl, mu, Ks, n0s, ths, dx_target=a.sweep_dx,
                                      n_quad=a.n_quad, use_q=a.growth_space, tol=a.tol,
                                      dilute=True, verbose=a.verbose)
            fields = ["n0", "theta", "theta_num", "L_exact", "L_dilute_monod",
                      "L_dilute_linear", "L_onset", "eps_dilute", "n0_over_Ks",
                      "in_fold", "n_states", "converged"]
            _write_rows_csv(os.path.join(outdir, "lstar_sweep.csv"), rows, fields)
            # Points inside a fold are EXCLUDED from the error statistic: there L* is
            # multi-valued, so "solve at L* and read theta back" has several right answers and
            # a mismatch says nothing about the formula.
            good = [r for r in rows
                    if np.isfinite(r["theta_num"]) and not r["in_fold"]]
            errs = [abs(r["theta_num"] - r["theta"]) / r["theta"] for r in good]
            for v, win in sorted(folds.items()):
                if win is not None:
                    print("  FOLD at n0=%g: L in [%.4f, %.4f] um carries 3 steady states"
                          % (v, win[0], win[1]))
            figure_lstar_general(rows, cl, mu, Ks,
                                 os.path.join(outdir, "steady_lstar_general.png"))
            berr = check_printed_bracket(cl, 15.0, cl.YDn * 15.0 * 0.5)
            figure_lstar_dilute(rows, cl, mu, Ks,
                                os.path.join(outdir, "steady_lstar_dilute.png"), berr)
            if errs:
                nf = sum(1 for r in rows if r["in_fold"])
                print("  general L* round-trip in theta:  median %.2e   worst %.2e   "
                      "(%d single-valued pts, %d excluded as multi-valued)"
                      % (float(np.median(errs)), max(errs), len(errs), nf))
            print("  printed dilute closed form vs its defining integral: %.2e" % berr)

        print("\n  wrote -> %s   [%.1f s]" % (outdir, time.time() - t_sweep))
        return

    if L < L_on:
        print(f"\n  L = {L:g} um is BELOW L_onset = {L_on:.4g} um.")
        print("  Only the trivial state rho == 0 exists there (tex: linear stability).")
        return

    t0 = time.time()
    res = solve_steady(L, N, p, cl, mu, Ks, use_q=a.growth_space, tol=a.tol, verbose=a.verbose)
    res["strain_label"] = label
    res["noise_floor"] = float(a.noise_floor)
    info = res["info"]
    print(f"\n  Newton: seed '{info.get('seed','')}', {info['iters']} iters, "
          f"|F|={info['residual']:.3e} (tol {info['tol_eff']:.3e}), "
          f"{info['message']}, branch={info['branch']}   [{time.time()-t0:.2f} s]")
    print(f"  rho(L)={float(res['rho'][-1]):.10g}   rho_max/rho_jam="
          f"{float(res['rho'].max())/p.rho_jam:.6g}   n(L)={float(res['n'][-1]):.4e} uM")

    # A non-converged iterate must never be reported as a result.  The seeds are built to
    # satisfy the invariant exactly and to carry rho(L) = Psi^-1(A), so an unconverged run
    # still produces a plausible closed-end density and still passes several checks -- it
    # looks like an answer.  Say so loudly and withhold the validation verdicts.
    ok = info["converged"] and info["branch"] == "colonised"
    if not ok:
        print()
        print("  " + "!" * 74)
        print(f"  !!  SOLVE FAILED: converged={info['converged']}, branch={info['branch']}")
        print(f"  !!  {info['message']}")
        print("  !!  The profile below is an UNCONVERGED ITERATE, not a steady state.")
        print("  !!  Its rho(L) and its invariant look right because the SEED is built that")
        print("  !!  way -- do not read anything into them.  Validation is skipped.")
        print("  " + "!" * 74)

    val_rows = []
    if a.validate and ok:
        val_rows, _ = validate(res, cl, a)
    elif a.validate:
        val_rows = [("validation", "SKIPPED -- solve did not converge")]

    branch_rows, quad_curve = None, None
    if a.continuation:
        print("\n  continuation from onset")
        branch_rows = continuation(N, p, cl, mu, Ks, L, L_min_factor=a.Lmin_factor,
                                   steps=a.Lsteps, use_q=a.growth_space, tol=a.tol,
                                   verbose=a.verbose)
        try:
            S = _source_table(cl, n0, mu, Ks, cl.p_cap(n0), a.n_quad)
            rr = np.linspace(1e-4, cl.rho_ceiling(n0) * 0.999, 60)
            qq = np.array([quad_length(S, float(cl.psi(v))) for v in rr])
            quad_curve = (rr, qq)
        except Exception:                                             # noqa: BLE001
            quad_curve = None

    outdir = a.outdir or os.path.join(HERE, f"steady_n0-{n0:g}_L{L:g}_dx{dx:g}")
    os.makedirs(outdir, exist_ok=True)
    write_csvs(outdir, res, cl, val_rows, branch_rows)
    figure_closures(res, cl, os.path.join(outdir, "steady_closures.png"))
    figure_nutrient(res, cl, os.path.join(outdir, "steady_nutrient.png"))
    figure_density(res, cl, os.path.join(outdir, "steady_density.png"))
    figure_diffusivity_x(res, cl, os.path.join(outdir, "steady_diffusivity_x.png"))
    figure_growth(res, cl, os.path.join(outdir, "steady_growth.png"))
    figure_flux(res, cl, os.path.join(outdir, "steady_flux.png"))
    figure_validation(res, cl, os.path.join(outdir, "steady_validation.png"))
    if branch_rows:
        figure_branch(branch_rows, res, cl, os.path.join(outdir, "steady_branch.png"),
                      quad_curve)
    print(f"\n  wrote -> {outdir}")


if __name__ == "__main__":
    main()
