"""
closures.py -- the physics of the two-strain E. coli / glucose cavity.  NO SOLVER HERE.

This is a clean re-implementation of `src/closures.py`.  What changed, and why:

  * NO CHEMOTAXIS.  No chi, sensing, mobility, volume filling, face gating or limiter.
  * NO RUNTIME PATCHING.  `self_diffusivity` is the CONTINUOUS closure, period -- it is not a
    swappable global.  (`src/dself_drift.use_dself()` rebound this name in two namespaces, so the
    physics in force depended on which `with` block you were inside.)
  * NO PRESSURE.  `use_pressure`, `M_pressure` and the projection are gone; jamming is carried by
    D_col alone, as in Slepukhin/Karita.
  * NEW: `growth_space_factor` -- growth cannot proceed into occupied space.  This is what makes
    rho <= 1 an invariant instead of a hope; see its docstring.
  * PARAMETERS ARE MEASURED, not designed: n* is an OUTPUT of (mu_i, Ks_i), so there is no
    `mu1_for_requested_crossover` and no `--design-crossing`.
  * `choose_grid_size` warns instead of silently clipping.

Units: x in um, t in h, n in uM, rho_i are packing fractions (dimensionless), Y in packing/uM.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


# =============================================================================
# Parameters
# =============================================================================
@dataclass(frozen=True)
class ModelParams:
    """E. coli on glucose.  Every growth/yield/diffusivity value is measured, not fitted here.

    Strain 1 = scavenger (high affinity, low mu_max);  strain 2 = glutton.
    The Monod crossover n* = (mu1 Ks2 - mu2 Ks1)/(mu2 - mu1) = 5.2526 uM is DERIVED.
    """

    # --- Monod growth: E. coli MG1655 dptsG vs dmglBAC (Steinsiek 2012; Mgl Kd / PtsG Km) ---
    mu1: float = 0.21          # 1/h   scavenger
    Ks1: float = 0.2           # uM    Mgl Kd
    mu2: float = 0.78          # 1/h   glutton
    Ks2: float = 15.0          # uM    PtsG Km

    # --- nutrient ---
    n0: float = 15.0           # uM    reservoir concentration at the open end
    Dn: float = 1.2e6          # um^2/h  0.5 x free-solution glucose
    Y: float = 2.895e-7        # packing fraction per uM, from Y_X/S = 0.45 g/g

    # --- collective transport and jamming (Slepukhin/Karita) ---
    D0: float = 376.0          # um^2/h
    eta: float = 5.8           # Richardson-Zaki exponent
    rho_jam: float = 0.64      # random close packing

    # --- label (self) diffusion ---
    # DEFAULT is the DASpaper form AS DISTRIBUTED: discontinuous at rho_jam, floor 1.5e-3.
    # `dself_shape="continuous"` selects the smooth variant (eps + (1-eps)max(1-rho/rho_jam,0)),
    # which is normally paired with Dself_floor = 1.5e-7 -- see self_diffusivity.
    dself_shape: str = "discontinuous"     # "discontinuous" | "continuous"
    Dself_floor: float = 1.5e-3            # jammed floor, D_self/D0
    Dself_jam_slope: float = 0.95          # alpha; discontinuous branch only
    use_self_diffusion: bool = True

    # --- growth space limitation (see growth_space_factor) ---
    rho_c: float = 0.95        # growth is UNDAMPED below this; ramps to 0 at rho_max
    rho_max: float = 1.0       # hard packing ceiling

    # --- finite cell size: the continuum's own smallest scale (see smoothing_matrix) ---
    d_cell: float = 0.0        # um, bacterium length along x.  0 = OFF (point cells)
    ell_factor: float = 1.5    # smoothing width ell = ell_factor * d_cell (IBM's 3x3 of dx=0.5 d)

    # --- numerical tolerances ---
    tiny_density: float = 1.0e-14


@dataclass(frozen=True)
class Numerics:
    T_end: float = 120.0
    dt: float = 0.05
    dx_target: float = 5.0
    n_save: int = 241
    min_cells: int = 40
    max_cells: int = 4000          # was 600 in src/: that capped dx at L/600
    advection_cfl: float = 0.45
    max_advection_substeps: int = 2000


@dataclass(frozen=True)
class InitialConditions:
    rho_seed: float = 1.0e-6
    seed_center_um: float = 25.0
    seed_width_um: float = 20.0
    seed_f1: float = 0.5
    rho_mode: str = "opening_gaussian"     # opening_gaussian | uniform | deep_gaussian
    nutrient_mode: str = "filled"          # filled | empty | linear


# =============================================================================
# Growth
# =============================================================================
def monod(n, mu: float, Ks: float):
    arr = np.maximum(n, 0.0)
    out = mu * arr / (Ks + arr + 1.0e-300)
    return float(out) if np.isscalar(n) else out


def g1(n, p: ModelParams):
    return monod(n, p.mu1, p.Ks1)


def g2(n, p: ModelParams):
    return monod(n, p.mu2, p.Ks2)


def monod_crossover(p: ModelParams) -> Optional[float]:
    """The n* where g1 = g2, if a positive finite one exists.  DERIVED, never imposed."""
    denom = p.mu2 - p.mu1
    if abs(denom) < 1.0e-14:
        return None
    n_star = (p.mu1 * p.Ks2 - p.mu2 * p.Ks1) / denom
    if not np.isfinite(n_star) or n_star <= 0.0:
        return None
    return float(n_star)


def growth_space_factor(rho, p: ModelParams):
    """q(rho): cells cannot divide into occupied space.  Growth becomes q(rho) * g_i(n).

        q = 1                                  rho <= rho_c
          = (rho_max - rho)/(rho_max - rho_c)  rho_c < rho < rho_max
          = 0                                  rho >= rho_max

    WHY THIS EXISTS.  In a closed cavity the only sink is the mouth, so D_col REDISTRIBUTES but
    never removes: at high n0 the reaction outproduces the mouth outflux and rho accumulates
    uniformly.  Measured in src/: rho = 1.9897, flat across 35 cells -- a packing fraction of
    199%.  D_col cannot bound that (its post-jamming branch is linear, D_col(1)/D0 = 360.7,
    finite), and clipping rho would leave the nutrient consumption running, scrubbing the jammed
    zone to n ~ 0 and making "jammed" and "starved" the same state by construction.  The source
    has to be limited, and this is that limit.

    WHY THE ONSET IS AT rho_c = 0.95 AND NOT AT rho_jam.  The jamming region rho_jam..rho_c is
    D_col's job; damping growth there would be a second, unjustified mechanism for the same
    physics.  Starting at 0.95 also keeps q == 1 EXACTLY for every sub-0.95 run -- including the
    n0=1000 jammed reference -- so those remain bit-identical to the old stack.

    WHY IT RAMPS RATHER THAN STEPS.  A hard step (q=1 up to rho_max, 0 above) lets a cell at
    rho = 0.999 grow at full rate for one whole step, overshooting to 1.039 at dt=0.05; the ramp
    makes the density equation logistic near the ceiling, so rho approaches rho_max
    asymptotically without crossing.  See check_growth_ceiling for the dt condition this needs.
    """
    arr = np.asarray(rho, dtype=float)
    span = p.rho_max - p.rho_c
    if span <= 0.0:
        raise ValueError(f"rho_c={p.rho_c} must be strictly below rho_max={p.rho_max}")
    ramp = np.clip((p.rho_max - arr) / span, 0.0, 1.0)
    out = np.where(arr <= p.rho_c, 1.0, ramp)     # exactly 1.0 below rho_c -> bit-identity
    return float(out) if np.isscalar(rho) else out


def growth_ceiling_danger_band(p: ModelParams, dt: float) -> Tuple[bool, float]:
    """(is dt able to jump the ramp at all, lowest rho from which it could).

    A cell at rho grows undamped for one step to rho*exp(dt*mu_max) whenever rho <= rho_c, so it
    can only overshoot rho_max if it starts above rho_danger = rho_max*exp(-dt*mu_max).  That band
    (rho_danger, rho_c] is non-empty iff rho_danger < rho_c -- the static condition.

    Reported rather than enforced up front, because the static condition alone is far too blunt:
    it would refuse dt=0.1 for a dilute run whose density never exceeds 0.013 and which can never
    approach the ceiling.  The solver checks membership of the band per step instead.
    """
    mu_max = max(p.mu1, p.mu2)
    rho_danger = p.rho_max * np.exp(-dt * mu_max)
    return bool(rho_danger < p.rho_c), float(rho_danger)


def check_growth_ceiling(p: ModelParams, dt: float) -> Tuple[bool, float]:
    """Warn (do not refuse) when dt is large enough to jump the growth ramp if rho ever gets there.

    Returns (risky, rho_danger) so the caller can police it per step.
    """
    risky, rho_danger = growth_ceiling_danger_band(p, dt)
    if risky:
        warnings.warn(
            f"dt={dt:g} with rho_c={p.rho_c:g}, mu_max={max(p.mu1, p.mu2):g}: a cell starting "
            f"above rho={rho_danger:.4f} could step past rho_max={p.rho_max:g} before the ramp "
            f"damps it. Harmless unless the cavity actually reaches that density -- the solver "
            f"fails the run if it does. To use this dt at high density, lower rho_c below "
            f"{rho_danger:.4f} or lower dt.", RuntimeWarning, stacklevel=2)
    return risky, rho_danger


def smoothing_matrix(N: int, dx: float, p: ModelParams):
    """Cell-scale averaging operator S (rows sum to 1).  Returns None when it would be the identity.

    WHY.  A bacterium is not a point: one of length d senses and consumes nutrient over its own
    body, so it cannot respond to structure finer than d.  Without this the model has NO intrinsic
    length, dx becomes the smallest scale by default, and dx therefore sets the physics -- which is
    exactly why dx=10 and dx=0.5 disagreed.

    The scale is read off the IBM this continuum was coarse-grained from
    (self-organised-seclusion, General_Code/main.py): it bins cell positions at dx = 0.5 cell
    diameters and then averages density over a 3x3 stencil, i.e. its continuum fields are DEFINED
    as an average over 1.5 cell diameters.  Hence ell = ell_factor * d_cell with ell_factor = 1.5.
    (Caveat: that is the IBM's convention for DENSITY.  It has no nutrient field at all -- birth
    there is a constant rate -- so extending it to the nutrient coupling is physically motivated
    but not something the IBM validates.)

    S is an exact box average of width ell, truncated and RENORMALISED at the boundaries, so a
    bacterium at the closed end samples only inward and no boundary value is invented.
    """
    ell = p.ell_factor * p.d_cell
    if ell <= dx:
        return None                       # sub-cell smoothing is the identity on this grid
    h = 0.5 * ell
    K = int(np.ceil(h / dx))
    offs, wts = [], []
    for k in range(-K, K + 1):            # overlap of [-h, h] with cell k spanning [(k-.5)dx,(k+.5)dx]
        lo = max(-h, (k - 0.5) * dx)
        hi = min(h, (k + 0.5) * dx)
        w = max(0.0, hi - lo) / ell
        if w > 0.0:
            offs.append(k); wts.append(w)
    S = diags(wts, offs, shape=(N, N), format="lil")
    rs = np.asarray(S.sum(axis=1)).ravel()          # truncated rows near the walls
    S = diags(1.0 / rs, 0, format="csc") @ S.tocsc()
    return S.tocsc()


def local_fraction_exact(rho1, rho2, tol: float = 1.0e-30) -> np.ndarray:
    """f = rho1/(rho1+rho2), preserving pure states exactly and defaulting empty cells to 1/2."""
    r1 = np.maximum(rho1, 0.0)
    r2 = np.maximum(rho2, 0.0)
    rho = r1 + r2
    f = np.full_like(rho, 0.5)
    only1 = (r1 > tol) & (r2 <= tol)
    only2 = (r2 > tol) & (r1 <= tol)
    both = (r1 > tol) & (r2 > tol)
    f[only1] = 1.0
    f[only2] = 0.0
    f[both] = r1[both] / rho[both]
    return np.clip(f, 0.0, 1.0)


# =============================================================================
# Transport closures
# =============================================================================
def collective_diffusivity(rho, p: ModelParams):
    """Slepukhin Eq 16: Richardson-Zaki mobility x Carnahan-Starling modulus, with a CONTINUOUS
    additive post-jamming branch.

        rho <= rho_jam :  D0 (1-rho)^(eta-4) (1 + 4rho + 4rho^2 - 4rho^3 + rho^4)
        rho >  rho_jam :  D0 [ 1000 (rho - rho_jam) + D_col(rho_jam)/D0 ]

    (The published Eq 16 divides by D_col(rho_jam) rather than adding it -- a typo that makes
    D_col crash to 0 at jamming.  The authors' own IBM code uses the additive form.)
    """
    arr = np.asarray(rho, dtype=float)
    phi = np.clip(arr, 0.0, 1.0 - 1.0e-10)

    def _sub(f):
        poly = 1.0 + 4.0 * f + 4.0 * f**2 - 4.0 * f**3 + f**4
        return (1.0 - f) ** (p.eta - 4.0) * poly

    # The sub-jamming branch MUST use the clipped phi -- (1-phi)^1.8 is NaN for phi>1 and np.where
    # evaluates both branches eagerly.  The post branch must NOT be clipped: it is linear, well
    # defined for any rho, and is the restoring force that has to keep growing.
    sub = _sub(phi)
    dcol_jam = _sub(min(p.rho_jam, 1.0 - 1.0e-10))          # = 0.686412
    post = 1000.0 * (arr - p.rho_jam) + dcol_jam
    out = p.D0 * np.where(arr <= p.rho_jam, sub, post)
    return float(out) if np.isscalar(rho) else out


def collective_diffusivity_deriv(rho, p: ModelParams):
    """dD_col/drho, analytic.  Used by the Newton collective-diffusion solve.

    Sub-jamming: D/D0 = (1-f)^(eta-4) P,  P = 1+4f+4f^2-4f^3+f^4
        => dD/df = D0 (1-f)^(eta-5) [ -(eta-4) P + (1-f) P' ],  P' = 4+8f-12f^2+4f^3
    Post-jamming: dD/drho = 1000 D0.  Kinked at rho_jam; the line search handles it.
    """
    arr = np.asarray(rho, dtype=float)
    f = np.clip(arr, 0.0, 1.0 - 1.0e-10)
    P = 1.0 + 4.0 * f + 4.0 * f**2 - 4.0 * f**3 + f**4
    Pp = 4.0 + 8.0 * f - 12.0 * f**2 + 4.0 * f**3
    sub = (1.0 - f) ** (p.eta - 5.0) * (-(p.eta - 4.0) * P + (1.0 - f) * Pp)
    out = p.D0 * np.where(arr <= p.rho_jam, sub, 1000.0)
    return float(out) if np.isscalar(rho) else out


def self_diffusivity(rho, p: ModelParams):
    """Label/self diffusivity.  TWO forms; `p.dself_shape` selects.

    "discontinuous" (DEFAULT) -- Slepukhin/Karita as their code is distributed:

        D_self/D0 = 1 - alpha rho/rho_jam    rho <  rho_jam      (alpha = 0.95)
                  = Dself_floor              rho >= rho_jam      (1.5e-3)

    "continuous" -- the smooth variant, normally with Dself_floor = 1.5e-7:

        D_self/D0 = eps + (1 - eps) max(1 - rho/rho_jam, 0)

    WHY BOTH EXIST.  The DASpaper source computes ratio 1.5e-3 at jamming (D_0 appears in both
    branches with D_0 = 1), but the authors' own docstring says "0.0001 otherwise", i.e.
    1.5e-3 * 1e-4 = 1.5e-7.  The code and its docstring disagree and we cannot resolve it from
    outside, so both are available and the as-distributed one is the default.

    THE CHOICE IS LOAD-BEARING, not cosmetic.  In the jammed phase it sets how far a label moves:

        floor 1.5e-3 -> D_self(jam) = 0.564 um^2/h  -> sqrt(2 D T) = 58 um over 3000 h
        floor 1.5e-7 -> D_self(jam) = 5.64e-5       -> sqrt(2 D T) = 0.58 um over 3000 h

    i.e. whether the jammed graveyard mixes over the whole growing region or is frozen solid.
    Do not treat it as a tolerance.

    The discontinuous form jumps by (1-alpha)/Dself_floor = 33x at rho_jam, and jammed runs sit
    right on that contour (rho ~ 0.66 vs rho_jam = 0.64) -- the reason the continuous variant was
    written.  eps > 0 either way is required by the ancestry kernel, which needs B = D_self rho > 0
    on the occupied region for the Green's measure to have a density at all.
    """
    arr = np.asarray(rho, dtype=float)
    if not p.use_self_diffusion:
        out = np.zeros_like(arr, dtype=float)
    elif p.dself_shape == "discontinuous":
        phi = arr / (p.rho_jam + 1.0e-300)
        out = p.D0 * np.where(arr < p.rho_jam,
                              np.maximum(1.0 - p.Dself_jam_slope * phi, 0.0),
                              p.Dself_floor)
    elif p.dself_shape == "continuous":
        eps = p.Dself_floor
        out = p.D0 * (eps + (1.0 - eps) * np.maximum(1.0 - arr / p.rho_jam, 0.0))
    else:
        raise ValueError(f"unknown dself_shape {p.dself_shape!r} "
                         f"(use 'discontinuous' or 'continuous')")
    return float(out) if np.isscalar(rho) else out


def validate_model_params(p: ModelParams) -> Dict[str, object]:
    if min(p.D0, p.Dn, p.rho_jam, p.Y) <= 0:
        raise ValueError("D0, Dn, rho_jam, Y must be positive")
    if min(p.mu1, p.mu2, p.Ks1, p.Ks2) <= 0:
        raise ValueError("mu_i and Ks_i must be positive")
    n_star = monod_crossover(p)
    return {
        "n_star": float("nan") if n_star is None else n_star,
        "g1_n0": float(g1(p.n0, p)),
        "g2_n0": float(g2(p.n0, p)),
        "low_slope1": p.mu1 / p.Ks1,
        "low_slope2": p.mu2 / p.Ks2,
        "high_n_winner": "strain1" if p.mu1 > p.mu2 else "strain2",
        "low_n_winner": "strain1" if p.mu1 / p.Ks1 > p.mu2 / p.Ks2 else "strain2",
        "crossover_message": ("no positive finite crossover" if n_star is None
                              else f"positive crossover at n*={n_star:.6g}"),
    }


# =============================================================================
# Grid and finite-volume operators
# =============================================================================
def choose_grid_size(L: float, num: Numerics) -> int:
    """N from dx_target, WARNING when the cell limits prevent honouring it.

    src/closures.py clipped silently, so asking for dx=1 on L=1000 quietly gave dx=1.667 -- a 67%
    error in the one parameter you were trying to control.
    """
    want = int(np.ceil(L / num.dx_target))
    N = int(np.clip(want, num.min_cells, num.max_cells))
    if N != want:
        warnings.warn(
            f"dx_target={num.dx_target:g} um needs N={want} cells on L={L:g} um, but "
            f"min_cells={num.min_cells}/max_cells={num.max_cells} clip it to N={N} "
            f"(realized dx={L / N:.4g} um). Raise Numerics.max_cells to honour dx_target.",
            RuntimeWarning, stacklevel=2)
    return N


def cell_grid(L: float, N: int) -> Tuple[np.ndarray, float]:
    dx = L / N
    return (np.arange(N) + 0.5) * dx, dx


def build_diffusion_matrix_dirichlet_left_neumann_right(
    D_cell: np.ndarray, dx: float, left_value: float
) -> Tuple[csc_matrix, np.ndarray]:
    """A, b for q_t = A q + b = d_x(D d_x q).  Dirichlet at x=0 (half cell), no flux at x=L."""
    N = len(D_cell)
    D_face = np.zeros(N + 1)
    D_face[0] = D_cell[0]
    if N > 1:
        D_face[1:N] = 0.5 * (D_cell[:-1] + D_cell[1:])
    D_face[N] = D_cell[-1]

    # Assembled with diags rather than a Python loop over cells: this is called ~10-20x per step
    # inside the Newton residual, and at N=2000 the loop version cost ~0.4 s/step (6.8 h for a
    # jammed run).  Same arithmetic in the same order, so the matrix is bit-identical.
    inv_dx2 = 1.0 / dx**2
    diag = np.empty(N)
    diag[0] = -(2.0 * D_face[0] + (D_face[1] if N > 1 else 0.0)) * inv_dx2
    if N > 2:
        diag[1:N - 1] = -(D_face[1:N - 1] + D_face[2:N]) * inv_dx2
    if N > 1:
        diag[N - 1] = -D_face[N - 1] * inv_dx2
    off = D_face[1:N] * inv_dx2          # sub-diagonal (rows 1..N-1) and super (rows 0..N-2)
    A = diags([off, diag, off], [-1, 0, 1], format="csc")
    b = np.zeros(N, dtype=float)
    b[0] = 2.0 * D_face[0] * left_value * inv_dx2
    return A, b


def build_diffusion_matrix_neumann_neumann(B_cell: np.ndarray, dx: float) -> csc_matrix:
    """A for q_t = d_x(B d_x q), no flux at both ends (the no-relative-label-flux BC)."""
    N = len(B_cell)
    B_face = np.zeros(N + 1)
    if N > 1:
        B_face[1:N] = 0.5 * (B_cell[:-1] + B_cell[1:])
    inv_dx2 = 1.0 / dx**2
    diag = -(B_face[:N] + B_face[1:N + 1]) * inv_dx2
    off = B_face[1:N] * inv_dx2
    return diags([off, diag, off], [-1, 0, 1], format="csc")


def diffusive_flux_dirichlet_left_neumann_right(
    q: np.ndarray, D_cell: np.ndarray, dx: float, left_value: float
) -> np.ndarray:
    """J = -D q_x on the N+1 faces, consistent with the matrix above."""
    N = len(q)
    J = np.zeros(N + 1)
    J[0] = -D_cell[0] * (q[0] - left_value) / (0.5 * dx)
    if N > 1:
        D_face = 0.5 * (D_cell[:-1] + D_cell[1:])
        J[1:N] = -D_face * (q[1:] - q[:-1]) / dx
    J[N] = 0.0
    return J


# =============================================================================
# Strain transport
# =============================================================================
def compute_advection_substeps(rho, J, dt, dx, p: ModelParams, num: Numerics) -> int:
    """Substeps keeping the explicit upwind partition of J_total positive."""
    outgoing = np.maximum(J[1:], 0.0) + np.maximum(-J[:-1], 0.0)
    mask = (rho > p.tiny_density) & (outgoing > 0.0)
    if not np.any(mask):
        return 1
    hmax = float(np.min(num.advection_cfl * rho[mask] * dx / outgoing[mask]))
    if not np.isfinite(hmax) or hmax <= 0.0:
        return num.max_advection_substeps
    return max(1, min(int(np.ceil(dt / hmax)), num.max_advection_substeps))


def upwind_fraction_faces(f: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Donor-cell upwind of f on faces.  J>0 means transport toward increasing x."""
    N = len(f)
    ff = np.zeros(N + 1)
    ff[0] = f[0]        # the opening carries zero biomass; only outflow (J[0]<0) is physical
    if N > 1:
        ff[1:N] = np.where(J[1:N] >= 0.0, f[:N - 1], f[1:N])
    ff[N] = f[-1]
    return np.clip(ff, 0.0, 1.0)


def advect_strain_by_total_flux(rho1, rho_total_start, J_total, dt, dx,
                                p: ModelParams, num: Numerics):
    """Conservatively advect strain-1 mass by the realized total flux, CFL-subcycled.

    The total density is advanced by the same J_total inside the loop so that f stays defined.
    Relative (label) diffusion is handled separately and implicitly.
    """
    nsub = compute_advection_substeps(rho_total_start, J_total, dt, dx, p, num)
    h = dt / nsub
    rho = rho_total_start.copy()
    m1 = np.minimum(np.maximum(rho1.copy(), 0.0), rho)
    max_clip = 0.0
    for _ in range(nsub):
        f = local_fraction_exact(m1, np.maximum(rho - m1, 0.0), tol=p.tiny_density)
        J1 = upwind_fraction_faces(f, J_total) * J_total
        m1_next = m1 - h * (J1[1:] - J1[:-1]) / dx
        rho_next = np.maximum(rho - h * (J_total[1:] - J_total[:-1]) / dx, 0.0)
        before = m1_next.copy()
        m1_next = np.minimum(np.maximum(m1_next, 0.0), rho_next)
        max_clip = max(max_clip, float(np.max(np.abs(m1_next - before))))
        m1, rho = m1_next, rho_next
    return m1, rho, {"advection_substeps": float(nsub), "advection_clip_linf": max_clip}


def implicit_self_diffusion_fraction(rho1_adv, rho_total, p: ModelParams, dt, dx):
    """Implicit conservative label diffusion:  rho f - dt d_x(D_self rho d_x f) = rho1_adv."""
    rho = np.maximum(rho_total, 0.0)
    rho1_adv = np.minimum(np.maximum(rho1_adv, 0.0), rho)
    rho2_adv = rho - rho1_adv

    if np.sum(rho) > p.tiny_density:
        if np.max(rho2_adv) <= p.tiny_density:
            return rho.copy(), {"selfdiff_clip_linf": 0.0}
        if np.max(rho1_adv) <= p.tiny_density:
            return np.zeros_like(rho), {"selfdiff_clip_linf": 0.0}

    if (not p.use_self_diffusion) or np.all(rho <= p.tiny_density):
        f = local_fraction_exact(rho1_adv, rho2_adv, tol=p.tiny_density)
        return f * rho, {"selfdiff_clip_linf": 0.0}

    B = np.asarray(self_diffusivity(rho, p)) * rho
    A = build_diffusion_matrix_neumann_neumann(B, dx)
    M = diags(np.maximum(rho, p.tiny_density), 0, format="csc") - dt * A
    f = spsolve(M, np.minimum(np.maximum(rho1_adv, 0.0), rho))
    before = f.copy()
    f = np.clip(f, 0.0, 1.0)
    clip = float(np.max(np.abs(f - before))) if len(f) else 0.0
    return f * rho, {"selfdiff_clip_linf": clip}


# =============================================================================
# Initial conditions
# =============================================================================
def initial_total_density(x, L, p: ModelParams, ic: InitialConditions) -> np.ndarray:
    if ic.rho_mode == "opening_gaussian":
        xc = min(ic.seed_center_um, 0.1 * L)
        w = max(ic.seed_width_um, 1.0)
        rho = ic.rho_seed * np.exp(-0.5 * ((x - xc) / w) ** 2)
    elif ic.rho_mode == "uniform":
        rho = ic.rho_seed * np.ones_like(x)
    elif ic.rho_mode == "deep_gaussian":
        rho = ic.rho_seed * np.exp(-0.5 * ((x - 0.70 * L) / max(0.08 * L, 1.0)) ** 2)
    else:
        raise ValueError(f"unknown rho_mode {ic.rho_mode!r}")
    return np.clip(rho, 0.0, p.rho_jam)


def initial_fraction(x, ic: InitialConditions) -> np.ndarray:
    if ic.seed_f1 <= 0.0:
        return np.zeros_like(x)
    if ic.seed_f1 >= 1.0:
        return np.ones_like(x)
    return np.clip(ic.seed_f1 * np.ones_like(x), 0.0, 1.0)


def initial_nutrient(x, p: ModelParams, ic: InitialConditions) -> np.ndarray:
    if ic.nutrient_mode == "filled":
        return p.n0 * np.ones_like(x)
    if ic.nutrient_mode == "empty":
        return np.zeros_like(x)
    if ic.nutrient_mode == "linear":
        return p.n0 * np.maximum(0.0, 1.0 - x / x[-1])
    raise ValueError(f"unknown nutrient_mode {ic.nutrient_mode!r}")
