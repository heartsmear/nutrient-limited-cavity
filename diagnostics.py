"""
diagnostics.py -- everything measured FROM a run, in one place.

This module exists because `src/` had **six copies** of the n* front locator with **two different
behaviours** (one snapped to cell centres, producing a dx-amplitude sawtooth; the rest
interpolated), so the same quantity depended on which script you asked.  One implementation here,
imported by figures.py, ancestry.py and validate().

Two corrections relative to `src/`:

  * The front is interpolated in **log n**, not linearly.  n falls exponentially with a decay
    length far below dx at high n0 -- 542x across ONE cell at n0=1e6 -- so a linear interpolant is
    systematically biased (+4.4 um there, +0.5 um at n0=15).
  * The resolution check measures lambda **at the front**, never in the starved tail.  The two
    differ by ~4x and only the front carries selection: deep in the tail n ~ 1e-12 and s ~ Cn, so
    under-resolving a region where nothing happens is harmless.  Measuring in the tail is what
    made the n0=15 reference look under-resolved when it is not.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from closures import ModelParams, monod_crossover, g1, g2, local_fraction_exact


# =============================================================================
# The n* front
# =============================================================================
def nstar_front(x: np.ndarray, n: np.ndarray, nstar: float) -> float:
    """x of the first n = n* crossing, interpolated in LOG n.  NaN if the profile never crosses."""
    b = np.where(n <= nstar)[0]
    if b.size == 0 or b[0] == 0:
        return float("nan")
    k = int(b[0])
    n1, n2 = float(n[k - 1]), float(n[k])
    if n1 <= 0.0 or n2 <= 0.0 or n1 == n2:
        return float(0.5 * (x[k - 1] + x[k]))
    w = (np.log(n1) - np.log(nstar)) / (np.log(n1) - np.log(n2))
    return float(x[k - 1] + w * (x[k] - x[k - 1]))


def nstar_front_series(x: np.ndarray, NN: np.ndarray, nstar: float) -> np.ndarray:
    """nstar_front per saved frame.  NN has shape (n_cells, n_frames)."""
    return np.array([nstar_front(x, NN[:, j], nstar) for j in range(NN.shape[1])])


def front_resolution(x: np.ndarray, n: np.ndarray, nstar: float, dx: float) -> Dict[str, float]:
    """Nutrient screening length AT THE FRONT, and how many cells resolve it.

        lambda = dx / ln(n[k-1]/n[k])      k = first cell with n <= n*

    `dx/lambda > 1` means less than one cell per decay length: the layer is unresolved and the
    front position, its width, and anything read off it are grid artifacts.
    """
    b = np.where(n <= nstar)[0]
    if b.size == 0 or b[0] == 0 or n[b[0]] <= 0.0:
        return {"lambda_front": float("nan"), "dx_over_lambda": float("nan"),
                "front_drop_factor": float("nan")}
    k = int(b[0])
    drop = float(n[k - 1] / n[k])
    lam = dx / np.log(drop) if drop > 1.0 else float("inf")
    return {"lambda_front": float(lam), "dx_over_lambda": float(dx / lam),
            "front_drop_factor": drop}


# =============================================================================
# Composition
# =============================================================================
def inverse_decomposition(res: Dict, p: ModelParams):
    """Biomass-weighted mouth / deep inverse fractions and global F1, per frame.

    mouth = strain 1 (h>0) where n > n*  (locally strain-2-favoured)
    deep  = strain 2 (h<0) where n < n*  (locally strain-1-favoured -- the graveyard)

    Uses THIS parameter set's own n*.  `src/chemotaxis_cavity.inverse_decomposition` hard-codes
    NSTAR = 1.5 and silently writes wrong numbers for any other pair.
    """
    nstar = monod_crossover(p)
    R1, R2, NN = res["rho1_profiles"], res["rho2_profiles"], res["n_profiles"]
    nt = R1.shape[1]
    mo = np.zeros(nt); dp = np.zeros(nt); F1 = np.zeros(nt)
    if nstar is None:
        for j in range(nt):
            tot = (R1[:, j] + R2[:, j]).sum()
            if tot > 0:
                F1[j] = R1[:, j].sum() / tot
        return mo, dp, F1, None
    for j in range(nt):
        r1, r2 = R1[:, j], R2[:, j]
        rho = r1 + r2
        tot = rho.sum()
        if tot <= 0:
            continue
        F1[j] = r1.sum() / tot
        occ = rho > 1e-9 * max(rho.max(), 1e-30)
        h = np.zeros_like(rho)
        h[occ] = (r1[occ] - r2[occ]) / rho[occ]
        mo[j] = rho[occ & (NN[:, j] > nstar) & (h > 0)].sum() / tot
        dp[j] = rho[occ & (NN[:, j] < nstar) & (h < 0)].sum() / tot
    return mo, dp, F1, float(nstar)


def delta_f_series(res: Dict, p: ModelParams) -> np.ndarray:
    """Delta_f = <f>_{n>n*} - <f>_{n<n*}, biomass-weighted.

    Monod alone predicts Delta_f < 0; Delta_f > 0 IS inverse sorting.  Scale-invariant, so global
    extinction cannot fake it -- unlike inv_deep, which is pinned at its ceiling whenever one
    strain is nearly gone and must not be read as enrichment.
    """
    nstar = monod_crossover(p)
    R1, R2, NN = res["rho1_profiles"], res["rho2_profiles"], res["n_profiles"]
    d = np.zeros(R1.shape[1])
    if nstar is None:
        return d
    for j in range(R1.shape[1]):
        r1, r2 = R1[:, j], R2[:, j]
        rho = r1 + r2
        if rho.sum() <= 0:
            continue
        occ = rho > 1e-9 * max(rho.max(), 1e-30)
        hi = occ & (NN[:, j] > nstar)
        lo = occ & (NN[:, j] < nstar)
        if hi.any() and lo.any() and rho[hi].sum() > 0 and rho[lo].sum() > 0:
            d[j] = r1[hi].sum() / rho[hi].sum() - r1[lo].sum() / rho[lo].sum()
    return d


def masses(res: Dict):
    """(M1, M2) per frame: sum(rho_i) * dx, in packing-fraction * um."""
    dx = res["dx"]
    return res["rho1_profiles"].sum(axis=0) * dx, res["rho2_profiles"].sum(axis=0) * dx


def bottleneck(res: Dict, establish_frac: float = 0.01):
    """Smallest M1, M2 reached AFTER the population establishes, with the times.

    The seeded initial condition is ~1e-5 by construction, so a naive min over all frames always
    reports the seed and never the interesting thing: a strain driven to sub-cell abundance by
    COMPETITION and then recovering.  Frames before the total biomass reaches `establish_frac` of
    its run maximum are therefore excluded.
    """
    M1, M2 = masses(res)
    t = res["t"]
    Mt = M1 + M2
    live = Mt >= establish_frac * max(Mt.max(), 1e-300)
    if not live.any():
        return dict(M1_min=float("nan"), M2_min=float("nan"),
                    t_M1_min=float("nan"), t_M2_min=float("nan"))
    i1 = int(np.argmin(np.where(live, M1, np.inf)))
    i2 = int(np.argmin(np.where(live, M2, np.inf)))
    return dict(M1_min=float(M1[i1]), M2_min=float(M2[i2]),
                t_M1_min=float(t[i1]), t_M2_min=float(t[i2]))


# =============================================================================
# Validation
# =============================================================================
#: One cell of E. coli is ~1 um^3.  M = sum(rho) dx is a volume per um^2 of cavity cross-section,
#: so M is roughly the cell count per um^2.  A cavity cross-section of A um^2 holds ~M*A cells.
#: Flag below this, for A = 1 um^2 -- deliberately conservative.
ONE_CELL_EQUIV = 1.0


def validate(res: Dict, p: ModelParams, verbose: bool = True) -> Dict[str, object]:
    """Every check that would have caught the src/ failures, in one call.

    Returns a dict of {check: (failed, detail)} plus `admissible`.
    """
    x, dx = res["x"], res["dx"]
    n_fin = res["n_profiles"][:, -1]
    rho_fin = res["rho1_profiles"][:, -1] + res["rho2_profiles"][:, -1]
    nstar = monod_crossover(p)

    resn = front_resolution(x, n_fin, nstar, dx) if nstar else {"dx_over_lambda": float("nan")}
    dol = resn["dx_over_lambda"]
    mo, dp, F1, _ = inverse_decomposition(res, p)
    df = delta_f_series(res, p)
    M1, M2 = masses(res)
    bn = bottleneck(res)
    t = res["t"]
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    int_deep = float(_trapz(dp, t))
    int_mouth = float(_trapz(mo, t))     # mouth-type inverse sorting is the PRIMARY signature;
                                         # src2 shipped only the deep integral.

    # Late-slope steady-state test (scope 4.3).  A run that has not reached steady state is
    # "converged" only to a transient; rho_peak at n0=1000 was still creeping at T=3000.
    rho_t = np.max(res["rho1_profiles"] + res["rho2_profiles"], axis=0)
    late = t >= (t[0] + 0.9 * (t[-1] - t[0]))
    if late.sum() >= 2 and t[-1] > t[0]:
        dF1_late = float(np.polyfit(t[late], F1[late], 1)[0]) * 1000.0
        drho_late = float(np.polyfit(t[late], rho_t[late], 1)[0]) * 1000.0
    else:
        dF1_late = drho_late = float("nan")

    checks = {
        "rho_exceeds_max": (res["rho_peak"] > p.rho_max * (1 + 1e-12),
                            f"rho_peak={res['rho_peak']:.6f} vs rho_max={p.rho_max:g}"),
        # Thresholds CALIBRATED against the n0=1000 dx sweep (sweep_dx_n0-1000_T500), where the
        # converged F1 is 1.0e-6 and coarser grids give:
        #     dx/lambda = 4.82 -> F1 = 1.6e-1   (5 orders wrong)
        #     dx/lambda = 1.13 -> F1 = 2.6e-3   (3 orders wrong)
        #     dx/lambda = 0.57 -> converged to 2%
        # So "1 cell per decay length" is nowhere near enough; ~2 is the practical floor.
        "unresolved_front": (np.isfinite(dol) and dol > 0.6,
                             f"dx/lambda={dol:.2f} > 0.6 (calibrated: 1.13 was 3 orders wrong)"),
        "marginal_front": (np.isfinite(dol) and 1.0 / 3.0 < dol <= 0.6,
                           f"dx/lambda={dol:.2f} (converged in one test at 0.57, but no margin)"),
        "strain_extinction": (min(bn["M1_min"], bn["M2_min"]) < ONE_CELL_EQUIV,
                              f"post-establishment bottleneck M1={bn['M1_min']:.3e} "
                              f"(t={bn['t_M1_min']:.0f} h), M2={bn['M2_min']:.3e} "
                              f"(t={bn['t_M2_min']:.0f} h) < {ONE_CELL_EQUIV:g} ~ one cell "
                              f"per um^2 cross-section"),
        "diffusion_nonconvergence": (res["diffusion_failures"] > 0,
                                     f"{res['diffusion_failures']} steps did not converge"),
        "metric_clash": (int_deep > 100.0 and df.max() <= 0.0,
                         f"int_inv_deep={int_deep:.1f} but delta_f_max={df.max():.4f} <= 0 "
                         f"-- inv_deep is at its ceiling, not enrichment"),
        # The fraction is bounded in [0,1] by construction; src/ tracked f_min/f_max and src2 had
        # dropped the check, so a scheme violation would have been silent.
        "fraction_out_of_range": (res.get("f_min", 0.0) < -1e-12 or res.get("f_max", 1.0) > 1 + 1e-12,
                                  f"f in [{res.get('f_min', float('nan')):.6g}, "
                                  f"{res.get('f_max', float('nan')):.6g}], must be [0,1]"),
        "advection_substep_cap": (res.get("adv_capped_steps", 0) > 0,
                                  f"{res.get('adv_capped_steps', 0)} steps hit the "
                                  f"{res.get('adv_substeps_max', 0)}-substep cap -- CFL not honoured"),
        "not_steady": (np.isfinite(drho_late) and abs(drho_late) > 1e-3,
                       f"d rho/dt over the last 10% = {drho_late:+.3e} per 1000 h "
                       f"(dF1/dt = {dF1_late:+.3e}) -- still transient"),
        "solver_failed": (not res["success"], res["message"]),
    }
    hard = ("rho_exceeds_max", "unresolved_front", "diffusion_nonconvergence", "solver_failed",
            "fraction_out_of_range")
    admissible = not any(checks[k][0] for k in hard)

    out = dict(checks, admissible=admissible, dx_over_lambda=dol,
               lambda_front=resn.get("lambda_front", float("nan")),
               front_drop_factor=resn.get("front_drop_factor", float("nan")),
               x_nstar=nstar_front(x, n_fin, nstar) if nstar else float("nan"),
               rho_peak=res["rho_peak"], q_min=res["q_min"],
               F1_final=float(F1[-1]), F1_min=float(F1.min()),
               delta_f_max=float(df.max()), delta_f_final=float(df[-1]),
               t_F1_min=float(t[int(np.argmin(F1))]),
               int_inv_deep=int_deep, int_inv_mouth=int_mouth,
               dF1_per_1000h_late=dF1_late, drho_per_1000h_late=drho_late,
               f_min=res.get("f_min", float("nan")), f_max=res.get("f_max", float("nan")),
               adv_substeps_max=res.get("adv_substeps_max", 0),
               adv_capped_steps=res.get("adv_capped_steps", 0),
               rho_max_over_jam=float(rho_fin.max() / p.rho_jam), **bn)

    if verbose:
        print(f"  n*={nstar:.4f} uM   x_n*={out['x_nstar']:.2f} um   "
              f"rho_peak={out['rho_peak']:.4f} (rho/rho_jam={out['rho_max_over_jam']:.3f})")
        print(f"  lambda_front={out['lambda_front']:.2f} um  dx={dx:g}  "
              f"dx/lambda={dol:.2f}   q_min={out['q_min']:.4f}")
        print(f"  F1 {out['F1_min']:.4g} (t={out['t_F1_min']:.0f} h) -> {out['F1_final']:.4f}   "
              f"delta_f {out['delta_f_max']:+.4f} max / {out['delta_f_final']:+.4f} final")
        print(f"  int_inv_mouth={int_mouth:.1f}  int_inv_deep={int_deep:.1f}   "
              f"f in [{out['f_min']:.4g}, {out['f_max']:.4g}]   "
              f"adv_substeps_max={out['adv_substeps_max']}")
        print(f"  late slope per 1000 h:  dF1={dF1_late:+.3e}   drho={drho_late:+.3e}")
        for k in checks:
            failed, detail = checks[k]
            if failed:
                tag = "FAIL" if k in hard else "warn"
                print(f"  [{tag}] {k}: {detail}")
        print(f"  ADMISSIBLE: {admissible}")
    return out
