"""
config.py -- the single source of truth for the E. coli / glucose cavity.

One parameter set, assembled in one place.  `src/` built parameters through four layers
(PAIRS dict -> make_params -> replace -> replace again for Dself_floor); here you call
`params(n0)` and that is the whole story.

Regimes carry their own dx, because dx is not a free choice: the nutrient boundary layer is
lambda = sqrt(D_n Y / (rho <mu/K>)), and it shrinks as the cavity fills.  At dx = 10 um the
n0 = 1000 uM jammed run has dx/lambda = 5.1 -- five cells' worth of structure inside one cell.

    py -3.11 src2/config.py          # self-test: parameters, n*, regimes, resolution estimate
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from closures import ModelParams, Numerics, InitialConditions, monod_crossover, g1, g2  # noqa: E402

# --- geometry -----------------------------------------------------------------------------
L = 1000.0            # um, cavity length

# --- regimes: (n0 [uM], dx [um], dt [h], T [h], label) -------------------------------------
# dx is chosen from the expected lambda, NOT inherited.  See resolution_estimate().
REGIMES = {
    "dilute":  dict(n0=15.0,   dx=5.0,  dt=0.05, T=2000.0,
                    note="grounded reference; rho/rho_jam ~ 0.02, never jams"),
    "jammed":  dict(n0=1000.0, dx=0.5,  dt=0.05, T=3000.0,
                    note="fully jammed; rho settles between rho_jam and rho_max. N=2000, slow."),
}
N0_DILUTE = REGIMES["dilute"]["n0"]
N0_JAMMED = REGIMES["jammed"]["n0"]

NSTAR = monod_crossover(ModelParams())     # 5.2526 uM, derived


#: The floor the CONTINUOUS D_self variant is normally paired with (the DASpaper docstring
#: intent, 1.5e-3 * 1e-4).  The discontinuous default uses ModelParams.Dself_floor = 1.5e-3,
#: which is what their code as distributed computes.  See closures.self_diffusivity.
DSELF_FLOOR_CONTINUOUS = 1.5e-7


def params(n0: float = N0_DILUTE, **overrides) -> ModelParams:
    """The E. coli/glucose parameter set at feed concentration n0."""
    from dataclasses import replace
    return replace(ModelParams(), n0=float(n0), **overrides)


def add_model_args(ap) -> None:
    """CLI flags shared by every driver, so the closure choice is settable from PyCharm."""
    ap.add_argument("--D_self_cont", "--d-self-cont", dest="dself_cont", action="store_true",
                    help="use the CONTINUOUS D_self, eps+(1-eps)max(1-rho/rho_jam,0) with "
                         f"eps={DSELF_FLOOR_CONTINUOUS:g}, instead of the default discontinuous "
                         "DASpaper form (alpha=0.95, floor=1.5e-3). NOT a tolerance: it changes "
                         "the jammed label diffusivity by 1e4 and hence whether the graveyard "
                         "mixes or is frozen.")
    ap.add_argument("--dself-floor", dest="dself_floor", type=float, default=None,
                    help="override the jammed D_self floor (defaults: 1.5e-3 discontinuous, "
                         f"{DSELF_FLOOR_CONTINUOUS:g} continuous)")


def params_from_args(a, n0: float, **overrides) -> ModelParams:
    """Build ModelParams honouring the shared closure flags from add_model_args."""
    kw = dict(overrides)
    if getattr(a, "dself_cont", False):
        kw.setdefault("dself_shape", "continuous")
        kw.setdefault("Dself_floor", DSELF_FLOOR_CONTINUOUS)
    if getattr(a, "dself_floor", None) is not None:
        kw["Dself_floor"] = float(a.dself_floor)
    return params(n0, **kw)


def numerics(T_end: float, dt: float, dx: float, n_save: int = 241, **overrides) -> Numerics:
    from dataclasses import replace
    return replace(Numerics(), T_end=float(T_end), dt=float(dt),
                   dx_target=float(dx), n_save=int(n_save), **overrides)


def initial_conditions(**overrides) -> InitialConditions:
    """De novo cavity invasion: Gaussian seed at the opening, neutral 50/50, filled nutrient.

    seed_f1 = 0.5 gives h0 == 0, so the inherited ancestry channel B0 vanishes exactly.
    """
    from dataclasses import replace
    return replace(InitialConditions(), **overrides)


def regime(name: str):
    """(p, num, ic) for a named regime."""
    r = REGIMES[name]
    return (params(r["n0"]),
            numerics(T_end=r["T"], dt=r["dt"], dx=r["dx"]),
            initial_conditions())


def resolution_estimate(p: ModelParams, rho: float, f1: float = 0.0) -> float:
    """A-priori nutrient screening length lambda = sqrt(D_n Y / (rho <mu/K>)).

    <mu/K> = f1 mu1/Ks1 + (1-f1) mu2/Ks2 is the low-n slope of the consumption, so this is the
    layer thickness where n << Ks.  Strain 1 has a 20x larger affinity slope than strain 2, so a
    strain-1-dominated deep region has a ~4.5x THINNER layer -- pass the f1 you expect.
    """
    slope = f1 * p.mu1 / p.Ks1 + (1.0 - f1) * p.mu2 / p.Ks2
    return float(np.sqrt(p.Dn * p.Y / max(rho * slope, 1e-300)))


def _self_test():
    p = ModelParams()
    v = monod_crossover(p)
    print("=== E. coli / glucose cavity ===")
    print(f"  mu1={p.mu1:g} Ks1={p.Ks1:g} (scavenger)   mu2={p.mu2:g} Ks2={p.Ks2:g} (glutton)")
    print(f"  n* = {v:.6f} uM   (derived, not designed)")
    print(f"  low-n affinity: mu1/Ks1 = {p.mu1/p.Ks1:.4g}  >>  mu2/Ks2 = {p.mu2/p.Ks2:.4g}")
    print(f"  high-n winner: strain {'1' if p.mu1 > p.mu2 else '2'}")
    print(f"  Y={p.Y:.4g}  Dn={p.Dn:.4g}  D0={p.D0:g}  rho_jam={p.rho_jam:g}")
    print(f"  growth ceiling: rho_c={p.rho_c:g} -> rho_max={p.rho_max:g}")
    print(f"  D_self(0)={p.D0:g}  D_self(rho_jam)={p.D0*p.Dself_floor:.4e} um^2/h")

    print("\n=== growth-space factor q(rho) ===")
    for r in (0.0, 0.5, 0.64, 0.95, 0.98, 0.999, 1.0):
        from closures import growth_space_factor
        print(f"  q({r:5.3f}) = {float(growth_space_factor(np.array([r]), p)[0]):.6f}")

    print("\n=== regimes and their resolution ===")
    for name, r in REGIMES.items():
        rho_exp = 0.02 * p.rho_jam if name == "dilute" else 0.9
        lam = resolution_estimate(params(r["n0"]), rho_exp, f1=0.5)
        print(f"  {name:8s} n0={r['n0']:>7g} uM  dx={r['dx']:g} um  dt={r['dt']:g} h  "
              f"T={r['T']:g} h")
        print(f"           expected rho~{rho_exp:.3f} -> lambda~{lam:.2f} um -> "
              f"dx/lambda~{r['dx']/lam:.2f}  {'OK' if r['dx']/lam <= 1.0 else 'UNDER-RESOLVED'}")
        print(f"           {r['note']}")


if __name__ == "__main__":
    _self_test()
