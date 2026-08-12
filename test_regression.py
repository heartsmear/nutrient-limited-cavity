"""
test_regression.py -- Stage 1: the new stack must reproduce the old one EXACTLY.

The only thing that makes a rewrite trustworthy is a bit-identity check against the code it
replaces, run before any behaviour is allowed to change.  This is that check.

WHY EXACT IDENTITY IS ACHIEVABLE.  At chi = 0 the old `chemotaxis_cavity.simulate_chemo` skips
`apply_chemotaxis` entirely, so it is literally `solver.one_step` in a loop -- the same arithmetic
src2/solver.py does.  And `growth_space_factor` is EXACTLY 1.0 for rho <= rho_c = 0.95, and
`1.0 * x == x` in IEEE754, so the new growth term is a no-op wherever the cavity stays below 0.95.
Both reference regimes do (dilute rho ~ 0.013, jammed n0=1000 rho ~ 0.66), so both must match to
0.0 -- not "to 1e-12".

Anything nonzero here means the port changed something by accident.

    py -3.11 src2/test_regression.py
    py -3.11 src2/test_regression.py --quick     # dilute case only
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(os.path.dirname(HERE), "src")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import closures as C2                      # noqa: E402  the new stack
import solver as S2                        # noqa: E402


def _old_stack():
    """Import the old stack with its own directory first on sys.path, then restore.

    The two packages share module names (config, closures, solver), so the import order matters
    and must not leave src/ shadowing src2/ afterwards.
    """
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.pop(k) for k in
                  ("config", "closures", "solver", "chemotaxis_cavity",
                   "sweep_grounded_chemotaxis", "dself_drift", "grounded_dself_run")
                  if k in sys.modules}
    sys.path.insert(0, OLD)
    try:
        import grounded_dself_run as G      # noqa: E402
        import sweep_grounded_chemotaxis as S   # noqa: E402
        import config as OC                 # noqa: E402
        return G, S, OC
    finally:
        sys.path[:] = saved_path
        for k in list(sys.modules):
            if k in ("config", "closures", "solver", "chemotaxis_cavity",
                     "sweep_grounded_chemotaxis", "dself_drift", "grounded_dself_run"):
                pass                        # keep the old modules loaded; they are self-consistent
        sys.modules.update({k: v for k, v in saved_mods.items() if k not in sys.modules})


def compare(n0, T, dt, dx, label, shape="discontinuous"):
    """Compare src2 against whichever src path uses the SAME D_self closure.

    src/ has both: its unpatched `solver.simulate` uses the discontinuous form (alpha=0.95,
    floor=1.5e-3) -- which is now src2's DEFAULT -- while the grounded runs monkey-patch it to the
    continuous form with floor=1.5e-7, which is src2's `--D_self_cont`.  Testing both closures
    against their own counterpart is strictly better coverage than testing one.
    """
    G, S, OC = _old_stack()
    num_old = OC.numerics(T_end=T, dt=dt, n_save=3, dx=dx)
    ic_old = S.make_ic()

    t0 = time.time()
    if shape == "continuous":
        # grounded_dself_run.simulate patches D_self -> continuous, floor 1.5e-7
        old = G.simulate(G.params(n0), num_old, 0.0)
        kw_new = dict(dself_shape="continuous", Dself_floor=1.5e-7)
    else:
        # src/solver.simulate UNPATCHED: closures.self_diffusivity is the discontinuous form,
        # and make_params leaves Dself_floor at the ModelParams default of 1.5e-3.
        p_old = S.make_params("ecoli_Kd", n0, S.Y_GROUNDED, S.DN_GROUNDED)
        old = OC.solver.simulate(OC.L, p_old, num_old, ic_old,
                                 OC.solver.RunConfig(case_name="regress"))
        old = {"rho1_profiles": old["rho1_profiles"], "rho2_profiles": old["rho2_profiles"],
               "n_profiles": old["n_profiles"]}
        kw_new = dict()
    t_old = time.time() - t0

    p_old = G.params(n0)          # for the parameter cross-check below (same biology either way)
    p_new = replace(C2.ModelParams(), n0=float(n0), **kw_new)
    # reaction_tol_relative=False selects src/'s absolute Newton tolerance.  src2's default (True)
    # is a scale-relative criterion that is ~13x faster and agrees to 12 significant figures, but
    # it takes a different number of Newton steps, so it cannot be BIT-identical.  This test is the
    # bit-identity guard, so it pins the legacy rule.
    num_new = replace(C2.Numerics(), T_end=T, dt=dt, dx_target=dx, n_save=3,
                      reaction_tol_relative=False)
    ic_new = C2.InitialConditions()
    t0 = time.time()
    new = S2.simulate(1000.0, p_new, num_new, ic_new)
    t_new = time.time() - t0

    # --- the parameters themselves must agree before the fields can ----------------------
    pmis = [(k, getattr(p_old, k), getattr(p_new, k))
            for k in ("mu1", "Ks1", "mu2", "Ks2", "n0", "Dn", "Y",
                      "D0", "eta", "rho_jam", "tiny_density")
            if getattr(p_old, k) != getattr(p_new, k)]

    d1 = float(np.max(np.abs(old["rho1_profiles"][:, -1] - new["rho1_final"])))
    d2 = float(np.max(np.abs(old["rho2_profiles"][:, -1] - new["rho2_final"])))
    dn = float(np.max(np.abs(old["n_profiles"][:, -1] - new["n_final"])))
    ok = (d1 == 0.0 and d2 == 0.0 and dn == 0.0 and not pmis)

    r1 = new["rho1_final"]; r2 = new["rho2_final"]
    print(f"--- {label}: n0={n0:g} uM, T={T:g} h, dt={dt:g}, dx={dx:g} (N={new['N']}), "
          f"D_self={shape} ---")
    if pmis:
        print("  PARAMETER MISMATCH:")
        for k, a, b in pmis:
            print(f"    {k}: old={a!r}  new={b!r}")
    print(f"  max|d rho1| = {d1:.3e}     max|d rho2| = {d2:.3e}     max|d n| = {dn:.3e}")
    print(f"  new: rho_peak={new['rho_peak']:.6f}  q_min={new['q_min']:.6f}  "
          f"F1={r1.sum()/max((r1+r2).sum(),1e-300):.6f}  success={new['success']}")
    print(f"  wall: old {t_old:.1f}s   new {t_new:.1f}s")
    print(f"  {'PASS (bit-identical)' if ok else '*** FAIL ***'}\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="dilute case only")
    ap.add_argument("--n0", type=float, default=None, help="run ONE custom case instead")
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--D_self_cont", "--d-self-cont", dest="dself_cont", action="store_true",
                    help="compare the CONTINUOUS closure instead of the default")
    ap.add_argument("--dx", type=float, default=10.0,
                    help="NOTE: src/ has max_cells=600, so dx below L/600=1.667 is silently "
                         "clipped there and the comparison becomes meaningless")
    a = ap.parse_args()

    print("=" * 78)
    print("STAGE 1 -- src2 must reproduce src EXACTLY at chi=0")
    print("=" * 78)
    if a.n0 is not None:
        cases = [(a.n0, a.T, a.dt, a.dx, "custom",
                  "continuous" if a.dself_cont else "discontinuous")]
    else:
        cases = [(15.0, 120.0, 0.05, 10.0, "dilute, DEFAULT closure", "discontinuous")]
        if not a.quick:
            cases += [
                (15.0, 120.0, 0.05, 10.0, "dilute, --D_self_cont", "continuous"),
                (1000.0, 120.0, 0.05, 10.0, "jammed, DEFAULT closure", "discontinuous"),
                (1000.0, 120.0, 0.05, 10.0, "jammed, --D_self_cont", "continuous"),
            ]
    ok = all(compare(*c) for c in cases)
    print("=" * 78)
    print("ALL BIT-IDENTICAL" if ok else "*** REGRESSION ***")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
