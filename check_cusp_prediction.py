"""Predict the jamming cusp from the pressure-plane time map -- no solver runs.

READ-ONLY.  Imports src2/closures.py + config.py; edits nothing; runs no simulation.

    py -3 src2/check_cusp_prediction.py            # validate, solve the cusp, test it
    py -3 src2/check_cusp_prediction.py --quick    # skip the amplitude sweep

THE CLAIM BEING TESTED
----------------------
draft2.tex eq:Lstar_exact_new already defines, at fixed n0, a map

    L(p_*; n0) = int_0^{p_*} dp / sqrt( 2 [ V(p_*) - V(p) ] ),
    V(p) = int_0^p g(n0 - s/(Y Dn)) Psi^-1(s) ds,     p = Psi(rho),

whose INVERSE is the steady branch of fig:steadyfold(a).  Everything about the fold
is then a statement about derivatives of that one function:

    tipping point   L' = 0
    cusp            L' = L'' = 0            <- two equations, two unknowns (rho_c, n0*)

Writing u = rho_* - rho_c, mu = n0 - n0*, and expanding Phi := dL/drho_* about the
cusp (where Phi and dPhi/du both vanish):

    Phi = c1 mu + (c3/2) u^2 + ...,   c1 := d2L/drho dn0,  c3 := d3L/drho^3

gives u_pm = +-sqrt(kappa mu) with kappa := -2 c1/c3, hence

    d_rho = 2 sqrt(kappa mu)                      ~ (n0-n0*)^{1/2}
    d_L   = (c3/12) d_rho^3 = (2c3/3) kappa^{3/2} mu^{3/2}  ~ (n0-n0*)^{3/2}

So 1/2 and 3/2 are FORCED, the amplitudes are predicted, and n0* comes from the
cusp conditions instead of from back-extrapolating the measured turning points --
which is what currently makes the draft's fitted 0.497/1.487 partly circular.

WHY THE VARIABLE IS rho_* AND NOT p_*
-------------------------------------
dp/drho = D_col > 0, so L' = 0 and L'' = 0 pick out the same points either way, and
the expansion above is generic in the variable.  Working in rho_* avoids ever
inverting Psi and predicts d_rho directly, which is what fig:steadyfold(d) measures.

TWO NUMERICAL TRAPS, BOTH HANDLED
---------------------------------
  * The integrand has a (rho_*-rho)^{-1/2} endpoint singularity AND the upper limit
    is the variable being differentiated.  The substitution rho = rho_*(1-s^2) with
    s in [0,1] removes both: fixed limits, smooth integrand, value G(p_*) at s=0.
  * D_col has a KINK at rho_jam (piecewise, slope jumps ~1000x).  Gauss panels that
    straddle it lose several digits.  Psi is built with rho_jam as an exact panel
    edge and the inner quadrature is split there.  Without this the upper turning
    point at n0=1500 comes out 37.699 instead of 37.687.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                                      # noqa: E402
import closures as C                                               # noqa: E402

#: measured elsewhere, for comparison only -- never used in the computation
REF_N0_STAR = 896.165          # steady_single_strain.py --fold (bisection + extrapolation)
REF_FOLD_1500 = (37.6871, 38.3404)   # L window at n0 = 1500

P = config.params(1000.0)      # n0 is passed per call; only the closure params are used
YDN = P.Y * P.Dn
RJ = P.rho_jam
RMAX = 0.72

g = lambda n: P.mu2 * n / (P.Ks2 + n)                              # noqa: E731  glutton
Dc = lambda r: C.collective_diffusivity(np.asarray(r, float), P)   # noqa: E731


# =============================================================================
# Psi(rho) = int_0^rho D_col, with rho_jam an exact panel edge
# =============================================================================
_gx, _gw = np.polynomial.legendre.leggauss(7)


def _panels(lo, hi, n):
    e = np.linspace(lo, hi, n + 1)
    a, b = e[:-1, None], e[1:, None]
    m, h = 0.5 * (a + b), 0.5 * (b - a)
    return e, h[:, 0] * (Dc(m + h * _gx[None, :]) * _gw[None, :]).sum(axis=1)


_e1, _p1 = _panels(0.0, RJ, 30000)
_e2, _p2 = _panels(RJ, RMAX, 12000)
PSI = CubicSpline(np.concatenate([_e1, _e2[1:]]),
                  np.concatenate([[0.0], np.cumsum(np.concatenate([_p1, _p2]))]))

NS, NIN = 220, 24
_sx, _sw = np.polynomial.legendre.leggauss(NS)
_sx, _sw = 0.5 * (_sx + 1.0), 0.5 * _sw
_ix, _iw = np.polynomial.legendre.leggauss(NIN)


def _seg(a, b, n0):
    """int_a^b g(n0 - Psi/YDn) r D_col(r) dr, vectorised over equal-length a, b."""
    m, h = 0.5 * (a + b), 0.5 * (b - a)
    r = m[:, None] + h[:, None] * _ix[None, :]
    return h * (g(n0 - PSI(r) / YDN) * r * Dc(r) * _iw[None, :]).sum(axis=1)


def _sigma_nodes(rs):
    """Gauss nodes for the outer s-integral on [0,1], SPLIT at the D_col kink.

    The integrand carries D_col(rho_*(1-s^2)), which has a slope discontinuity at
    s_jam = sqrt(1 - rho_jam/rho_*) whenever rho_* > rho_jam.  Gauss-Legendre across
    a kink converges at low order: leaving it unsplit makes the UPPER turning point
    at n0=1500 wander by ~0.02 um with the finite-difference step, while the lower
    one (which sits below rho_jam) is stable to 1e-6.
    """
    if rs <= RJ:
        return _sx, _sw
    sj = np.sqrt(max(1.0 - RJ / rs, 0.0))
    x, w = [], []
    for lo, hi in ((0.0, sj), (sj, 1.0)):
        if hi - lo < 1e-14:
            continue
        m, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        x.append(m + h * (2.0 * _sx - 1.0))
        w.append(h * 2.0 * _sw)
    return np.concatenate(x), np.concatenate(w)


def ell(rs, n0):
    """The time map L(rho_*; n0), eq:Lstar_exact_new in the rho variable."""
    sx, sw = _sigma_nodes(rs)
    lo = rs * (1.0 - sx ** 2)
    hi = np.full_like(lo, rs)
    straddle = (lo < RJ) & (hi > RJ)                  # split the D_col kink
    inner = np.where(straddle,
                     _seg(np.minimum(lo, RJ), np.full_like(lo, RJ), n0)
                     + _seg(np.full_like(lo, RJ), np.maximum(hi, RJ), n0),
                     _seg(lo, hi, n0))
    return float(np.sqrt(2.0 * rs) * (sw * Dc(lo) / np.sqrt(inner / (rs * sx ** 2))).sum())


# --- derivatives.  Steps chosen so the FD truncation and the quadrature noise
# --- are both well below the quantity; --quick prints the step sensitivity.
def d1(r, n, h=2e-4):
    return (ell(r + h, n) - ell(r - h, n)) / (2 * h)


def d2(r, n, h=2e-3):
    return (ell(r + h, n) - 2 * ell(r, n) + ell(r - h, n)) / h ** 2


def d3(r, n, h=6e-3):
    """f''' = [f(+2h) - 2f(+h) + 2f(-h) - f(-2h)] / (2h^3).  Sign order matters."""
    return (ell(r + 2 * h, n) - 2 * ell(r + h, n)
            + 2 * ell(r - h, n) - ell(r - 2 * h, n)) / (2 * h ** 3)


def cap(n0):
    """rho at which n(L) = 0; beyond it G(p_*) = 0 and the time map diverges."""
    return (brentq(lambda r: PSI(r) - YDN * n0, 1e-9, RMAX - 1e-6)
            if PSI(RMAX - 1e-6) > YDN * n0 else RMAX - 1e-6)


def turning(n0, npts=1100):
    rr = np.linspace(0.12, 0.985 * cap(n0), npts)
    dd = np.array([d1(r, n0) for r in rr])
    return [brentq(lambda r: d1(r, n0), rr[i], rr[i + 1], xtol=1e-13)
            for i in np.where(np.diff(np.sign(dd)))[0]]


def interior_min(n0):
    """(min of dL/drho over the interior, argmin).  Fold present iff the min < 0."""
    rr = np.linspace(0.12, 0.985 * cap(n0), 220)
    dv = np.array([d2(r, n0) for r in rr])
    best = None
    for i in np.where(np.diff(np.sign(dv)))[0]:
        if dv[i] < 0 < dv[i + 1]:                     # d2 crosses up => min of d1
            rc = brentq(lambda r: d2(r, n0), rr[i], rr[i + 1], xtol=1e-12)
            v = d1(rc, n0)
            if best is None or v < best[0]:
                best = (v, rc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the amplitude sweep")
    a = ap.parse_args()

    print("=" * 78)
    print("  cusp of the pressure-plane time map  (eq:Lstar_exact_new), n0 in uM")
    print("=" * 78)

    # ---- 1. validate the quadrature against the solver-based script ----------
    n0 = 1500.0
    t = turning(n0)
    print("\n  [1] validation at n0=1500 against steady_single_strain.py --fold")
    if len(t) == 2:
        Ls = sorted(ell(r, n0) for r in t)
        print("      L window  = [%.6f, %.6f]   script: [%.4f, %.4f]"
              % (Ls[0], Ls[1], REF_FOLD_1500[0], REF_FOLD_1500[1]))
        print("      rho_*     = %.6f, %.6f" % (min(t), max(t)))
        ok = (abs(Ls[0] - REF_FOLD_1500[0]) < 2e-3 and abs(Ls[1] - REF_FOLD_1500[1]) < 2e-3)
        print("      %s" % ("OK" if ok else "MISMATCH -- do not trust anything below"))
    else:
        raise SystemExit("      found %d turning points, expected 2" % len(t))

    # ---- 2. solve the cusp ---------------------------------------------------
    n0s = brentq(lambda n: interior_min(n)[0], 840.0, 960.0, xtol=1e-8)
    rc = interior_min(n0s)[1]
    c3 = d3(rc, n0s)
    c1 = (d1(rc, n0s + 2.0) - d1(rc, n0s - 2.0)) / 4.0
    kap = -2.0 * c1 / c3

    print("\n  [2] cusp, from  dL/drho = d2L/drho2 = 0  (no simulation, no fitting)")
    print("      n0*     = %.4f uM        measured elsewhere: %.3f  (%+.1e rel)"
          % (n0s, REF_N0_STAR, (n0s - REF_N0_STAR) / REF_N0_STAR))
    print("      rho_c   = %.6f            rho_c/rho_jam = %.4f" % (rc, rc / RJ))
    print("      L_cusp  = %.5f um" % ell(rc, n0s))
    print("      residuals: dL/drho = %+.2e   d2L/drho2 = %+.2e" % (d1(rc, n0s), d2(rc, n0s)))
    print("      c1 = %+.6e   c3 = %+.6e   kappa = %+.6e" % (c1, c3, kap))
    print("      kappa %s 0  =>  fold for n0 %s n0*"
          % (">" if kap > 0 else "<", ">" if kap > 0 else "<"))
    print("      step sensitivity: c3 %+.4e/%+.4e/%+.4e   c1 %+.4e/%+.4e/%+.4e"
          % (d3(rc, n0s, 4e-3), c3, d3(rc, n0s, 9e-3),
             (d1(rc, n0s + 1) - d1(rc, n0s - 1)) / 2, c1,
             (d1(rc, n0s + 4) - d1(rc, n0s - 4)) / 8))

    print("\n  [3] predicted amplitudes (parameter-free)")
    print("      d_rho = %.6f sqrt(mu)          d_L = %.6e mu^{3/2}"
          % (2 * np.sqrt(kap), 2 * c3 / 3 * kap ** 1.5))
    print("      d_L/(d_rho)^3 -> c3/12 = %.4f" % (c3 / 12))
    if a.quick:
        return

    # ---- 3. test them, and check the ERROR ORDER ----------------------------
    print("\n  [4] prediction vs measurement.  The relative error must scale as O(mu):")
    print("      doubling mu doubles it.  An O(mu^1/2) correction would give x1.41.")
    print("      %7s %11s %11s %9s %11s %11s %9s"
          % ("mu", "drho meas", "drho pred", "err", "dL meas", "dL pred", "err"))
    prev = None
    for mu in (0.5, 1.0, 2.0, 5.0, 10.0):
        t = turning(n0s + mu)
        if len(t) != 2:
            continue
        dr = abs(t[1] - t[0])
        dL = abs(ell(t[0], n0s + mu) - ell(t[1], n0s + mu))
        drp, dLp = 2 * np.sqrt(kap * mu), 2 * c3 / 3 * (kap * mu) ** 1.5
        e = dr / drp - 1.0
        print("      %7.2f %11.6f %11.6f %8.4f%% %11.4e %11.4e %8.4f%%"
              % (mu, dr, drp, 100 * e, dL, dLp, 100 * (dL / dLp - 1.0)))
        if prev is not None and prev[0] * 2 == mu:
            print("        %*s error ratio vs mu/2: %.2f (O(mu) => 2.00)" % (14, "", e / prev[1]))
        prev = (mu, e)

    # ---- 4. refit the exponents against the DERIVED n0* ---------------------
    mus = np.geomspace(0.15, 2.0, 9)
    rows = []
    for mu in mus:
        t = turning(n0s + mu)
        if len(t) == 2:
            rows.append((mu, abs(t[1] - t[0]),
                         abs(ell(t[0], n0s + mu) - ell(t[1], n0s + mu))))
    M = np.array(rows)
    fit = lambda x, y: np.polyfit(np.log(x), np.log(y), 1)[0]      # noqa: E731
    asep, awid = fit(M[:, 0], M[:, 1]), fit(M[:, 0], M[:, 2])
    print("\n  [5] exponents refitted against the DERIVED n0*, mu in [0.15, 2]")
    print("      alpha_sep   = %.4f   (normal form 1/2)" % asep)
    print("      alpha_width = %.4f   (normal form 3/2)" % awid)
    print("      ratio       = %.4f   (normal form exactly 3; needs no n0*)" % (awid / asep))
    print("      draft, fitted against its back-extrapolated n0*: 0.4966 / 1.4872")


if __name__ == "__main__":
    main()
