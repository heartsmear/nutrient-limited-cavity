"""
plot_onset_law.py -- validation of the onset amplitude law, including the D_col'(0) term.

Imports steady_single_strain read-only and solves the full nonlinear steady state; nothing in the
solver is changed.  The law under test (appendix, Colonisation Bifurcation):

    delta_L = 3 g0 kappa0 / (2 g1_eff) * (L - L_onset),
    g1_eff  = g1 + g0 * Dcol'(0) * Y Dn / (2 D0^2)          (= g1 (1 + R))

The dilute approximation (the draft's law) is the same with g1_eff -> g1, i.e. it drops the O(rho^2) term of Psi.  Their ratio
is 1 + R with

    R = g0 Dcol'(0) Y Dn / (2 D0^2 g1)
      = (Dcol'(0)/D0) (Y Dn n0 / 2 D0) (1 + n0/Ks)          for Monod kinetics.

WHY THIS VALIDATES THE LAW GENERALLY, NOT AT ONE n0.  The law has no free parameter, and n0, the
kinetics (mu, Ks) and the closure (eta, through Dcol'(0) = (8 - eta) D0) enter the ratio
measured/dilute-approximation ONLY through R:  measured/dilute = 1/(1+R).  So runs at any n0, for either strain and
for different closures, must all fall on that one curve.  Panel (c) tests exactly that; panel (d)
shows the residual of the general delta_L.  Panels (a) and (b) illustrate two cases: n0 = 15 uM (R = 0.03)
and n0 = 1000 uM (R = 69).

MEASUREMENT (not steady_single_strain.measure_onset_slope, which failed here in two ways).
  * Onset is the solver's own DISCRETE onset length L0 (check_onset_eigenvalue), so the small
    O(dx^2) offset of the analytic L_onset cannot bias delta_L/(L - L0) at small eps.
  * eps = 1e-4 ... 2e-3.  The old range 1.25e-3 ... 1e-2 is too coarse where Ks is small: for
    the scavenger (Ks = 0.2 uM) delta_L already exceeds Ks at eps = 0.01 and the ratio curves.
    Going below 1e-4 fails the other way: a nearly empty state then meets Newton's tolerance, so
    the amplitude is not determined by the solve.  The ratio delta_L/(L - L0) is fitted with a
    quadratic in eps; its intercept is the measured slope.
  * Every L is solved twice, seeded with the small-amplitude onset mode at 2x and 5x the
    general delta_L's amplitude, and kept only if both converge to the same delta_L (1e-4).  The
    seed therefore only selects the branch that bifurcates from the empty cavity; a result that
    echoed the seed would fail the check.  This matters: with the default seeds, eta = 7 at
    n0 >= 1000 converged to a second, dense branch that coexists with the small one right at
    onset (delta_L ~ 360 uM as L -> L0) -- real for that steeper closure, but not the branch
    the law describes.
N = 400 cells.

Writes beside the PNG: <stem>_curves.csv (panels a, b) and <stem>_collapse.csv (panels c, d).
No explanatory text is drawn into the figure: panel tags, axis labels and legends only.

    py -3.11 src2/plot_onset_law.py
    py -3.11 src2/plot_onset_law.py --out ../Paper/onset_law_validation.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                        # noqa: E402
import closures as C                                 # noqa: E402
import steady_single_strain as S                     # noqa: E402

N_CELLS = 400
EPS_CURVE = np.geomspace(1e-4, 10 ** -1.5, 12)       # (a), (b): dilute side of onset only
EPS_FIT = np.geomspace(1e-4, 2e-3, 7)                # (c), (d): slope extrapolation
SURFACE, INK, INK2, MUTED, GRID_C = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"
C_NEW, C_OLD, C_PTS = "#2a78d6", "#c1392b", "#0b0b0b"

#: (label, strain, eta, n0 values) -- strain 2 = glutton (mu 0.78, Ks 15), 1 = scavenger (0.21, 0.2)
FAMILIES = [
    (r"glutton, $\eta=5.8$",   2, 5.8, np.geomspace(0.1, 1e4, 16)),
    (r"scavenger, $\eta=5.8$", 1, 5.8, np.geomspace(0.01, 1e3, 11)),
    (r"glutton, $\eta=7$",     2, 7.0, np.geomspace(1.0, 1e4, 9)),
    (r"glutton, $\eta=4$",     2, 4.0, np.geomspace(1.0, 1e4, 9)),
]
MARKERS = ("o", "s", "^", "D")
COLORS = ("#2a78d6", "#eb6834", "#1a9e77", "#7a3fb8")


def dcol_prime0(p):
    """Dcol'(0) from the code's own closure, by a one-sided difference (not assumed)."""
    h = 1e-7
    d = C.collective_diffusivity(np.array([0.0, h]), p)
    return float((d[1] - d[0]) / h)


def laws(p, mu, Ks):
    """(s_dilute, s_general, R, L_onset) for parameters p and kinetics (mu, Ks)."""
    g0, g1 = float(S.g_of(p.n0, mu, Ks)), float(S.dg_of(p.n0, mu, Ks))
    kap = np.sqrt(g0 / p.D0)
    R = g0 * dcol_prime0(p) * p.Y * p.Dn / (2.0 * p.D0 ** 2 * g1)
    s_dilute = 1.5 * g0 * kap / g1
    return s_dilute, s_dilute / (1.0 + R), R, S.onset_length(p, mu, Ks)


#: Two deliberately wrong seed amplitudes, relative to the general delta_L.  Both are ABOVE it: a
#: seed below the solution (tested: 0.3x) falls into the empty state, which is always a
#: solution too.  2x and 5x stay valid even if the law were off by a factor of two.
SEED_FACTORS = (2.0, 5.0)
#: Both seeds must reach the same delta_L to this relative tolerance.  Near onset the solve pins
#: the amplitude to ~2e-6 (measured), so 1e-4 is strict for a test at the 1e-3 level.
SEED_AGREE = 1e-4


def onset_branch(p, cl, mu, Ks, eps_list, s_seed):
    """delta_L and rho_max/rho_jam on the branch that bifurcates from the empty cavity.

    Returns (L0, rows) with rows = (eps, L, L - L0, delta_L, rho_max/rho_jam, seed_spread).
    Each L is solved twice, from the onset mode delta = a sin(kappa0 x) with amplitude a at
    SEED_FACTORS x the general delta_L's prediction.  Both must converge onto a colonised state with
    the same delta_L (relative spread <= SEED_AGREE); otherwise delta_L is nan.  So the seed only
    picks the small-amplitude branch -- a result that merely echoed the seed would fail this.
    (Continuation from eps = 1e-5 was tried and rejected: that close to a transcritical onset a
    nearly empty state already meets Newton's tolerance, so the amplitude is not determined.)
    """
    L0, _ = S.check_onset_eigenvalue(p, mu, Ks, N_CELLS)
    kap = np.sqrt(float(S.g_of(p.n0, mu, Ks)) / p.D0)
    rows = []
    for e in sorted(eps_list):
        L = L0 * (1.0 + e)
        st = S.Steady(L, N_CELLS, p, cl, mu, Ks, False)
        got, rj = [], np.nan
        for fac in SEED_FACTORS:
            amp = min(fac * s_seed * (L - L0), 0.5 * p.n0)
            d = amp * np.sin(np.clip(kap * st.x, 0.0, 0.5 * np.pi))
            rho = np.asarray(cl.psi_inv(np.clip(cl.YDn * d, 0.0, cl.psi_max)), float)
            u0 = st.pack(np.clip(rho, 0.0, S.RHO_HI), p.n0 - d)
            r = S.solve_steady(L, N_CELLS, p, cl, mu, Ks, u0=u0, seed_label="onset-mode x%g" % fac)
            ok = r["info"]["converged"] and r["info"]["branch"] == "colonised"
            got.append(p.n0 - float(r["n"][-1]) if ok else np.nan)
            rj = float(np.max(r["rho"])) / p.rho_jam
        spread = abs(got[0] - got[1]) / max(abs(np.nanmean(got)), 1e-300)
        dl = float(np.mean(got)) if np.all(np.isfinite(got)) and spread <= SEED_AGREE else np.nan
        rows.append((e, L, L - L0, dl, rj, spread))
    return L0, rows


def measured_slope(p, cl, mu, Ks, s_seed):
    """Intercept of delta_L/(L - L0) against eps (quadratic fit).

    Returns (slope, worst relative fit residual, eps values used, worst seed spread among the
    USED values, rows).  Only eps values whose two seeds agree enter the fit; at least 4 needed.
    """
    L0, rows = onset_branch(p, cl, mu, Ks, EPS_FIT, s_seed)
    e = np.array([r[0] for r in rows]); ratio = np.array([r[3] / r[2] for r in rows])
    m = np.isfinite(ratio)
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum()), np.nan, rows
    c = np.polyfit(e[m], ratio[m], 2)
    spread = max(r[5] for r, ok in zip(rows, m) if ok)
    return (float(c[-1]), float(np.max(np.abs(np.polyval(c, e[m]) - ratio[m]) / c[-1])),
            int(m.sum()), float(spread), rows)


def curve(n0, cl, p_base, mu, Ks):
    """delta_L against L - L0 on the dilute side of onset, plus both laws."""
    p = S._with_n0(p_base, n0)
    s_dilute, s_general, R, L_on = laws(p, mu, Ks)
    L0, br = onset_branch(p, cl, mu, Ks, EPS_CURVE, s_general)
    rows = [(n0, e, L, dL, dl, s_dilute * dL, s_general * dL, rj) for e, L, dL, dl, rj, _ in br]
    return rows, (s_dilute, s_general, R, L0)


def style(ax, tag):
    ax.set_facecolor(SURFACE)
    ax.set_title(tag, loc="left", fontsize=11, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="both", color=GRID_C, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "onset_law_validation.png"))
    ap.add_argument("--n0", type=float, nargs=2, default=(15.0, 1000.0),
                    help="the two illustrated supplies for panels (a), (b) [uM]")
    a = ap.parse_args()

    closures = {}

    def closure_for(eta):
        if eta not in closures:
            closures[eta] = S.Closure(config.params(15.0, eta=eta), n_grid=200_001)
        return closures[eta]

    # ---- (a), (b): the two illustrated supplies, glutton, production closure ---------------
    p_g = config.params(15.0)
    mu_g, Ks_g, _ = S.strain_kinetics(p_g, 2)
    curves, laws_ab = [], []
    for n0 in a.n0:
        rows, lw = curve(n0, closure_for(5.8), p_g, mu_g, Ks_g)
        curves.append(rows); laws_ab.append(lw)

    # ---- (c), (d): the collapse across n0, kinetics and closure ----------------------------
    collapse = []
    for (lab, strain, eta, n0s), mk, col in zip(FAMILIES, MARKERS, COLORS):
        p_f = config.params(15.0, eta=eta)
        mu, Ks, _ = S.strain_kinetics(p_f, strain)
        cl = closure_for(eta)
        for n0 in n0s:
            p = S._with_n0(p_f, n0)
            s_dilute, s_general, R, L_on = laws(p, mu, Ks)
            s_meas, fit_res, n_used, spread, rows = measured_slope(p, cl, mu, Ks, s_general)
            rj = max(r[4] for r in rows)
            collapse.append(dict(family=lab, strain=strain, mu=mu, Ks=Ks, eta=eta,
                                 dcol0_over_D0=dcol_prime0(p) / p.D0, n0=n0, R=R, L_onset=L_on,
                                 s_meas=s_meas, s_dilute=s_dilute, s_general=s_general,
                                 meas_over_dilute=s_meas / s_dilute, meas_over_general=s_meas / s_general,
                                 fit_rel_residual=fit_res, eps_used=n_used,
                                 seed_spread_used=spread, rho_max_over_jam=rj,
                                 marker=mk, color=col))
            print(f"  {lab:24s} n0={n0:10.4g}  R={R:10.4g}  meas/dilute={s_meas / s_dilute:10.4g}"
                  f"  meas/general={s_meas / s_general:.6f}  fit {fit_res:.1e}  eps used {n_used}/"
                  f"{len(EPS_FIT)}  seeds {spread:.1e}  rho/jam {rj:.2e}", flush=True)

    # ---- figure ----------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 9.0))
    fig.patch.set_facecolor(SURFACE)
    for k, (rows, (s_dilute, s_general, R, L_on), n0) in enumerate(zip(curves, laws_ab, a.n0)):
        A = ax[0, k]; style(A, "(%s)" % "ab"[k])
        dL = np.array([r[3] for r in rows]); dl = np.array([r[4] for r in rows])
        xx = np.geomspace(dL.min() * 0.8, dL.max() * 1.25, 50)
        A.plot(xx, s_dilute * xx, ls="--", color=C_OLD, lw=1.8, label="dilute approximation")
        A.plot(xx, s_general * xx, ls="-", color=C_NEW, lw=1.8, label=r"general $\delta_L$")
        A.plot(dL, dl, "o", color=C_PTS, ms=6, mfc="white", mew=1.5, label="steady-state solver",
               zorder=5)
        A.set_xscale("log"); A.set_yscale("log")
        A.set_xlabel(r"$L-L_{\rm onset}$  [$\mu$m]", color=INK, fontsize=10)
        A.set_ylabel(r"$\delta_L=n_0-n(L)$  [$\mu$M]", color=INK, fontsize=10)
        A.legend(title=r"$n_0=%g\ \mu$M" % n0, frameon=False, fontsize=8.5, title_fontsize=9,
                 loc="upper left")

    B = ax[1, 0]; style(B, "(c)")
    Rs = np.geomspace(min(c["R"] for c in collapse) * 0.5, max(c["R"] for c in collapse) * 2, 200)
    B.plot(Rs, 1.0 / (1.0 + Rs), color=INK2, lw=1.6, label=r"$1/(1+R)$", zorder=2)
    D_ = ax[1, 1]; style(D_, "(d)")
    D_.axhline(1.0, color=INK2, lw=1.2, zorder=2)
    for lab, mk, col in zip([f[0] for f in FAMILIES], MARKERS, COLORS):
        cc = [c for c in collapse if c["family"] == lab and np.isfinite(c["s_meas"])]
        B.plot([c["R"] for c in cc], [c["meas_over_dilute"] for c in cc], mk, color=col, ms=6.5,
               mfc="none", mew=1.5, label=lab, zorder=4)
        D_.plot([c["R"] for c in cc], [c["meas_over_general"] for c in cc], mk, color=col, ms=6.5,
                mfc="none", mew=1.5, label=lab, zorder=4)
    B.set_xscale("log"); B.set_yscale("log")
    B.set_xlabel(r"$R$", color=INK, fontsize=10)
    B.set_ylabel("measured slope / dilute approximation", color=INK, fontsize=10)
    B.legend(frameon=False, fontsize=8.5, loc="lower left")
    D_.set_xscale("log")
    D_.set_ylim(0.995, 1.005)
    D_.set_xlabel(r"$R$", color=INK, fontsize=10)
    D_.set_ylabel(r"measured slope / general $\delta_L$", color=INK, fontsize=10)
    D_.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("\nwrote %s" % a.out)

    # ---- CSVs: every plotted array ----------------------------------------------------------
    stem = os.path.splitext(a.out)[0]
    with open(stem + "_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n0_uM", "eps", "L_um", "L_minus_L0_um", "deltaL_measured_uM",
                    "deltaL_dilute_approximation_uM", "deltaL_general_uM",
                    "rho_max_over_jam"])
        for rows in curves:
            for r in rows:
                w.writerow(["%g" % r[0], "%.6g" % r[1]] + ["%.10g" % v for v in r[2:]])
    keys = ["family", "strain", "mu", "Ks", "eta", "dcol0_over_D0", "n0", "R", "L_onset",
            "s_meas", "s_dilute", "s_general", "meas_over_dilute", "meas_over_general", "fit_rel_residual",
            "eps_used", "seed_spread_used", "rho_max_over_jam"]
    with open(stem + "_collapse.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for c in collapse:
            w.writerow([c[k] if isinstance(c[k], str) else "%.10g" % c[k] for k in keys])
    print("wrote %s_{curves,collapse}.csv" % stem)

    print("\nNUMBERS FOR THE CAPTION")
    for n0, (s_dilute, s_general, R, L_on) in zip(a.n0, laws_ab):
        print(f"  n0 = {n0:g} uM:  L_onset = {L_on:.3f} um,  R = {R:.4g},  "
              f"dilute approximation too steep by {1 + R:.4g}x")
    fin = [c for c in collapse if np.isfinite(c["s_meas"])]
    if not fin:
        print("  collapse: no run passed the seed-agreement check")
        return
    dev = max(abs(c["meas_over_general"] - 1.0) for c in fin)
    print(f"  collapse: {len(fin)} of {len(collapse)} runs, R = {min(c['R'] for c in fin):.3g} .. "
          f"{max(c['R'] for c in fin):.3g};  max |measured/general - 1| = {dev:.2e};  "
          f"max seed spread among used points {max(c['seed_spread_used'] for c in fin):.1e};  "
          f"eps values used per run {min(c['eps_used'] for c in fin)}..{len(EPS_FIT)}")
    for c in collapse:
        if not np.isfinite(c["s_meas"]):
            print(f"  EXCLUDED: {c['family']}, n0 = {c['n0']:g} uM (R = {c['R']:.3g}): only "
                  f"{c['eps_used']} of {len(EPS_FIT)} eps values had agreeing seeds")


if __name__ == "__main__":
    main()
