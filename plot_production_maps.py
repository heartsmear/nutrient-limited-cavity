"""
plot_production_maps.py -- where biomass is ACTUALLY made, not where it could be.  Simulates nothing.

    py -3.11 src2/plot_production_maps.py run_n0-15_L1000_dx0.5_T500
    py -3.11 src2/plot_production_maps.py run_n0-1000_dx0.5_T10000 --tmax 2000

THE DISTINCTION THIS EXISTS TO MAKE.  `plot_growth_maps.py` draws g_i(n(x,t)), the rate a cell
WOULD grow at if one were there.  That is a property of the environment, and in a nutrient-filled
cavity it is near-maximal everywhere for the first tens of hours -- which invites the reading that
the deep interior is productive early on.  It is not: at t = 12.5 h in the dilute run the cavity
at x = 900 um sits at full rate g1 = 0.207 /h with a density of 1.8e-21.  A rate needs something
to multiply.

What is plotted here is the PRODUCTION RATE, the actual source term of the density equation:

    P_i(x,t) = q(rho) * g_i(n) * rho_i          [density per hour]

and its time integral, which is the biomass each location has ever contributed:

    Y_i(x) = int P_i dt                          [density]

Y_i is the honest answer to "where does the biomass come from".  It is production, not standing
stock: transport then moves what was made, so Y_i(x) is deliberately NOT the final profile.

PANELS.  (a) and (b) are P_1 and P_2 on a shared logarithmic colour scale -- production spans
20+ decades across the plane, so a linear scale shows one bright pixel and nothing else.  (c) is
the net P_1 - P_2 on a symmetric-log diverging scale, so the sign structure survives the same
range problem.  (d) is the cumulative fraction of total production against x, which is the panel
that actually answers the question, with the 50% and 90% points marked.

TIME AXIS IS LOG by default (`--linear-t` to override), for the same reason as in
plot_growth_maps: the nutrient is drawn down in the first tens of hours of a 500-2000 h run.

WHY q IS INCLUDED even though it is 1 in both reference runs: P is meant to be the solver's
actual source term, and `closures.growth_space_factor` is part of it.  Leaving it out would make
this file silently wrong for any future run that reaches rho_c = 0.95.  The printout reports
whether q ever departed from 1.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                        # noqa: E402
import closures as C                                 # noqa: E402
import plot_fst as P                                 # noqa: E402  (n0_from_name)
import plot_growth_maps as G                         # noqa: E402  (read_columns, panel)

SURFACE, INK, INK2, MUTED, GRIDC = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"
COL1, COL2, COLT = "#2a78d6", "#c1392b", "#0b0b0b"


def quantile_x(w, x, qs):
    tot = float(np.sum(w))
    if not np.isfinite(tot) or tot <= 0:
        return [float("nan")] * len(qs)
    c = np.cumsum(w) / tot
    return [float(np.interp(q, c, x)) for q in qs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir")
    ap.add_argument("--out", default=None, help="default: <rundir>/production_maps.png")
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--no-contours", dest="contours", action="store_false")
    ap.add_argument("--linear-t", dest="log_t", action="store_false")
    ap.add_argument("--decades", type=float, default=8.0,
                    help="colour range below the peak, in decades (default 8); production spans "
                         "20+ decades and the deep tail is numerically meaningless")
    a = ap.parse_args()

    d = a.rundir if os.path.isabs(a.rundir) else os.path.join(HERE, a.rundir)
    dump = os.path.join(d, "full_dump.csv")
    if not os.path.exists(dump):
        sys.exit(f"no full_dump.csv in {d}; re-run figures.py with --dump")
    name = os.path.basename(os.path.normpath(d))
    n0 = P.n0_from_name(d)
    p = config.params(n0 if n0 is not None else config.N0_DILUTE)
    nstar = float(C.monod_crossover(p))

    t, x, col = G.read_columns(dump, ("n", "rho", "rho1", "rho2"), tmax=a.tmax)
    n, rho, r1, r2 = col["n"], col["rho"], col["rho1"], col["rho2"]
    q = np.asarray(C.growth_space_factor(rho, p))
    P1 = q * np.asarray(C.g1(n, p)) * r1
    P2 = q * np.asarray(C.g2(n, p)) * r2
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0

    # time integral -> biomass ever produced per unit volume at each x
    Y1 = np.trapezoid(P1, t, axis=1) if hasattr(np, "trapezoid") else np.trapz(P1, t, axis=1)
    Y2 = np.trapezoid(P2, t, axis=1) if hasattr(np, "trapezoid") else np.trapz(P2, t, axis=1)
    Yt = Y1 + Y2

    print(f"{name}   n0={float(p.n0):g} uM   n*={nstar:.4f} uM   "
          f"{len(x)} cells x {len(t)} frames   t=0..{t[-1]:g} h   L={x[-1]+0.5*dx:g} um")
    print(f"  q(rho) {'== 1 everywhere' if np.all(q == 1.0) else f'dips to {q.min():.4f}'}")
    print(f"\n  WHERE THE BIOMASS WAS MADE   (integral of q*g_i*rho_i over the whole run)")
    print(f"{'':>10}{'total':>13}{'x(50%)':>10}{'x(90%)':>10}{'x(99%)':>10}"
          f"{'<100um':>9}{'<200um':>9}{'>500um':>9}")
    for lab, Y in (("strain 1", Y1), ("strain 2", Y2), ("BOTH", Yt)):
        q50, q90, q99 = quantile_x(Y, x, (0.5, 0.9, 0.99))
        tot = float(Y.sum() * dx)
        f100 = float(Y[x < 100].sum() / max(Y.sum(), 1e-300))
        f200 = float(Y[x < 200].sum() / max(Y.sum(), 1e-300))
        f500 = float(Y[x > 500].sum() / max(Y.sum(), 1e-300))
        print(f"{lab:>10}{tot:>13.4e}{q50:>10.1f}{q90:>10.1f}{q99:>10.1f}"
              f"{f100:>9.4f}{f200:>9.4f}{f500:>9.4f}")

    # and when
    Tt = P1.sum(axis=0) + P2.sum(axis=0)
    tq = quantile_x(Tt, t, (0.5, 0.9))
    print(f"\n  WHEN: 50% of all production by t = {tq[0]:.1f} h, 90% by t = {tq[1]:.1f} h "
          f"(run is {t[-1]:g} h)")

    fig, ax = plt.subplots(1, 4, figsize=(6.0 * 4, 5.6), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    pmax = max(float(P1.max()), float(P2.max()))
    vmin = pmax * 10 ** (-a.decades)
    lv = pmax * 10.0 ** np.arange(-a.decades + 1, 0.0, 2.0)

    for k, (Pi, lab, mu) in enumerate(((P1, r"(a) $P_1=q\,g_1\rho_1$  strain 1 produced", None),
                                       (P2, r"(b) $P_2=q\,g_2\rho_2$  strain 2 produced", None))):
        im = G.panel(ax[k], t, x, np.maximum(Pi, vmin * 1e-3), lab, "magma",
                     norm=LogNorm(vmin=vmin, vmax=pmax), levels=lv,
                     cbar_label=r"production rate [h$^{-1}$]  (log, shared)",
                     contours=a.contours, fmt="%.0e", log_t=a.log_t)
        del im

    net = P1 - P2
    vs = float(np.max(np.abs(net)))
    G.panel(ax[2], t, x, net, r"(c) $P_1-P_2$   blue: strain 1 gaining faster", "coolwarm_r",
            norm=SymLogNorm(linthresh=max(vs * 1e-6, 1e-300), vmin=-vs, vmax=vs, base=10),
            levels=None, cbar_label=r"net production [h$^{-1}$]  (symlog)",
            contours=False, log_t=a.log_t)

    b = ax[3]
    for Y, c, lab in ((Y1, COL1, "strain 1"), (Y2, COL2, "strain 2"), (Yt, COLT, "both")):
        cum = np.cumsum(Y) / max(float(Y.sum()), 1e-300)
        b.plot(x, cum, color=c, lw=2.0 if lab == "both" else 1.6, label=lab,
               ls="-" if lab == "both" else "--")
    for lev, st in ((0.5, ":"), (0.9, "-.")):
        b.axhline(lev, color=GRIDC, lw=0.9, ls=st)
        xv = quantile_x(Yt, x, (lev,))[0]
        b.plot([xv], [lev], "o", color=COLT, ms=5, zorder=4)
        b.annotate(f"{lev*100:.0f}% by x = {xv:.0f} " + r"$\mu$m", (xv, lev),
                   textcoords="offset points", xytext=(8, -10), fontsize=9, color=COLT)
    b.set_xlabel(r"$x$ [$\mu$m]", color=INK2)
    b.set_ylabel("cumulative fraction of all biomass produced", color=INK2)
    b.set_ylim(0, 1.02); b.set_xlim(x[0], x[-1])
    b.legend(fontsize=9, frameon=False, labelcolor=INK2, loc="lower right")
    b.grid(alpha=.25, lw=.6)
    b.set_title("(d) where the biomass was actually made", fontsize=11,
                fontweight="bold", color=INK, loc="left")
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)

    fig.suptitle(f"Where biomass is actually produced   {name}   "
                 f"$n_0$={float(p.n0):g} " + r"$\mu$M, $L$=" + f"{x[-1]+0.5*dx:g} " + r"$\mu$m",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
    fig.text(0.5, 0.030,
             r"$P_i=q(\rho)\,g_i(n)\,\rho_i$ is the source term of the density equation, so (d) "
             r"is $\int P\,dt$ -- biomass PRODUCED at each $x$, not standing stock: transport "
             r"then moves it.  Colour is logarithmic over "
             f"{a.decades:g} decades below the peak.",
             ha="center", fontsize=8.5, color=MUTED)
    dtf = np.diff(t)
    fig.text(0.5, 0.006,
             f"Dump sampling: {len(t)} frames, first at t = {t[1] if len(t) > 1 else t[0]:.4g} h, "
             f"spacing {dtf.min():.4g}-{dtf.max():.4g} h.",
             ha="center", fontsize=8, color=MUTED)

    out = a.out or os.path.join(d, "production_maps.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"\nwrote {out}")

    # Every plotted array to CSV beside the PNG (rule: plotted arrays are always saved).
    # The heatmaps draw max(P, vmin*1e-3) on a clipped log scale; the CSV holds the raw P.
    stem = os.path.splitext(out)[0]
    tt, xx = np.meshgrid(t, x)                      # (x, t) layout, matching P1/P2
    with open(stem + ".csv", "w", newline="") as fh:                  # panels (a)-(c)
        fh.write("t_h,x_um,P1,P2,P1_minus_P2\n")
        np.savetxt(fh, np.column_stack([tt.ravel(), xx.ravel(), P1.ravel(), P2.ravel(),
                                        net.ravel()]), delimiter=",", fmt="%.8g")
    with open(stem + "_cumulative.csv", "w", newline="") as fh:       # panel (d)
        fh.write("x_um,Y1,Y2,Y_total,cum_frac_1,cum_frac_2,cum_frac_total\n")
        cums = [np.cumsum(Y) / max(float(Y.sum()), 1e-300) for Y in (Y1, Y2, Yt)]
        np.savetxt(fh, np.column_stack([x, Y1, Y2, Yt] + cums), delimiter=",", fmt="%.10g")
    with open(stem + "_summary.csv", "w", newline="") as fh:          # marks and colour range
        fh.write("quantity,value\n")
        x50, x90 = quantile_x(Yt, x, (0.5, 0.9))
        for k, v in (("n0_uM", float(p.n0)), ("nstar_uM", nstar), ("q_min", float(q.min())),
                     ("x50_total_um", x50), ("x90_total_um", x90),
                     ("t50_production_h", tq[0]), ("t90_production_h", tq[1]),
                     ("P_max", pmax), ("colour_vmin", vmin), ("colour_decades", a.decades),
                     ("frames", len(t)), ("T_h", float(t[-1]))):
            fh.write(f"{k},{v:.10g}\n")
    print(f"wrote {stem}.csv, {stem}_cumulative.csv, {stem}_summary.csv")


if __name__ == "__main__":
    main()
