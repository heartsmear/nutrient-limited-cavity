"""Grid convergence of the jammed van Leer runs at MATCHED t = 2000 h: F1, f(x), h(x).

Reads the four existing run folders (dx = 2, 1, 0.5, 0.25); runs no simulation.

WHY t = 2000 AND NOT EACH RUN'S FINAL TIME.  Three of the runs end at T = 6000 h and one at
T = 2000 h, so "final" is not a common state.  It does not matter much -- F1 drifts by only
-3e-5 .. -6e-5 between t = 2000 and t = 6000 on every grid that got there -- but comparing at a
matched time costs nothing and removes the objection.

WHY THE LIMIT COMES FROM THE THREE FINEST GRIDS.  dx = 2 fails the front-resolution check
(dx/lambda = 0.70 > 0.6), and including it gives p = 1.64 and a limit of 0.32055 -- which the
measured dx = 0.25 value 0.32027 then undershoots, i.e. the sequence passes straight through its
own extrapolated limit.  The admissible triplet (1, 0.5, 0.25) gives p = 0.92: first order, which
is what a TVD limiter degrades to at extrema, and the composition front is exactly an extremum.

The profile limit uses that same scalar p pointwise rather than a per-cell order, because a
pointwise p is unusable wherever the two differences cross zero.

    py -3 src2/plot_grid_convergence_T2000.py
    py -3 src2/plot_grid_convergence_T2000.py --out ../Paper/grid_convergence_T2000.png

Writes beside the PNG: <stem>_F1.csv (panels a, b), <stem>_fit.csv (orders, limits, inset
window) and <stem>_profiles.csv (panels c, d and their insets, incl. the Richardson profile).
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.cm import viridis                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T_MATCH = 2000.0

RUNS = [(2.00, "run_n0-1000_L1000_dx2_T6000_seed0.0001_lim-vanleer"),
        (1.00, "run_n0-1000_L1000_dx1_T6000_seed0.0001_lim-vanleer"),
        (0.50, "run_n0-1000_L1000_dx0.5_T6000_seed0.0001_lim-vanleer"),
        (0.25, "run_n0-1000_L1000_dx0.25_T2000_seed0.0001_lim-vanleer")]

COL = {2.00: viridis(0.86), 1.00: viridis(0.62), 0.50: viridis(0.36), 0.25: viridis(0.06)}
LIMC = "#c0392b"


def f1_at_match(d):
    """Global strain-1 fraction at t = T_MATCH, from timeseries.csv."""
    rows = list(csv.DictReader(open(os.path.join(HERE, d, "timeseries.csv"))))
    t = np.array([float(q["t_h"]) for q in rows])
    f = np.array([float(q["F1"]) for q in rows])
    return float(f[int(np.argmin(np.abs(t - T_MATCH)))])


def profile_at_match(d):
    """(x, f1, h) at t = T_MATCH, streamed out of full_dump.csv by row prefix."""
    path = os.path.join(HERE, d, "full_dump.csv")
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split(",")
        ix, if1, ih = head.index("x_um"), head.index("f1"), head.index("h")
        pre = "%g," % T_MATCH
        x, f, h = [], [], []
        for line in fh:
            if line.startswith(pre):
                p = line.split(",")
                x.append(float(p[ix])); f.append(float(p[if1])); h.append(float(p[ih]))
    if not x:
        raise SystemExit("no t = %g frame in %s" % (T_MATCH, path))
    return np.array(x), np.array(f), np.array(h)


def richardson(fine, coarse, p, r=2.0):
    """phi_exact ~ phi_fine + (phi_fine - phi_coarse) / (r**p - 1)."""
    return fine + (fine - coarse) / (r ** p - 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "grid_convergence_T2000.png"),
                    help="PNG path; <stem>_F1/_fit/_profiles.csv are written beside it")
    args = ap.parse_args()
    F1 = {dx: f1_at_match(d) for dx, d in RUNS}
    P = {dx: profile_at_match(d) for dx, d in RUNS}

    # --- orders from successive differences -------------------------------------------------
    d_coarse = F1[2.00] - F1[1.00]
    d_mid = F1[1.00] - F1[0.50]
    d_fine = F1[0.50] - F1[0.25]
    p_coarse = math.log2(d_coarse / d_mid)          # uses the inadmissible dx = 2
    p_fine = math.log2(d_mid / d_fine)              # admissible triplet
    lim_fine = richardson(F1[0.25], F1[0.50], p_fine)
    lim_coarse = richardson(F1[0.50], F1[1.00], p_coarse)

    # --- pointwise profile limit on the finest grid ------------------------------------------
    xf, ff, hf = P[0.25]
    xc, fc, hc = P[0.50]
    f_lim = richardson(ff, np.interp(xf, xc, fc), p_fine)
    h_lim = 2.0 * f_lim - 1.0

    # front = where the finest-grid composition crosses f = 1/2, for the inset windows
    k = int(np.argmin(np.abs(ff - 0.5)))
    x0, x1 = max(0.0, xf[k] - 90.0), min(xf[-1], xf[k] + 90.0)

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 9.2))
    dxs = [dx for dx, _ in RUNS]

    # ============================ (a) F1 vs dx ==============================================
    a = ax[0, 0]
    a.axhline(lim_fine, color=LIMC, ls="--", lw=1.6, zorder=1,
              label=r"Richardson limit %.5f  ($p$=%.2f, $\Delta x$=1/0.5/0.25)" % (lim_fine, p_fine))
    a.axhline(lim_coarse, color="0.55", ls=":", lw=1.5, zorder=1,
              label=r"limit %.5f from $\Delta x$=2/1/0.5 ($p$=%.2f) -- overshot" % (lim_coarse, p_coarse))
    a.plot(dxs, [F1[d] for d in dxs], "-", color="0.7", lw=1.0, zorder=2)
    for dx in dxs:
        # dx = 2 is drawn hollow: it is plotted but excluded from the fit
        kw = (dict(mfc="white", mec=COL[dx], mew=2.0) if dx == 2.00
              else dict(color=COL[dx], mec="white", mew=1.2))
        a.plot([dx], [F1[dx]], "o", ms=9, zorder=3, **kw)
        a.annotate("%+.2f%%" % (100 * (F1[dx] - lim_fine) / lim_fine),
                   (dx, F1[dx]), textcoords="offset points", xytext=(9, -3),
                   fontsize=8.5, color=COL[dx])
    a.plot([], [], "o", ms=8, mfc="white", mec="0.4", mew=2.0,
           ls="none", label="open = excluded (unresolved front)")
    a.set_xscale("log", base=2)
    a.set_xticks(dxs); a.set_xticklabels(["2", "1", "0.5", "0.25"])
    a.invert_xaxis()
    a.set_xlabel(r"$\Delta x$ [$\mu$m]  (refining $\rightarrow$)")
    a.set_ylabel(r"$F_1$ at $t=2000$ h")
    a.set_title("(a)  global composition converges first order", fontsize=10, loc="left")
    a.legend(fontsize=7.6, loc="upper right")
    a.grid(alpha=0.25)

    # ============================ (b) error, log-log ========================================
    b = ax[0, 1]
    err = np.array([abs(F1[d] - lim_fine) for d in dxs])
    for dx, e in zip(dxs, err):
        kw = (dict(mfc="white", mec=COL[dx], mew=2.0) if dx == 2.00
              else dict(color=COL[dx], mec="white", mew=1.2))
        b.plot([dx], [e], "o", ms=9, zorder=4, **kw)
    ref = np.array([2.0, 0.25])
    b.plot(ref, err[-1] * (ref / 0.25) ** 1.0, "-", color="0.55", lw=1.2,
           label=r"slope 1 (first order)")
    b.plot(ref, err[-1] * (ref / 0.25) ** 2.0, ":", color="0.55", lw=1.2,
           label=r"slope 2 (second order)")
    b.set_xscale("log", base=2); b.set_yscale("log")
    b.set_xticks(dxs); b.set_xticklabels(["2", "1", "0.5", "0.25"])
    b.invert_xaxis()
    b.set_xlabel(r"$\Delta x$ [$\mu$m]  (refining $\rightarrow$)")
    b.set_ylabel(r"$|F_1-F_1^{\,\rm lim}|$")
    b.set_title("(b)  the error halves per refinement, it does not quarter",
                fontsize=10, loc="left")
    b.legend(fontsize=8, loc="lower right")
    b.grid(alpha=0.25, which="both")

    # ============================ (c) f(x) and (d) h(x) =====================================
    for pane, key, lim, lab in ((ax[1, 0], 1, f_lim, r"$f_1(x)$"),
                                (ax[1, 1], 2, h_lim, r"$h(x)=2f_1-1$")):
        for dx, _ in RUNS:
            pane.plot(P[dx][0], P[dx][key], lw=1.5, color=COL[dx],
                      label=r"$\Delta x=%g$" % dx)
        pane.plot(xf, lim, "--", color=LIMC, lw=1.7, label="Richardson limit")
        pane.set_xlabel(r"$x$ [$\mu$m]")
        pane.set_ylabel(lab)
        pane.grid(alpha=0.25)
        if key == 2:
            pane.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        pane.legend(fontsize=8, loc="center left" if key == 1 else "lower left")

        # both insets go top-right, the one region empty in each panel
        ins = pane.inset_axes([0.52, 0.52, 0.45, 0.44] if key == 1
                              else [0.52, 0.54, 0.45, 0.42])
        for dx, _ in RUNS:
            ins.plot(P[dx][0], P[dx][key], lw=1.4, color=COL[dx])
        ins.plot(xf, lim, "--", color=LIMC, lw=1.5)
        ins.set_xlim(x0, x1)
        sel = (xf >= x0) & (xf <= x1)
        lo, hi = float(min(lim[sel].min(), P[2.00][key].min())), float(lim[sel].max())
        pad = 0.06 * (hi - lo + 1e-12)
        ins.set_ylim(lo - pad, hi + pad)
        ins.tick_params(labelsize=7)
        ins.set_title("front, %d-%d $\\mu$m" % (x0, x1), fontsize=7.5)
        ins.grid(alpha=0.2)

    ax[1, 0].set_title("(c)  composition profile", fontsize=10, loc="left")
    ax[1, 1].set_title("(d)  dominance variable", fontsize=10, loc="left")

    fig.suptitle("Grid convergence, jammed $n_0=1000\\ \\mu$M, $L=1000\\ \\mu$m, van Leer limiter, "
                 r"$\rho_{\rm seed}=10^{-4}$, $\Delta t=0.05$ h, all at $t=2000$ h"
                 "\n"
                 r"$\Delta x=2$ is outside the validated front resolution "
                 r"($\Delta x/\lambda=0.70>0.6$) and is excluded from the fit",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.out
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print("wrote %s" % out)
    # Every plotted array to CSV beside the PNG (rule: plotted arrays are always saved).
    stem = os.path.splitext(out)[0]
    with open(stem + "_F1.csv", "w", newline="") as fh:            # panels (a), (b)
        w = csv.writer(fh)
        w.writerow(["dx_um", "F1_at_t2000", "abs_err_vs_lim", "rel_err_vs_lim_pct", "in_fit"])
        for dx in dxs:
            w.writerow(["%g" % dx, "%.10g" % F1[dx], "%.6e" % abs(F1[dx] - lim_fine),
                        "%.6g" % (100 * (F1[dx] - lim_fine) / lim_fine), int(dx != 2.00)])
    with open(stem + "_fit.csv", "w", newline="") as fh:           # lines and inset window
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in (("T_match_h", T_MATCH),
                     ("p_fine_dx_1_0.5_0.25", p_fine), ("F1_limit_fine", lim_fine),
                     ("p_coarse_dx_2_1_0.5", p_coarse), ("F1_limit_coarse", lim_coarse),
                     ("inset_x0_um", x0), ("inset_x1_um", x1)):
            w.writerow([k, "%.10g" % v])
    with open(stem + "_profiles.csv", "w", newline="") as fh:      # panels (c), (d) + insets
        w = csv.writer(fh)
        w.writerow(["series", "dx_um", "x_um", "f1", "h"])
        for dx, _ in RUNS:
            xs, fs, hs = P[dx]
            for i in range(xs.size):
                w.writerow(["dx=%g" % dx, "%g" % dx, "%.6g" % xs[i], "%.10g" % fs[i],
                            "%.10g" % hs[i]])
        for i in range(xf.size):                                   # on the dx = 0.25 grid
            w.writerow(["richardson_limit", "", "%.6g" % xf[i], "%.10g" % f_lim[i],
                        "%.10g" % h_lim[i]])
    print("wrote %s_{F1,fit,profiles}.csv\n" % stem)
    print("%-7s %12s %12s %10s" % ("dx", "F1(t=2000)", "difference", "err vs lim"))
    prev = None
    for dx in dxs:
        d = "" if prev is None else "%12.6f" % (prev - F1[dx])
        print("%-7g %12.7f %12s %9.2f%%"
              % (dx, F1[dx], d, 100 * (F1[dx] - lim_fine) / lim_fine))
        prev = F1[dx]
    print("\norder from dx=2,1,0.5    : p = %.3f  -> limit %.6f  (dx=2 inadmissible)"
          % (p_coarse, lim_coarse))
    print("order from dx=1,0.5,0.25 : p = %.3f  -> limit %.6f  <-- quote this"
          % (p_fine, lim_fine))
    print("measured dx=0.25 (%.6f) is BELOW the coarse-triplet limit (%.6f) by %.1e"
          % (F1[0.25], lim_coarse, lim_coarse - F1[0.25]))
    mx = float(np.abs(P[0.50][2] - np.interp(P[0.50][0], xf, h_lim)).max())
    print("\nmax |h - h_lim| at dx=0.5 : %.4f     at dx=0.25 : %.4f"
          % (mx, float(np.abs(hf - h_lim).max())))


if __name__ == "__main__":
    main()
