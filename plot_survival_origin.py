"""
plot_survival_origin.py -- where the SURVIVING biomass was produced.  Simulates nothing.

    py -3.11 src2/plot_survival_origin.py survival_n0-15_L1000_dx0.5_T500 \
                                          survival_n0-1000_L1000_dx0.5_T2000
    py -3.11 src2/plot_survival_origin.py <dirs...> --out src2/survival_origin.png

Reads the CSVs written by `survival_origin.py`, one row of panels per run:

  (a) origin PROFILES        where biomass was made, and where the survivors were made,
                             against where the biomass now sits -- each normalised to its own
                             integral, so the three shapes are comparable despite totals that
                             differ by more than an order of magnitude.
  (b) CUMULATIVE fractions   the same three as running integrals, with the 90% points marked.
                             This is the panel the thesis number comes from.
  (c) sigma(y)               the fraction of biomass created at y that is still in the cavity at
                             T.  This is the mechanism behind the gap between the first two
                             curves in (b), not a separate result.
  (d) survivors by strain    where each strain's surviving biomass was made.

LOG X THROUGHOUT.  Production lives in the first ~100 um of a 1000 um cavity while the standing
stock is spread over the whole of it; on a linear axis the production curves collapse onto the
axis and the comparison that motivates the figure is invisible.

WHY (a) AND (d) ARE NORMALISED BUT (c) IS NOT.  (a) and (d) compare SHAPES of distributions whose
totals differ by design -- 88-97% of production washes out, so "survived" is a small fraction of
"produced" and plotting raw would flatten it.  sigma in (c) is already a fraction in [0,1] and
normalising it would destroy its meaning.

n* FRONT.  Taken from the matching `run_<same suffix>/summary.csv` (`x_nstar_final_um`) when that
directory exists; drawn as a vertical line, omitted silently otherwise.  It is context, not data
from this computation.
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

SURFACE, INK, INK2, MUTED, GRIDC = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"
C_MADE, C_SURV, C_STOCK = "#7a7975", "#2a78d6", "#c1392b"
C_S1, C_S2 = "#2a78d6", "#c1392b"


def load(d):
    path = os.path.join(d, "survival_origin.csv")
    if not os.path.exists(path):
        sys.exit(f"no survival_origin.csv in {d}; run survival_origin.py first")
    R = list(csv.DictReader(open(path)))
    out = {k: np.array([float(r[k]) for r in R]) for k in R[0]}
    return out


def nstar_x(d):
    """x of the n* front at T, from the matching run directory if there is one."""
    base = os.path.basename(os.path.normpath(d))
    if not base.startswith("survival_"):
        return None
    cand = os.path.join(os.path.dirname(os.path.abspath(d)),
                        "run_" + base[len("survival_"):])
    s = os.path.join(cand, "summary.csv")
    if not os.path.exists(s):
        return None
    for row in csv.reader(open(s)):
        if row and row[0] == "x_nstar_final_um":
            try:
                return float(row[1])
            except ValueError:
                return None
    return None


def qx(w, x, q):
    tot = float(np.sum(w))
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    return float(np.interp(q, np.cumsum(w) / tot, x))


def norm(w):
    s = float(np.sum(w))
    return w / s if s > 0 else w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "survival_origin.png"))
    ap.add_argument("--linear-x", dest="log_x", action="store_false")
    a = ap.parse_args()

    runs = []
    for d in a.dirs:
        p = d if os.path.isabs(d) else os.path.join(HERE, d)
        runs.append((os.path.basename(os.path.normpath(p)), load(p), nstar_x(p)))

    nr = len(runs)
    fig, ax = plt.subplots(nr, 4, figsize=(21.0, 4.7 * nr), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    print(f"{'run':>36}{'':>4}{'x50':>8}{'x90':>8}{'x99':>8}   quantity")
    for k, (name, z, xns) in enumerate(runs):
        x = z["x_um"]
        made, surv, stock = z["produced_total"], z["survived_total"], z["rho_final"]
        sig = z["sigma_first_step"]
        frac = float(surv.sum() / max(made.sum(), 1e-300))

        for lab, w in (("produced", made), ("SURVIVED", surv), ("now sits", stock)):
            print(f"{name if lab == 'produced' else '':>36}{'':>4}"
                  f"{qx(w, x, .5):>8.1f}{qx(w, x, .9):>8.1f}{qx(w, x, .99):>8.1f}   {lab}")

        # ---- (a) profiles ----------------------------------------------------------------
        b = ax[k][0]
        for w, c, lab in ((made, C_MADE, "produced here (all)"),
                          (surv, C_SURV, "produced here, still present at $T$"),
                          (stock, C_STOCK,
                           r"standing $\rho(x,T)$ -- CURRENT position, not origin")):
            b.plot(x, norm(w), color=c, lw=1.9, label=lab)
        b.set_yscale("log"); b.set_ylim(1e-8, None)
        b.set_ylabel("normalised profile", color=INK2)
        b.set_title(f"(a) {name}\n     origin profiles   (survival {100*frac:.1f}%)",
                    fontsize=10, fontweight="bold", color=INK, loc="left")

        # ---- (b) cumulative --------------------------------------------------------------
        c_ = ax[k][1]
        for w, col, lab in ((made, C_MADE, "produced here (all)"),
                            (surv, C_SURV, "produced here, still present at $T$"),
                            (stock, C_STOCK, "standing stock (CURRENT position)")):
            c_.plot(x, np.cumsum(w) / max(float(w.sum()), 1e-300), color=col, lw=1.9, label=lab)
        c_.axhline(0.9, color=GRIDC, lw=0.9, ls="--")
        # the two 90% points differ by only 5-12 um, so stagger the labels rather than let
        # them overprint each other
        # the two 90% points are 5-12 um apart and the curve passes right between them, so put
        # one label left-and-above and the other right-and-below rather than stacking them
        for (w, col), off, ha in (((made, C_MADE), (-7, 9), "right"),
                                  ((surv, C_SURV), (7, -17), "left")):
            xv = qx(w, x, 0.9)
            c_.plot([xv], [0.9], "o", color=col, ms=6, zorder=4)
            c_.annotate(f"{xv:.0f} " + r"$\mu$m", (xv, 0.9), textcoords="offset points",
                        xytext=off, ha=ha, fontsize=9.5, color=col, fontweight="bold")
        c_.set_ylim(0, 1.04)
        c_.set_ylabel("cumulative fraction", color=INK2)
        c_.set_title("(b) 90% of it was made shallower than...", fontsize=10,
                     fontweight="bold", color=INK, loc="left")

        # ---- (c) survival by production site ---------------------------------------------
        # NOT sigma at m=0.  T = diag(rho_new) P diag(1/rho_r) is undefined where rho_r = 0, so
        # apply_TT returns 0 there; at m=0 the cavity is empty beyond the seed and sigma^0 is
        # therefore 0 over most of the domain -- which would read as "nothing made deep ever
        # survives".  That is a convention, not physics.  It never touches the result, because
        # c_i^m is also 0 wherever there is no mass, but it must not be plotted as if it were a
        # survival probability.  What IS meaningful is the production-weighted ratio
        #     A(y)/Y(y) = sum_m c^m(y) sigma^m(y) / sum_m c^m(y),
        # i.e. of the biomass actually made at y, the fraction still present at T.
        s_ = ax[k][2]
        ok = made > 1e-12 * made.max()
        ratio = np.where(ok, surv / np.where(ok, made, 1.0), np.nan)
        s_.plot(x[ok], ratio[ok], color=INK, lw=1.9)
        s_.axhline(frac, color=C_SURV, lw=1.1, ls=":", label=f"run-average {frac:.3f}")
        s_.set_ylim(0, 1.02)
        s_.set_ylabel("fraction still present at $T$", color=INK2)
        s_.set_title(r"(c) of biomass made at $y$, how much survives", fontsize=10,
                     fontweight="bold", color=INK, loc="left")

        # ---- (d) by strain ---------------------------------------------------------------
        d_ = ax[k][3]
        for w, col, lab in ((z["survived_s1"], C_S1, "strain 1 (scavenger)"),
                            (z["survived_s2"], C_S2, "strain 2 (glutton)")):
            d_.plot(x, norm(w), color=col, lw=1.9, label=lab)
            q10, q90 = qx(w, x, 0.1), qx(w, x, 0.9)
            d_.axvspan(q10, q90, color=col, alpha=.10, lw=0)
            print(f"{'':>36}{'':>4}{qx(w, x, .5):>8.1f}{q90:>8.1f}{qx(w, x, .99):>8.1f}"
                  f"   survivors {lab.split()[0]} {lab.split()[1]}  [10-90%: {q10:.0f}-{q90:.0f}]")
        d_.set_yscale("log"); d_.set_ylim(1e-8, None)
        d_.set_ylabel("normalised profile", color=INK2)
        d_.set_title("(d) survivors by strain  (bands = 10-90%)", fontsize=10,
                     fontweight="bold", color=INK, loc="left")

        for b_ in ax[k]:
            if a.log_x:
                b_.set_xscale("log")
                b_.set_xlim(max(x[0], 1.0), x[-1])
            # The grey and blue curves are indexed by ORIGIN (where biomass was made); the red
            # one by CURRENT position (where it now sits).  Those are two different meanings of
            # x, so the axis gets a neutral label and the distinction lives in the legend.
            b_.set_xlabel(r"$x$ [$\mu$m]", color=INK2)
            if xns is not None:
                b_.axvline(xns, color="0.45", lw=1.0, ls="-.")
            b_.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="best")
            b_.grid(alpha=.22, lw=.6)
            b_.tick_params(colors=MUTED, labelsize=9)
            for sp in ("top", "right"):
                b_.spines[sp].set_visible(False)
        if xns is not None:
            ax[k][0].annotate(r"$n^\ast$ front at $T$", (xns, 1), xycoords=("data", "axes fraction"),
                              textcoords="offset points", xytext=(3, -11), fontsize=8,
                              color="0.45", rotation=90, va="top")

    fig.suptitle("Where the surviving biomass was produced   "
                 "(exact split of the growth chain into create + pure transport)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.get_layout_engine().set(rect=(0, 0.028, 1, 0.955))
    fig.text(0.5, 0.006,
             r"Blue in (a),(b) is $\sum_m c_i^m\sigma^m$: biomass created at $y$ that is still in "
             r"the cavity at $T$.  It sits DEEPER than production as a whole because loss is at "
             r"the mouth, where production is concentrated.  Red is standing stock -- where that "
             r"biomass is now, not where it came from.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(a.out, dpi=165, facecolor=SURFACE)
    print(f"\nwrote {a.out}")
    for path in write_csvs(a.out, runs):
        print(f"wrote {path}")


def write_csvs(out, runs):
    """Every array the figure draws, beside the PNG (rule: plotted arrays are always saved).

    <stem>.csv          one block per run: the normalised profiles of (a) and (d), the
                        cumulative fractions of (b) and the survival ratio of (c) -- 'nan'
                        exactly where (c) masks it
    <stem>_summary.csv  the marked points: run-average survival, 90% points, 10-90% bands, n*
    """
    stem = os.path.splitext(out)[0]
    cum = lambda w: np.cumsum(w) / max(float(w.sum()), 1e-300)
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "x_um", "produced_norm", "survived_norm", "stock_norm",
                    "produced_cum", "survived_cum", "stock_cum", "survival_ratio",
                    "survived_s1_norm", "survived_s2_norm"])
        for name, z, _ in runs:
            x = z["x_um"]
            made, surv, stock = z["produced_total"], z["survived_total"], z["rho_final"]
            ok = made > 1e-12 * made.max()
            ratio = np.where(ok, surv / np.where(ok, made, 1.0), np.nan)
            cols = (norm(made), norm(surv), norm(stock), cum(made), cum(surv), cum(stock), ratio,
                    norm(z["survived_s1"]), norm(z["survived_s2"]))
            for i in range(x.size):
                w.writerow([name, f"{x[i]:.6g}"] + [f"{c[i]:.10g}" for c in cols])
    with open(stem + "_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "survival_fraction", "x90_produced_um", "x90_survived_um",
                    "s1_x10_um", "s1_x90_um", "s2_x10_um", "s2_x90_um", "x_nstar_um"])
        for name, z, xns in runs:
            x = z["x_um"]
            frac = float(z["survived_total"].sum() / max(z["produced_total"].sum(), 1e-300))
            vals = [frac, qx(z["produced_total"], x, 0.9), qx(z["survived_total"], x, 0.9),
                    qx(z["survived_s1"], x, 0.1), qx(z["survived_s1"], x, 0.9),
                    qx(z["survived_s2"], x, 0.1), qx(z["survived_s2"], x, 0.9)]
            w.writerow([name] + [f"{v:.8g}" for v in vals] + ["" if xns is None else f"{xns:g}"])
    return [stem + ".csv", stem + "_summary.csv"]


if __name__ == "__main__":
    main()
