"""
plot_f1_extrapolation.py -- when does the high-affinity strain take over completely?

Read-only: fits the LATE-TIME trend of F1(t) from a run's timeseries.csv and extrapolates it.

Three kinetics are fitted, because the honest result is that the simulated window cannot tell
them apart:

  linear         F1 = a + b t                     -- the naive read; F1 is BOUNDED by 1, so a
                                                     straight line crossing it is meaningless
                                                     past the crossing.  Lower bound only.
  logistic       dF1/dt = r F1 (1-F1)             -- the replicator / competitive-exclusion form.
  exp-decay      F1 = 1 - A exp(-k t)             -- constant per-capita loss of the loser.

All three reach R^2 > 0.999 on the observed window and then disagree by ~4x on the answer, so
the extrapolation is MODEL-limited, not data-limited.  The figure is built to show that: panel
B plots 1-F1 on log axes, where the three separate visibly, and shades everything past the last
datum so no one reads the projection as measurement.

F1 = 1 exactly is attained only by the linear model, as an artifact of extrapolating a straight
line past a bounded quantity.  The bounded models approach 1 asymptotically, so a threshold
(0.9 / 0.99 / 0.999) is required for the question to have an answer at all.

Usage:
    py -3.11 src2/plot_f1_extrapolation.py --run src2/run_n0-1000_dx0.5_T50000
    py -3.11 src2/plot_f1_extrapolation.py --run <dir> --fit-from 35000 --thresh 0.99
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Categorical slots 1-3 of the documented default palette.  Validated (light, surface #fcfcfb):
# lightness PASS, chroma PASS, CVD worst adjacent dE 9.2 deutan PASS, normal-vision 27.6 PASS,
# contrast WARN on the green (2.74:1) -> relieved by the direct labels + the printed table.
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED = "#1a1a19", "#4a4a47", "#8a8a85"
SURFACE = "#fcfcfb"
HOURS_PER_YEAR = 8766.0


def load(run):
    r = list(csv.DictReader(open(os.path.join(run, "timeseries.csv"))))
    g = lambda k: np.array([float(v[k]) if v[k] not in ("", None) else np.nan for v in r])
    return g("t_h"), g("F1")


# --- the three kinetics -------------------------------------------------------
def _logistic(x, r, t0):
    return 1.0 / (1.0 + np.exp(-r * (x - t0)))


def _expdecay(x, A, k):
    return 1.0 - A * np.exp(-k * x)


def fit_models(t, F1, t_from):
    m = t >= t_from
    T, F = t[m], F1[m]
    models = {}

    b, a = np.polyfit(T, F, 1)
    models["linear"] = (lambda x, a=a, b=b: a + b * x, f"$b$={b:.3g} h$^{{-1}}$")

    p, _ = curve_fit(_logistic, T, F, p0=[4.5e-6, 1.5e5], maxfev=400000)
    models["logistic"] = (lambda x, p=p: _logistic(x, *p), f"$r$={p[0]:.3g} h$^{{-1}}$")

    q, _ = curve_fit(_expdecay, T, F, p0=[0.7, 1e-6], maxfev=400000)
    models["exp-decay"] = (lambda x, q=q: _expdecay(x, *q), f"$k$={q[1]:.3g} h$^{{-1}}$")

    out = {}
    for name, (f, lab) in models.items():
        pred = f(T)
        r2 = 1 - ((F - pred) ** 2).sum() / ((F - F.mean()) ** 2).sum()
        out[name] = (f, lab, r2)
    return out, T


def crossing(f, thresh, lo, hi=1e11):
    """First t where f(t) >= thresh, by bisection.  None if never reached."""
    if f(hi) < thresh:
        return None
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) < thresh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fmt_t(v):
    if v is None:
        return "never"
    return f"{v:.3g} h ({v / HOURS_PER_YEAR:.3g} yr)"


def figure(run, out_path, t_from, thresholds=(0.9, 0.99, 0.999), headline=0.99):
    t, F1 = load(run)
    models, T = fit_models(t, F1, t_from)
    t_end = float(t[-1])

    tab = {n: {th: crossing(f, th, t_end) for th in thresholds} for n, (f, _, _) in models.items()}
    hi_t = max([v for d in tab.values() for v in d.values() if v is not None] + [t_end * 10])
    xs = np.logspace(np.log10(max(t[t > 0].min(), 1.0)), np.log10(hi_t * 1.6), 1200)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=SURFACE)
    fig.suptitle("When does the high-affinity strain take over?   "
                 f"$n_0$=1000 $\\mu$M, $\\Delta x$=0.5 $\\mu$m   |   "
                 f"simulated to {t_end:.0f} h, fitted on $t\\geq${t_from:.0f} h",
                 fontsize=11.5, fontweight="bold", color=INK)

    for a in ax:
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
        a.tick_params(colors=INK2, labelsize=9)
        a.grid(alpha=0.18, which="both", color=MUTED, lw=0.6)
        a.set_axisbelow(True)
        a.axvspan(t_end, xs[-1], color=MUTED, alpha=0.10, lw=0)     # projection is not data

    # ---- (A) F1 -------------------------------------------------------------
    A = ax[0]
    for (name, (f, lab, r2)), col in zip(models.items(), CAT):
        A.plot(xs, np.clip(f(xs), 0, 1.02), color=col, lw=2, ls="--", zorder=2)
    A.plot(t, F1, color=INK, lw=2.6, zorder=4)
    A.axhline(headline, color=MUTED, lw=1.0, ls=":")
    A.text(xs[0] * 1.3, headline, f" $F_1$={headline:g}", va="bottom", ha="left",
           fontsize=8.5, color=INK2)
    A.set_xscale("log"); A.set_xlim(xs[0], xs[-1]); A.set_ylim(0, 1.05)
    A.set_xlabel("t [h]", color=INK2); A.set_ylabel("$F_1$   (strain-1 mass fraction)", color=INK2)
    A.set_title("(A) the projection", fontsize=9.5, fontweight="bold", color=INK, loc="left")
    A.text(t_end * 1.25, 0.045, "extrapolated\n(not simulated)", fontsize=8, color=MUTED, ha="left")
    A.annotate(f"last datum\n$F_1$={F1[-1]:.4f}", (t_end, F1[-1]), textcoords="offset points",
               xytext=(-8, 16), ha="right", fontsize=8.5, color=INK)
    A.plot([t_end], [F1[-1]], "o", color=INK, ms=6, zorder=5,
           markeredgecolor=SURFACE, markeredgewidth=2)

    # Direct labels: text in ink, a colored dash beside it carries identity.
    # Anchor each at a DIFFERENT height on its own curve -- at a common height the three
    # crossing times are within a factor ~2 and the labels land on top of each other.
    for (name, (f, lab, r2)), col, yl in zip(models.items(), CAT, (0.80, 0.60, 0.40)):
        xt = crossing(f, yl, t_end) or hi_t
        A.plot([xt * 0.55, xt * 0.80], [yl, yl], color=col, lw=2.4, ls="--",
               solid_capstyle="butt", zorder=6)
        A.text(xt * 0.88, yl, name, fontsize=8.5, color=INK, va="center", ha="left", zorder=6,
               bbox=dict(fc=SURFACE, ec="none", pad=1.2, alpha=0.85))
    # the linear model runs off the top: say so rather than let the clip look like a plateau
    xc1 = crossing(models["linear"][0], 1.0, t_end)
    if xc1 is not None and xc1 < xs[-1]:
        A.annotate("linear crosses $F_1$=1\n(unphysical: $F_1$ is bounded)", (xc1, 1.0),
                   textcoords="offset points", xytext=(10, -26), fontsize=7.8, color=CAT[0],
                   arrowprops=dict(arrowstyle="-", color=CAT[0], lw=0.8))

    # ---- (B) 1 - F1, where the models actually separate ---------------------
    B = ax[1]
    for (name, (f, lab, r2)), col in zip(models.items(), CAT):
        B.plot(xs, np.maximum(1 - f(xs), 1e-6), color=col, lw=2, ls="--",
               label=f"{name}   {lab}   $R^2$={r2:.4f}", zorder=2)
    B.plot(t, 1 - F1, color=INK, lw=2.6, zorder=4, label=f"simulated (to {t_end:.0f} h)")
    B.axhline(1 - headline, color=MUTED, lw=1.0, ls=":")
    B.text(xs[0] * 1.3, 1 - headline, f" $1-F_1$={1 - headline:g}", va="bottom", ha="left",
           fontsize=8.5, color=INK2)
    B.set_xscale("log"); B.set_yscale("log")
    B.set_xlim(xs[0], xs[-1]); B.set_ylim(1e-4, 1.4)
    B.set_xlabel("t [h]", color=INK2); B.set_ylabel("$1-F_1$   (remaining strain-2 share)", color=INK2)
    B.set_title("(B) why the answer is model-limited, not data-limited",
                fontsize=9.5, fontweight="bold", color=INK, loc="left")
    leg = B.legend(fontsize=8, loc="lower left", framealpha=0.95, edgecolor=MUTED)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.text(0.5, 0.005,
             "All three fit the simulated window to $R^2>0.999$ and then disagree by ~4x. "
             "$F_1=1$ exactly is reached only by the linear model, which is an artifact of "
             "extrapolating a straight line past a bounded quantity.",
             ha="center", fontsize=8.2, color=MUTED)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---- the table view (also the contrast relief for the green slot) -------
    print(f"\n  simulated to t={t_end:.0f} h, F1={F1[-1]:.6f};  fits on t >= {t_from:.0f} h\n")
    hdr = f"  {'model':<12}{'R^2':>9}   {'parameter':<22}" + "".join(f"{'F1=' + str(th):>26}" for th in thresholds)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, (f, lab, r2) in models.items():
        cells = "".join(f"{fmt_t(tab[name][th]):>26}" for th in thresholds)
        print(f"  {name:<12}{r2:>9.5f}   {lab.replace('$', '').replace('^{-1}', '/h'):<22}{cells}")
    vals = [tab[n][headline] for n in tab if tab[n][headline] is not None]
    if vals:
        print(f"\n  F1={headline:g}:  {min(vals):.3g} h .. {max(vals):.3g} h "
              f"({min(vals)/HOURS_PER_YEAR:.3g} .. {max(vals)/HOURS_PER_YEAR:.3g} yr)"
              f"  -- spread {max(vals)/min(vals):.1f}x across models")
    print(f"\n  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fit-from", type=float, default=35000.0,
                    help="fit the late trend from this time [h] (default 35000)")
    ap.add_argument("--thresh", type=float, default=0.99,
                    help="headline threshold drawn on the figure (default 0.99)")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    figure(run, a.out or os.path.join(run, "f1_extrapolation.png"), a.fit_from, headline=a.thresh)


if __name__ == "__main__":
    main()
