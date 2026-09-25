"""
plot_fst_vs_L.py -- peak F_ST against cavity length, dilute vs jammed.  Simulates nothing.

Reads the eight runs of the completed length x density grid and plots the two series as points.
F_ST per frame comes from the run's own timeseries.csv where present (written by
`diagnostics.fst_series` during the run) and is otherwise recomputed from full_dump.csv through
the same function via plot_fst's streaming loader -- verified identical on the two runs that
have both, so the numbers match every other F_ST in the project rather than merely resembling
them.

THE PLOTTED VALUE IS A ROLLING-MEDIAN PEAK, not the raw maximum -- see MEDIAN_FRAMES.

    py -3.11 src2/plot_fst_vs_L.py
    py -3.11 src2/plot_fst_vs_L.py --out somewhere.png

AXES ARE LOG-LOG, and both for a reason rather than by habit:
  * y spans 0.0012 to 0.90, nearly three decades.  On a linear axis every dilute point lies on
    zero and the dilute series is invisible.
  * x is one decade with three of four points below 300, which a linear axis crowds to the left.

T DIFFERS BETWEEN RUNS (500/12000 dilute, 3000/6000 jammed) because the regimes evolve on
different timescales, and because the two L=1000 points reuse the long main runs.  Only the PEAK
is plotted, and every peak sits well inside its own run (the latest, jammed L=1000, is 1350 h of
6000), so the comparison is sound -- but the *final* values are not comparable across this grid
and are deliberately not shown.  dx = 0.5 um throughout, so resolution is matched.
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
import diagnostics as D                              # noqa: E402
import plot_fst as P                                 # noqa: E402  (loader only)

# (L, directory) per regime; dx = 0.5 for all eight.
# Repointed 2026-09-22 at the PRODUCTION config (rho_seed=1e-4, van Leer); the old entries were
# rho_seed=1e-6 donor-cell runs.  The two L=1000 points reuse the long main runs rather than
# dedicated sweeps: the metric is a PEAK and both peaks sit far inside those runs (jammed 1375 h
# of 6000, dilute 150 h of 12000), so the 9 h T=10000 jammed sweep is not needed.
GRID = {
    "dilute": (15.0, [
        (100,  "run_n0-15_L100_dx0.5_T500_seed0.0001_lim-vanleer"),
        (200,  "run_n0-15_L200_dx0.5_T500_seed0.0001_lim-vanleer"),
        (300,  "run_n0-15_L300_dx0.5_T500_seed0.0001_lim-vanleer"),
        (1000, "run_n0-15_L1000_dx0.5_T12000_seed0.0001_lim-vanleer"),
    ]),
    "jammed": (1000.0, [
        (100,  "run_n0-1000_L100_dx0.5_T3000_seed0.0001_lim-vanleer"),
        (200,  "run_n0-1000_L200_dx0.5_T3000_seed0.0001_lim-vanleer"),
        (300,  "run_n0-1000_L300_dx0.5_T3000_seed0.0001_lim-vanleer"),
        (1000, "run_n0-1000_L1000_dx0.5_T6000_seed0.0001_lim-vanleer"),
    ]),
}

#: Frames in the rolling median applied before taking the peak.  F_ST is a VARIANCE RATIO and
#: carries frame-to-frame noise whose maximum the raw peak then samples.  Measured at jammed
#: L=200: mean |frame-to-frame| = 5.3% of the peak, while F1 and x_n* run smooth and monotone
#: through the same window -- so it is the metric oscillating, not the solution.  The bias is
#: L-dependent (raw vs median: 0.5% at L=100, 4.3% at L=200, 0.01% at L=1000), so it distorts
#: the SHAPE of the trend and not merely its level.  Set to 1 to recover the raw peak.
MEDIAN_FRAMES = 5


def series_from_timeseries(d):
    """(t, F_ST) from a run's timeseries.csv, or None.

    WHY THIS PATH EXISTS.  `plot_fst.load` needs full_dump.csv to rebuild the profiles and
    recompute F_ST.  The six short-cavity sweep runs were launched with --fst but not --dump
    (the dump is ~100 MB per run and nothing else needs it), so they have no trajectory and the
    loader skips them.  Their timeseries.csv, however, already carries F_ST per frame, written
    by the SAME `diagnostics.fst_series` during the run -- verified equal to the recomputed
    value on the two L=1000 runs that have both (0.9092 / 0.0797 either way).  So this is the
    same number by a cheaper route, not a substitute for it.
    """
    path = os.path.join(d, "timeseries.csv")
    if not os.path.exists(path):
        return None
    t, f = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if "F_ST" not in r or r["F_ST"] in ("", "nan"):
                continue
            t.append(float(r["t_h"])); f.append(float(r["F_ST"]))
    if not t:
        return None
    return np.asarray(t), np.asarray(f)


def smoothed_peak(fst, t, w=MEDIAN_FRAMES):
    """(peak, t_peak) of the w-frame rolling median, NaN frames dropped first."""
    ok = np.isfinite(np.asarray(fst, float))
    f, tt = np.asarray(fst, float)[ok], np.asarray(t, float)[ok]
    if f.size == 0:
        return float("nan"), float("nan")
    if w > 1:
        h = w // 2
        f = np.array([np.median(f[max(i - h, 0):i + h + 1]) for i in range(f.size)])
    i = int(np.argmax(f))
    return float(f[i]), float(tt[i])
# dataviz reference palette, categorical slots 1-2 (validated --pairs all, light)
COL = {"dilute": "#2a78d6", "jammed": "#eb6834"}
MRK = {"dilute": "o", "jammed": "s"}
LAB = {"dilute": r"dilute   $n_0=15\ \mu$M", "jammed": r"jammed   $n_0=1000\ \mu$M"}
SURFACE, INK, INK2, MUTED, GRID_C = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "fst_peak_vs_L.png"))
    a = ap.parse_args()

    data, table = {}, []
    print(f"{'regime':>8}{'L [um]':>9}{'F_ST med'+str(MEDIAN_FRAMES):>12}{'t_peak [h]':>12}"
          f"{'raw peak':>12}{'raw bias':>10}{'run':>52}")
    for reg, (n0, entries) in GRID.items():
        p = config.params(n0)
        Ls, pk = [], []
        for L, d in entries:
            path = os.path.join(HERE, d)
            ts = series_from_timeseries(path)
            if ts is not None:
                t_arr, fst = ts
            else:
                got = P.load(path)
                if got is None:
                    print(f"{reg:>8}{L:>9}   MISSING: {d}")
                    continue
                res, kind = got
                if kind != "trajectory":
                    print(f"{reg:>8}{L:>9}   no trajectory in {d} -- skipped")
                    continue
                fst, _, _, _ = D.fst_series(res, p)
                t_arr = res["t"]
            raw = float(np.nanmax(fst)) if np.any(np.isfinite(fst)) else float("nan")
            sm, t_sm = smoothed_peak(fst, t_arr)
            Ls.append(L); pk.append(sm)
            table.append((reg, L, sm, t_sm, raw, d))
            bias = 100.0 * (raw / sm - 1.0) if np.isfinite(sm) and sm > 0 else float("nan")
            print(f"{reg:>8}{L:>9}{sm:>12.4f}{t_sm:>12.0f}{raw:>12.4f}{bias:>9.2f}%  {d}")
        data[reg] = (np.array(Ls, float), np.array(pk, float))

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for reg in ("dilute", "jammed"):
        L, pk = data[reg]
        if L.size == 0:
            continue
        ax.plot(L, pk, MRK[reg], color=COL[reg], ms=10, mec=SURFACE, mew=1.8,
                ls="none", zorder=3, label=LAB[reg])
        # value beside every point; the rightmost sits at the axis edge so it labels leftwards
        for Li, v in zip(L, pk):
            right = Li >= L.max()
            ax.annotate(f"{v:.4f}", (Li, v), textcoords="offset points",
                        xytext=(-10 if right else 10, -3),
                        ha="right" if right else "left", va="center",
                        fontsize=9, color=COL[reg], zorder=5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([100, 200, 300, 1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"cavity length  $L$  [$\mu$m]", color=INK2, fontsize=11)
    ax.set_ylabel(r"peak  $F_{ST}$", color=INK2, fontsize=11)
    ax.set_title("Spatial structure against cavity length", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="lower right")
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID_C)
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.text(0.5, 0.015, r"peak over the whole run, $dx=0.5\ \mu$m throughout; "
                         "log-log because the dilute series spans three decades in $F_{ST}$",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(a.out, dpi=170, facecolor=SURFACE)
    # Every plotted point, to CSV beside the PNG (rule: plotted arrays are always saved).
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "L_um", "F_ST_peak_median%d" % MEDIAN_FRAMES, "t_peak_h",
                    "F_ST_peak_raw", "run"])
        for reg, L, sm, tp, raw, d in table:
            w.writerow([reg, L, "%.8g" % sm, "%.6g" % tp, "%.8g" % raw, d])
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
