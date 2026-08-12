"""
plot_front_lag.py -- the composition front vs the selection front, and the three phases.

Read-only: it consumes the CSVs a run has already written (`timeseries.csv`, `full_dump.csv`)
and simulates nothing.  The one thing it computes that no other script does is the COMPOSITION
FRONT x_comp(t) -- the deepest point where f1 = 0.5 -- which requires the full space-time dump.
Its distance behind the selection front (the n* crossing) is the lag that IS inverse sorting:

    lag(t) = x_comp(t) - x_nstar(t)

Because the dump pass is the expensive part (~100 MB), the extracted fronts are cached to
`front_lag.csv` in the run directory and reused on subsequent calls (--refresh to recompute).

Usage:
    py -3.11 src2/plot_front_lag.py --run src2/run_n0-1000_dx0.5_T10000
    py -3.11 src2/plot_front_lag.py --run <dir> --refresh --tmin 5
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

C_S1 = "#1f77b4"      # strain 1, scavenger
C_S2 = "#d62728"      # strain 2, glutton
C_SEL = "#2ca02c"     # selection front (n*)
C_LAG = "#9467bd"


# ---------------------------------------------------------------- data
def load_summary(run):
    d = dict(r for r in csv.reader(open(os.path.join(run, "summary.csv"))) if len(r) == 2)
    return d


def load_timeseries(run):
    rows = list(csv.DictReader(open(os.path.join(run, "timeseries.csv"))))
    out = {}
    for k in rows[0]:
        out[k] = np.array([float(v[k]) if v[k] not in ("", None) else np.nan for v in rows])
    return out


def extract_fronts(run, nstar, refresh=False):
    """Composition front (deepest f1=0.5), selection front (n* crossing), and the
    biomass-weighted <f1> inside the n>n* zone, per saved frame.  Cached."""
    cache = os.path.join(run, "front_lag.csv")
    if os.path.exists(cache) and not refresh:
        r = list(csv.DictReader(open(cache)))
        return {k: np.array([float(v[k]) if v[k] != "" else np.nan for v in r]) for k in r[0]}

    dump = os.path.join(run, "full_dump.csv")
    if not os.path.exists(dump):
        raise SystemExit(f"need {dump} (run figures.py with --dump) or an existing front_lag.csv")

    with open(dump) as fh:
        hdr = fh.readline().strip().split(",")
        need = ("t_h", "x_um", "f1", "n", "rho")
        miss = [k for k in need if k not in hdr]
        if miss:
            raise SystemExit(f"full_dump.csv lacks {miss} -- regenerate with the current figures.py")
        ix = {k: hdr.index(k) for k in need}
        T, XC, XN, FM = [], [], [], []
        cur, buf = None, []

        def flush(t, buf):
            a = np.asarray(buf)
            x, f, n, ro = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            j = np.where(f > 0.5)[0]
            hi = n > nstar
            T.append(t)
            XC.append(x[j].max() if len(j) else np.nan)
            XN.append(x[hi].max() if hi.any() else np.nan)
            w = ro[hi]
            FM.append(float((w * f[hi]).sum() / w.sum()) if hi.any() and w.sum() > 0 else np.nan)

        for line in fh:
            p = line.split(",")
            t = float(p[ix["t_h"]])
            if cur is None:
                cur = t
            if t != cur:
                flush(cur, buf)
                buf = []
                cur = t
            buf.append([float(p[ix["x_um"]]), float(p[ix["f1"]]),
                        float(p[ix["n"]]), float(p[ix["rho"]])])
        if buf:
            flush(cur, buf)

    out = {"t_h": np.array(T), "x_comp": np.array(XC),
           "x_nstar": np.array(XN), "f1_in_highn": np.array(FM)}
    out["lag"] = out["x_comp"] - out["x_nstar"]
    with open(cache, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(out))
        for i in range(len(out["t_h"])):
            w.writerow(["" if not np.isfinite(out[k][i]) else "%.6g" % out[k][i] for k in out])
    print(f"  cached {cache}")
    return out


# ---------------------------------------------------------------- phases
def find_phases(ts, fr):
    """(t_crash_end, t_recovery_end): the F1 bottleneck, and where the lag stops growing."""
    t, F1 = ts["t_h"], ts["F1"]
    t_bot = float(t[int(np.argmin(F1))])
    lag = fr["lag"]
    ok = np.isfinite(lag)
    if ok.sum() < 3:
        return t_bot, float(t[-1])
    tl, ll = fr["t_h"][ok], lag[ok]
    fin = ll[-1]
    reach = np.where(ll >= 0.99 * fin)[0]          # first time within 1% of the final lag
    return t_bot, float(tl[reach[0]]) if len(reach) else float(tl[-1])


def shade(ax, t0, t1, tmin, tmax):
    for (a, b, c, lab) in ((tmin, t0, "#d62728", "1. crash"),
                           (t0, t1, "#2ca02c", "2. recovery"),
                           (t1, tmax, "#7f7f7f", "3. arrest")):
        ax.axvspan(a, b, color=c, alpha=0.06, lw=0)
        ax.axvline(b, color="0.55", lw=0.7, ls=":", zorder=0)
    return


def phase_labels(ax, t0, t1, tmin, tmax):
    y = ax.get_ylim()[1]
    for (a, b, lab) in ((tmin, t0, "1. crash"), (t0, t1, "2. recovery"), (t1, tmax, "3. arrest")):
        ax.text(np.sqrt(a * b), y, lab, ha="center", va="top", fontsize=7.5,
                color="0.25", fontweight="bold")


# ---------------------------------------------------------------- figure
def figure(run, out_path, tmin=5.0, refresh=False):
    S = load_summary(run)
    nstar = float(S["n_star_uM"])
    n0, dx, dt = float(S["n0_uM"]), float(S["dx_um"]), float(S["dt_h"])
    ts = load_timeseries(run)
    fr = extract_fronts(run, nstar, refresh=refresh)
    t0, t1 = find_phases(ts, fr)
    t = ts["t_h"]
    tmax = float(t[-1])
    tm = max(tmin, float(t[1]) if len(t) > 1 else tmin)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Front lag and the three phases   |   $n_0$={n0:g} $\\mu$M, "
        f"$\\Delta x$={dx:g} $\\mu$m, $\\Delta t$={dt:g} h, "
        f"$n^*$={nstar:.3f} $\\mu$M, $T$={tmax:g} h",
        fontsize=11, fontweight="bold")

    # (A) composition -------------------------------------------------
    a = ax[0, 0]
    a.plot(t, ts["F1"], color=C_S1, lw=2, label="$F_1$ (strain-1 mass fraction)")
    a.set_yscale("log"); a.set_xscale("log"); a.set_xlim(tm, tmax)
    a.set_ylabel("$F_1$"); a.set_xlabel("t [h]")
    shade(a, t0, t1, tm, tmax)
    i = int(np.argmin(ts["F1"]))
    a.plot(t[i], ts["F1"][i], "v", color="k", ms=7, zorder=5)
    a.annotate(f"bottleneck\n$F_1$={ts['F1'][i]:.2e}\nt={t[i]:.0f} h",
               (t[i], ts["F1"][i]), textcoords="offset points", xytext=(8, 14),
               fontsize=7.5, ha="left")
    a.axhline(ts["F1"][-1], color="0.4", lw=0.8, ls="--")
    a.text(tmax, ts["F1"][-1], f"  {ts['F1'][-1]:.4f}", va="center", fontsize=8, color="0.3")
    a.grid(alpha=0.25, which="both"); a.legend(fontsize=8, loc="lower right")
    a.set_title("(A) composition: crash, recovery, arrest", fontsize=9, fontweight="bold")
    phase_labels(a, t0, t1, tm, tmax)

    # (B) the two fronts ----------------------------------------------
    b = ax[0, 1]
    b.plot(fr["t_h"], fr["x_nstar"], color=C_SEL, lw=2, label="selection front ($n=n^*$)")
    b.plot(fr["t_h"], fr["x_comp"], color=C_LAG, lw=2, label="composition front ($f_1=0.5$)")
    ok = np.isfinite(fr["x_comp"]) & np.isfinite(fr["x_nstar"])
    b.fill_between(fr["t_h"][ok], fr["x_nstar"][ok], fr["x_comp"][ok],
                   color=C_LAG, alpha=0.18, lw=0, label="lag = inverse-sorted zone")
    b.set_xscale("log"); b.set_xlim(tm, tmax)
    b.set_xlabel("t [h]"); b.set_ylabel("x [$\\mu$m]")
    shade(b, t0, t1, tm, tmax)
    b.grid(alpha=0.25, which="both"); b.legend(fontsize=8, loc="upper left")
    b.set_title("(B) the composition front lags the selection front", fontsize=9, fontweight="bold")

    # (C) the lag, and the mouth flip ---------------------------------
    c = ax[1, 0]
    c.plot(fr["t_h"], fr["lag"], color=C_LAG, lw=2)
    c.set_xscale("log"); c.set_xlim(tm, tmax)
    c.set_xlabel("t [h]"); c.set_ylabel("lag  $x_{\\rm comp}-x_{n^*}$  [$\\mu$m]", color=C_LAG)
    c.tick_params(axis="y", colors=C_LAG)
    fin = fr["lag"][np.isfinite(fr["lag"])]
    if len(fin):
        c.axhline(fin[-1], color=C_LAG, lw=0.8, ls="--")
        # inside the axes: at the right edge this collides with the twin axis' top tick label
        c.text(tm * 1.4, fin[-1], f"final lag {fin[-1]:.0f} $\\mu$m", va="bottom", ha="left",
               fontsize=8, color=C_LAG, fontweight="bold")
    shade(c, t0, t1, tm, tmax)
    c2 = c.twinx()
    c2.plot(fr["t_h"], fr["f1_in_highn"], color=C_S1, lw=1.6, ls="-.",
            label="$\\langle f_1\\rangle$ where $n>n^*$")
    c2.axhline(0.5, color="k", lw=0.8, ls=":")
    c2.set_ylabel("$\\langle f_1\\rangle$ in the $n>n^*$ zone", color=C_S1)
    c2.tick_params(axis="y", colors=C_S1); c2.set_ylim(-0.03, 1.03)
    fl = fr["t_h"][np.isfinite(fr["f1_in_highn"]) & (fr["f1_in_highn"] > 0.5)]
    if len(fl):
        c2.axvline(fl[0], color=C_S1, lw=1.0, ls="--")
        c2.annotate(f"mouth inverts\nt={fl[0]:.0f} h", (fl[0], 0.5),
                    textcoords="offset points", xytext=(10, -30), fontsize=7.5, color=C_S1)
    c.grid(alpha=0.25, which="both")
    c.set_title("(C) lag growth, and when the mouth zone inverts", fontsize=9, fontweight="bold")

    # (D) the inverse-sorted fractions --------------------------------
    d = ax[1, 1]
    d.plot(t, ts["inv_deep"], color=C_S2, lw=2, label="deep graveyard ($n<n^*$, $f_1<0.5$)")
    d.plot(t, ts["inv_mouth"], color=C_S1, lw=2, label="mouth inverse ($n>n^*$, $f_1>0.5$)")
    d.plot(t, ts["delta_f"], color="0.35", lw=1.4, ls="--",
           label="$\\Delta f=\\langle f\\rangle_{\\rm mouth}-\\langle f\\rangle_{\\rm deep}$")
    d.set_xscale("log"); d.set_xlim(tm, tmax); d.set_ylim(-0.03, 1.03)
    d.set_xlabel("t [h]"); d.set_ylabel("biomass fraction")
    shade(d, t0, t1, tm, tmax)
    for arr, col in ((ts["inv_deep"], C_S2), (ts["inv_mouth"], C_S1)):
        i = int(np.nanargmax(arr))
        d.plot(t[i], arr[i], "o", color=col, ms=5, zorder=5)
        d.annotate(f"{arr[i]:.3f}\n@{t[i]:.0f} h", (t[i], arr[i]), textcoords="offset points",
                   xytext=(6, 6), fontsize=7, color=col)
    d.grid(alpha=0.25, which="both"); d.legend(fontsize=7.5, loc="center left")
    d.set_title("(D) inverse-sorted biomass: both types persist", fontsize=9, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"  phases: crash 0-{t0:.0f} h | recovery {t0:.0f}-{t1:.0f} h | arrest {t1:.0f}-{tmax:.0f} h")
    if len(fin):
        print(f"  final lag = {fin[-1]:.1f} um ;  final F1 = {ts['F1'][-1]:.4f}")
    if len(fl):
        print(f"  mouth zone inverts at t = {fl[0]:.0f} h")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing timeseries.csv + full_dump.csv")
    ap.add_argument("--out", default=None, help="output png (default <run>/front_lag.png)")
    ap.add_argument("--tmin", type=float, default=5.0, help="left edge of the log time axis [h]")
    ap.add_argument("--refresh", action="store_true", help="re-scan full_dump.csv, ignore the cache")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    out = a.out or os.path.join(run, "front_lag.png")
    figure(run, out, tmin=a.tmin, refresh=a.refresh)


if __name__ == "__main__":
    main()
