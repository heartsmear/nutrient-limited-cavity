"""
plot_nstar_front.py -- the selection front x_n*(t) from a full_dump.csv.

Read-only: consumes a dump a run has already written and simulates nothing.

The front is located with `diagnostics.nstar_front` (first n = n* crossing, interpolated in
log n) rather than a private reimplementation, so the curve here is the same quantity that
`summary.csv` and `timeseries.csv` report.  Two traps this avoids:

  * Do NOT find the front by interpolating on s = g1 - g2.  s has TWO roots, n = n* and n = 0,
    so a sign scan on s locks onto the starved tail and returns x = 0.
  * The front position is only meaningful where the nutrient boundary layer is RESOLVED.
    lambda = dx / ln(n[k-1]/n[k]) is the screening length at the front; once dx/lambda > 0.6 the
    discrete profile cannot decay as fast as the true one and the front sits too deep.  Frames
    that fail this are shaded, because the curve is a grid artifact there, not a measurement.

Usage:
    py -3.11 src2/plot_nstar_front.py --run src2/run_n0-1000_dx0.5_T10000
    py -3.11 src2/plot_nstar_front.py --dump path/to/full_dump.csv --nstar 5.2526 --linear
    py -3.11 src2/plot_nstar_front.py --run <dir> --csv fronts.csv --no-resolution
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

from diagnostics import nstar_front, front_resolution     # canonical definitions

# dumps from src/ used different headers; accept both.
ALIAS = {"t_h": ("t_h", "t", "time_h"),
         "x_um": ("x_um", "x"),
         "n": ("n", "n_uM", "nutrient")}
DX_OVER_LAMBDA_FAIL = 0.6      # calibrated in diagnostics.validate


def resolve_columns(hdr):
    out = {}
    for key, cands in ALIAS.items():
        hit = next((c for c in cands if c in hdr), None)
        if hit is None:
            raise SystemExit(f"full_dump.csv has no column for {key!r} (tried {cands}); "
                             f"header is: {hdr}")
        out[key] = hdr.index(hit)
    return out


def find_nstar(run_dir, override=None):
    """CLI value > summary.csv of the run > the model default from config."""
    if override is not None:
        return float(override), "--nstar"
    s = os.path.join(run_dir, "summary.csv")
    if os.path.exists(s):
        d = dict(r for r in csv.reader(open(s)) if len(r) == 2)
        if d.get("n_star_uM"):
            return float(d["n_star_uM"]), "summary.csv"
    import config
    from closures import monod_crossover
    ns = monod_crossover(config.params())
    if ns is None:
        raise SystemExit("no n* for the default parameters; pass --nstar explicitly")
    return float(ns), "config default"


def scan_dump(path, nstar):
    """One streaming pass: per frame -> (t, x_nstar, lambda_front, dx_over_lambda)."""
    T, XF, LAM, DOL = [], [], [], []
    with open(path) as fh:
        hdr = fh.readline().strip().split(",")
        ix = resolve_columns(hdr)
        cur, xs, ns_ = None, [], []

        def flush(t, xs, ns_):
            x = np.asarray(xs)
            n = np.asarray(ns_)
            o = np.argsort(x)                       # dumps are already ordered; be safe
            x, n = x[o], n[o]
            dx = float(np.median(np.diff(x))) if x.size > 1 else float("nan")
            r = front_resolution(x, n, nstar, dx)
            T.append(t)
            XF.append(nstar_front(x, n, nstar))
            LAM.append(r["lambda_front"])
            DOL.append(r["dx_over_lambda"])

        for line in fh:
            p = line.split(",")
            t = float(p[ix["t_h"]])
            if cur is None:
                cur = t
            if t != cur:
                flush(cur, xs, ns_)
                xs, ns_ = [], []
                cur = t
            xs.append(float(p[ix["x_um"]]))
            ns_.append(float(p[ix["n"]]))
        if xs:
            flush(cur, xs, ns_)
    return (np.array(T), np.array(XF), np.array(LAM), np.array(DOL))


def figure(t, xf, lam, dol, nstar, out_path, title="", logx=True,
           show_resolution=True, tmin=None):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ok = np.isfinite(xf)

    if show_resolution:
        bad = np.isfinite(dol) & (dol > DX_OVER_LAMBDA_FAIL)
        if bad.any():                                # contiguous runs -> one span each
            e = np.diff(np.concatenate(([0], bad.view(np.int8), [0])))
            for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]):
                ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#d62728", alpha=0.13, lw=0,
                           zorder=0)
            ax.plot([], [], "s", color="#d62728", alpha=0.35, ms=9,
                    label=f"unresolved ($\\Delta x/\\lambda>{DX_OVER_LAMBDA_FAIL}$) -- front is a grid artifact")

    ax.plot(t[ok], xf[ok], color="#2ca02c", lw=2,
            label=f"$x_{{n^*}}(t)$   ($n^*$={nstar:.4g} $\\mu$M)")

    if ok.sum():
        i = int(np.nanargmin(np.where(ok, xf, np.inf)))
        ax.plot(t[i], xf[i], "v", color="k", ms=7, zorder=5)
        # up-and-left: below the marker is the x-axis, to the right is the rising limb
        ax.annotate(f"closest to mouth\n{xf[i]:.1f} $\\mu$m @ {t[i]:.0f} h",
                    (t[i], xf[i]), textcoords="offset points", xytext=(-10, 22),
                    ha="right", fontsize=8)
        ax.axhline(xf[ok][-1], color="0.45", lw=0.8, ls="--")
        lo = ax.get_xlim()[0]
        ax.text(lo * 1.6 if logx else lo, xf[ok][-1], f"final {xf[ok][-1]:.1f} $\\mu$m",
                va="bottom", ha="left", fontsize=8, color="0.3")

    gaps = int((~ok).sum())
    if gaps:
        ax.plot([], [], " ",
                label=f"({gaps} frame{'s' if gaps != 1 else ''} with no $n^*$ crossing)")

    if logx:
        ax.set_xscale("log")
        lo = tmin if tmin is not None else (t[t > 0].min() if (t > 0).any() else 1.0)
        ax.set_xlim(lo, t.max())
    else:
        ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("t [h]")
    ax.set_ylabel("$x_{n^*}$  [$\\mu$m]   (distance from the opening)")
    ax.set_title(title or "Selection front $n=n^*$ over time", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory containing full_dump.csv (+ summary.csv for n*)")
    src.add_argument("--dump", help="path to a full_dump.csv directly")
    ap.add_argument("--nstar", type=float, default=None, help="override n* [uM]")
    ap.add_argument("--out", default=None, help="output png (default <dir>/nstar_front.png)")
    ap.add_argument("--csv", default=None, help="also write the extracted series here")
    ap.add_argument("--linear", action="store_true", help="linear time axis (default log)")
    ap.add_argument("--tmin", type=float, default=None, help="left edge of a log time axis [h]")
    ap.add_argument("--no-resolution", action="store_true",
                    help="do not shade frames where dx/lambda > %.2g" % DX_OVER_LAMBDA_FAIL)
    a = ap.parse_args()

    dump = os.path.abspath(a.dump) if a.dump else os.path.join(os.path.abspath(a.run), "full_dump.csv")
    if not os.path.exists(dump):
        raise SystemExit(f"no such file: {dump}  (run figures.py with --dump)")
    d = os.path.dirname(dump)
    nstar, where = find_nstar(d, a.nstar)
    print(f"  dump  : {dump}  ({os.path.getsize(dump)/2**20:.1f} MB)")
    print(f"  n*    : {nstar:.6g} uM   (from {where})")

    t, xf, lam, dol = scan_dump(dump, nstar)
    ok = np.isfinite(xf)
    print(f"  frames: {len(t)}   t = {t.min():.4g} .. {t.max():.4g} h   "
          f"({(~ok).sum()} with no crossing)")
    if ok.any():
        print(f"  x_n*  : min {np.nanmin(xf[ok]):.2f} um @ t={t[ok][np.nanargmin(xf[ok])]:.0f} h   "
              f"max {np.nanmax(xf[ok]):.2f} um   final {xf[ok][-1]:.2f} um")
    if np.isfinite(dol).any():
        nb = int((dol[np.isfinite(dol)] > DX_OVER_LAMBDA_FAIL).sum())
        print(f"  dx/lam: median {np.nanmedian(dol):.3f}  max {np.nanmax(dol):.3f}  "
              f"-> {nb}/{int(np.isfinite(dol).sum())} frames UNRESOLVED")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_h", "x_nstar_um", "lambda_front_um", "dx_over_lambda"])
            for i in range(len(t)):
                w.writerow(["%.6g" % t[i]] + ["" if not np.isfinite(v[i]) else "%.6g" % v[i]
                                              for v in (xf, lam, dol)])
        print(f"  wrote {a.csv}")

    out = a.out or os.path.join(d, "nstar_front.png")
    figure(t, xf, lam, dol, nstar, out, title=f"Selection front $n=n^*$   |   {os.path.basename(d)}",
           logx=not a.linear, show_resolution=not a.no_resolution, tmin=a.tmin)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
