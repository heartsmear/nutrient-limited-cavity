"""Rebuild run.png for an EXISTING run folder from its full_dump.csv -- no re-simulation.

`figures.figure_run` needs exactly six things from the solver result: x, dx, t and the three
profile arrays (rho1, rho2, n).  full_dump.csv carries all of them at every saved frame, so a
figure can be regenerated exactly after a plotting change without paying for the solve again --
which matters here, where a jammed run is hours and dx=0.25 is most of a day.

The model parameters are rebuilt from config and then OVERRIDDEN field by field with whatever
summary.csv recorded, so a run made with non-default closure flags replots with its own closures
rather than today's defaults.

    py -3 src2/replot_run_from_dump.py run_n0-15_L1000_dx0.5_T1000_seed0.0001_lim-vanleer
    py -3 src2/replot_run_from_dump.py <dir> --out /tmp/preview.png     # don't overwrite
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                                      # noqa: E402
import figures                                                     # noqa: E402

#: summary.csv key -> ModelParams field.  Anything absent from a given summary is left at the
#: config default; anything present wins, so the replot uses the run's own parameters.
_PMAP = {"mu1": "mu1", "Ks1": "Ks1", "mu2": "mu2", "Ks2": "Ks2", "Y": "Y", "Dn": "Dn",
         "D0": "D0", "rho_jam": "rho_jam", "rho_c": "rho_c", "rho_max": "rho_max",
         "Dself_floor": "Dself_floor", "Dself_jam_slope": "Dself_jam_slope"}


def read_summary(d):
    return {r[0]: r[1] for r in csv.reader(open(os.path.join(d, "summary.csv"))) if len(r) == 2}


def params_from_summary(s):
    p = config.params(float(s["n0_uM"]))
    kw, changed = {}, []
    for key, field in _PMAP.items():
        if key in s and s[key] != "":
            old, new = getattr(p, field), float(s[key])
            if old != new:
                changed.append("%s %g->%g" % (field, old, new))
            kw[field] = new
    if s.get("dself_shape"):
        if getattr(p, "dself_shape") != s["dself_shape"]:
            changed.append("dself_shape %s->%s" % (p.dself_shape, s["dself_shape"]))
        kw["dself_shape"] = s["dself_shape"]
    if changed:
        print("  params overridden from summary.csv: " + ", ".join(changed))
    else:
        print("  params match the config defaults exactly")
    return replace(p, **kw)


def profiles_from_dump(d):
    """(x, dx, t, R1, R2, NN) from full_dump.csv, shaped (N, n_save) like the solver returns."""
    path = os.path.join(d, "full_dump.csv")
    if not os.path.isfile(path):
        raise SystemExit("no full_dump.csv in %s -- the run must have been made with --dump" % d)
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=["t_h", "x_um", "rho1", "rho2", "n"])
        cols = {k: df[k].to_numpy() for k in ("t_h", "x_um", "rho1", "rho2", "n")}
    except ImportError:                                   # pandas-free fallback
        raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 7))
        cols = dict(zip(("t_h", "x_um", "rho1", "rho2", "n"), raw.T))

    t = np.unique(cols["t_h"]); x = np.unique(cols["x_um"])
    ns, N = len(t), len(x)
    if ns * N != len(cols["t_h"]):
        raise SystemExit("dump is not a complete %d x %d grid (%d rows)"
                         % (N, ns, len(cols["t_h"])))
    # the writer emits one frame at a time, x ascending within the frame -- verify, don't assume
    if not (np.all(cols["t_h"][:N] == cols["t_h"][0]) and np.allclose(cols["x_um"][:N], x)):
        raise SystemExit("dump row order is not frame-major with x ascending; cannot reshape")

    def grid(k):
        return cols[k].reshape(ns, N).T

    return x, float(x[1] - x[0]), t, grid("rho1"), grid("rho2"), grid("n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="run folder(s) containing full_dump.csv + summary.csv")
    ap.add_argument("--out", default=None,
                    help="write here instead of <dir>/run.png (single dir only)")
    ap.add_argument("--force", action="store_true",
                    help="plot even if the dump disagrees with summary.csv")
    a = ap.parse_args()
    if a.out and len(a.dirs) > 1:
        raise SystemExit("--out takes a single run directory")

    for d in a.dirs:
        d = d.rstrip("/\\")
        full = d if os.path.isabs(d) else os.path.join(HERE, d)
        print(d)
        s = read_summary(full)
        p = params_from_summary(s)
        x, dx, t, R1, R2, NN = profiles_from_dump(full)
        # rho_peak is the run maximum over ALL frames, not the final one -- recompute it from the
        # dump rather than reading summary.csv, so the replot stands alone.
        res = {"x": x, "dx": dx, "t": t,
               "rho1_profiles": R1, "rho2_profiles": R2, "n_profiles": NN,
               "rho_peak": float((R1 + R2).max())}
        print("  rebuilt N=%d, n_save=%d, dx=%g, T=%g h" % (len(x), len(t), dx, t[-1]))
        # Integrity check against the numbers the original run recorded.  A mismatch means the
        # reshape did not recover the solver's arrays, so the figure would be wrong -- refuse to
        # overwrite on it rather than printing a warning nobody reads.
        bad = []
        for key, got in (("F1_final", R1[:, -1].sum() / max((R1[:, -1] + R2[:, -1]).sum(), 1e-300)),
                         ("rho_peak", float((R1 + R2).max()))):
            if key in s:
                want = float(s[key])
                ok = abs(got - want) <= 1e-6 * max(1.0, abs(want))
                if not ok:
                    bad.append(key)
                print("  %-9s dump %.10g vs summary %.10g   %s"
                      % (key, got, want, "OK" if ok else "MISMATCH"))
        if bad and not a.force:
            raise SystemExit("  refusing to overwrite: %s disagree with summary.csv "
                             "(pass --force to plot anyway)" % ", ".join(bad))
        out = a.out or os.path.join(full, "run.png")
        figures.figure_run(res, p, out)
        print("  wrote %s\n" % out)


if __name__ == "__main__":
    main()
