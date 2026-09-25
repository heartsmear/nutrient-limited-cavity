"""
plot_fst.py -- F_ST(t) for runs that are already on disk.  Simulates nothing.

`--fst` on figures.py only helps runs made after it existed.  This recovers the same curve from
`full_dump.csv`, which carries the space-time fields any completed run with `--dump` already has,
and computes it by calling `diagnostics.fst_series` on a reconstructed frame stack -- the same
code path as a live run, so the numbers are identical rather than merely similar.

    py -3.11 src2/plot_fst.py --all
    py -3.11 src2/plot_fst.py run_n0-15_L1000_dx0.5_T500 run_n0-1000_dx0.5_T10000
    py -3.11 src2/plot_fst.py --all --out fst_all.png

WHAT IS AND IS NOT RECOVERABLE
  * `full_dump.csv`   -> the whole F_ST(t) trajectory.
  * `profile_final.csv` only -> ONE value at the final frame.  That is not enough: F_ST is a
    transient and its final value understates it (dilute peaks near 0.14 and decays ~89% within
    800 h).  Such runs are reported as a single point and explicitly flagged, not plotted as a
    curve.

n0 IS READ FROM THE DIRECTORY NAME.  F_ST itself does not depend on any model parameter -- it is
a moment of f -- so this only affects the labelling and the n* used for context.  A directory
renamed away from the `n0-<value>` convention falls back to the config default and says so.

MEMORY.  The dumps run to ~120 MB, so the file is streamed frame by frame rather than loaded.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
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


def n0_from_name(d: str):
    m = re.search(r"n0-([0-9.e+]+)", os.path.basename(os.path.normpath(d)))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def frames_from_dump(path):
    """Stream full_dump.csv into (t, rho1_profiles, rho2_profiles) without loading it whole."""
    ts, R1, R2 = [], [], []
    cur, a1, a2 = None, [], []
    with open(path) as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        it, ir = hdr.index("t_h"), hdr.index("rho")
        # Older runs dumped (rho, f1); newer ones dump rho1/rho2 as well.  Reconstruct from the
        # fraction when the per-strain columns are absent, so every run on disk is readable.
        i1 = hdr.index("rho1") if "rho1" in hdr else None
        i2 = hdr.index("rho2") if "rho2" in hdr else None
        iff = hdr.index("f1") if "f1" in hdr else None
        if i1 is None and iff is None:
            raise ValueError(f"{path}: needs rho1, or rho and f1")
        for row in rd:
            t = float(row[it])
            if t != cur:
                if cur is not None and a1:
                    ts.append(cur); R1.append(a1); R2.append(a2)
                cur, a1, a2 = t, [], []
            try:
                rho = float(row[ir])
                if i1 is not None:
                    r1 = float(row[i1])
                    r2 = float(row[i2]) if i2 is not None else rho - r1
                else:
                    f1 = float(row[iff])
                    r1 = rho * f1
                    r2 = rho - r1
            except ValueError:
                r1 = r2 = np.nan
            a1.append(r1); a2.append(r2)
        if cur is not None and a1:
            ts.append(cur); R1.append(a1); R2.append(a2)
    # frames are rows here; fst_series wants (cells, frames)
    return np.array(ts), np.array(R1).T, np.array(R2).T


def frames_from_final(path):
    """One frame from profile_final.csv -- a point, not a trajectory."""
    R = list(csv.DictReader(open(path)))
    r1 = np.array([float(r["rho1"]) for r in R])
    rho = np.array([float(r["rho"]) for r in R])
    return np.array([np.nan]), r1[:, None], (rho - r1)[:, None]


def load(d):
    dump = os.path.join(d, "full_dump.csv")
    fin = os.path.join(d, "profile_final.csv")
    if os.path.exists(dump):
        t, R1, R2 = frames_from_dump(dump); kind = "trajectory"
    elif os.path.exists(fin):
        t, R1, R2 = frames_from_final(fin); kind = "final-frame only"
    else:
        return None
    return dict(t=t, rho1_profiles=R1, rho2_profiles=R2), kind


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories")
    ap.add_argument("--all", action="store_true", help="every run_*/ with a full_dump.csv")
    ap.add_argument("--out", default=os.path.join(HERE, "fst_from_csv.png"))
    ap.add_argument("--min-sd-max", dest="min_sd_max", type=float, default=0.05,
                    help="conditioning floor; F_ST is 0/0 as F1 -> 1 (default 0.05)")
    ap.add_argument("--no-csv", action="store_true", help="do not write fst.csv per run")
    a = ap.parse_args()

    runs = list(a.runs)
    if a.all:
        runs += sorted(os.path.dirname(p) for p in glob.glob(os.path.join(HERE, "run_*", "full_dump.csv")))
    runs = [r for r in dict.fromkeys(runs) if r]
    if not runs:
        ap.error("no runs given; pass directories or --all")

    print(f"{'run':>40}{'frames':>8}{'peak':>9}{'t_peak':>9}{'final':>9}"
          f"{'slope/1000h':>13}{'valid':>8}")
    curves = []
    for d in runs:
        d = d if os.path.isabs(d) else os.path.join(HERE, d)
        got = load(d)
        name = os.path.basename(os.path.normpath(d))
        if got is None:
            print(f"{name:>40}   no full_dump.csv or profile_final.csv -- skipped")
            continue
        res, kind = got
        n0 = n0_from_name(d)
        p = config.params(n0 if n0 is not None else config.N0_DILUTE)
        fst, sd, sdmax, _ = D.fst_series(res, p, min_sd_max=a.min_sd_max)
        nvalid = int(np.isfinite(fst).sum())

        if kind == "trajectory":
            s = D.fst_summary(res, p, min_sd_max=a.min_sd_max)
            print(f"{name:>40}{len(res['t']):>8}{s['F_ST_peak']:>9.4f}"
                  f"{s['t_F_ST_peak']:>9.0f}{s['F_ST_final']:>9.4f}"
                  f"{s['dF_ST_per_1000h_late']:>13.3e}{nvalid:>5}/{len(fst)}")
            curves.append((name, res["t"], fst))
        else:
            v = fst[0] if np.isfinite(fst[0]) else float("nan")
            print(f"{name:>40}{1:>8}{v:>9.4f}{'--':>9}{v:>9.4f}{'--':>13}{nvalid:>5}/1"
                  f"   [{kind}: no trajectory, value is at T only]")

        if not a.no_csv and kind == "trajectory":
            with open(os.path.join(d, "fst.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_h", "F_ST", "sd_f", "sd_f_max"])
                for j in range(len(res["t"])):
                    w.writerow([f"{res['t'][j]:.6g}",
                                "" if not np.isfinite(fst[j]) else f"{fst[j]:.8g}",
                                f"{sd[j]:.8g}", f"{sdmax[j]:.8g}"])

    if not curves:
        print("\nnothing with a trajectory to plot")
        return

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(curves)))
    for (name, t, fst), c in zip(curves, cols):
        ax.plot(t, fst, color=c, lw=2.0, label=name)
        i = int(np.nanargmax(fst)) if np.isfinite(fst).any() else 0
        if np.isfinite(fst[i]):
            ax.plot([t[i]], [fst[i]], "o", color=c, ms=6, mfc="none", mew=1.5)
    for lv, lab in ((0.05, "little"), (0.15, "great")):
        ax.axhline(lv, color="0.75", lw=0.8, ls=":")
        ax.annotate(lab, (ax.get_xlim()[1], lv), fontsize=7.5, color="0.55",
                    va="bottom", ha="right")
    ax.set_xscale("symlog", linthresh=100)
    ax.set(xlabel="t [h]", ylabel=r"$F_{ST}$", ylim=(0, None))
    ax.set_title("Spatial structure of the composition, recovered from existing dumps",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.25, lw=0.6)
    fig.text(0.5, 0.01, "circles mark each peak; gaps are frames below the conditioning floor "
                        r"(sd$_{\max}<$" f"{a.min_sd_max:g}), where $F_{{ST}}$ is 0/0",
             ha="center", fontsize=8, color="0.4")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(a.out, dpi=160)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
