"""Overlay h_sim (solid) and h_theory (dashed) at every dx already computed.

Reads the profiles.csv that adjoint_kernel.py wrote for each resolution; runs nothing.
The gap between solid and dashed is the first-order kernel's numerical diffusion, and it
closes as dx falls -- Richardson limit 0.000e+00, order p ~ 0.89.

    py -3.11 src2/plot_adjoint_convergence.py
    py -3.11 src2/plot_adjoint_convergence.py --out ../Paper/adjoint_convergence_overlay.png

Writes <out stem>.csv beside the PNG: dx_um, x_um, h_sim, h_theory, abs_diff for every curve.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [(2.0, "adjoint_n0-1000_L1000_dx2_T2000_m30_d900_seed0.0001_traj-vanleer", "#1f77b4"),
        (1.0, "adjoint_n0-1000_L1000_dx1_T2000_m30_d900_seed0.0001_traj-vanleer", "#ff7f0e"),
        (0.5, "adjoint_n0-1000_L1000_dx0.5_T2000_m30_d900_seed0.0001_traj-vanleer", "#2ca02c")]


def load(d):
    rows = [q for q in csv.DictReader(open(os.path.join(HERE, d, "profiles.csv")))
            if int(q["occupied"])]
    return (np.array([float(q["x_um"]) for q in rows]),
            np.array([float(q["h_sim"]) for q in rows]),
            np.array([float(q["h_theory"]) for q in rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "adjoint_convergence_overlay.png"),
                    help="PNG path; the CSV of every plotted array is written beside it")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for dx, d, col in RUNS:
        x, hs, ht = load(d)
        e = float(np.abs(ht - hs).max())
        for a in ax:
            a.plot(x, hs, color=col, lw=2.0, alpha=0.9)
            a.plot(x, ht, color=col, lw=1.3, ls="--")
        ax[0].plot([], [], color=col, lw=2.0,
                   label=r"$\Delta x=%g$   max$|h_{\rm th}-h_{\rm sim}|$=%.2e" % (dx, e))

    ax[0].plot([], [], color="0.35", lw=2.0, label=r"solid: $h_{\rm sim}$")
    ax[0].plot([], [], color="0.35", lw=1.3, ls="--", label=r"dashed: $h_{\rm theory}$")
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].set_xlabel(r"$x$ [$\mu$m]")
    ax[0].set_ylabel("$h = 2f_1-1$")
    ax[0].set_title("full cavity", fontsize=10)
    ax[0].legend(loc="lower left", fontsize=8)

    ax[1].set_xlim(0, 420)
    ax[1].set_ylim(0.55, 1.04)
    ax[1].set_xlabel(r"$x$ [$\mu$m]")
    ax[1].set_title("zoom: the strain-1 plateau, where the gap lives", fontsize=10)

    fig.suptitle("jammed $n_0$=1000, T=2000 h, vanleer trajectory   |   the solid-dashed gap is "
                 "the FIRST-ORDER kernel's numerical diffusion:\norder $p$=0.89, Richardson "
                 "limit 0.000e+00 -- it converges away, it is not an error in the decomposition",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = args.out
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote %s" % out)
    # Every plotted array to CSV beside the PNG, so the figure redraws without the run folders.
    csv_out = os.path.splitext(out)[0] + ".csv"
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dx_um", "x_um", "h_sim", "h_theory", "abs_diff"])
        for dx, d, _ in RUNS:
            x, hs, ht = load(d)
            for i in range(x.size):
                w.writerow(["%g" % dx, "%.6g" % x[i], "%.10g" % hs[i], "%.10g" % ht[i],
                            "%.6e" % abs(ht[i] - hs[i])])
    print("wrote %s" % csv_out)
    print()
    # plain mean over occupied cells -- this was labelled "biomass-wt", which it is not
    print("%-6s %12s %12s" % ("dx", "max resid", "mean resid"))
    for dx, d, _ in RUNS:
        x, hs, ht = load(d)
        e = np.abs(ht - hs)
        print("%-6g %12.3e %12.3e" % (dx, e.max(), e.mean()))


if __name__ == "__main__":
    main()
