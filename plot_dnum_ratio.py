"""
plot_dnum_ratio.py -- the numerical label diffusivity against the physical one, in space and time.
Simulates nothing; reads `full_dump.csv` from runs that already exist.

    py -3.11 src2/plot_dnum_ratio.py
    py -3.11 src2/plot_dnum_ratio.py --tmax 3000 --out somewhere.png

WHAT IS PLOTTED.  Donor-cell upwinding of the strain label does not solve the advection equation
it is meant to; its modified equation carries an extra diffusion

    D_num = |u| dx (1 - |nu|) / 2 ,

which is not in the model.  It acts on the label in the same slot as the physical D_self, so the
two ADD, and the ratio D_num/D_self says which one is actually moving the labels.  Ratio > 1 means
the discretisation error dominates the physics it is supposed to be a correction to.

WHY THE JAMMED REGION IS WHERE IT BITES.  D_self collapses discontinuously at rho_jam (18.8 ->
0.564 um^2/h, a factor 33), so behind the jamming front the physical term is nearly switched off
while D_num is unchanged.  The rho = rho_jam contour is drawn on each panel for exactly that
reason: the ratio should jump as it is crossed.

TWO HONEST LIMITS ON THE NUMBER.
  * nu -> 0 is assumed, i.e. D_num = |u| dx / 2, which is an UPPER bound.  The advection is
    CFL-sub-cycled to nu <= 0.45, so the true coefficient is 0.55-1x of what is drawn.  The ratio
    BETWEEN grids is unaffected, since both are sub-cycled by the same rule.
  * The formula is the modified equation of the scalar model problem.  The solver transports mass
    m1 by J1 = f_face * J_tot with variable u = J/rho, sub-cycled, with the total advancing too,
    so this is a scale estimate for the real scheme and not its exact coefficient.

ONLY DONOR-CELL RUNS ARE SHOWN.  With the van Leer limiter the face value is reconstructed rather
than taken from the upstream centre, so D_num above is no longer the error being made and the
ratio would be meaningless.  The limiter's effect is measured elsewhere, by what it changes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

RHO_JAM = 0.64
#: below this density the cell is effectively empty, u = J/rho is meaningless and the ratio is
#: masked rather than plotted -- otherwise the mouth boundary layer dominates the colour scale
RHO_MIN = 1.0e-2

SURFACE, INK, INK2, MUTED, GRID_C = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"

RUNS = [(2.0, "run_n0-1000_L1000_dx2_T6000"),
        (0.5, "run_n0-1000_dx0.5_T10000")]


def grid(path, dx, tmax):
    """(t, x, ratio, rho) on the dump's own space-time grid, masked where nearly empty."""
    d = pd.read_csv(os.path.join(HERE, path),
                    usecols=["t_h", "x_um", "rho", "u", "Dself"])
    d = d[d.t_h <= tmax]
    t = np.sort(d.t_h.unique())
    x = np.sort(d.x_um.unique())
    shape = (len(x), len(t))
    piv = lambda c: d.pivot(index="x_um", columns="t_h", values=c).to_numpy()
    rho, u, ds = piv("rho"), piv("u"), piv("Dself")
    dnum = 0.5 * np.abs(u) * dx                      # nu -> 0 upper bound
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dnum / ds
    r = np.where(rho > RHO_MIN, r, np.nan)
    assert rho.shape == shape, f"{rho.shape} != {shape}"
    return t, x, r, rho


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID_C)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tmax", type=float, default=3000.0)
    ap.add_argument("--out", default=os.path.join(HERE, "dnum_over_dself.png"))
    a = ap.parse_args()

    data = []
    for dx, d in RUNS:
        p = os.path.join(d, "full_dump.csv")
        if not os.path.exists(os.path.join(HERE, p)):
            sys.exit(f"missing {p}")
        print(f"  reading {d} ...", flush=True)
        data.append((dx,) + grid(p, dx, a.tmax))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), facecolor=SURFACE,
                             layout="constrained")
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=1.0)
    im = None
    for k, (dx, t, x, r, rho) in enumerate(data):
        ax = axes[k]
        im = ax.pcolormesh(t, x, np.log10(r), cmap="RdBu_r", norm=norm, shading="auto")
        ax.contour(t, x, rho, levels=[RHO_JAM], colors="k", linewidths=1.2)
        frac = 100.0 * np.nanmean(r > 1.0)
        ax.set_ylim(0, 1000)
        style(ax, "t  [h]", r"x  [$\mu$m]", rf"({'ab'[k]})  $\Delta x={dx:g}\ \mu$m")
        ax.annotate(rf"$D_{{\rm num}}>D_{{\rm self}}$ over {frac:.1f}%" "\nof occupied space-time",
                    (0.03, 0.96), xycoords="axes fraction", ha="left", va="top",
                    color=INK, fontsize=8.5)
        ax.annotate("black contour: " r"$\rho=\rho_{\rm jam}$" "\njammed side is below",
                    (0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
                    color=INK, fontsize=8)
    cb = fig.colorbar(im, ax=axes[:2].tolist(), pad=0.015, fraction=0.035)
    cb.set_label(r"$\log_{10}\left(D_{\rm num}/D_{\rm self}\right)$", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    ax = axes[2]
    for (dx, t, x, r, rho), c in zip(data, ("#c0392b", "#2a78d6")):
        for tq, ls in ((300.0, "-"), (1000.0, "--")):
            j = int(np.argmin(np.abs(t - tq)))
            ax.plot(x, r[:, j], ls=ls, lw=1.5, color=c,
                    label=rf"$\Delta x={dx:g}$, $t={t[j]:.0f}$ h")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.annotate("numerical = physical", (0.98, 1.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_yscale("log"); ax.set_xlim(0, 1000)
    ax.legend(frameon=False, fontsize=7.6, labelcolor=INK2, loc="lower right")
    style(ax, r"x  [$\mu$m]", r"$D_{\rm num}/D_{\rm self}$",
          "(c)  refining $4\\times$ drops the ratio $\\approx4\\times$: "
          "$D_{\\rm num}\\propto\\Delta x$")

    fig.suptitle("Numerical label diffusivity against the physical one, donor-cell upwinding "
                 r"($n_0=1000\ \mu$M)", color=INK, fontsize=12.5, x=0.006, ha="left")
    fig.get_layout_engine().set(rect=(0.0, 0.075, 1.0, 0.94))
    fig.text(0.006, 0.015,
             r"$D_{\rm num}=\frac{1}{2}|u|\Delta x$ is the $\nu\!\to\!0$ upper bound; CFL "
             r"sub-cycling ($\nu\leq0.45$) puts the true value at 0.55--1$\times$ this, equally "
             r"on both grids.  Cells below $\rho=10^{-2}$ are masked, where $u=J/\rho$ is "
             "meaningless.",
             ha="left", fontsize=8.3, color=MUTED)
    fig.savefig(a.out, dpi=170, facecolor=SURFACE)
    print(f"\nwrote {a.out}")
    for path in write_csvs(a.out, data):
        print(f"wrote {path}")


def write_csvs(out, data):
    """Every array the figure draws, beside the PNG (rule: plotted arrays are always saved).

    <stem>.csv           the (a)/(b) heatmap fields in long format, with rho for the contour;
                         the ratio is 'nan' exactly where the figure masks it (rho < RHO_MIN)
    <stem>_profiles.csv  the (c) slices, at the frame actually drawn (nearest to 300/1000 h)
    <stem>_summary.csv   the percentage printed in (a)/(b)
    """
    stem = os.path.splitext(out)[0]
    with open(stem + ".csv", "w", newline="") as fh:
        fh.write("dx_um,t_h,x_um,rho,Dnum_over_Dself\n")
        for dx, t, x, r, rho in data:
            tt, xx = np.meshgrid(t, x)               # (x, t) layout, matching r and rho
            np.savetxt(fh, np.column_stack([np.full(r.size, dx), tt.ravel(), xx.ravel(),
                                            rho.ravel(), r.ravel()]),
                       delimiter=",", fmt="%.8g")
    with open(stem + "_profiles.csv", "w", newline="") as fh:
        fh.write("dx_um,t_requested_h,t_h,x_um,Dnum_over_Dself\n")
        for dx, t, x, r, rho in data:
            for tq in (300.0, 1000.0):
                j = int(np.argmin(np.abs(t - tq)))
                np.savetxt(fh, np.column_stack([np.full(x.size, dx), np.full(x.size, tq),
                                                np.full(x.size, t[j]), x, r[:, j]]),
                           delimiter=",", fmt="%.8g")
    with open(stem + "_summary.csv", "w", newline="") as fh:
        fh.write("dx_um,pct_occupied_spacetime_Dnum_gt_Dself,rho_min_mask\n")
        for dx, t, x, r, rho in data:
            fh.write(f"{dx:g},{100.0 * np.nanmean(r > 1.0):.6g},{RHO_MIN:g}\n")
    return [stem + ".csv", stem + "_profiles.csv", stem + "_summary.csv"]


if __name__ == "__main__":
    main()
