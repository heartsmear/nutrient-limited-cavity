"""plot_kernel_figures.py -- the eight draft2 kernel panels, from adjoint_kernel.py output.

WHY THIS EXISTS
---------------
The draft's eight kernel panels were made by `ancestry.py`:

    h_theory_vs_numericdil / h_theory_vs_numeric      dominance: theory vs numerics
    criteriondilute        / criterioninversejam      exposure decomposition A+ vs A-
    kerneldilutemouth/deep / kerneljammouth/deep      K(y,t) space-time heatmaps

`ancestry.py` CANNOT be re-run at the production config.  It recovers P_m by eps-probing, which
requires transport linear in the label, which only donor-cell upwinding is -- so `forward()`
refuses a limiter outright (F15).  Its replacement, `adjoint_kernel.py`, solves for K directly as
the adjoint field and runs limited, but emits two figures in its own layout (adjoint.png,
profiles.png), not these eight.  This script re-cuts the eight from what the adjoint runs stored.

Simulates nothing.  Reads only files that already exist; imports no solver module.

WHAT EACH PANEL NOW SHOWS, and how it differs from the ancestry.py original
--------------------------------------------------------------------------
* h_theory_vs_*: h_sim against h_theory, plus the residual on a log axis.  The ancestry.py
  version was an ALGEBRAIC IDENTITY -- realized increments pushed back through realized matrices,
  so agreement was guaranteed and the residual was round-off (F3).  The adjoint kernel is solved
  independently of the forward label field, so the residual here is a MEASURED disagreement.
  That is the point of the figure now, and the caption has to change with it.
* criterion*: A+ and A- per cell with the inverse-sorted set shaded, and underneath BOTH SIDES OF
  THE IDENTITY sigma_T*h = A+ - A- on one axis, so the gap between them is the residual.
  Do NOT plot (A- - A+) against sigma_T*h: they are the same quantity negated (corr = -1.000000),
  so the panel would show two mirror-image curves and read as disagreement where there is an
  identity.  On the missing 1/2 see the CONVENTION note in fig_criterion -- the manuscript's
  continuum A+- are exactly twice these discrete ones (measured 2.000-2.006), so both
  sigma_T h = 1/2(A+ - A-) and sigma_T h = A+ - A- are correct, each in its own convention.
* kernel*: K(y,t) for one target, with the mean ancestral trajectory ybar(t) and the n* front
  overlaid.  The front comes from the matching base run's timeseries.csv (same config, same
  trajectory), since fields.npz stores only the final nutrient profile.

    py -3.11 src2/plot_kernel_figures.py                 # all eight into src2/
    py -3.11 src2/plot_kernel_figures.py --outdir ../Paper
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
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))

# Project palette, matching plot_fst_vs_L.py / plot_production_maps.py.
SURFACE, INK, INK2, MUTED, GRID_C = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"
C_SIM, C_TH, C_WEAK = "#0b0b0b", "#eb6834", "#7a7975"
C_PLUS, C_MINUS = "#2a78d6", "#eb6834"
C_TRAJ, C_FRONT = "#2ca02c", "#ffffff"
SHADE = "#2a78d6"

#: case -> (adjoint dir, base run dir for the n* front, label).  The label carries T, so a panel
#: always states its own time and cannot be mis-captioned.
CASES = {
    "dil96": ("adjoint_n0-15_L1000_dx0.5_T96_m30_d900_seed0.0001_traj-vanleer",
              "run_n0-15_L1000_dx0.5_T96_seed0.0001_lim-vanleer",
              r"dilute  $n_0=15\ \mu$M,  $T=96$ h"),
    "dil500": ("adjoint_n0-15_L1000_dx0.5_T500_m30_d900_seed0.0001_traj-vanleer",
               "run_n0-15_L1000_dx0.5_T500_seed0.0001_lim-vanleer",
               r"dilute  $n_0=15\ \mu$M,  $T=500$ h"),
    "jam": ("adjoint_n0-1000_L1000_dx0.5_T2000_m30_d900_seed0.0001_traj-vanleer",
            "run_n0-1000_L1000_dx0.5_T2000_seed0.0001_lim-vanleer",
            r"jammed  $n_0=1000\ \mu$M,  $T=2000$ h"),
}

#: (case, panel, draft figure stem).
#:
#: WHY THE DILUTE PANELS SPLIT ACROSS TWO TIMES.  The dilute cavity is ~recovered by T=500:
#: F1 = 0.98, h > 0 nearly everywhere, and only a thin mouth sliver is inverse-sorted -- so the
#: h and criterion panels had almost nothing to show.  At T=96, which is also the time of the
#: draft's MAIN dilute figure (run_noise_dilute), 69.3% of occupied cells are inverse-sorted:
#: 72 mouth cells (sigma_T<0) plus 1314 deep (sigma_T>0), i.e. the double-inverse structure the
#: text describes.  The two kernel HEATMAPS stay at T=500 because their job is different -- they
#: trace where the deep biomass came from, and the long sweep from the opening out to x=900 is
#: only visible once the trajectory has had time to run.
PANELS = [
    ("dil96",  "h",     "h_theory_vs_numericdil"),
    ("dil96",  "crit",  "criteriondilute"),
    ("dil500", "mouth", "kerneldilutemouth"),
    ("dil500", "deep",  "kerneldilutedeep"),
    ("jam",    "h",     "h_theory_vs_numeric"),
    ("jam",    "crit",  "criterioninversejam"),
    ("jam",    "mouth", "kerneljammouth"),
    ("jam",    "deep",  "kerneljamdeep"),
]
KIND = "vanleer"        # the second-order, FCT-limited adjoint; "none" is the first-order one


def _read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
            for k in rows[0]}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID_C, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _shade_inverse(ax, x, inverse):
    """Shade the contiguous runs where the composition is inverse-sorted."""
    inv = np.asarray(inverse) > 0.5
    if not inv.any():
        return
    edges = np.flatnonzero(np.diff(np.r_[0, inv.view(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(x[a], x[min(b, len(x) - 1)], color=SHADE, alpha=0.10, lw=0, zorder=1)


def _write_csv(path, header, cols):
    """Write equal-length columns to CSV.  Every figure here gets a sibling CSV of exactly the
    arrays it draws, so it can be redrawn -- or re-plotted in another tool -- from disk alone."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in zip(*cols):
            w.writerow(["%.10g" % v for v in row])


def fig_h(pr, label, out):
    """h_sim vs h_theory, and the residual."""
    occ = pr["occupied"] > 0.5
    x, hs, ht = pr["x_um"][occ], pr["h_sim"][occ], pr["h_theory"][occ]
    hw = pr["h_weak"][occ] if "h_weak" in pr else None

    fig, ax = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.12))
    fig.patch.set_facecolor(SURFACE)
    for a in ax:
        _style(a)
    _shade_inverse(ax[0], x, pr["inverse"][occ]); _shade_inverse(ax[1], x, pr["inverse"][occ])

    ax[0].axhline(0.0, color=MUTED, lw=0.8, zorder=2)
    ax[0].plot(x, hs, color=C_SIM, lw=2.1, zorder=4, label=r"$h_{\rm sim}$  (direct numerics)")
    ax[0].plot(x, ht, color=C_TH, lw=1.5, ls="--", zorder=5,
               label=r"$h_{\rm theory}$  (adjoint kernel, Duhamel)")
    ax[0].set_ylim(-1.15, 1.15)
    ax[0].set_ylabel(r"$h = 2f_1-1$", color=INK, fontsize=11)
    ax[0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax[0].set_title(label, color=INK, fontsize=11, loc="left")

    # h_weak is drawn only when it FITS.  The weak-selection form drops the (1-h^2) factor, so it
    # is fine while |h| stays small and diverges as |h| -> 1: measured max |h_weak| is 0.63 at
    # dilute T=96, 4.00 at dilute T=500, ~29 at jammed T=2000.
    hwmax = float("nan")
    if hw is not None and np.isfinite(hw).any():
        hwmax = float(np.nanmax(np.abs(hw)))
        if hwmax <= 1.10:
            ax[0].plot(x, hw, color=C_WEAK, lw=1.3, ls=":", zorder=3,
                       label=r"$h_{\rm weak}=\frac{1}{2}A_K$  (weak selection)")
            ax[0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    res = np.abs(ht - hs)
    ax[1].semilogy(x, np.maximum(res, 1e-16), color=C_TH, lw=1.3, zorder=4)
    ax[1].set_ylabel(r"residual $|\Delta h|$", color=INK2, fontsize=10)
    ax[1].set_xlabel(r"$x$  [$\mu$m]", color=INK, fontsize=11)

    # NO explanatory text is drawn into the figure -- it belongs in the LaTeX caption, written by
    # the author.  The numbers a caption would need are RETURNED instead and printed by main().
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    hw_col = hw if hw is not None else np.full_like(hs, np.nan)
    _write_csv(os.path.splitext(out)[0] + ".csv",
               ["x_um", "h_sim", "h_theory", "h_weak", "residual_abs", "inverse"],
               [x, hs, ht, hw_col, res, pr["inverse"][occ]])
    return dict(res_max=float(res.max()), res_mean=float(res.mean()),
                h_weak_max=hwmax, h_sim_absmax=float(np.abs(hs).max()),
                h_weak_plotted=bool(np.isfinite(hwmax) and hwmax <= 1.10),
                inv_frac=float(np.mean(pr["inverse"][occ] > 0.5)))


def fig_criterion(pr, label, out):
    """A+ vs A-, and the criterion against what it predicts."""
    occ = pr["occupied"] > 0.5
    x = pr["x_um"][occ]
    Ap, Am = pr["A_plus"][occ], pr["A_minus"][occ]
    sig, hs = pr["sigma_T"][occ], pr["h_sim"][occ]

    fig, ax = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.12))
    fig.patch.set_facecolor(SURFACE)
    for a in ax:
        _style(a); _shade_inverse(a, x, pr["inverse"][occ])

    ax[0].plot(x, Ap, color=C_PLUS, lw=1.8, zorder=4, label=r"$A_+$  reinforcing")
    ax[0].plot(x, Am, color=C_MINUS, lw=1.8, zorder=4, label=r"$A_-$  opposing")
    ax[0].set_ylabel("exposure", color=INK, fontsize=11)
    ax[0].legend(loc="best", fontsize=9, framealpha=0.95)
    ax[0].set_title(label, color=INK, fontsize=11, loc="left")

    # The lower panel VERIFIES the exposure identity, it does not compare two independent things.
    # sigma_T*h = A+ - A- is an identity, so plotting (A- - A+) against sigma_T*h would draw a
    # curve against its own negative (corr = -1.000000) and read as disagreement.  Both sides of
    # the identity go on the same axis instead; the visible gap IS the residual.
    # CONVENTION -- read this before "correcting" the missing 1/2.  BOTH forms are right:
    #   manuscript (continuum):  sigma_T h = 1/2 (A+ - A-),  A+- = int int K (1-h^2)|s| dy dt
    #   here      (discrete)  :  sigma_T h =      A+ - A- ,  A+- built from D_SEL = h^r - h
    # D_SEL is the exact per-step reaction increment and ALREADY equals 1/2 (1-h^2) s dt, so the
    # discrete A+- are the continuum ones halved.  Measured on this run: ratio 2.006/2.001 (mouth)
    # and 2.000/2.005 (deep).  Plugging these A+- into the manuscript's formula misses by ~0.5 --
    # which is a unit error, NOT a defect in the derivation.  Same trap as CLAUDE.md 14.3.3, where
    # the 2*B_0 criterion was carried across the same convention change.
    ax[1].axhline(0.0, color=MUTED, lw=0.8, zorder=2)
    ax[1].plot(x, sig * hs, color=C_SIM, lw=2.0, zorder=4, label=r"$\sigma_T h$  (observed)")
    ax[1].plot(x, Ap - Am, color=C_TH, lw=1.4, ls="--", zorder=5,
               label=r"$A_+-A_-$  (exposure decomposition)")
    ax[1].set_ylabel(r"$\sigma_T h$", color=INK2, fontsize=10)
    ax[1].set_xlabel(r"$x$  [$\mu$m]", color=INK, fontsize=11)
    ax[1].legend(loc="best", fontsize=8.5, framealpha=0.95)

    ident = float(np.abs(sig * hs - (Ap - Am)).max())
    agree = float(np.mean((Am > Ap) == (sig * hs < 0)) * 100.0)

    # NO explanatory text is drawn into the figure -- it belongs in the LaTeX caption.  The
    # numbers are RETURNED and printed by main() so the caption can be written from them.
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    _write_csv(os.path.splitext(out)[0] + ".csv",
               ["x_um", "A_plus", "A_minus", "sigma_T", "h_sim", "sigmaT_h", "Aplus_minus_Aminus",
                "inverse"],
               [x, Ap, Am, sig, hs, sig * hs, Ap - Am, pr["inverse"][occ]])
    inv = pr["inverse"][occ] > 0.5
    return dict(identity_resid=ident, criterion_agree_pct=agree,
                inv_frac=float(np.mean(inv)),
                n_inv_mouth=int(np.sum(inv & (sig < 0))),
                n_inv_deep=int(np.sum(inv & (sig > 0))), n_occ=int(occ.sum()))


def fig_kernel(z, traj, front, target, label, out):
    """K(y,t) heatmap for one target, with ybar(t) and the n* front."""
    anc = np.asarray(z[f"anc_{target}_{KIND}"])          # (nb, N)
    t_blk, x = np.asarray(z["t_blk"]), np.asarray(z["x"])
    tgt_x = float(np.asarray(z["x"])[int(np.asarray(z["targets"])[0 if target == "mouth" else 1])])

    K = np.maximum(anc, 0.0)
    pos = K[K > 0]
    vmax = float(pos.max()) if pos.size else 1.0
    vmin = max(vmax * 1e-6, float(np.percentile(pos, 5)) if pos.size else 1e-12)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    fig.patch.set_facecolor(SURFACE)
    im = ax.pcolormesh(x, t_blk, np.maximum(K, vmin), cmap="magma",
                       norm=LogNorm(vmin=vmin, vmax=vmax), shading="auto", zorder=1)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(r"ancestry kernel  $K(y,t\mid x,T)$", color=INK2, fontsize=10)
    cb.ax.tick_params(colors=INK2, labelsize=8)

    tt, yb = traj["t_h"], traj[f"ybar_{target}_{KIND}"]
    ok = np.isfinite(yb)
    ax.plot(yb[ok], tt[ok], color=C_TRAJ, lw=2.0, zorder=4,
            label=r"mean ancestral trajectory $\bar y(t)$")
    if front is not None:
        ft, fx = front
        ax.plot(fx, ft, color=C_FRONT, lw=1.4, ls="--", alpha=0.85, zorder=3,
                label=r"$n^*$ front")
    ax.plot([tgt_x], [t_blk[-1]], marker="v", ms=10, color=C_TRAJ, mec=SURFACE, mew=1.4,
            ls="none", zorder=6, clip_on=False, label="target $(x,T)$")

    # Zoom to where the kernel actually lives.  K is strongly localised -- the jammed mouth's
    # ancestry never leaves y < 200 um of a 1000 um cavity -- so the full domain spends most of
    # its width on exact zeros and compresses the structure into a sliver.
    colmax = K.max(axis=0)
    sup = np.flatnonzero(colmax > 1e-4 * vmax)
    if sup.size:
        lo, hi = float(x[sup[0]]), float(x[sup[-1]])
        pad = max(0.08 * (hi - lo), 5.0)
        ax.set_xlim(max(0.0, lo - pad), min(float(x[-1]), hi + pad))

    ax.set_xlabel(r"ancestral position  $y$  [$\mu$m]", color=INK, fontsize=11)
    ax.set_ylabel(r"time  $t$  [h]", color=INK, fontsize=11)
    ax.set_title(f"{label}   -- ancestry of the {target}", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=9)
    leg = ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # The FULL field, not just the zoomed window: K is on disk otherwise only in fields.npz,
    # which is binary.  Long format (t_h, y_um, K).  The overlays are already CSV at source --
    # ybar(t) in the adjoint run's trajectory.csv, the n* front in the base run's timeseries.csv.
    tt, yy = np.meshgrid(t_blk, x, indexing="ij")
    _write_csv(os.path.splitext(out)[0] + ".csv", ["t_h", "y_um", "K"],
               [tt.ravel(), yy.ravel(), anc.ravel()])


def nstar_front(base_dir):
    """(t, x_nstar) from the matching base run, or None.  fields.npz has only n_final."""
    path = os.path.join(HERE, base_dir, "timeseries.csv")
    if not os.path.exists(path):
        return None
    d = _read_csv(path)
    if "x_nstar_um" not in d:
        return None
    ok = np.isfinite(d["x_nstar_um"])
    return d["t_h"][ok], d["x_nstar_um"][ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--only", default=None,
                    help="comma-separated figure stems to rebuild (default: all eight)")
    a = ap.parse_args()
    outdir = os.path.abspath(os.path.join(HERE, a.outdir))
    os.makedirs(outdir, exist_ok=True)
    want = None if a.only is None else {s.strip() for s in a.only.split(",") if s.strip()}

    cache = {}
    caption_facts = []
    for case, panel, stem in PANELS:
        if want is not None and stem not in want:
            continue
        if case not in cache:
            adir, bdir, label = CASES[case]
            A = os.path.join(HERE, adir)
            if not os.path.isdir(A):
                print(f"  MISSING adjoint run: {adir}"); cache[case] = None; continue
            front = nstar_front(bdir)
            if front is None:
                print(f"  note: no n* front for {case} (missing {bdir}/timeseries.csv)")
            cache[case] = (_read_csv(os.path.join(A, "profiles.csv")),
                           _read_csv(os.path.join(A, "trajectory.csv")),
                           np.load(os.path.join(A, "fields.npz")), front, label)
        if cache[case] is None:
            continue
        pr, traj, z, front, label = cache[case]
        p = os.path.join(outdir, stem + ".png")
        facts = None
        if panel == "h":
            facts = fig_h(pr, label, p)
        elif panel == "crit":
            facts = fig_criterion(pr, label, p)
        else:
            fig_kernel(z, traj, front, panel, label, p)
        print(f"  wrote {stem}.png   [{case}]")
        if facts:
            caption_facts.append((stem, case, facts))

    if caption_facts:
        print("\n" + "=" * 78)
        print("NUMBERS FOR THE CAPTIONS  (deliberately NOT drawn into the figures)")
        print("=" * 78)
        for stem, case, f in caption_facts:
            print(f"\n{stem}.png   [{CASES[case][2]}]")
            if "res_max" in f:
                print("  residual |h_theory - h_sim|: max %.3e, mean over occupied %.3e"
                      % (f["res_max"], f["res_mean"]))
                print("  inverse-sorted: %.1f%% of occupied cells" % (100 * f["inv_frac"]))
                if np.isfinite(f["h_weak_max"]):
                    print("  max |h_weak| = %.2f  vs  max |h_sim| = %.2f   (%s)"
                          % (f["h_weak_max"], f["h_sim_absmax"],
                             "plotted -- in bounds" if f["h_weak_plotted"]
                             else "NOT plotted -- off scale, outside |h| <= 1"))
                else:
                    print("  h_weak unavailable (run predates the column)")
            else:
                print("  identity |sigma_T h - (A+ - A-)|: max %.3e" % f["identity_resid"])
                print("  sign criterion agrees in %.2f%% of %d occupied cells"
                      % (f["criterion_agree_pct"], f["n_occ"]))
                print("  inverse-sorted: %.1f%% of occupied  (%d mouth, %d deep)"
                      % (100 * f["inv_frac"], f["n_inv_mouth"], f["n_inv_deep"]))
        print("\nNote: A+- here are the DISCRETE ones (built from Delta h_sel, which already")
        print("carries the 1/2).  The manuscript's continuum A+- are exactly 2x these, which is")
        print("why its identity reads sigma_T h = 1/2 (A+ - A-).  Both are correct.")


if __name__ == "__main__":
    main()
