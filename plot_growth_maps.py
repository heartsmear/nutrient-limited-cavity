"""
plot_growth_maps.py -- g1, g2 and s = g1 - g2 as space-time heat maps.  Simulates nothing.

    py -3.11 src2/plot_growth_maps.py run_n0-15_L1000_dx0.5_T500
    py -3.11 src2/plot_growth_maps.py run_n0-1000_L1000_dx0.5_T2000 --tmax 2000
    py -3.11 src2/plot_growth_maps.py <rundir> --out somewhere.png --no-contours

Three panels per run, all on the same (t, y) axes:

    g1(n(x,t))      Monod growth of strain 1, the scavenger   [1/h]
    g2(n(x,t))      Monod growth of strain 2, the glutton     [1/h]
    s = g1 - g2     the selection coefficient                 [1/h]

VALUES ARE MEANT TO BE READ OFF, which a bare heat map does not allow -- a colour maps to a value
only via a bar the eye has to walk back to.  So each panel carries LABELLED CONTOURS on top of the
image, and the level lists are chosen for the quantity rather than by an algorithm:

  * g1, g2 get contours at fractions of that strain's own mu_i, so a line reads directly as "this
    strain is at half its maximum rate here".
  * s gets a symmetric set plus a HEAVY ZERO CONTOUR, which is the n* front -- the single most
    important curve in the model, and the one every mechanism argument refers to.

THE MONOD LAW VERSUS THE REALIZED RATE.  The solver multiplies growth by the volume-filling
factor q(rho) (`closures.growth_space_factor`), so the realized per-capita rate is q*g_i.  For the
two reference runs this distinction turns out to be VACUOUS, and the caption says so rather than
hedging: q ramps down only above rho_c = 0.95, and the n0=1000 "jammed" run peaks at rho = 0.659
-- jammed relative to rho_jam = 0.64, but far below the growth ceiling.  So q == 1 identically in
both, and the contour values ARE the realized rates.

  (An earlier draft of this file asserted the opposite -- "in a jammed cavity q falls well below
  1, so the realized rates are smaller than these contours".  That confused rho_jam with rho_c.
  q is now computed for every run and the caption reports which case it is, so the claim cannot
  drift from the data again.)

Where q < 1 does occur, note that n* is defined by g1(n*) = g2(n*), so q cancels there: the zero
contour of s is the zero contour of q*s at every density.  q scales the magnitude of selection,
never its sign or the position of the front.  `--show-q` adds a fourth panel of q.

n IS READ FROM THE DUMP, not recomputed.  `full_dump.csv` carries the realized nutrient field, so
these maps are the growth rates the run actually experienced.  Runs dumped before `n` was a column
are rejected with a message rather than silently reconstructed.

MEMORY.  Dumps reach ~110 MB; the file is streamed frame by frame and only n(x,t) is retained.
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
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                        # noqa: E402
import closures as C                                 # noqa: E402
import plot_fst as P                                 # noqa: E402  (n0_from_name only)

SURFACE, INK, INK2, MUTED, GRIDC = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"


def read_columns(path, cols, tmax=None):
    """Stream full_dump.csv once, returning t, x and {name: array of shape (cells, frames)}.

    One streaming implementation for every consumer -- the dumps reach ~110 MB, so a second pass
    per extra column is the difference between seconds and tens of seconds, and two independent
    parsers would be two places for the frame-boundary logic to drift.
    """
    cols = list(cols)
    with open(path) as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        for need in ["t_h", "x_um"] + cols:
            if need not in hdr:
                sys.exit(f"{path}: no '{need}' column.  This dump predates it; re-run\n"
                         f"  py -3.11 src2/figures.py --n0 <n0> --L <L> --dx <dx> --T <T> --dump")
        it, ix = hdr.index("t_h"), hdr.index("x_um")
        idx = [hdr.index(c) for c in cols]
        ts, out, xs = [], [[] for _ in cols], []
        cur, buf, xcol = None, [[] for _ in cols], []
        for row in rd:
            t = float(row[it])
            if t != cur:
                if cur is not None and buf[0]:
                    ts.append(cur)
                    for a, b in zip(out, buf):
                        a.append(b)
                    if not xs:
                        xs = xcol
                cur, buf, xcol = t, [[] for _ in cols], []
                if tmax is not None and t > tmax:
                    break
            for b, j in zip(buf, idx):
                b.append(float(row[j]))
            if not xs:
                xcol.append(float(row[ix]))
        if cur is not None and buf[0] and (tmax is None or cur <= tmax):
            ts.append(cur)
            for a, b in zip(out, buf):
                a.append(b)
            if not xs:
                xs = xcol
    return np.array(ts), np.array(xs), {c: np.array(a).T for c, a in zip(cols, out)}


def read_fields(path, tmax=None):
    """t, x, n, rho -- the pair this module needs, via `read_columns`."""
    t, x, d = read_columns(path, ("n", "rho"), tmax=tmax)
    return t, x, d["n"], d["rho"]


def levels_for(vmax, mu):
    """Contour levels at readable fractions of the strain's own maximum rate."""
    fr = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    lv = fr * mu
    return lv[(lv > 0) & (lv < vmax * 0.999)]


def panel(ax, t, x, Z, title, cmap, norm=None, levels=None, cbar_label="",
          contours=True, zero_contour=False, fmt="%.3f", log_t=True):
    im = ax.pcolormesh(t, x, Z, cmap=cmap, shading="nearest", norm=norm)
    if log_t:
        # Nutrient is drawn down within ~50 h of a 500-2000 h run, so on a linear axis ~90% of
        # the panel is depleted bulk and every feature is crushed into a thin L at the corner.
        ax.set_xscale("log")
        pos = t[t > 0]
        if pos.size:
            ax.set_xlim(pos[0], t[-1])
    if contours and levels is not None and len(levels):
        cs = ax.contour(t, x, Z, levels=levels, colors="k", linewidths=0.6, alpha=.65)
        ax.clabel(cs, fmt=fmt, fontsize=7, inline=True, inline_spacing=3)
    if zero_contour:
        cz = ax.contour(t, x, Z, levels=[0.0], colors="k", linewidths=2.0)
        ax.clabel(cz, fmt={0.0: r"$s=0$  ($n^\ast$)"}, fontsize=8.5, inline=True)
    ax.set_xlabel(r"$t$ [h]", color=INK2)
    ax.set_ylabel(r"$x$ [$\mu$m]", color=INK2)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=9)
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label(cbar_label, color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    if levels is not None and len(levels):
        cb.add_lines(cs if contours else cz)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir")
    ap.add_argument("--out", default=None, help="default: <rundir>/growth_maps.png")
    ap.add_argument("--tmax", type=float, default=None, help="truncate the time axis")
    ap.add_argument("--no-contours", dest="contours", action="store_false")
    ap.add_argument("--linear-t", dest="log_t", action="store_false",
                    help="linear time axis; the default is log, because the nutrient is "
                         "drawn down in the first tens of hours of a 500-2000 h run")
    ap.add_argument("--show-q", action="store_true",
                    help="add a fourth panel with the volume-filling factor q(rho), the factor "
                         "by which the realized rate falls below the Monod value")
    a = ap.parse_args()

    d = a.rundir if os.path.isabs(a.rundir) else os.path.join(HERE, a.rundir)
    dump = os.path.join(d, "full_dump.csv")
    if not os.path.exists(dump):
        sys.exit(f"no full_dump.csv in {d}; re-run figures.py with --dump")
    name = os.path.basename(os.path.normpath(d))
    n0 = P.n0_from_name(d)
    p = config.params(n0 if n0 is not None else config.N0_DILUTE)
    nstar = float(C.monod_crossover(p))

    t, x, n, rho = read_fields(dump, tmax=a.tmax)
    g1 = np.asarray(C.g1(n, p))
    g2 = np.asarray(C.g2(n, p))
    s = g1 - g2
    q = np.asarray(C.growth_space_factor(rho, p))
    q_unity = bool(np.all(q == 1.0))

    print(f"{name}   n0={float(p.n0):g} uM   n*={nstar:.4f} uM   "
          f"{len(x)} cells x {len(t)} frames   t=0..{t[-1]:g} h")
    print(f"  mu1={p.mu1:g}  Ks1={p.Ks1:g}   mu2={p.mu2:g}  Ks2={p.Ks2:g}")
    print(f"{'quantity':>10}{'min':>12}{'max':>12}{'at t=0':>12}{'at t=T':>12}")
    for lab, Z in (("n [uM]", n), ("g1 [1/h]", g1), ("g2 [1/h]", g2), ("s [1/h]", s)):
        print(f"{lab:>10}{Z.min():>12.4g}{Z.max():>12.4g}"
              f"{Z[:, 0].max():>12.4g}{Z[:, -1].max():>12.4g}")
    frac = float((s > 0).mean())
    print(f"  fraction of the (x,t) plane with s>0 (strain 1 favoured): {frac:.4f}")

    ncol = 4 if a.show_q else 3
    fig, ax = plt.subplots(1, ncol, figsize=(6.0 * ncol, 5.4), layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    # g1 and g2 share one colour scale, so the two panels are comparable by eye -- on
    # independent scales strain 1 would look as fast as strain 2 despite mu1/mu2 = 0.27.
    gmax = max(float(g1.max()), float(g2.max()))
    im0 = panel(ax[0], t, x, g1, r"(a) $g_1(n)$  strain 1, scavenger", "viridis",
                levels=levels_for(float(g1.max()), p.mu1),
                cbar_label=r"$g_1$ [h$^{-1}$]  (scale shared with b)", contours=a.contours,
                log_t=a.log_t)
    im0.set_clim(0, gmax)
    im1 = panel(ax[1], t, x, g2, r"(b) $g_2(n)$  strain 2, glutton", "viridis",
                levels=levels_for(float(g2.max()), p.mu2),
                cbar_label=r"$g_2$ [h$^{-1}$]  (scale shared with a)", contours=a.contours,
                log_t=a.log_t)
    im1.set_clim(0, gmax)

    smin, smax = float(s.min()), float(s.max())
    fr = np.array([0.03, 0.1, 0.25, 0.5, 0.75])
    lv = np.sort(np.concatenate([smin * fr, smax * fr]))
    panel(ax[2], t, x, s, r"(c) $s=g_1-g_2$   $s>0$: strain 1 favoured", "coolwarm_r",
          norm=TwoSlopeNorm(0.0, min(smin, -1e-30), max(smax, 1e-30)), levels=lv,
          cbar_label=r"$s$ [h$^{-1}$]", contours=a.contours, zero_contour=True,
          log_t=a.log_t)

    if a.show_q:
        im3 = panel(ax[3], t, x, q,
                    r"(d) $q(\rho)$  volume-filling factor"
                    + (r"  -- $\equiv 1$ here" if q_unity else ""), "cividis",
                    levels=np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
                    cbar_label=r"$q$  (realized rate $=q\,g_i$)", contours=a.contours,
                    fmt="%.2f", log_t=a.log_t)
        im3.set_clim(0, 1)

    for aa in ax:
        for sp in ("top", "right"):
            aa.spines[sp].set_visible(False)

    fig.suptitle(f"Growth rates and selection over space and time   {name}   "
                 f"$n_0$={float(p.n0):g} " + r"$\mu$M, $n^\ast$=" + f"{nstar:.3f} " + r"$\mu$M",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
    # The realized rate is q*g_i, but q == 1 whenever rho stays below rho_c = 0.95 -- which BOTH
    # reference runs do (the n0=1000 "jammed" case peaks at rho = 0.659, jammed relative to
    # rho_jam = 0.64 but far below the growth ceiling).  Saying "q discounts these rates" would
    # be false here, so the caption reports which case this run is.
    qtxt = (r"$q(\rho)\equiv 1$ throughout this run, so the contour values ARE the realized "
            r"per-capita rates." if q_unity else
            rf"$q(\rho)$ falls to {q.min():.3f} here, so realized rates are $q\,g_i$ -- lower "
            r"than the contours; $q$ scales magnitude but not sign, so $s=0$ is unaffected.")
    dtf = np.diff(t)
    fig.text(0.5, 0.030,
             r"Monod laws evaluated on the realized $n(x,t)$ from full_dump.csv.  "
             r"Contours are labelled in h$^{-1}$; the heavy line in (c) is $s=0$, i.e. $n=n^\ast$.  "
             + qtxt, ha="center", fontsize=8.5, color=MUTED)
    fig.text(0.5, 0.006,
             f"Dump sampling: {len(t)} frames, first at t = {t[1] if len(t) > 1 else t[0]:.4g} h, "
             f"spacing {dtf.min():.4g}-{dtf.max():.4g} h.  "
             "Structure faster than the spacing is not resolved.",
             ha="center", fontsize=8, color=MUTED)

    out = a.out or os.path.join(d, "growth_maps.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
