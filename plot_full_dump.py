"""
plot_full_dump.py -- plot everything in a `full_dump.csv` written by grounded_dself_figure.py.

NEW FILE.  Standalone: reads only the CSV, imports none of the solver stack, runs nothing.

WHAT IT DRAWS
  For EVERY field in the dump, the full spatial profile at EVERY stored time step, overlaid in
  one panel and coloured by time.  One sheet, `profiles_all.png`, with a shared time colourbar.

  Overlaying 10^4 curves is done with a LineCollection rather than 10^4 `plot` calls -- the
  difference is minutes versus seconds, and it is the only reason "every time step" is drawable
  at all.  `--stride k` thins the DRAWN curves (the data is untouched); the panel title always
  reports how many of how many are shown, so a thinned plot can never be mistaken for a full one.

THE n* CROSSOVER is overlaid on every panel (`--no-nstar` to suppress):
  * the shaded band  = the range swept by the n* front x_{n*}(t) over the whole run,
  * the dashed line  = its final position,
  * on the `n` panel = a horizontal line at n* itself, in the same units as the y axis.
  n* is read OUT OF THE DUMP (from its own g1/g2 columns), never hard-coded, so it is correct for
  whatever parameter set produced the file.  Built-in consistency check, visible in the figure:
  the zero crossing of `s` must sit exactly on the dashed front line, since n = n* <=> s = 0.

AXIS SCALES are chosen per field from the data, not hard-coded, because the dump mixes quantities
spanning 15 decades (n, rho) with sign-changing ones (u, J, the d_t f1 budget):
  * strictly positive with a wide dynamic range  -> log y, floored at the smallest positive value
  * changes sign                                 -> linear y with a zero line
  * f1                                           -> linear, pinned to [0,1]

    py -3.11 src/plot_full_dump.py <dir-or-csv>
    py -3.11 src/plot_full_dump.py <dir> --stride 20          # thin the drawn curves
    py -3.11 src/plot_full_dump.py <dir> --fields rho1,rho2,n
    py -3.11 src/plot_full_dump.py <dir> --frames --frames-stride 200   # one PNG per time step
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.collections import LineCollection      # noqa: E402
from matplotlib.colors import Normalize                # noqa: E402
from matplotlib.cm import ScalarMappable               # noqa: E402

# Nicer axis labels than the raw column names; anything absent falls back to the column name.
LABEL = {
    "rho1": r"$\rho_1$  (scavenger)",   "rho2": r"$\rho_2$  (glutton)",
    "rho": r"$\rho$  (total)",          "f1": r"$f_1$",
    "h": r"$h=2f_1-1$",                 "n": r"$n$  [$\mu$M]",
    "g1": r"$g_1(n)$  [h$^{-1}$]",      "g2": r"$g_2(n)$  [h$^{-1}$]",
    "s": r"$s=g_1-g_2$  [h$^{-1}$]",
    "Dcol": r"$D_{\rm col}(\rho)$  [$\mu$m$^2$/h]",
    "Dself": r"$D_{\rm self}(\rho)$  [$\mu$m$^2$/h]",
    "u": r"$u=J/\rho$  [$\mu$m/h]",     "Jtot": r"$J_{\rm tot}$",
    "J1": r"$J_1$",                     "J2": r"$J_2$",
    "ADV": r"$\partial_tf_1$: ADV",     "SELF": r"$\partial_tf_1$: SELF",
    "SEL": r"$\partial_tf_1$: SEL",
    # src/ names kept so this script still reads OLD dumps; src2 renamed these and dropped the
    # chemotaxis ones entirely.
    "D_col": r"$D_{\rm col}(\rho)$  [$\mu$m$^2$/h]",
    "D_self": r"$D_{\rm self}(\rho)$  [$\mu$m$^2$/h]",
    "J_tot": r"$J_{\rm tot}$",          "v_drift": r"$v_{\rm drift}$  [$\mu$m/h]",
    "dtf1_ADV": r"$\partial_tf_1$: ADV",    "dtf1_SELF": r"$\partial_tf_1$: SELF",
    "dtf1_SEL": r"$\partial_tf_1$: SEL",    "dtf1_DRIFT": r"$\partial_tf_1$: DRIFT",
}

# Widest log axis allowed, in decades below the field's peak.  See scale_for().
MAX_DECADES = 12.0


def nstar_from_dump(fields):
    """The Monod crossover n*, read OUT OF THE DUMP rather than hard-coded.

    s = g1 - g2 is a function of n alone, and (for a crossing pair) decreases monotonically
    through zero, so interpolating n against s at s = 0 recovers n* exactly -- for whatever
    parameter set produced the file.  Hard-coding 5.2526 would silently mislabel any run with a
    different pair, which is the trap `cc.inverse_decomposition` fell into with NSTAR=1.5.

    Returns None if the run never spans the crossover (then there is nothing to draw).
    """
    if "s" not in fields or "n" not in fields:
        return None
    s = fields["s"].ravel()
    n = fields["n"].ravel()
    ok = np.isfinite(s) & np.isfinite(n) & (n > 0.0)
    s, n = s[ok], n[ok]
    if s.size == 0 or s.max() <= 0.0 or s.min() >= 0.0:
        return None                      # no sign change: n* is outside the realized range

    # Scan the sign change in n, NOT in s.  s(n) = mu1 n/(K1+n) - mu2 n/(K2+n) has TWO roots:
    # the crossover n*, and the trivial n = 0 where both Monod terms vanish (CLAUDE.md Sec 12.6).
    # Interpolating n against s at s = 0 therefore locks onto whichever root has more samples --
    # and a run whose interior is starved to n ~ 1e-300 has overwhelmingly many points at the
    # trivial root, returning n* = 0.  Scanning upward in n is immune: s > 0 for all
    # 0 < n < n* and s < 0 for n > n*, so the first sign change encountered IS the crossover.
    order = np.argsort(n)
    n_s, s_s = n[order], s[order]
    neg = np.where(s_s < 0.0)[0]
    if neg.size == 0 or neg[0] == 0:
        return None
    k = int(neg[0])
    n1, n2, s1, s2 = n_s[k - 1], n_s[k], s_s[k - 1], s_s[k]
    if s1 == s2:
        return float(n2)
    return float(n1 + (0.0 - s1) * (n2 - n1) / (s2 - s1))


def nstar_front(x, N, nstar):
    """x where n crosses n*, per frame, LINEARLY INTERPOLATED.

    Same construction as grounded_dself_figure.nstar_front: snapping to the cell centre
    quantises the front to multiples of dx and turns a front creeping by a fraction of a cell
    into a spurious dx-amplitude sawtooth.  NaN where the whole profile is on one side.
    """
    out = np.full(N.shape[0], np.nan)
    for j in range(N.shape[0]):
        nu = N[j]
        b = np.where(nu <= nstar)[0]
        if b.size == 0 or b[0] == 0:
            continue
        k = int(b[0])
        n1, n2 = nu[k - 1], nu[k]
        frac = (n1 - nstar) / (n1 - n2) if n1 != n2 else 0.0
        out[j] = x[k - 1] + frac * (x[k] - x[k - 1])
    return out


def resolve(path):
    """Accept either the CSV itself or the directory containing it."""
    if os.path.isdir(path):
        path = os.path.join(path, "full_dump.csv")
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}\n(expected a full_dump.csv, or a directory holding one)")
    return path


def load(path):
    """Long CSV -> (x, t, {field: array of shape (n_t, n_x)}).

    Rows are lexsorted by (t, x) before reshaping rather than trusting the writer's ordering:
    the reshape is silent if the order is wrong, and a transposed field would look like a
    plausible physical result.
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = list(df.columns)
    for req in ("t_h", "x_um"):
        if req not in cols:
            sys.exit(f"{path} has no '{req}' column -- is this a full_dump.csv?")
    t_all = df["t_h"].to_numpy()
    x_all = df["x_um"].to_numpy()
    order = np.lexsort((x_all, t_all))
    x = np.unique(x_all)
    t = np.unique(t_all)
    if x.size * t.size != len(df):
        sys.exit(f"{path} is not a complete rectangular grid: {len(df)} rows but "
                 f"{t.size} times x {x.size} positions = {t.size * x.size}")
    fields = {c: df[c].to_numpy()[order].reshape(t.size, x.size)
              for c in cols if c not in ("t_h", "x_um")}
    return x, t, fields


def scale_for(name, Z):
    """('log'|'linear', ylim) decided from the data."""
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        return "linear", None
    lo, hi = float(finite.min()), float(finite.max())
    if name == "f1":
        return "linear", (-0.02, 1.02)
    if lo < 0.0:                                    # sign-changing: linear, symmetric-ish
        return "linear", None
    pos = finite[finite > 0.0]
    # decades = log10(hi) - log10(min positive).  Done in log space on purpose: the dump reaches
    # ~1e-300 in starved/empty cells, and the direct ratio hi/min overflows to inf there.
    if pos.size and hi > 0.0 and (np.log10(hi) - np.log10(pos.min())) > 3.0:
        # Floor the axis at MAX_DECADES below the peak.  Using the true positive minimum instead
        # gives a ~300-decade axis on which every physical feature is squashed into the top tick:
        # the tail below this floor is numerical underfill of empty cells, not structure.
        floor = max(float(pos.min()), hi * 10.0 ** (-MAX_DECADES))
        return "log", (floor * 0.5, hi * 2.0)
    return "linear", None


def mark_nstar(ax, nstar, front, name, legend=False):
    """Overlay the n* crossover: the swept front band + its final position on every panel, and
    the threshold itself on the `n` panel (where the y axis is in the same units as n*)."""
    if nstar is None:
        return
    if name == "n":
        ax.axhline(nstar, color="crimson", ls=":", lw=1.4, zorder=4,
                   label=rf"$n^\ast$ = {nstar:.4g} $\mu$M")
    if front is not None and np.isfinite(front).any():
        f = front[np.isfinite(front)]
        lo, hi = float(f.min()), float(f.max())
        if hi > lo:
            ax.axvspan(lo, hi, color="0.55", alpha=0.18, lw=0, zorder=0,
                       label=r"$n^\ast$ front sweep")
        ax.axvline(f[-1], color="k", ls="--", lw=1.1, zorder=4,
                   label=rf"$n^\ast$ front, final ({f[-1]:.0f} $\mu$m)")
    if legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.85)


def draw_panel(ax, x, t, Z, name, stride, norm, cmap, nstar=None, front=None, legend=False):
    rows = np.arange(0, Z.shape[0], stride)
    scale, ylim = scale_for(name, Z)
    Zp = Z
    if scale == "log":                              # keep zeros off a log axis without dropping rows
        floor = ylim[0]
        Zp = np.where(Z > 0.0, Z, floor)
    segs = [np.column_stack([x, Zp[j]]) for j in rows]
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.6, alpha=0.85)
    lc.set_array(t[rows])
    ax.add_collection(lc)
    ax.set_yscale(scale)
    ax.set_xlim(x.min(), x.max())
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        finite = Zp[np.isfinite(Zp)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            pad = 0.05 * (hi - lo) if hi > lo else (abs(hi) or 1.0) * 0.05
            ax.set_ylim(lo - pad, hi + pad)
        if finite.size and finite.min() < 0.0 < finite.max():
            ax.axhline(0.0, color="0.6", lw=0.7, zorder=0)
    if scale == "linear":
        # Kill the shared-offset tick format.  Left on, a panel whose values sit at 15.0 +- 1e-3
        # is labelled "+1.4999e1" with ticks running -0.001..0.001, which reads as a field that
        # changes sign -- the opposite of what the data does.
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
    ax.set_xlabel(r"$x$ [$\mu$m]")
    ax.set_ylabel(LABEL.get(name, name), fontsize=9)
    ax.set_title(f"{name}   [{rows.size}/{Z.shape[0]} steps]", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.2)
    mark_nstar(ax, nstar, front, name, legend=legend)


def figure_all(x, t, fields, names, stride, path, subtitle="", nstar=None, front=None):
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    # constrained_layout, not tight_layout + bbox_inches: with a figure-level colorbar spanning
    # every axes, tight_layout is silently skipped and the panel titles land on the row above.
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow), squeeze=False,
                             constrained_layout=True)
    norm = Normalize(float(t.min()), float(t.max()))
    cmap = plt.get_cmap("viridis")
    for k, name in enumerate(names):
        # legend on the first panel and on `n` only -- repeating it 20 times buys nothing and
        # covers data.
        draw_panel(axes[k // ncol][k % ncol], x, t, fields[name], name, stride, norm, cmap,
                   nstar=nstar, front=front, legend=(k == 0 or name == "n"))
    for k in range(len(names), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.set_label(r"$t$ [h]")
    fig.suptitle("Full spatial profile at every stored time step" + subtitle,
                 fontweight="bold", fontsize=14)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_frames(x, t, fields, names, outdir, stride, nstar=None, front=None):
    """One PNG per time step: all fields, single profile each.  Fixed axes across frames."""
    d = os.path.join(outdir, "frames")
    os.makedirs(d, exist_ok=True)
    rows = np.arange(0, t.size, stride)
    lims = {}
    for name in names:
        scale, ylim = scale_for(name, fields[name])
        if ylim is None:
            f = fields[name][np.isfinite(fields[name])]
            lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
            pad = 0.05 * (hi - lo) if hi > lo else (abs(hi) or 1.0) * 0.05
            ylim = (lo - pad, hi + pad)
        lims[name] = (scale, ylim)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    for j in rows:
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow), squeeze=False)
        for k, name in enumerate(names):
            ax = axes[k // ncol][k % ncol]
            scale, ylim = lims[name]
            y = fields[name][j]
            ax.plot(x, np.where(y > 0, y, ylim[0]) if scale == "log" else y, color="#1f77b4", lw=1.5)
            ax.set_yscale(scale); ax.set_ylim(*ylim); ax.set_xlim(x.min(), x.max())
            ax.set_xlabel(r"$x$ [$\mu$m]"); ax.set_ylabel(LABEL.get(name, name), fontsize=8)
            ax.set_title(name, fontsize=9); ax.grid(alpha=0.2)
            # per-frame the front is a single position, so mark THIS frame's crossover, not the
            # whole-run sweep band.
            if nstar is not None:
                if name == "n":
                    ax.axhline(nstar, color="crimson", ls=":", lw=1.3, zorder=4)
                if front is not None and np.isfinite(front[j]):
                    ax.axvline(front[j], color="k", ls="--", lw=1.1, zorder=4)
        for k in range(len(names), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"t = {t[j]:.4g} h   (step {j} of {t.size - 1})",
                     fontweight="bold", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(d, f"frame_{j:06d}.png"), dpi=100)
        plt.close(fig)
    return d, rows.size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="full_dump.csv, or the directory containing it")
    ap.add_argument("--stride", type=int, default=1,
                    help="draw every k-th time step (default 1 = all). Data is never thinned.")
    ap.add_argument("--fields", default=None,
                    help="comma-separated subset of columns (default: all)")
    ap.add_argument("--outdir", default=None, help="default: alongside the CSV")
    ap.add_argument("--frames", action="store_true",
                    help="also write one PNG per time step into <outdir>/frames/")
    ap.add_argument("--no-nstar", dest="no_nstar", action="store_true",
                    help="do not overlay the Monod crossover n* / its front")
    ap.add_argument("--frames-stride", dest="frames_stride", type=int, default=100,
                    help="stride for --frames, in FRAMES not hours (default 100). The hour "
                         "interval it gives is stride x the dump's own frame spacing, which is "
                         "T/(n_save-1) -- use --frames-every to ask in hours instead.")
    ap.add_argument("--frames-every", dest="frames_every", type=float, default=None,
                    help="frame interval in HOURS. Converted to a stride using the dump's actual "
                         "frame spacing, and CANNOT be finer than it: a dump saved with n_save "
                         "frames over T hours has spacing T/(n_save-1), and no plotting option "
                         "can recover times that were never saved.")
    a = ap.parse_args()

    csv_path = resolve(a.path)
    outdir = a.outdir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    x, t, fields = load(csv_path)
    print(f"read {csv_path}  ({os.path.getsize(csv_path)/1e6:.1f} MB) in {time.time()-t0:.1f}s")
    print(f"  {t.size} time steps x {x.size} positions, {len(fields)} fields")
    print(f"  t: {t.min():g} -> {t.max():g} h   x: {x.min():g} -> {x.max():g} um")

    names = list(fields)
    if a.fields:
        want = [c.strip() for c in a.fields.split(",") if c.strip()]
        missing = [c for c in want if c not in fields]
        if missing:
            sys.exit(f"unknown field(s): {', '.join(missing)}\navailable: {', '.join(names)}")
        names = want

    nstar = None if a.no_nstar else nstar_from_dump(fields)
    front = None
    if nstar is not None and "n" in fields:
        front = nstar_front(x, fields["n"], nstar)
        f = front[np.isfinite(front)]
        print(f"  n* = {nstar:.4f} uM (from the dump's own g1/g2)", end="")
        print(f"   front x_n*: {f.min():.1f} -> {f.max():.1f} um, final {f[-1]:.1f} um"
              if f.size else "   front: never crosses inside the domain")
    elif not a.no_nstar:
        print("  n*: not marked -- s = g1-g2 does not change sign anywhere in this run")

    stride = max(1, a.stride)
    t1 = time.time()
    path = os.path.join(outdir, "profiles_all.png")
    figure_all(x, t, fields, names, stride, path,
               subtitle=f"   ({os.path.basename(os.path.dirname(csv_path) or '.')})",
               nstar=nstar, front=front)
    print(f"  wrote {path}  in {time.time()-t1:.1f}s")

    if a.frames:
        t2 = time.time()
        fs = max(1, a.frames_stride)
        spacing = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        if a.frames_every is not None and spacing > 0:
            fs = max(1, int(round(a.frames_every / spacing)))
            if abs(fs * spacing - a.frames_every) > 1e-9:
                print(f"  --frames-every {a.frames_every:g} h is not attainable: this dump's frame "
                      f"spacing is {spacing:g} h, so the closest is stride {fs} = every "
                      f"{fs * spacing:g} h. Re-run the simulation with "
                      f"--n-save {int(round(t[-1] / a.frames_every)) + 1} for {a.frames_every:g} h "
                      f"frames.")
        print(f"  frames: stride {fs} x {spacing:g} h spacing = one PNG every {fs * spacing:g} h")
        d, nf = figure_frames(x, t, fields, names, outdir, fs,
                              nstar=nstar, front=front)
        print(f"  wrote {nf} frames into {d}  in {time.time()-t2:.1f}s")


if __name__ == "__main__":
    main()
