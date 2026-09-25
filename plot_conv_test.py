"""
plot_conv_test.py -- the four convergence runs made at the OPERATING POINT (dx=0.5, dt=0.05),
for n0 = 1000 and n0 = 15.  Simulates nothing; reads the CSVs those runs wrote.

    conv_dx_n0-1000_T1000/sweep_dx.csv    dx = 1, 0.5, 0.25   (dt = 0.05, T = 1000 h)
    conv_dt_n0-1000_dx0.5_T1000/sweep_dt.csv  dt = 0.1, 0.05, 0.025 (dx = 0.5, T = 1000 h)
    conv_dx_n0-15_T500/sweep_dx.csv       dx = 1, 0.5, 0.25   (dt = 0.05, T = 500 h)
    conv_dt_n0-15_dx0.5_T500/sweep_dt.csv dt = 0.1, 0.05, 0.025 (dx = 0.5, T = 500 h)

    py -3.11 src2/plot_conv_test.py
    py -3.11 src2/plot_conv_test.py --out somewhere.png

WHAT IS PLOTTED.  Columns 1 and 2 show the relative error of three metrics -- F1, rho_peak and
x_n* -- against the refinement parameter.  Column 3 shows F1 itself approaching its extrapolated
limit, with the operating point marked, because that is the question the runs were made to answer.

THREE HONEST CONSTRAINTS, all visible in the figure rather than buried here.

  1. THE EXTRAPOLATION USES THE TWO FINEST POINTS, so those two lie on a slope-1 line BY
     CONSTRUCTION.  The coarsest point is the only independent check that the sequence really is
     first order.  With three points there is no way around this; it is marked in every panel.

  2. RICHARDSON ASSUMES p = 1, the scheme's design order, not a fitted exponent -- three points
     give too noisy an exponent to invert.  The MEASURED order is printed in each panel so the
     assumption can be judged: it is 0.99 (dx, n0=1000) and 0.95 (dt, n0=15), and nonsense where
     the sequence is not asymptotic.

  3. A NON-MONOTONE SEQUENCE GETS NO EXTRAPOLATION.  Two occur: F1 vs dt at n0=1000 (differences
     +5.8e-4 then -2.0e-4, a sign flip) and rho_peak vs dx at n0=15 (differences -9e-8 then
     +5e-8, i.e. round-off).  For those the error is measured against the FINEST run instead and
     the series is drawn open with a dotted line, so a bound is never read as a bias.
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

SURFACE, INK, INK2, MUTED, GRID_C = "#ffffff", "#0b0b0b", "#52514e", "#7a7975", "#e3e2df"
C_F1, C_RHO, C_XNS = "#c0392b", "#2a78d6", "#6f6e6b"
OPERATING = {"dx": 0.5, "dt": 0.05}

RUNS = {
    ("jammed", "dx"): ("conv_dx_n0-1000_T1000/sweep_dx.csv", "dx"),
    ("jammed", "dt"): ("conv_dt_n0-1000_dx0.5_T1000/sweep_dt.csv", "dt"),
    ("dilute", "dx"): ("conv_dx_n0-15_T500/sweep_dx.csv", "dx"),
    ("dilute", "dt"): ("conv_dt_n0-15_dx0.5_T500/sweep_dt.csv", "dt"),
}
TITLE = {"jammed": r"$n_0=1000\ \mu$M, $T=1000$ h", "dilute": r"$n_0=15\ \mu$M, $T=500$ h"}
METRICS = (("F1_final", C_F1, "o", r"$F_1$"),
           ("rho_peak", C_RHO, "s", r"$\rho_{\rm peak}$"),
           ("x_nstar", C_XNS, "^", r"$x_{n^*}$"))


def load(rel, hkey):
    """Rows coarse -> fine, plus the refinement parameter as `h`."""
    rows = []
    with open(os.path.join(HERE, rel), newline="") as fh:
        for r in csv.DictReader(fh):
            row = {k: ((v == "True") if k in ("admissible", "success") else float(v))
                   for k, v in r.items()}
            row["h"] = row[hkey]
            rows.append(row)
    return sorted(rows, key=lambda r: -r["h"])


def extrapolate(v):
    """(limit, measured order, valid, note) for values ordered coarse -> fine, h halving.

    Richardson at the SCHEME's order p = 1:  v* = v_fine - (v_coarse - v_fine)/(2^p - 1), i.e.
    v* = v_fine + d_fine/(2^p - 1) with d_fine = v_fine - v_coarse, which at p = 1 is the usual
    2 v_fine - v_coarse.
    Declared invalid when the two differences do not share a sign (not asymptotic) or their ratio
    is outside [1.1, 6] (no recognisable order), in which case the caller falls back to the finest
    value as the reference.
    """
    v = np.asarray(v, float)
    d = np.diff(v)
    if d[0] == 0.0 or d[1] == 0.0:
        return v[-1], np.nan, False, "a difference is exactly zero"
    ratio = d[0] / d[1]
    order = float(np.log2(abs(ratio)))
    if ratio < 0:
        return v[-1], order, False, "non-monotone (sign flip)"
    if not (1.1 <= ratio <= 6.0):
        return v[-1], order, False, f"difference ratio {ratio:.2f} outside [1.1, 6]"
    return float(v[-1] + d[-1] / (2.0 ** 1 - 1)), order, True, ""


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID_C)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def panel_error(ax, rows, which, title):
    """Relative error of each metric against its extrapolated limit (or the finest run)."""
    h = np.array([r["h"] for r in rows])
    op = OPERATING[which]
    notes = []
    for col, c, mk, lab in METRICS:
        v = np.array([r[col] for r in rows])
        lim, order, ok, why = extrapolate(v)
        err = 100.0 * np.abs(v / lim - 1.0)
        hh, ee = (h, err) if ok else (h[:-1], err[:-1])       # finest is the ref when invalid
        ax.plot(hh, np.maximum(ee, 1e-7), marker=mk, ms=5.5, lw=1.4, color=c,
                ls="-" if ok else ":", mfc=c if ok else "white", zorder=3,
                label=(f"{lab}  (p={order:.2f})" if ok
                       else f"{lab}  [not asymptotic]"))
        if not ok:
            notes.append(lab)
    ax.axvline(op, color="#c0392b", lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.annotate("operating\npoint", (op, 0.97), xycoords=("data", "axes fraction"),
                color="#c0392b", fontsize=7.8, ha="center", va="top",
                xytext=(0, -2), textcoords="offset points")
    lo, hi = ax.get_ylim()
    g = np.array([h.min(), h.max()])
    ax.plot(g, hi * 0.20 * (g / g.max()), color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.annotate("slope 1", (g.max() * 0.55, hi * 0.145), color=MUTED, fontsize=8, rotation=20)
    if notes:
        ax.annotate(", ".join(notes) + ": error vs the finest run", (0.03, 0.03),
                    xycoords="axes fraction", color=MUTED, fontsize=7.0, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sorted(h))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    leg = ax.legend(fontsize=7.4, labelcolor=INK2, loc="lower right", frameon=True,
                    framealpha=0.9, edgecolor="none", facecolor=SURFACE)
    leg.set_zorder(6)
    style(ax, rf"$\Delta {which[1]}$" + ("  [$\\mu$m]" if which == "dx" else "  [h]"),
          "relative error  [%]", title)


def panel_F1(ax, rows_dx, rows_dt, title):
    """F1 itself against refinement, both sweeps on a shared axis of h / h_operating."""
    for rows, which, c, mk, lab in ((rows_dx, "dx", "#8a5cd0", "o", r"$\Delta x$ sweep"),
                                    (rows_dt, "dt", "#2e8b57", "s", r"$\Delta t$ sweep")):
        h = np.array([r["h"] for r in rows]) / OPERATING[which]
        v = np.array([r["F1_final"] for r in rows])
        lim, order, ok, why = extrapolate(v)
        ax.plot(h, v, marker=mk, ms=6, lw=1.5, color=c, zorder=3, label=lab)
        i_op = int(np.argmin(np.abs(h - 1.0)))
        ax.plot([h[i_op]], [v[i_op]], marker=mk, ms=12, mfc="none", mec="#c0392b", mew=1.4,
                ls="none", zorder=5)
        if ok:
            ax.axhline(lim, color=c, lw=1.1, ls="--", alpha=0.85, zorder=2)
            gap = 100.0 * (v[i_op] / lim - 1.0)
            ax.annotate(f"extrapolated {lim:.5f}\noperating point {gap:+.2f}%",
                        (h.max(), lim), textcoords="offset points", xytext=(-4, 8),
                        ha="right", color=c, fontsize=7.6)
        else:
            ax.annotate(f"no extrapolation:\n{why}", (h.max(), v[0]), textcoords="offset points",
                        xytext=(-4, -6), ha="right", va="top", color=c, fontsize=7.6)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    handles, labels = ax.get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], marker="o", ms=10, mfc="none",
                                           mec="#c0392b", mew=1.4, ls="none"))
    labels.append("operating point")
    leg = ax.legend(handles, labels, fontsize=7.8, labelcolor=INK2, loc="upper left",
                    frameon=True, framealpha=0.9, edgecolor="none", facecolor=SURFACE)
    leg.set_zorder(6)
    style(ax, r"$h\,/\,h_{\rm operating}$   (1 = $\Delta x{=}0.5$, $\Delta t{=}0.05$)",
          r"$F_1^{\rm final}$", title)


def report(data):
    print(f"{'regime':>8}{'sweep':>7}{'metric':>12}{'order':>8}{'limit':>14}"
          f"{'err @ operating':>17}{'valid':>7}")
    for regime in ("jammed", "dilute"):
        for which in ("dx", "dt"):
            rows = data[(regime, which)]
            op = OPERATING[which]
            i_op = int(np.argmin([abs(r["h"] - op) for r in rows]))
            for col, _c, _m, lab in METRICS:
                v = [r[col] for r in rows]
                lim, order, ok, why = extrapolate(v)
                err = 100.0 * (v[i_op] / lim - 1.0)
                print(f"{regime:>8}{which:>7}{col:>12}{order:>8.2f}{lim:>14.6f}"
                      f"{err:>+16.3f}%{('yes' if ok else 'NO'):>7}"
                      + ("" if ok else f"   <- {why}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "conv_test_operating_point.png"))
    a = ap.parse_args()

    data = {k: load(rel, hk) for k, (rel, hk) in RUNS.items()}
    for k, rows in data.items():
        assert len(rows) == 3, f"{k}: expected 3 rows, got {len(rows)}"
    report(data)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), facecolor=SURFACE)
    for i, regime in enumerate(("jammed", "dilute")):
        tag = "(a)" if i == 0 else "(d)"
        panel_error(axes[i, 0], data[(regime, "dx")], "dx",
                    f"{tag}  grid, {TITLE[regime]}")
        panel_error(axes[i, 1], data[(regime, "dt")], "dt",
                    f"{'(b)' if i == 0 else '(e)'}  time step, {TITLE[regime]}")
        panel_F1(axes[i, 2], data[(regime, "dx")], data[(regime, "dt")],
                 f"{'(c)' if i == 0 else '(f)'}  $F_1$ vs its extrapolated limit")

    fig.suptitle("Is the operating point converged?  "
                 r"$\Delta x=0.5\ \mu$m, $\Delta t=0.05$ h, tested by halving each",
                 color=INK, fontsize=13, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.030,
             r"Richardson at the scheme's order $p=1$; the MEASURED order is in each legend. "
             "The limit is built from the two finest runs, so those two lie on slope 1 by "
             "construction — the coarsest point is the only independent check.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.text(0.008, 0.009,
             r"Verdict: at $n_0=15$ every metric is inside 0.1% of its limit; at $n_0=1000$ "
             r"$\Delta t=0.05$ is inside 0.11% but $\Delta x=0.5$ leaves $F_1$ +15.1% high, "
             r"while $\rho_{\rm peak}$ and $x_{n^*}$ are inside 0.7%.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.048, 1, 0.963))
    fig.savefig(a.out, dpi=170, facecolor=SURFACE)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
