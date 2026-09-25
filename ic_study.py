"""
ic_study.py -- how much of the two-strain result is the initial condition?

Read-only against the stack (imports config/solver/closures/diagnostics, reimplements no
physics).  Written for the two `Parameter dependence` stubs in the manuscript:

    Varying seed        seed density; f1(0) != 0.5; uniform seed rather than invasion
    Initial conditions  changes in the nutrient IC

Four axes, each swept about the SAME baseline so the panels are comparable:

    --axis f1         seed_f1        0.1 .. 0.9      composition of the inoculum
    --axis mode       rho_mode       gaussian | uniform | deep      geometry of the inoculum
    --axis density    rho_seed       1e-6 .. 1e-2    size of the inoculum
    --axis nutrient   nutrient_mode  filled | empty | linear
    --axis all        (default)      all four

WHAT THIS IS TESTING, AND THE PREDICTION IN EACH CASE.  Stating these first makes the run a
test rather than a fishing trip.

  f1        seed_f1 = 0.5 makes h0 == 0 identically, which is why the baseline has an exactly
            vanishing inherited ancestry channel B0.  Moving off 0.5 switches B0 on, so the
            inverse-sorting criterion becomes A- > A+ + 2 B0 rather than A- > A+.  PREDICTION:
            a seed biased toward the glutton (low f1) should DEEPEN the bottleneck and delay
            recovery without changing its existence, because the bottleneck is driven by
            selection, not by the starting ratio.

  mode      invasion (opening_gaussian) versus a pre-colonised cavity (uniform).  PREDICTION:
            the deep graveyard requires biomass to be CARRIED inward before the crossover
            sweeps back, so a uniform seed -- which starts with biomass already deep -- should
            weaken or remove the mouth-type signature while keeping the deep one.

  density   sets the growth lag tau_g ~ ln(rho_sig/rho_seed)/g, hence the DURATION of the early
            nutrient-rich window in which the glutton is favoured everywhere.  PREDICTION:
            larger seed -> shorter window -> shallower bottleneck -> higher final F1.

  nutrient  PREDICTION: a near-null result, by timescale separation.  Nutrient relaxes on
            tau_n ~ L^2/Dn = 0.83 h at L=1000 um, Dn=1.2e6 um^2/h, while biomass needs
            tau_g ~ 1/mu = 1.3 h (glutton) to 4.8 h (scavenger) merely to e-fold from a seed
            that is 5-6 orders below the signal.  An empty cavity is therefore ~filled long
            before biomass matters.  A LARGE effect here would mean the separation is not
            operating and would need explaining.

TRAPS, all of which cost something to find:

  (1) THE UNIFORM/GAUSSIAN COMPARISON IS CONFOUNDED AT EQUAL rho_seed.  The unit-amplitude
      inoculum masses are NOT equal:

          opening_gaussian    44.87 um        1.00x
          deep_gaussian      200.51 um        4.47x
          first_mode         636.62 um       14.19x
          uniform           1000.00 um       22.29x

      So "uniform vs invasion" at fixed rho_seed varies geometry AND a 22x larger inoculum at
      once, and by the `density` axis above the amount alone moves the answer.  `--match-mass`
      rescales rho_seed per mode so the total initial biomass matches the Gaussian, isolating
      geometry.  Ratios are measured at run time from closures.initial_total_density itself,
      at an amplitude low enough that the rho_jam clip is inactive -- measuring them at
      amplitude 1.0 silently returns 640 um for uniform, because the clip bites.

  (2) READ delta_f, NOT inv_deep.  diagnostics.delta_f_series is scale-invariant; inv_deep is
      pinned at its ceiling whenever one strain is nearly gone, and must not then be read as
      enrichment.  diagnostics.validate flags exactly this as `metric_clash`.

  (3) A RUN WHOSE F1 HAS NOT TURNED IS A LOWER BOUND, NOT A MEASUREMENT.  If the F1 minimum
      sits at the end of the run the composition never recovered, and delta_f/inv_* cannot
      support a statement either way.  Such rows are printed with a `!` and drawn dotted; they
      are EXCLUDED from the trend summary rather than averaged in.  Extend --T instead.

  (4) The density axis overlaps `seed_study.py`, which grounds rho_seed in CELL COUNTS (the
      1e-6 default is a fortieth of one bacterium; ~1e-4 is a physical 2-cell inoculum).  This
      script sweeps the axis for the manuscript figure; seed_study.py is the authority on what
      the values MEAN.  Quote cell counts from there, not from here.

Outputs, into an auto-tagged directory:

    ic_study.csv        one row per run: the IC, every diagnostics.validate scalar, warnings
    ic_series.csv       long format time series: axis, value, t_h, F1, delta_f, inv_*, M1, M2
    ic_<axis>.png       4 panels per axis: F1(t), delta_f(t), inverse decomposition, final f1(x)
    ic_summary.png      one panel per axis: delta_f_max and F1_final against the swept value

Usage:
    py -3.11 src2/ic_study.py                                   # all four, dilute baseline
    py -3.11 src2/ic_study.py --axis mode --match-mass
    py -3.11 src2/ic_study.py --regime jammed --T 3000          # slow: dx=0.5, N=2000
    py -3.11 src2/ic_study.py --axis f1 --T 500 --n-save 61     # quick look
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                        # noqa: E402
import solver                                        # noqa: E402
import closures as C                                 # noqa: E402
import diagnostics as D                              # noqa: E402


# =============================================================================
# The four axes
# =============================================================================
AXES = {
    "f1": dict(
        field="seed_f1",
        values=[0.1, 0.25, 0.5, 0.75, 0.9],
        label=r"seed composition $f_1(0)$",
        fmt=lambda v: f"{v:g}",
        numeric=True,
    ),
    "mode": dict(
        field="rho_mode",
        values=["opening_gaussian", "uniform", "deep_gaussian"],
        label="seed geometry",
        fmt=lambda v: {"opening_gaussian": "invasion", "uniform": "uniform",
                       "deep_gaussian": "deep"}.get(v, str(v)),
        numeric=False,
    ),
    "density": dict(
        field="rho_seed",
        values=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        label=r"seed density $\rho_{\mathrm{seed}}$",
        fmt=lambda v: f"{v:g}",
        numeric=True,
    ),
    "nutrient": dict(
        field="nutrient_mode",
        values=["filled", "empty", "linear"],
        label="nutrient initial condition",
        fmt=lambda v: str(v),
        numeric=False,
    ),
}

#: The baseline every axis is swept about (config.initial_conditions defaults).
BASELINE = dict(seed_f1=0.5, rho_mode="opening_gaussian", rho_seed=1.0e-6,
                nutrient_mode="filled")


# =============================================================================
# Inoculum mass, for --match-mass
# =============================================================================
def unit_masses(p, num, modes) -> dict:
    """Total initial biomass per unit rho_seed, for each rho_mode, on the ACTUAL grid.

    Measured at an amplitude small enough that the rho_jam clip inside
    closures.initial_total_density is inactive: at amplitude 1.0 the clip bites and uniform
    reports 640 um instead of 1000 um.
    """
    N = C.choose_grid_size(config.L, num)
    x, dx = C.cell_grid(config.L, N)
    probe = 1.0e-9
    out = {}
    for m in modes:
        ic = config.initial_conditions(rho_mode=m, rho_seed=probe)
        rho0 = C.initial_total_density(x, config.L, p, ic)
        if float(rho0.max()) >= p.rho_jam * (1.0 - 1e-9):
            raise RuntimeError(f"rho_jam clip active while probing {m!r}; lower the probe")
        out[m] = float(rho0.sum() * dx) / probe
    return out


# =============================================================================
# One run
# =============================================================================
def run_one(axis: str, value, a, mass_ratio: float = 1.0):
    """Simulate one point on one axis.  Returns (row, series)."""
    spec = AXES[axis]
    ic_kw = dict(BASELINE)
    ic_kw[spec["field"]] = value

    # --match-mass: hold total initial biomass fixed while geometry varies.
    rho_seed_used = ic_kw["rho_seed"] * mass_ratio
    ic_kw["rho_seed"] = rho_seed_used

    p = config.params_from_args(a, a.n0)
    num = config.numerics(T_end=a.T, dt=a.dt, dx=a.dx, n_save=a.n_save,
                          label_limiter=(getattr(a, "limiter", None) or "none"))
    ic = config.initial_conditions(**ic_kw)

    t0 = time.time()
    res = solver.simulate(config.L, p, num, ic)
    wall = time.time() - t0

    v = D.validate(res, p, verbose=False)
    mo, dp, F1, nstar = D.inverse_decomposition(res, p)
    df = D.delta_f_series(res, p)
    M1, M2 = D.masses(res)

    warnings = sorted(k for k, val in v.items()
                      if isinstance(val, tuple) and bool(val[0]))
    scalars = {k: val for k, val in v.items() if not isinstance(val, tuple)}

    row = dict(
        axis=axis, value=spec["fmt"](value), value_raw=value,
        seed_f1=ic.seed_f1, rho_mode=ic.rho_mode, rho_seed=rho_seed_used,
        nutrient_mode=ic.nutrient_mode, mass_ratio=mass_ratio,
        n0=a.n0, dx=res["dx"], dt=res["dt"], T=a.T, N=res["N"],
        label_limiter=num.label_limiter,
        **scalars,
        warnings=";".join(warnings), wall_s=round(wall, 1),
    )
    series = dict(t=res["t"], F1=F1, delta_f=df, inv_mouth=mo, inv_deep=dp, M1=M1, M2=M2,
                  x=res["x"], f1_final=_final_fraction(res), n_final=res["n_final"])
    return row, series


def _final_fraction(res):
    r1, r2 = res["rho1_final"], res["rho2_final"]
    rho = r1 + r2
    f = np.full_like(rho, np.nan)
    occ = rho > 1e-9 * max(float(rho.max()), 1e-30)
    f[occ] = r1[occ] / rho[occ]
    return f


# =============================================================================
# Figures
# =============================================================================
def _colors(k):
    return plt.get_cmap("viridis")(np.linspace(0.08, 0.88, max(k, 2)))


def _coincident(seriess, keys=("F1", "delta_f", "inv_deep"), tol=0.01) -> bool:
    """Do all runs trace the same curve to within `tol` of the signal's own range?

    A null axis (e.g. the nutrient IC) plots N curves exactly on top of one another, and only the
    last one drawn is visible -- which reads as "we ran one simulation" rather than as the result.
    """
    if len(seriess) < 2:
        return False
    for k in keys:
        Y = np.array([s[k] for s in seriess])
        rng = float(Y.max() - Y.min())
        if rng <= 0:
            continue
        if float(np.abs(Y - Y[0]).max()) > tol * rng:
            return False
    return True


def _interleaved(n, period=9.0):
    """Dash patterns that TILE one shared path: n coincident curves become one striped line.

    Each series is on for period/n and off for the rest, with its phase shifted so the segments
    abut rather than overlap.  The composite reads as a single curve in alternating colours.
    """
    on = period / n
    return [(i * on, (on, period - on)) for i in range(n)]


def figure_axis(axis, rows, seriess, nstar, path, solid=False, no_text=False):
    """One IC axis, four panels.

    solid    draw every run solid, ignoring the inadmissible/never-turned dotting.  Legitimate ONLY
             when the flagged resolution has been validated for the comparison being drawn -- the
             jammed dx=2 study was, against dx=0.5 in plot_ic_dx_validation.py (IC spread kept to
             within 9% although every run fails the front check).  The rows' own `admissible`
             value is untouched; only the rendering changes.
    no_text  no suptitle and no footer note: panel tags (A)-(D), axis labels and legends only, so
             the explanation lives in the manuscript caption.  Panel (C) gains a small legend,
             because with the footer gone nothing else says that line style separates mouth from
             deep.
    """
    spec = AXES[axis]
    cols = _colors(len(rows))
    overlap = _coincident(seriess)
    dashes = _interleaved(len(rows)) if overlap else None
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
    if not no_text:
        fig.suptitle(f"Initial-condition dependence: {spec['label']}", fontsize=12)

    for i, (row, s) in enumerate(zip(rows, seriess)):
        ok = solid or (bool(row.get("F1_turned", True)) and bool(row.get("admissible", True)))
        ls = dashes[i] if overlap else ("-" if ok else ":")
        style = dict(color=cols[i], lw=2.4 if overlap else 1.7, ls=ls,
                     alpha=1.0 if ok else 0.65,
                     label=f"{row['value']}" + ("" if ok else "  (!)"))
        ax[0, 0].plot(s["t"], s["F1"], **style)
        ax[0, 1].plot(s["t"], s["delta_f"], **style)
        ax[1, 0].plot(s["t"], s["inv_mouth"], **style)
        ax[1, 0].plot(s["t"], s["inv_deep"], color=cols[i], alpha=0.75,
                      lw=1.3 if overlap else 1.1, ls=ls if overlap else "--")
        ax[1, 1].plot(s["x"], s["f1_final"], **style)

    if no_text:
        titles = ("(A)", "(B)", "(C)", "(D)")
    else:
        titles = ("global composition",
                  r"inverse sorting  $\Delta_f=\langle f\rangle_{n>n^*}-\langle f\rangle_{n<n^*}$",
                  "inverse-sorted mass: mouth (thick) / deep (thin)" if overlap
                  else "inverse-sorted mass: mouth (solid) / deep (dashed)",
                  "final composition profile")
    tkw = dict(loc="left") if no_text else {}
    ax[0, 0].set(xlabel="t [h]", ylabel=r"$F_1$"); ax[0, 0].set_title(titles[0], **tkw)
    ax[0, 0].axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax[0, 0].set_ylim(-0.02, 1.02)

    ax[0, 1].set(xlabel="t [h]", ylabel=r"$\Delta_f$"); ax[0, 1].set_title(titles[1], **tkw)
    ax[0, 1].axhline(0.0, color="0.6", lw=0.8, ls=":")

    ax[1, 0].set(xlabel="t [h]", ylabel="biomass fraction"); ax[1, 0].set_title(titles[2], **tkw)

    ax[1, 1].set(xlabel=r"$x$ [$\mu$m]", ylabel=r"$f_1(x,T)$"); ax[1, 1].set_title(titles[3], **tkw)
    ax[1, 1].axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax[1, 1].set_ylim(-0.02, 1.02)

    for A in ax.ravel():
        A.grid(alpha=0.25, lw=0.6)
    ax[0, 0].legend(fontsize=8, title=spec["label"], title_fontsize=8, frameon=False)
    if no_text:
        # The footer used to carry the mouth/deep line-style key; without it, say it here.
        from matplotlib.lines import Line2D
        key = [Line2D([], [], color="0.3", lw=2.4 if overlap else 1.7, label="mouth"),
               Line2D([], [], color="0.3", lw=1.3 if overlap else 1.1,
                      ls="-" if overlap else "--", label="deep")]
        ax[1, 0].legend(handles=key, fontsize=8, frameon=False)

    if not no_text:
        if overlap:
            note = ("all runs coincide to <1% of the signal range, so the curves are drawn as "
                    "interleaved dashes: the striped line IS every run, overlaid")
        else:
            note = "dotted = F1 never turned or inadmissible: lower bound, not a measurement"
        if nstar is not None:
            note = rf"$n^*={nstar:.3f}\ \mu$M    " + note
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="0.35")
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _as_bool(v):
    """CSV booleans come back as the strings 'True'/'False' -- and bool('False') is True."""
    return str(v).strip().lower() in ("true", "1", "yes")


def replot(outdir, axes, nstar, solid=False, no_text=False):
    """Redraw the per-axis figures from the CSVs of a finished study, simulating nothing."""
    need = ("ic_study.csv", "ic_series.csv", "ic_profiles.csv")
    missing = [f for f in need if not os.path.exists(os.path.join(outdir, f))]
    if missing:
        sys.exit(f"--replot: {outdir} lacks {', '.join(missing)}.  Studies run before "
                 f"2026-09-23 never wrote ic_profiles.csv, so panel (D) cannot be redrawn "
                 f"without rerunning them once.")
    rows_all = list(csv.DictReader(open(os.path.join(outdir, "ic_study.csv"))))
    ser, prof = {}, {}
    for r in csv.DictReader(open(os.path.join(outdir, "ic_series.csv"))):
        d = ser.setdefault((r["axis"], r["value"]), {k: [] for k in
                            ("t", "F1", "delta_f", "inv_mouth", "inv_deep")})
        d["t"].append(float(r["t_h"])); d["F1"].append(float(r["F1"]))
        d["delta_f"].append(float(r["delta_f"])); d["inv_mouth"].append(float(r["inv_mouth"]))
        d["inv_deep"].append(float(r["inv_deep"]))
    for r in csv.DictReader(open(os.path.join(outdir, "ic_profiles.csv"))):
        d = prof.setdefault((r["axis"], r["value"]), {"x": [], "f1_final": []})
        d["x"].append(float(r["x_um"])); d["f1_final"].append(float(r["f1_final"]))
    for axis in axes:
        rows = [dict(r, F1_turned=_as_bool(r["F1_turned"]), admissible=_as_bool(r["admissible"]))
                for r in rows_all if r["axis"] == axis]
        if not rows:
            continue
        seriess = []
        for r in rows:
            s = {k: np.asarray(v) for k, v in ser[(axis, r["value"])].items()}
            s.update({k: np.asarray(v) for k, v in prof[(axis, r["value"])].items()})
            seriess.append(s)
        path = os.path.join(outdir, f"ic_{axis}.png")
        figure_axis(axis, rows, seriess, nstar, path, solid=solid, no_text=no_text)
        print(f"  replotted {path}")


def figure_summary(by_axis, path):
    """Two measures, two ROWS -- never a twin y-axis.

    delta_f_max (0..0.25) and F1_final (0..1) live on different scales.  Overlaying them on a
    shared frame with two y-scales lets the axis limits manufacture whatever crossings and
    apparent correlations you like; small multiples on a common x say the same thing honestly.
    """
    axes_present = [k for k in AXES if k in by_axis]
    ncol = len(axes_present)
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 6.4), squeeze=False, sharex="col")
    fig.suptitle("What each initial condition changes", fontsize=12)

    for j, axis in enumerate(axes_present):
        rows = by_axis[axis]
        pos = np.arange(len(rows))
        good = [bool(r.get("F1_turned", True)) and bool(r.get("admissible", True)) for r in rows]
        face = ["#2a78d6" if g else "#b9c3bd" for g in good]
        hatch = ["" if g else "//" for g in good]

        A = axs[0, j]
        A.bar(pos, [r["delta_f_max"] for r in rows], color=face, hatch=hatch, width=0.62)
        A.set_title(AXES[axis]["label"], fontsize=10)
        A.axhline(0.0, color="0.5", lw=0.8)
        if j == 0:
            A.set_ylabel(r"$\Delta_f^{\max}$  (inverse sorting)")

        B = axs[1, j]
        B.bar(pos, [r["F1_final"] for r in rows], color=face, hatch=hatch, width=0.62)
        B.set_ylim(0.0, 1.02)
        B.axhline(1.0, color="0.5", lw=0.8, ls=":")
        B.set_xticks(pos)
        B.set_xticklabels([r["value"] for r in rows], rotation=20, ha="right", fontsize=8)
        if j == 0:
            B.set_ylabel(r"$F_1$ final  (competitive outcome)")

        for X in (A, B):
            X.grid(axis="y", alpha=0.25, lw=0.6)
            X.set_axisbelow(True)

    fig.text(0.5, 0.005, "hatched = F1 never turned or inadmissible (lower bound only); "
                         "dotted line = Tilman $R^*$ exclusion, $F_1=1$",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", default="all",
                    choices=list(AXES) + ["all"], help="which axis to sweep")
    ap.add_argument("--regime", default="dilute", choices=list(config.REGIMES),
                    help="baseline regime; --n0/--dx/--dt/--T override it")
    ap.add_argument("--n0", type=float, default=None, help="uM")
    ap.add_argument("--dx", type=float, default=None, help="um")
    ap.add_argument("--dt", type=float, default=None, help="h")
    ap.add_argument("--T", type=float, default=None, help="h")
    ap.add_argument("--n-save", dest="n_save", type=int, default=121)
    ap.add_argument("--match-mass", action="store_true",
                    help="on the `mode` axis, rescale rho_seed so total initial biomass matches "
                         "the Gaussian -- isolates geometry from the 22x inoculum difference")
    ap.add_argument("--values", default=None,
                    help="comma-separated override of the swept values (single --axis only)")
    ap.add_argument("--rho-seed", dest="rho_seed", type=float, default=None,
                    help="baseline inoculum amplitude (default: the BASELINE value, 1e-6). "
                         "Production runs use 1e-4, which is ~7 cells; 1e-6 is far below one "
                         "cell.  NOTE this sets the BASELINE the other three axes are swept "
                         "about -- the `density` axis sweeps rho_seed itself and is unaffected.")
    ap.add_argument("--limiter", default=None,
                    choices=["none", "minmod", "vanleer", "superbee"],
                    help="TVD flux limiter for the strain-label advection (default: the "
                         "closures.Numerics default, 'none').  Production runs use vanleer.")
    ap.add_argument("--solid", action="store_true",
                    help="draw every run solid (no dotting, no '(!)').  Only after the flagged "
                         "resolution has been validated -- see plot_ic_dx_validation.py.")
    ap.add_argument("--no-text", dest="no_text", action="store_true",
                    help="no suptitle and no footer; panel tags (A)-(D) instead of titles.  The "
                         "explanation belongs in the manuscript caption.")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the per-axis figures from the CSVs already in the output "
                         "directory, simulating nothing.  Needs ic_profiles.csv, which runs "
                         "made before 2026-09-23 did not write.")
    ap.add_argument("--outdir", default=None)
    config.add_model_args(ap)
    a = ap.parse_args()

    r = config.REGIMES[a.regime]
    a.n0 = r["n0"] if a.n0 is None else a.n0
    a.dx = r["dx"] if a.dx is None else a.dx
    a.dt = r["dt"] if a.dt is None else a.dt
    a.T = r["T"] if a.T is None else a.T

    axes = list(AXES) if a.axis == "all" else [a.axis]
    if a.values is not None:
        if len(axes) != 1:
            ap.error("--values requires a single --axis")
        spec = AXES[axes[0]]
        raw = [v.strip() for v in a.values.split(",")]
        spec["values"] = [float(v) for v in raw] if spec["numeric"] else raw

    # The tag must include the axis: ic_study.csv / ic_series.csv are written per RUN, not per
    # axis, so two single-axis sweeps at the same n0/dx/T would silently overwrite each other's
    # data (the per-axis PNGs survive, because those are named for the axis -- which is exactly
    # what makes the loss easy to miss).
    # --rho-seed moves the BASELINE the other axes are swept about.  The `density` axis sweeps
    # rho_seed itself, so it is unaffected by this and only the limiter changes there.
    if a.rho_seed is not None:
        BASELINE["rho_seed"] = a.rho_seed

    tag = f"ic_study_n0-{a.n0:g}_dx{a.dx:g}_T{a.T:g}"
    if a.axis != "all":
        tag += f"_{a.axis}"
    if a.match_mass:
        tag += "_massmatched"
    # Tag seed and limiter, as figures.py and adjoint_kernel.py already do.  Without this a
    # production-config sweep writes straight into the folder of an older default-config one
    # and destroys it -- the two are otherwise indistinguishable after the fact.
    if a.rho_seed is not None:
        tag += f"_seed{a.rho_seed:g}"
    if a.limiter is not None and a.limiter != "none":
        tag += f"_lim-{a.limiter}"
    outdir = a.outdir or os.path.join(HERE, tag)
    os.makedirs(outdir, exist_ok=True)

    p0 = config.params_from_args(a, a.n0)
    num0 = config.numerics(T_end=a.T, dt=a.dt, dx=a.dx, n_save=a.n_save,
                           label_limiter=(a.limiter or "none"))
    nstar = C.monod_crossover(p0)

    if a.replot:
        replot(outdir, axes, nstar, solid=a.solid, no_text=a.no_text)
        return

    print(f"=== initial-condition dependence ===")
    print(f"  regime={a.regime}  n0={a.n0:g} uM  dx={a.dx:g} um  dt={a.dt:g} h  T={a.T:g} h"
          f"  n*={nstar:.4f} uM")
    print(f"  baseline: " + ", ".join(f"{k}={v}" for k, v in BASELINE.items())
          + f", label_limiter={num0.label_limiter}")
    print(f"  outdir: {outdir}")

    masses = unit_masses(p0, num0, AXES["mode"]["values"])
    base_mass = masses[BASELINE["rho_mode"]]
    print("\n  unit-amplitude inoculum mass by geometry:")
    for m, v in masses.items():
        print(f"    {m:18s} {v:9.2f} um   {v/base_mass:6.2f}x")
    if "mode" in axes and not a.match_mass:
        print("    NOTE: --match-mass is OFF, so the `mode` axis varies geometry AND inoculum "
              "size together.")

    all_rows, by_axis, all_series = [], {}, []
    for axis in axes:
        spec = AXES[axis]
        print(f"\n--- axis: {axis}  ({spec['field']}) ---")
        hdr = (f"{'value':>18}{'rho_seed':>11}{'rho/rjam':>10}{'F1_min':>10}{'F1_fin':>9}"
               f"{'df_max':>9}{'Imouth':>9}{'Ideep':>9}{'adm':>5}{'s':>7}")
        print(hdr, flush=True)
        rows, seriess = [], []
        for value in spec["values"]:
            ratio = 1.0
            if axis == "mode" and a.match_mass:
                ratio = base_mass / masses[value]
            row, s = run_one(axis, value, a, mass_ratio=ratio)
            rows.append(row); seriess.append(s)
            flag = "" if row["F1_turned"] else "!"
            print(f"{row['value']:>18}{row['rho_seed']:>11.3g}{row['rho_max_over_jam']:>10.4f}"
                  f"{row['F1_min']:>10.3e}{row['F1_final']:>9.4f}{row['delta_f_max']:>9.4f}"
                  f"{row['int_inv_mouth']:>9.1f}{row['int_inv_deep']:>9.1f}"
                  f"{str(bool(row['admissible']))[0]+flag:>5}{row['wall_s']:>7.0f}", flush=True)
            if row["warnings"]:
                print(f"{'':>18}  warn: {row['warnings']}", flush=True)

        by_axis[axis] = rows
        all_rows.extend(rows)
        for row, s in zip(rows, seriess):
            all_series.append((axis, row["value"], s))

        figure_axis(axis, rows, seriess, nstar, os.path.join(outdir, f"ic_{axis}.png"),
                    solid=a.solid, no_text=a.no_text)
        print(f"  wrote ic_{axis}.png")

    # --- CSVs -----------------------------------------------------------------------------
    path = os.path.join(outdir, "ic_study.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {path}")

    path = os.path.join(outdir, "ic_series.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "value", "t_h", "F1", "delta_f", "inv_mouth", "inv_deep", "M1", "M2"])
        for axis, value, s in all_series:
            for k in range(len(s["t"])):
                w.writerow([axis, value, f"{s['t'][k]:.6g}", f"{s['F1'][k]:.6g}",
                            f"{s['delta_f'][k]:.6g}", f"{s['inv_mouth'][k]:.6g}",
                            f"{s['inv_deep'][k]:.6g}", f"{s['M1'][k]:.6g}", f"{s['M2'][k]:.6g}"])
    print(f"wrote {path}")

    # Final composition profiles.  Panel (D) of every per-axis figure plots these, and until
    # 2026-09-23 they were never written -- so no figure could be redrawn without rerunning the
    # whole study.  With this file present, --replot redraws everything from disk.
    path = os.path.join(outdir, "ic_profiles.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "value", "x_um", "f1_final"])
        for axis, value, s in all_series:
            for k in range(len(s["x"])):
                w.writerow([axis, value, f"{s['x'][k]:.6g}", f"{s['f1_final'][k]:.8g}"])
    print(f"wrote {path}")

    figure_summary(by_axis, os.path.join(outdir, "ic_summary.png"))
    print(f"wrote {os.path.join(outdir, 'ic_summary.png')}")

    # --- verdict per axis -----------------------------------------------------------------
    print("\n=== how much does each initial condition matter? ===")
    print("    (spread over the axis, USABLE rows only: F1 turned and admissible)")
    for axis, rows in by_axis.items():
        good = [r for r in rows
                if bool(r.get("F1_turned", True)) and bool(r.get("admissible", True))]
        skipped = len(rows) - len(good)
        if len(good) < 2:
            print(f"  {axis:9s} only {len(good)} usable row(s) of {len(rows)} -- "
                  f"no trend can be read.  Extend --T.")
            continue
        df = np.array([r["delta_f_max"] for r in good])
        f1 = np.array([r["F1_final"] for r in good])
        scale = max(abs(df).max(), 1e-12)
        note = f"  ({skipped} row(s) skipped)" if skipped else ""
        print(f"  {axis:9s} delta_f_max {df.min():+.4f} .. {df.max():+.4f}   "
              f"(spread {100*(df.max()-df.min())/scale:5.1f}% of max)   "
              f"F1_final {f1.min():.4f} .. {f1.max():.4f}{note}")


if __name__ == "__main__":
    main()
