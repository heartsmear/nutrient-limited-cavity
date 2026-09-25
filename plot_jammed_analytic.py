"""Fully jammed analytic nutrient profile against the numerical steady state.

Reads an existing `steady_*` run directory (profile.csv + summary.csv) read-only and
overlays the closed-form profile of the fully jammed approximation,

    n(x) = n0 * cosh(kappa_j (L - x)) / cosh(kappa_j L),
    kappa_j = sqrt(k rho_j / (Y Dn)),

which follows from freezing the density at rho_j and linearising Monod growth as
g(n) ~ k n with k = mu / Ks.  Both assumptions are stated in the derivation; this
script measures where they actually hold.

The nutrient spans ~17 decades, so error is reported three ways: relative error on the
trustworthy window (n above the solver noise floor), the error in decades (the natural
measure for a near-exponential profile), and the local decay rate -dln(n)/dx against
kappa_j, which separates whether the *rate* is right from whether the *prefactor* is.

    py -3.11 src2/plot_jammed_analytic.py [rundir] [--rho-jam] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "steady_n0-1000_L1000_dx0.5"

C1, C2, CN = "#1f77b4", "#d62728", "#2ca02c"


def read_summary(path: Path) -> dict:
    with path.open(newline="") as fh:
        return {r[0]: r[1] for r in csv.reader(fh) if len(r) >= 2}


def read_profile(path: Path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["x_um"], d["rho"], d["n_uM"]


def n_jammed(x, n0, kappa, L):
    """cosh(kappa(L-x)) / cosh(kappa L), evaluated without overflowing.

    kappa*L is ~314 here, so the ratio is rewritten as
        exp(-kappa x) * (1 + exp(-2 kappa (L-x))) / (1 + exp(-2 kappa L))
    which is exact and stays in range for any kappa*L.
    """
    return n0 * np.exp(-kappa * x) * (
        (1.0 + np.exp(-2.0 * kappa * (L - x))) / (1.0 + np.exp(-2.0 * kappa * L))
    )


def local_rate(x, n, floor):
    """-d ln n / dx by central differences, masked where n sits at the noise floor."""
    ok = n > floor
    r = np.full_like(x, np.nan)
    lg = np.log(np.where(ok, n, np.nan))
    r[1:-1] = -(lg[2:] - lg[:-2]) / (x[2:] - x[:-2])
    r[~ok] = np.nan
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", nargs="?", default=str(DEFAULT_RUN))
    ap.add_argument("--rho-jam", action="store_true",
                    help="use rho_jam=0.64 instead of the run's realised jammed density")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    run = Path(args.rundir).resolve()
    out = Path(args.outdir).resolve() if args.outdir else run
    out.mkdir(parents=True, exist_ok=True)

    s = read_summary(run / "summary.csv")
    x, rho, n_num = read_profile(run / "profile.csv")

    mu, Ks = float(s["mu"]), float(s["Ks"])
    n0, Y, Dn = float(s["n0_uM"]), float(s["Y"]), float(s["Dn"])
    L, rho_jam = float(s["L_um"]), float(s["rho_jam"])
    floor = float(s["noise_floor_uM"])

    # k is the low-nutrient slope of the Monod law, g(n) = mu n/(Ks+n) -> (mu/Ks) n.
    k = mu / Ks
    rho_j = rho_jam if args.rho_jam else float(s["rho_at_L"])
    kappa = np.sqrt(k * rho_j / (Y * Dn))

    n_ana = n_jammed(x, n0, kappa, L)

    # Only where the numerical nutrient is above the solver noise floor does it carry
    # information; deeper than that the run pins n at ~1e-14 and comparison is meaningless.
    trust = n_num > floor
    rel = np.abs(n_ana - n_num) / n_num
    dec = np.abs(np.log10(np.maximum(n_ana, 1e-300)) - np.log10(np.maximum(n_num, 1e-300)))

    rate_num = local_rate(x, n_num, floor)
    tail = trust & (n_num < 0.1 * Ks)      # exponential regime
    head = n_num > 10.0 * Ks               # saturated regime

    x_trust = x[trust].max()
    i_Ks = int(np.argmin(np.abs(n_num - Ks)))

    # The formula fixes the amplitude at the mouth, n(0)=n0, but the head is saturated
    # (n >> Ks, so g ~ mu, not k n) and the density is still climbing there, so the real
    # profile leaves the head with a different amplitude.  Fitting that amplitude to the
    # tail separates "is the decay rate right" from "is the mouth amplitude right".
    amp = np.exp(np.mean(np.log(n_num[tail]) + kappa * x[tail])) if tail.any() else np.nan
    n_fit = amp * np.exp(-kappa * x)
    rel_fit = np.abs(n_fit - n_num) / n_num

    # Where the density plateau is reached: the other assumption of the derivation.
    rho_plateau = float(s["rho_at_L"])
    i_plat = int(np.argmax(rho >= 0.99 * rho_plateau))

    def band(mask, label):
        if not mask.any():
            return f"  {label:<32s} (empty)"
        return (f"  {label:<32s} rel {rel[mask].min():9.2e} - {rel[mask].max():9.2e}"
                f"   median {np.median(rel[mask]):9.2e}   max {dec[mask].max():5.2f} dec")

    print(f"run            {run}")
    print(f"mu={mu:g} 1/h  Ks={Ks:g} uM  n0={n0:g} uM  L={L:g} um  dx={float(s['dx_um']):g} um")
    print(f"k = mu/Ks    = {k:.6g} 1/(uM h)")
    print(f"rho_j        = {rho_j:.10g}  ({'rho_jam' if args.rho_jam else 'realised, rho(L)'};"
          f" rho_max/rho_jam = {float(s['rho_max_over_jam']):.4f})")
    print(f"kappa_j      = {kappa:.6g} 1/um     1/kappa_j = {1.0 / kappa:.4f} um")
    print(f"kappa_j L    = {kappa * L:.4g}  -> the cosh ratio is a pure exponential to "
          f"{np.exp(-2 * kappa * L):.1e}")
    print()
    print(f"noise floor    {floor:g} uM, reached at x = {float(s['x_below_noise_floor_um']):g} um"
          f"  ({100 * float(s['cavity_fraction_below_noise_floor']):.1f}% of cavity below)")
    print(f"trustworthy    x in [0, {x_trust:g}] um   ({int(trust.sum())} of {x.size} cells)")
    print(f"n = Ks at      x = {x[i_Ks]:g} um   (Monod linearisation valid deeper than this)")
    print(f"rho plateau    x = {x[i_plat]:g} um   (rho within 1% of {rho_plateau:.4f})")
    print()
    print("relative error |n_ana - n_num| / n_num")
    print(band(trust, "whole trustworthy window"))
    print(band(head, "saturated head (n > 10 Ks)"))
    print(band(tail, "exponential tail (n < 0.1 Ks)"))
    print()
    r_tail = rate_num[tail]
    r_tail = r_tail[np.isfinite(r_tail)]
    if r_tail.size:
        print(f"local decay rate in the tail   {r_tail.mean():.6g} +- {r_tail.std():.2g} 1/um")
        print(f"kappa_j                        {kappa:.6g} 1/um"
              f"   -> rate ratio {r_tail.mean() / kappa:.4f}"
              f"  ({100 * (r_tail.mean() / kappa - 1):+.2f}%)")
    print()
    print(f"tail amplitude fit   n ~ A exp(-kappa_j x),  A = {amp:.6g} uM")
    print(f"  vs the formula amplitude n0 = {n0:g} uM   ->  A/n0 = {amp / n0:.4g}"
          f"  ({np.log10(n0 / amp):.2f} decades low)")
    print(f"  equivalent head offset  x0 = ln(n0/A)/kappa_j = {np.log(n0 / amp) / kappa:.2f} um")
    print("relative error of the amplitude-fitted exponential")
    if tail.any():
        print(f"  exponential tail (n < 0.1 Ks)    max {rel_fit[tail].max():.2e}"
              f"   median {np.median(rel_fit[tail]):.2e}")

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.6))
    for a in ax.ravel():
        a.grid(alpha=.2)

    a = ax[0, 0]
    a.plot(x, np.maximum(n_num, floor * 1e-1), color=C1, lw=2.0, label="numerical")
    a.plot(x, np.maximum(n_ana, floor * 1e-1), color=C2, lw=1.4, ls="--",
           label=r"analytic $n_0\,\cosh[\kappa_j(L-x)]/\cosh(\kappa_j L)$")
    a.plot(x, np.maximum(n_fit, floor * 1e-1), color="0.35", lw=1.2, ls="-.",
           label=rf"same $\kappa_j$, amplitude fitted ($A$={amp:.3g} $\mu$M)")
    a.axhspan(floor * 1e-1, floor, color="grey", alpha=.15)
    a.axhline(floor, color="grey", lw=1, label=rf"noise floor ({floor:g} $\mu$M)")
    a.axhline(Ks, color="k", ls=":", lw=1, label=rf"$K_s={Ks:g}$ $\mu$M")
    a.set_yscale("log")
    a.set_ylim(floor * 1e-1, n0 * 3.0)
    a.set_xlim(0, min(2.0 * x_trust, L))
    a.set_xlabel(r"$x$ [$\mu$m]")
    a.set_ylabel(r"$n$ [$\mu$M]")
    a.legend(fontsize=7, loc="upper right")
    a.set_title("(A) full profile, log scale", fontsize=9)

    a = ax[0, 1]
    m = x <= 40.0
    a.plot(x[m], n_num[m], color=C1, lw=2.0, label="numerical")
    a.plot(x[m], n_ana[m], color=C2, lw=1.4, ls="--", label="analytic")
    a.axhline(Ks, color="k", ls=":", lw=1, label=rf"$K_s={Ks:g}$ $\mu$M")
    a.set_xlabel(r"$x$ [$\mu$m]")
    a.set_ylabel(r"$n$ [$\mu$M]")
    a.legend(fontsize=7)
    a.set_title("(B) mouth region, linear scale -- where the linearisation fails", fontsize=9)

    a = ax[1, 0]
    a.plot(x[trust], rel[trust], color=CN, lw=1.8, label="formula as written")
    a.plot(x[trust], rel_fit[trust], color="0.35", lw=1.4, ls="-.",
           label="amplitude fitted to the tail")
    a.axvline(x[i_Ks], color="k", ls=":", lw=1, label=rf"$n=K_s$ at $x$={x[i_Ks]:.1f} $\mu$m")
    a.axhline(1.0, color="grey", lw=1, ls="--", label="100% error")
    a.set_yscale("log")
    a.set_xlabel(r"$x$ [$\mu$m]")
    a.set_ylabel(r"$|n_{\rm ana}-n_{\rm num}|/n_{\rm num}$")
    a.legend(fontsize=7)
    a.set_title("(C) relative error on the trustworthy window", fontsize=9)

    a = ax[1, 1]
    a.plot(x, rate_num, color=C1, lw=1.8, label=r"numerical $-\,d\ln n/dx$")
    a.axhline(kappa, color=C2, lw=1.4, ls="--", label=rf"$\kappa_j$={kappa:.4f} 1/$\mu$m")
    a.axvline(x[i_Ks], color="k", ls=":", lw=1, label=rf"$n=K_s$ at $x$={x[i_Ks]:.1f} $\mu$m")
    a.set_xlim(0, x_trust)
    a.set_ylim(0, max(kappa * 1.6, float(np.nanmax(rate_num[trust])) * 1.1))
    a.set_xlabel(r"$x$ [$\mu$m]")
    a.set_ylabel(r"[1/$\mu$m]")
    a.legend(fontsize=7, loc="lower right")
    a.set_title(r"(D) local decay rate: is $\kappa_j$ the right slope?", fontsize=9)

    fig.suptitle(
        f"fully jammed analytic $n(x)$ vs numerical steady state -- "
        f"$n_0$={n0:g} $\\mu$M, $\\mu$={mu:g} 1/h, $K_s$={Ks:g} $\\mu$M, "
        f"$L$={L:g} $\\mu$m, $dx$={float(s['dx_um']):g} $\\mu$m, "
        f"$\\rho_j$={rho_j:.4f}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .95))
    png = out / ("jammed_analytic_nutrient" + ("_rhojam" if args.rho_jam else "") + ".png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\nwrote {png}")

    # Every plotted array to CSV beside the PNG (rule: plotted arrays are always saved).
    # The figure clips n at floor/10 for display only; the CSV holds the unclipped values.
    stem = png.with_suffix("")
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_um", "rho", "n_num_uM", "n_analytic_uM", "n_fit_uM",
                    "rel_err_analytic", "rel_err_fit", "decay_rate_num_per_um", "trustworthy"])
        for i in range(x.size):
            w.writerow([f"{x[i]:.6g}", f"{rho[i]:.10g}", f"{n_num[i]:.10g}", f"{n_ana[i]:.10g}",
                        f"{n_fit[i]:.10g}", f"{rel[i]:.8g}", f"{rel_fit[i]:.8g}",
                        "" if not np.isfinite(rate_num[i]) else f"{rate_num[i]:.8g}",
                        int(trust[i])])
    with open(f"{stem}_params.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in (("mu_per_h", mu), ("Ks_uM", Ks), ("n0_uM", n0), ("Y", Y), ("Dn", Dn),
                     ("L_um", L), ("rho_j", rho_j), ("rho_j_source",
                                                     "rho_jam" if args.rho_jam else "rho(L)"),
                     ("k_mu_over_Ks", k), ("kappa_j_per_um", kappa), ("tail_amplitude_A_uM", amp),
                     ("noise_floor_uM", floor), ("x_n_eq_Ks_um", x[i_Ks]),
                     ("x_trust_max_um", x_trust), ("x_rho_plateau_um", x[i_plat])):
            w.writerow([k, v if isinstance(v, str) else f"{v:.10g}"])
    print(f"wrote {stem}.csv, {stem}_params.csv")


if __name__ == "__main__":
    main()
