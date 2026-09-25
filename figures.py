"""
figures.py -- the standard figure set for the E. coli / glucose cavity, plus CSV output.

Ports `src/grounded_dself_figure.py` + `src/plot_full_dump.py` with chemotaxis removed.

WHAT CHANGED vs src/:
  * The d_t f1 budget has THREE terms (ADV, SELF, SEL), not four -- there is no drift.  All three
    now get a heatmap; previously SELF was computed but never shown.
  * `chemo_diag.png` is gone.  Its two non-chemotaxis panels (term shares, delta_f) moved here.
  * NEW resolution panel: dx/lambda over time, against the fail (1) and caution (1/3) thresholds.
    The absence of this diagnostic is what let runs with a 500x nutrient drop across ONE cell look
    fine, so it belongs in the standard set rather than in a validator nobody runs.
  * n* fronts come from diagnostics.nstar_front (log-interpolated, one implementation).

    py -3.11 src2/figures.py --regime dilute
    py -3.11 src2/figures.py --n0 1000 --dx 0.5 --T 3000 --outdir jammed_run
    py -3.11 src2/figures.py --regime dilute --no-dump       # skip full_dump.csv (on by default)
    py -3.11 src2/figures.py --replot <rundir> [--outdir D]  # redraw from CSVs, simulate nothing

EVERY PLOTTED ARRAY IS ON DISK AS CSV: run.png draws from profile_final.csv, timeseries.csv,
full_dump.csv and run_monod.csv; initial_state.png from initial_state.csv; fst.png from fst.csv.
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
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.collections import LineCollection                 # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm    # noqa: E402
from matplotlib.cm import ScalarMappable                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config                                                     # noqa: E402
import solver                                                     # noqa: E402
import closures as C                                              # noqa: E402
import diagnostics as D                                           # noqa: E402

C1, C2, CN = "#1f77b4", "#d62728", "#2ca02c"
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Below this nutrient concentration the solve is not trustworthy -- the implicit nutrient step
#: is driven by differences of numbers ~n0, so once n falls ~15 decades below the feed the
#: profile is rounding noise, not a solution.  It is NOT a hard floor: nothing is clipped, the
#: value is used only to SHADE the affected region of the log-scale nutrient panel so the figure
#: says what the draft's captions already assert in words.
N_NOISE_FLOOR = 1.0e-12


# =============================================================================
# Field reconstruction (post-hoc, from the saved frames)
# =============================================================================
def fields(res, p, mask=True):
    """Fluxes, velocities and the d_t f1 budget, rebuilt from saved frames.

        d_t f1 = -u d_x f1                              ADV
               + (1/rho) d_x(D_self rho d_x f1)         SELF
               + f1(1-f1) s(n)                          SEL

    u = J/rho = -D_col d_x ln rho.  `mask=False` writes every cell (for the CSV dump); the default
    blanks unoccupied cells, where d_t f1 is meaningless and would swamp every colour scale.
    """
    x, dx = res["x"], res["dx"]
    R1, R2, NN = res["rho1_profiles"], res["rho2_profiles"], res["n_profiles"]
    N, nt = R1.shape
    rho = R1 + R2
    f1 = np.where(rho > 0, R1 / np.maximum(rho, 1e-300), 0.0)
    floor = max(1e-9, 1e-3 * rho.max())
    keys = ("ADV", "SELF", "SEL", "u", "s", "Jtot", "J1", "J2", "Dcol", "Dself")
    out = {k: np.full((N, nt), np.nan) for k in keys}

    for c in range(nt):
        r, fr, nu = rho[:, c], f1[:, c], NN[:, c]
        occ = (r > max(floor, 0.03 * max(r.max(), 1e-30))) if mask else np.ones(N, bool)
        Dc = np.asarray(C.collective_diffusivity(r, p))
        Ds = np.asarray(C.self_diffusivity(r, p))
        Jf = C.diffusive_flux_dirichlet_left_neumann_right(r, Dc, dx, left_value=0.0)
        Jt = 0.5 * (Jf[:-1] + Jf[1:])
        u = np.where(r > floor, Jt / np.maximum(r, floor), 0.0)
        B = Ds * r
        Bf = 0.5 * (B[:-1] + B[1:])
        Jself = np.zeros(N + 1)
        Jself[1:N] = Bf * (fr[1:] - fr[:-1]) / dx        # = D_self rho d_x f on interior faces
        Jrel = np.zeros(N + 1)
        Jrel[1:N] = -Jself[1:N]
        s = np.asarray(C.g1(nu, p)) - np.asarray(C.g2(nu, p))
        vals = (
            ("ADV", -u * np.gradient(fr, x)),
            ("SELF", np.where(r > floor, (Jself[1:] - Jself[:-1]) / dx / np.maximum(r, floor), 0.0)),
            ("SEL", fr * (1.0 - fr) * s),
            ("u", u), ("s", s), ("Jtot", Jt),
            ("J1", Jt * fr + 0.5 * (Jrel[:-1] + Jrel[1:])),
            ("J2", Jt * (1.0 - fr) - 0.5 * (Jrel[:-1] + Jrel[1:])),
            ("Dcol", Dc), ("Dself", Ds),
        )
        for k, v in vals:
            out[k][occ, c] = np.asarray(v)[occ]

    out["rho1"] = R1; out["rho2"] = R2; out["rho"] = rho
    out["f1"] = f1; out["h"] = 2.0 * f1 - 1.0; out["n"] = NN
    _check_fields(out)
    return out


def _heat(fig, ax, x, t, Z, title, cmap, diverging=False, clip_pct=None, log=False):
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    Zp = np.array(Z, dtype=float)
    if log:
        Zp = np.log10(np.maximum(Zp, 1e-20))
    if diverging:
        a = (float(np.nanpercentile(np.abs(Zp), clip_pct)) if clip_pct
             else float(np.nanmax(np.abs(Zp)))) or 1.0
        Zp = np.clip(Zp, -a, a)
        norm = TwoSlopeNorm(0.0, -a, a)
    else:
        norm = Normalize(np.nanmin(Zp), np.nanmax(Zp))
    im = ax.imshow(Zp.T, origin="lower", aspect="auto", cmap=cm, norm=norm,
                   extent=[x[0], x[-1], t[0], t[-1]])
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(r"$x$ [$\mu$m]"); ax.set_ylabel(r"$t$ [h]")
    fig.colorbar(im, ax=ax)
    return im


# =============================================================================
# run.png
# =============================================================================
def figure_run(res, p, path):
    x, t, dx = res["x"], res["t"], res["dx"]
    R1, R2, NN = res["rho1_profiles"], res["rho2_profiles"], res["n_profiles"]
    nstar = float(C.monod_crossover(p))
    mo, dp, F1, _ = D.inverse_decomposition(res, p)
    df = D.delta_f_series(res, p)
    M1, M2 = D.masses(res)
    rho_t = (R1 + R2).max(axis=0) / p.rho_jam
    xf = D.nstar_front_series(x, NN, nstar)
    r1, r2 = R1[:, -1], R2[:, -1]
    rho = r1 + r2
    f1 = np.where(rho > 0, r1 / np.maximum(rho, 1e-300), 0.0)
    nf = NN[:, -1]

    fig, ax = plt.subplots(3, 3, figsize=(16, 12))

    a = ax[0, 0]
    a.plot(x, r1, color=C1, lw=1.8, label=r"$\rho_1$ (scavenger)")
    a.plot(x, r2, color=C2, lw=1.8, label=r"$\rho_2$ (glutton)")
    a.plot(x, rho, color="0.3", lw=1.0, ls="--", label=r"$\rho$")
    if p.rho_jam <= 1.5 * rho.max():
        a.axhline(p.rho_jam, color="0.6", ls=":", lw=.9)
        a.text(x[-1], p.rho_jam, r" $\rho_{\rm jam}$", va="bottom", ha="right",
               fontsize=8, color="0.5")
        a.axhline(p.rho_c, color="crimson", ls=":", lw=.9)
        a.text(x[-1], p.rho_c, r" $\rho_c$", va="bottom", ha="right", fontsize=8, color="crimson")
    else:
        a.set_ylim(0, 1.18 * max(rho.max(), 1e-12))
        a.text(0.98, 0.95, rf"$\rho_{{\max}}/\rho_{{\rm jam}}$={rho.max()/p.rho_jam:.4f}",
               transform=a.transAxes, ha="right", va="top", fontsize=8, color="0.45")
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel("packing fraction")
    a.legend(fontsize=8); a.set_title("(A) final densities", fontsize=10)

    a = ax[0, 1]
    a.plot(x, f1, color=C1, lw=1.8)
    a.axhline(0.5, color="0.6", ls=":", lw=.9)
    if np.isfinite(xf[-1]):
        a.axvline(xf[-1], color=CN, ls="--", lw=1.2, label=r"$n^\ast$ front")
        a.legend(fontsize=8)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$f_1$")
    a.set_title("(B) final composition", fontsize=10)

    a = ax[0, 2]
    # Draw the whole profile faded, then overplot only the trustworthy stretch solid, so the
    # curve itself shows where it stops being a solution.  Nothing is clipped or altered.
    _np = np.maximum(nf, 1e-20)
    _ok = _np >= N_NOISE_FLOOR
    a.semilogy(x, _np, color=CN, lw=1.1, ls=":", alpha=0.45)
    a.semilogy(x, np.where(_ok, _np, np.nan), color=CN, lw=1.8)
    a.axhline(nstar, color="k", ls="--", lw=1.2)
    a.axhline(p.n0, color="0.6", ls=":", lw=.9)
    _lo = a.get_ylim()[0]
    if _lo < N_NOISE_FLOOR:
        a.axhspan(_lo, N_NOISE_FLOOR, color="0.75", alpha=0.35, lw=0, zorder=0)
        a.axhline(N_NOISE_FLOOR, color="0.35", ls="-.", lw=1.0, zorder=1)
        a.text(0.02, 0.02, "numerical noise:  $n<10^{-12}$ $\\mu$M\n(profile not trustworthy)",
               transform=a.transAxes, ha="left", va="bottom", fontsize=7.5, color="0.25",
               bbox=dict(fc="white", ec="0.7", lw=.6, alpha=.8))
        if (~_ok).any():                      # where the profile first enters the noise floor
            _xn = float(x[int(np.argmax(~_ok))])
            a.axvline(_xn, color="0.45", ls="-.", lw=0.9, zorder=1)
            a.annotate(r"$n<10^{-12}$ beyond %.0f $\mu$m" % _xn, xy=(_xn, N_NOISE_FLOOR),
                       xytext=(6, 10), textcoords="offset points", fontsize=7.5, color="0.3")
    a.text(.985, .955, rf"$n_0$={p.n0:g} $\mu$M" "\n" rf"$n^\ast$={nstar:.2f} $\mu$M",
           transform=a.transAxes, ha="right", va="top", fontsize=8,
           bbox=dict(fc="white", ec="0.8", lw=.6, alpha=.85))
    if np.isfinite(xf[-1]):
        a.axvline(xf[-1], color=CN, ls="--", lw=1.0)
        a.annotate(rf"$x_{{n^\ast}}$={xf[-1]:.1f} $\mu$m", xy=(xf[-1], nstar),
                   xytext=(14, -26), textcoords="offset points", fontsize=8, color=CN)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$n$ [$\mu$M]")
    a.set_title(r"(C) final nutrient and the $n^\ast$ crossover", fontsize=10)

    a = ax[1, 0]
    xlo, xhi = min(p.Ks1, nstar) * 1e-2, max(p.n0 * 3.0, nstar * 4.0)
    ng = np.logspace(np.log10(xlo), np.log10(xhi), 400)
    gg1, gg2 = C.g1(ng, p), C.g2(ng, p)
    a.semilogx(ng, gg1, color=C1, lw=1.8, label=rf"$g_1$: $\mu_1$={p.mu1:g}, $K_1$={p.Ks1:g}")
    a.semilogx(ng, gg2, color=C2, lw=1.8, label=rf"$g_2$: $\mu_2$={p.mu2:g}, $K_2$={p.Ks2:g}")
    a.axvspan(xlo, nstar, color=C1, alpha=.07); a.axvspan(nstar, xhi, color=C2, alpha=.07)
    a.axvline(nstar, color="k", ls="--", lw=1.2); a.axvline(p.n0, color="0.5", ls=":", lw=1.0)
    a.set_xlim(xlo, xhi)
    ytop = float(max(gg1.max(), gg2.max())) * 1.15
    a.set_ylim(0, ytop)
    a.text(nstar, ytop * .99, rf" $n^\ast$={nstar:.2f}", fontsize=8, va="top")
    a.text(xlo * 1.6, ytop * .06, "strain 1 wins\n(high affinity)", fontsize=8, color=C1)
    a.text(nstar * 1.35, ytop * .06, "strain 2 wins\n(high rate)", fontsize=8, color=C2)
    a.set_xlabel(r"$n$ [$\mu$M]"); a.set_ylabel(r"$g_i(n)$ [h$^{-1}$]")
    a.legend(fontsize=7, loc="upper left")
    a.set_title("(D) Monod growth and the crossover", fontsize=10)

    a = ax[1, 1]
    a.plot(t, F1, color=C1, lw=2.0)
    a.axhline(0.5, color="0.6", ls=":", lw=.9)
    j = int(np.argmin(F1))
    a.plot(t[j], F1[j], "o", color=C2, ms=6)
    a.annotate(rf"min {F1[j]:.4g} at {t[j]:.0f} h", xy=(t[j], F1[j]), xytext=(12, 10),
               textcoords="offset points", fontsize=8, color=C2)
    a.plot(t[-1], F1[-1], "o", color=C1, ms=6)
    a.annotate(rf"final {F1[-1]:.4f}", xy=(t[-1], F1[-1]), xytext=(-70, 8),
               textcoords="offset points", fontsize=8, color=C1)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"$F_1$")
    a.set_title("(E) global composition over time", fontsize=10)

    a = ax[1, 2]
    a.plot(t, mo, color=C1, lw=1.6, label="mouth inverse")
    a.plot(t, dp, color=C2, lw=1.6, label="deep inverse")
    a.plot(t, df, color="k", lw=1.3, ls="-.", label=r"$\Delta f$ (mouth$-$deep)")
    a.axhline(0.0, color="0.7", lw=.7)
    a2 = a.twinx()
    a2.plot(t, rho_t, color="0.5", lw=1.1, ls=":")
    a2.axhline(1.0, color="0.7", ls=":", lw=.8)
    a2.set_ylabel(r"$\rho_{\max}/\rho_{\rm jam}$ (dotted)", fontsize=8, color="0.4")
    a.set_xlabel(r"$t$ [h]"); a.set_ylabel("inverse-sorted fraction")
    a.legend(fontsize=7.5, loc="upper left")
    a.set_title(r"(F) inverse sorting, $\Delta f$, jamming", fontsize=10)
    a.text(.5, -.30, r"$\Delta f>0$ is inverse sorting and cannot be faked by extinction; "
                     r"the deep fraction is pinned at its ceiling",
           transform=a.transAxes, ha="center", va="top", fontsize=6.8, color="0.45")

    a = ax[2, 0]
    a.plot(t, M1 + M2, color="0.2", lw=1.8, label=r"$M_{\rm tot}$")
    a.plot(t, M1, color=C1, lw=1.3, label=r"$M_1$")
    a.plot(t, M2, color=C2, lw=1.3, label=r"$M_2$")
    a.set_yscale("log"); a.set_xlabel(r"$t$ [h]"); a.set_ylabel("biomass")
    a.legend(fontsize=8); a.set_title("(G) biomass totals", fontsize=10)

    a = ax[2, 1]
    F = np.where((R1 + R2) > 1e-12 * max((R1 + R2).max(), 1e-30),
                 R1 / np.maximum(R1 + R2, 1e-300), np.nan)
    im = a.pcolormesh(t, x, F, cmap="coolwarm_r", vmin=0, vmax=1, shading="auto")
    a.plot(t, xf, color="k", lw=1.4, label=r"$n^\ast$ front")
    a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"$x$ [$\mu$m]")
    a.legend(fontsize=8, loc="upper right"); fig.colorbar(im, ax=a, label=r"$f_1$")
    a.set_title("(H) composition in space-time", fontsize=10)

    a = ax[2, 2]
    with np.errstate(divide="ignore"):
        Ln = np.log10(np.maximum(NN, 1e-20))
    im = a.pcolormesh(t, x, Ln, cmap="viridis", shading="auto")
    a.contour(t, x, NN, levels=[nstar], colors="w", linewidths=1.6)
    a.plot([], [], color="w", lw=1.6, label=r"$n=n^\ast$")
    a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"$x$ [$\mu$m]")
    a.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=a, label=r"$\log_{10} n$ [$\mu$M]")
    a.set_title("(I) nutrient in space-time", fontsize=10)

    for row in ax:
        for a in row:
            a.grid(alpha=.2)
    fig.suptitle(rf"E. coli / glucose cavity — $n_0$={p.n0:g} $\mu$M, $n^\ast$={nstar:.3f} $\mu$M, "
                 rf"$T$={t[-1]:g} h, $dx$={dx:g} $\mu$m, $\rho_{{\rm peak}}$={res['rho_peak']:.4f}",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.005, 1, 0.965))
    fig.savefig(path, dpi=150); plt.close(fig)

    # Panel (D) is the only one drawn from arrays that exist nowhere else: the Monod curves are
    # evaluated here, not saved by the run.  Every other panel is in the run's own CSVs --
    # (A)-(C) profile_final.csv, (E)-(G) timeseries.csv, (H)-(I) full_dump.csv (f1, n) -- so
    # `figures.py --replot <rundir>` redraws the whole figure without simulating.
    with open(os.path.splitext(path)[0] + "_monod.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_uM", "g1_per_h", "g2_per_h", "n_star_uM"])
        for k in range(ng.size):
            w.writerow([f"{ng[k]:.8g}", f"{gg1[k]:.10g}", f"{gg2[k]:.10g}", f"{nstar:.10g}"])


def figure_initial_state(res, p, ic, path):
    x = res["x"]
    rho0 = C.initial_total_density(x, res["L"], p, ic)
    f0 = C.initial_fraction(x, ic)
    n0 = C.initial_nutrient(x, p, ic)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].semilogy(x, np.maximum(rho0 * f0, 1e-30), color=C1, label=r"$\rho_1$")
    ax[0].semilogy(x, np.maximum(rho0 * (1 - f0), 1e-30), color=C2, label=r"$\rho_2$")
    ax[0].semilogy(x, np.maximum(rho0, 1e-30), color="0.3", ls="--", label=r"$\rho$")
    ax[0].set_ylim(1e-10, 1); ax[0].set_title("initial biomass (log-y)"); ax[0].legend(fontsize=8)
    ax[1].plot(x, f0, color=C1); ax[1].set_ylim(0, 1); ax[1].set_title(r"initial $f_1$")
    ax[2].plot(x, n0, color=CN); ax[2].set_title(r"initial $n$ [$\mu$M]")
    for a in ax:
        a.set_xlabel(r"$x$ [$\mu$m]"); a.grid(alpha=.25)
    fig.suptitle(rf"initial state: {ic.rho_mode}, $\rho_{{\rm seed}}$={ic.rho_seed:g}, "
                 rf"seed $f_1$={ic.seed_f1:g}, {ic.nutrient_mode}", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    # the arrays drawn, unclipped (the log panel floors them at 1e-30 for display only)
    with open(os.path.splitext(path)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_um", "rho1", "rho2", "rho", "f1", "n_uM"])
        for i in range(x.size):
            w.writerow([f"{x[i]:.6g}", f"{rho0[i] * f0[i]:.10g}", f"{rho0[i] * (1 - f0[i]):.10g}",
                        f"{rho0[i]:.10g}", f"{f0[i]:.8g}", f"{n0[i]:.10g}"])


# =============================================================================
# flux_bump.png -- 4x3, with the resolution panel replacing DRIFT
# =============================================================================
def figure_flux_bump(res, p, F, path):
    x, t, dx = res["x"], res["t"], res["dx"]
    nstar = float(C.monod_crossover(p))
    NN = res["n_profiles"]
    xf = D.nstar_front_series(x, NN, nstar)
    df = D.delta_f_series(res, p)

    xb = np.full(t.size, np.nan)
    for c in range(t.size):
        occ = np.isfinite(F["ADV"][:, c])
        if occ.sum() > 4:
            idx = np.where(occ)[0]
            xb[c] = x[idx[int(np.argmax(F["f1"][idx, c]))]]

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))
    _heat(fig, ax[0, 0], x, t, np.where(np.isfinite(F["ADV"]), F["f1"], np.nan),
          r"(A) $f_1$   (bump black, $n^\ast$ cyan)", "coolwarm_r")
    ax[0, 0].plot(xb, t, "k", lw=1.2); ax[0, 0].plot(xf, t, color="c", lw=1.2)
    _heat(fig, ax[0, 1], x, t, F["rho"], r"(B) $\rho$", "viridis")
    ax[0, 1].plot(xf, t, color="c", lw=1.2)
    _heat(fig, ax[0, 2], x, t, F["n"], r"(C) $\log_{10} n$", "viridis", log=True)
    ax[0, 2].plot(xf, t, color="w", lw=1.2)

    _heat(fig, ax[1, 0], x, t, F["ADV"], r"(D) ADV $=-u\,\partial_x f_1$",
          "coolwarm", diverging=True, clip_pct=97)
    _heat(fig, ax[1, 1], x, t, F["SELF"],
          r"(E) SELF $=\frac{1}{\rho}\partial_x(D_{\rm self}\rho\,\partial_x f_1)$",
          "coolwarm", diverging=True, clip_pct=97)
    _heat(fig, ax[1, 2], x, t, F["SEL"], r"(F) SEL $=f_1(1-f_1)s(n)$",
          "coolwarm", diverging=True, clip_pct=97)

    cc = int(np.argmax(df)) if np.isfinite(df).any() and df.max() > 0 else t.size // 2
    a = ax[2, 0]
    occ = np.isfinite(F["ADV"][:, cc])
    tot = np.full_like(F["ADV"][:, cc], np.nan)
    if occ.any():
        tot[occ] = sum(F[k][occ, cc] for k in ("ADV", "SELF", "SEL"))
    for k, col in (("ADV", "#1f77b4"), ("SELF", "#9467bd"), ("SEL", "#d62728")):
        a.plot(x, F[k][:, cc], color=col, lw=1.5, label=k)
    a.plot(x, tot, color="0.5", ls="--", lw=1.0, label="sum")
    if np.isfinite(xf[cc]):
        a.axvline(xf[cc], color="c", ls="--", lw=1.0)
    a.axhline(0, color="0.7", lw=.7)
    if occ.any():
        a.set_xlim(x[occ].min() - 10, x[occ].max() + 10)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$\partial_t f_1$ [h$^{-1}$]")
    a.legend(fontsize=7)
    a.set_title(rf"(G) $\partial_t f_1$ budget at $t$={t[cc]:.0f} h (peak $\Delta f$)",
                fontsize=9, fontweight="bold")

    _heat(fig, ax[2, 1], x, t, F["u"], r"(H) $u=J/\rho=-D_{\rm col}\partial_x\ln\rho$",
          "coolwarm", diverging=True, clip_pct=97)
    ax[2, 1].plot(xb, t, "k", lw=1.0)

    a = ax[2, 2]
    a.plot(x, F["Jtot"][:, -1], color="0.2", lw=1.6, label=r"$J_{\rm tot}$")
    a.plot(x, F["J1"][:, -1], color=C1, lw=1.3, label=r"$J_1$")
    a.plot(x, F["J2"][:, -1], color=C2, lw=1.3, label=r"$J_2$")
    a.axhline(0, color="0.7", lw=.7)
    a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel("flux")
    a.legend(fontsize=8); a.set_title("(I) final biomass fluxes", fontsize=9, fontweight="bold")
    a.grid(alpha=.25)

    # --- (J) THE RESOLUTION PANEL -- new, and the reason the src/ failures went unseen ---
    a = ax[3, 0]
    dol = np.array([D.front_resolution(x, NN[:, j], nstar, dx)["dx_over_lambda"]
                    for j in range(t.size)])
    a.plot(t, dol, color="k", lw=1.7)
    a.axhline(1.0, color=C2, ls="--", lw=1.2)
    a.axhline(1 / 3, color="darkorange", ls=":", lw=1.2)
    a.axhspan(1.0, max(2.0, np.nanmax(dol) * 1.1 if np.isfinite(dol).any() else 2.0),
              color=C2, alpha=.10)
    a.axhspan(1 / 3, 1.0, color="darkorange", alpha=.08)
    a.set_yscale("log")
    a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"$\Delta x/\lambda_{\rm front}$")
    fin = dol[np.isfinite(dol)]
    a.text(.98, .04,
           (rf"final $\Delta x/\lambda$={fin[-1]:.2f}" "\n"
            rf"$\lambda_{{\rm front}}$="
            rf"{D.front_resolution(x, NN[:, -1], nstar, dx)['lambda_front']:.2f} $\mu$m"
            "\n" rf"$\Delta x$={dx:g} $\mu$m") if fin.size else "no crossing",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8,
           bbox=dict(fc="white", ec="0.8", lw=.6, alpha=.9))
    a.set_title(r"(J) is the nutrient layer resolved?  ($>1$ fails, $>1/3$ caution)",
                fontsize=9, fontweight="bold")
    a.grid(alpha=.25)

    a = ax[3, 1]
    totals = {k: np.nansum(np.abs(F[k]), axis=0) for k in ("ADV", "SELF", "SEL")}
    den = sum(totals.values()); den[den == 0] = np.nan
    for k, col in (("ADV", "#1f77b4"), ("SELF", "#9467bd"), ("SEL", "#d62728")):
        a.plot(t, totals[k] / den, color=col, lw=1.5, label=k)
    a.set_ylim(0, 1); a.legend(fontsize=8)
    a.set_xlabel(r"$t$ [h]"); a.set_ylabel(r"share of $\sum|{\rm term}|$")
    a.set_title(r"(K) which term drives $\partial_t f_1$?", fontsize=9, fontweight="bold")
    a.grid(alpha=.25)
    a.text(.5, -.30, "report ORDERING, not calibrated magnitudes — the ranking is mesh-stable, "
                     "the value is not", transform=a.transAxes, ha="center", va="top",
           fontsize=6.8, color="0.45")

    a = ax[3, 2]
    a.plot(t, df, color="k", lw=1.8)
    a.axhline(0, color="0.7", lw=.8)
    a.fill_between(t, 0, df, where=(df > 0), color=C1, alpha=.15)
    a.set_xlabel(r"$t$ [h]")
    a.set_ylabel(r"$\Delta f=\langle f\rangle_{\rm mouth}-\langle f\rangle_{\rm deep}$")
    a.set_title(r"(L) sorting index ($>0$ = inverse; extinction-proof)",
                fontsize=9, fontweight="bold")
    a.grid(alpha=.25)

    fig.suptitle(rf"Flux and budget — $n_0$={p.n0:g} $\mu$M, $n^\ast$={nstar:.3f} $\mu$M, "
                 rf"$dx$={dx:g} $\mu$m", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(path, dpi=150); plt.close(fig)


# =============================================================================
# profiles_all.png -- every stored profile, coloured by time
# =============================================================================
MAX_DECADES = 12.0
PROFILE_FIELDS = ("rho1", "rho2", "rho", "f1", "h", "n", "s", "Dcol", "Dself",
                  "u", "Jtot", "J1", "J2", "ADV", "SELF", "SEL")


def _check_fields(out):
    """Every PROFILE_FIELD must actually be produced.

    The dump and profiles_all both do `[c for c in PROFILE_FIELDS if c in F]`, which
    SILENTLY drops anything missing -- that is how rho1/rho2 vanished from a 23.7 MB
    dump and from the figure without a word.  Fail loudly instead.
    """
    missing = [k for k in PROFILE_FIELDS if k not in out]
    if missing:
        raise KeyError(f"fields() did not produce {missing}, but PROFILE_FIELDS lists "
                       f"them; they would be silently dropped from the dump and figure")


def _scale_for(name, Z):
    fin = Z[np.isfinite(Z)]
    if fin.size == 0:
        return "linear", None
    lo, hi = float(fin.min()), float(fin.max())
    if name == "f1":
        return "linear", (-0.02, 1.02)
    if lo < 0.0:
        return "linear", None
    pos = fin[fin > 0.0]
    if pos.size and hi > 0 and (np.log10(hi) - np.log10(pos.min())) > 3.0:
        floor = max(float(pos.min()), hi * 10.0 ** (-MAX_DECADES))
        return "log", (floor * .5, hi * 2.0)
    return "linear", None


def figure_profiles_all(res, p, F, path, stride=1):
    """Every stored time step's full profile, overlaid, coloured by t.

    LineCollection rather than one plot() per curve: the difference is minutes vs seconds and it
    is the only reason "every time step" is drawable.  Panel titles report drawn/total, so a
    thinned plot cannot be mistaken for a complete one.
    """
    x, t = res["x"], res["t"]
    names = [n for n in PROFILE_FIELDS if n in F]
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow), squeeze=False,
                             constrained_layout=True)
    norm = Normalize(float(t.min()), float(t.max()))
    cmap = plt.get_cmap("viridis")
    nstar = float(C.monod_crossover(p))
    front = D.nstar_front_series(x, res["n_profiles"], nstar)
    fr = front[np.isfinite(front)]

    for k, name in enumerate(names):
        a = axes[k // ncol][k % ncol]
        Z = F[name].T                                   # (n_frames, n_cells)
        rows = np.arange(0, Z.shape[0], stride)
        scale, ylim = _scale_for(name, Z)
        Zp = np.where(Z > 0.0, Z, ylim[0]) if scale == "log" else Z
        lc = LineCollection([np.column_stack([x, Zp[j]]) for j in rows],
                            cmap=cmap, norm=norm, linewidths=.6, alpha=.85)
        lc.set_array(t[rows]); a.add_collection(lc)
        a.set_yscale(scale); a.set_xlim(x.min(), x.max())
        if ylim is not None:
            a.set_ylim(*ylim)
        else:
            fin = Zp[np.isfinite(Zp)]
            if fin.size:
                lo, hi = float(fin.min()), float(fin.max())
                pad = .05 * (hi - lo) if hi > lo else (abs(hi) or 1.) * .05
                a.set_ylim(lo - pad, hi + pad)
                if lo < 0 < hi:
                    a.axhline(0, color="0.6", lw=.7, zorder=0)
        if scale == "linear":
            a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        if name == "n":
            a.axhline(nstar, color="crimson", ls=":", lw=1.4, zorder=4,
                      label=rf"$n^\ast$={nstar:.3g} $\mu$M")
        if fr.size:
            if fr.max() > fr.min():
                a.axvspan(fr.min(), fr.max(), color="0.55", alpha=.18, lw=0, zorder=0,
                          label=r"$n^\ast$ front sweep")
            a.axvline(fr[-1], color="k", ls="--", lw=1.1, zorder=4,
                      label=rf"$n^\ast$ front, final ({fr[-1]:.0f} $\mu$m)")
        if k == 0 or name == "n":
            a.legend(fontsize=7, loc="best", framealpha=.85)
        a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(name, fontsize=9)
        a.set_title(f"{name}   [{rows.size}/{Z.shape[0]} steps]", fontsize=9, fontweight="bold")
        a.grid(alpha=.2)
    for k in range(len(names), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=.015, pad=.01).set_label(r"$t$ [h]")
    fig.suptitle(rf"Full spatial profile at every stored time step — $n_0$={p.n0:g} $\mu$M",
                 fontweight="bold", fontsize=14)
    fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
# CSV output
# =============================================================================
def figure_fst(res, p, path):
    """F_ST(t) with the raw sd beside it, and the conditioning floor made visible.

    Two panels because F_ST alone is not safe to read: it is 0/0 in the fixation limit, so the
    right-hand panel shows sd(f) against sd_max = sqrt(<f>(1-<f>)).  Where sd_max approaches the
    floor the ratio is meaningless and F_ST is drawn broken rather than plausible.
    """
    fst, sd, sdmax, fbar = D.fst_series(res, p)
    t = res["t"]
    summ = D.fst_summary(res, p)

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax[0].plot(t, fst, color=C1, lw=2.2)
    if np.isfinite(summ["F_ST_peak"]):
        ax[0].plot([summ["t_F_ST_peak"]], [summ["F_ST_peak"]], "o", color=C1, ms=8,
                   mfc="none", mew=1.8,
                   label=f"peak {summ['F_ST_peak']:.3f} at t={summ['t_F_ST_peak']:.0f} h")
        ax[0].legend(frameon=False, fontsize=9)
    for lv, lab in ((0.05, "little"), (0.15, "great")):
        ax[0].axhline(lv, color="0.75", lw=0.8, ls=":")
        ax[0].annotate(lab, (t[-1], lv), fontsize=7.5, color="0.55", va="bottom", ha="right")
    ax[0].set(xlabel="t [h]", ylabel=r"$F_{ST}$", ylim=(0, None))
    ax[0].set_title("spatial structure of the composition", fontsize=10, loc="left")

    ax[1].plot(t, sd, color=C1, lw=2.2, label=r"sd$(f)$ realised")
    ax[1].plot(t, sdmax, color="0.55", lw=1.6, ls="--", label=r"sd$_{\max}=\sqrt{f(1-f)}$")
    ax[1].axhline(0.05, color=C2, lw=1.0, ls=":", label="conditioning floor")
    ax[1].set(xlabel="t [h]", ylabel="composition spread")
    ax[1].set_title(r"$F_{ST}$ is their squared ratio", fontsize=10, loc="left")
    ax[1].legend(frameon=False, fontsize=8.5)
    for A in ax:
        A.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)

    # fst.csv: the arrays both panels draw.  The --fst help text always promised this file; until
    # 2026-09-23 only the PNG was written.
    with open(os.path.splitext(path)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_h", "F_ST", "sd_f", "sd_f_max", "f_bar"])
        for k in range(len(t)):
            w.writerow([f"{t[k]:.6g}", f"{fst[k]:.8g}", f"{sd[k]:.8g}", f"{sdmax[k]:.8g}",
                        f"{fbar[k]:.8g}"])
    return path


def write_csvs(outdir, res, p, val, dump=False, F=None, fst=False, num=None):
    x, t, dx = res["x"], res["t"], res["dx"]
    R1, R2, NN = res["rho1_profiles"], res["rho2_profiles"], res["n_profiles"]
    nstar = float(C.monod_crossover(p))
    mo, dp, F1, _ = D.inverse_decomposition(res, p)
    df = D.delta_f_series(res, p)
    M1, M2 = D.masses(res)
    xf = D.nstar_front_series(x, NN, nstar)

    with open(os.path.join(outdir, "timeseries.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        fs, fsd, fsm, _fb = D.fst_series(res, p)
        w.writerow(["t_h", "F1", "inv_mouth", "inv_deep", "delta_f", "x_nstar_um",
                    "rho_max", "rho_max_over_jam", "M1", "M2", "dx_over_lambda",
                    "F_ST", "sd_f", "sd_f_max"])
        for j in range(len(t)):
            rho = R1[:, j] + R2[:, j]
            r = D.front_resolution(x, NN[:, j], nstar, dx)["dx_over_lambda"]
            w.writerow([f"{t[j]:.6g}", f"{F1[j]:.8g}", f"{mo[j]:.8g}", f"{dp[j]:.8g}",
                        f"{df[j]:.8g}", "" if not np.isfinite(xf[j]) else f"{xf[j]:.6g}",
                        f"{rho.max():.8g}", f"{rho.max()/p.rho_jam:.8g}",
                        f"{M1[j]:.8g}", f"{M2[j]:.8g}",
                        "" if not np.isfinite(r) else f"{r:.6g}",
                        "" if not np.isfinite(fs[j]) else f"{fs[j]:.8g}",
                        f"{fsd[j]:.8g}", f"{fsm[j]:.8g}"])

    with open(os.path.join(outdir, "profile_final.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_um", "rho1", "rho2", "rho", "f1", "n"])
        r1, r2 = R1[:, -1], R2[:, -1]
        rho = r1 + r2
        f1 = np.where(rho > 0, r1 / np.maximum(rho, 1e-300), 0.0)
        for i in range(len(x)):
            w.writerow([f"{x[i]:.6g}", f"{r1[i]:.10g}", f"{r2[i]:.10g}", f"{rho[i]:.10g}",
                        f"{f1[i]:.8g}", f"{NN[i, -1]:.10g}"])

    # biomass in CELLS: M_i [um] * A [um^2] / V [um^3].  A is an ANALYSIS parameter (the chamber
    # cross-section), not part of the 1D equations -- if it is absent these come out blank.
    _A = res.get("cross_section_um2"); _V = res.get("cell_volume_um3")
    if _A and _V:
        _c1, _c2 = M1 * _A / _V, M2 * _A / _V
        _mn1, _mn2 = float(_c1.min()), float(_c2.min())
        _tm1 = float(t[int(np.argmin(_c1))])
        _nbelow = int(((_c1 < 1.0) | (_c2 < 1.0)).sum())
    else:
        _mn1 = _mn2 = _tm1 = _nbelow = ""

    # Same quantities on the PER-STEP mass history.  The frame-based ones above are a minimum over
    # n_save points -- on the jammed run a 25 h grid, which steps clean over the t ~ 1.6 h washout
    # dip and reports the seed as the minimum.  Kept side by side so the difference is visible.
    _MH = res.get("mass_history")
    _sm1 = _sm2 = _stm1 = _sbelow = ""
    if _MH is not None and len(_MH) and _A and _V:
        _st, _sM1, _sM2 = _MH[:, 0], _MH[:, 1], _MH[:, 2]
        _sc1, _sc2 = _sM1 * _A / _V, _sM2 * _A / _V
        _sm1, _sm2 = float(_sc1.min()), float(_sc2.min())
        _stm1 = float(_st[int(np.argmin(_sc1))])
        _sbelow = int(((_sc1 < 1.0) | (_sc2 < 1.0)).sum())
        with open(os.path.join(outdir, "mass_history.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_h", "M1", "M2", "cells1", "cells2"])
            for _k in range(len(_st)):
                w.writerow([f"{_st[_k]:.6g}", f"{_sM1[_k]:.10g}", f"{_sM2[_k]:.10g}",
                            f"{_sc1[_k]:.8g}", f"{_sc2[_k]:.8g}"])

    rows = [("n0_uM", p.n0), ("n_star_uM", nstar), ("mu1", p.mu1), ("Ks1", p.Ks1),
            ("mu2", p.mu2), ("Ks2", p.Ks2), ("Y", p.Y), ("Dn", p.Dn), ("D0", p.D0),
            ("rho_jam", p.rho_jam), ("rho_c", p.rho_c), ("rho_max", p.rho_max),
            ("dself_shape", p.dself_shape), ("Dself_floor", p.Dself_floor),
            ("Dself_jam_slope", p.Dself_jam_slope),
            ("rho_mode", res.get("rho_mode", "")), ("fraction_mode", res.get("fraction_mode", "")),
            ("nutrient_mode", res.get("nutrient_mode", "")), ("seed_f1", res.get("seed_f1", "")),
            ("rho_seed", res.get("rho_seed", "")),
            ("cross_section_um2", res.get("cross_section_um2", "")),
            ("cell_volume_um3", res.get("cell_volume_um3", "")),
            ("inoculum_cells", res.get("inoculum_cells", "")),
            ("min_cells1", _mn1), ("min_cells2", _mn2),
            ("t_min_cells1_h", _tm1), ("frames_below_one_cell", _nbelow),
            ("min_cells1_perstep", _sm1), ("min_cells2_perstep", _sm2),
            ("t_min_cells1_perstep_h", _stm1), ("steps_below_one_cell", _sbelow),
            ("L_um", res["L"]), ("dx_um", dx), ("N", res["N"]), ("dt_h", res["dt"]),
            ("T_end_h", t[-1]),
            # Scheme provenance.  Until 2026-09-21 summary.csv recorded rho_seed but NOT the
            # limiter, so an untagged folder was ambiguous about the one setting that changes
            # the label transport -- referee M4 ("the scheme behind each figure is ambiguous").
            # The CFL evidence M4 also asked for is already below as adv_substeps_max /
            # adv_capped_steps (from the validator, which sees EVERY step -- do not recompute
            # them from history_rows, which is frame-sampled and understates the max ~10x).
            # These three record the settings those numbers should be read against.
            ("label_limiter", getattr(num, "label_limiter", "")),
            ("advection_cfl", getattr(num, "advection_cfl", "")),
            ("max_advection_substeps", getattr(num, "max_advection_substeps", "")),
            ("F1_final", val["F1_final"]), ("F1_min", val["F1_min"]),
            ("t_F1_min_h", val["t_F1_min"]), ("F1_turned", val["F1_turned"]),
            ("delta_f_max", val["delta_f_max"]), ("delta_f_final", val["delta_f_final"]),
            ("int_inv_mouth", val["int_inv_mouth"]), ("int_inv_deep", val["int_inv_deep"]),
            ("peak_inv_mouth", float(mo.max())), ("peak_inv_deep", float(dp.max())),
            ("x_nstar_final_um", val["x_nstar"]),
            ("rho_peak", val["rho_peak"]), ("rho_max_over_jam", val["rho_max_over_jam"]),
            ("q_min", val["q_min"]),
            ("lambda_front_um", val["lambda_front"]), ("dx_over_lambda", val["dx_over_lambda"]),
            ("front_drop_factor", val["front_drop_factor"]),
            ("M1_bottleneck", val["M1_min"]), ("t_M1_bottleneck_h", val["t_M1_min"]),
            ("M2_bottleneck", val["M2_min"]), ("t_M2_bottleneck_h", val["t_M2_min"]),
            ("dF1_per_1000h_late", val["dF1_per_1000h_late"]),
            ("drho_per_1000h_late", val["drho_per_1000h_late"]),
            ("f_min", val["f_min"]), ("f_max", val["f_max"]),
            ("adv_substeps_max", val["adv_substeps_max"]),
            ("adv_capped_steps", val["adv_capped_steps"]),
            ("reaction_tol_relative", res.get("reaction_tol_relative", "")),
            ("boundary_flux_linf", res["max_resource_drift"]),
            ("diffusion_failures", res["diffusion_failures"]),
            ("admissible", val["admissible"]), ("success", res["success"]),
            ("message", res["message"])]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["quantity", "value"])
        for k, v in rows:
            w.writerow([k, v])

    # history.csv -- the per-step numerics the solver computes and src2 was discarding entirely:
    # advection substep counts, Newton iteration counts, clip magnitudes, budget residuals.
    # Without this an advection substep cap or a silently clipped label is invisible after the run.
    hist = res.get("history_rows") or []
    if hist:
        keys = []
        for row in hist:                       # union, in first-seen order (row 0 has no diag keys)
            keys += [k for k in row if k != "t_h" and k not in keys]
        with open(os.path.join(outdir, "history.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_h"] + keys)
            for row in hist:
                w.writerow([f"{row['t_h']:.6g}"]
                           + ["" if k not in row else f"{row[k]:.10g}" for k in keys])

    if dump and F is not None:
        cols = ["t_h", "x_um"] + [c for c in PROFILE_FIELDS if c in F]
        path = os.path.join(outdir, "full_dump.csv")
        with open(path, "w", newline="") as fh:
            fh.write(",".join(cols) + "\n")
            buf = []
            for c in range(len(t)):
                blk = [np.full(len(x), t[c]), x] + [F[k][:, c] for k in cols[2:]]
                buf.append(np.column_stack(blk))
                if len(buf) >= 200:
                    np.savetxt(fh, np.vstack(buf), delimiter=",", fmt="%.10g"); buf = []
            if buf:
                np.savetxt(fh, np.vstack(buf), delimiter=",", fmt="%.10g")
        return path
    return None


# =============================================================================
# --replot: redraw a finished run from its own CSVs, simulating nothing
# =============================================================================
#: ModelParams fields that summary.csv records.  Anything else the run used is a dataclass default.
_SUMMARY_PARAMS = ("mu1", "Ks1", "mu2", "Ks2", "Y", "Dn", "D0", "rho_jam", "rho_c", "rho_max",
                   "Dself_floor", "Dself_jam_slope")


def replot(rundir, outdir):
    """Redraw run.png and initial_state.png (and fst.png if the run wrote fst.csv) from CSVs.

    Needs summary.csv and full_dump.csv, i.e. any run made since the dump became the default.
    The figures come out of the same `figure_*` functions as a live run, so a restyle is a
    seconds-long redraw, not a rerun.  The only difference from the live figure is the dump's
    10-significant-figure rounding, which is invisible at any plotted scale.
    """
    from dataclasses import replace as _replace
    import plot_growth_maps as _G                  # streaming dump reader (read_columns)

    S = {r[0]: r[1] for r in csv.reader(open(os.path.join(rundir, "summary.csv"))) if len(r) >= 2}
    dump = os.path.join(rundir, "full_dump.csv")
    if not os.path.exists(dump):
        raise SystemExit(f"--replot needs {dump}; this run was made without the dump, so its "
                         f"space-time panels exist only as pixels and it must be re-run.")
    kw = {k: float(S[k]) for k in _SUMMARY_PARAMS if S.get(k, "") not in ("", None)}
    if S.get("dself_shape"):
        kw["dself_shape"] = S["dself_shape"]
    p = config.params(float(S["n0_uM"]), **kw)
    ns_saved, ns_now = float(S["n_star_uM"]), float(C.monod_crossover(p))
    if abs(ns_now - ns_saved) > 1e-9 * max(1.0, abs(ns_saved)):
        raise SystemExit(f"parameter reconstruction failed: n* = {ns_now:.10g}, run had "
                         f"{ns_saved:.10g}")

    ic_kw = {k: S[k] for k in ("rho_mode", "fraction_mode", "nutrient_mode") if S.get(k)}
    for k in ("seed_f1", "rho_seed"):
        if S.get(k, "") not in ("", None):
            ic_kw[k] = float(S[k])
    ic = config.initial_conditions(**ic_kw)

    print(f"  reading {dump} ...", flush=True)
    t, x, col = _G.read_columns(dump, ("rho1", "rho2", "n"))
    L = float(S["L_um"])
    res = dict(x=x, t=t, dx=float(S["dx_um"]), L=L, N=int(float(S["N"])),
               rho1_profiles=col["rho1"], rho2_profiles=col["rho2"], n_profiles=col["n"],
               rho_peak=float(S["rho_peak"]))
    os.makedirs(outdir, exist_ok=True)
    out = [os.path.join(outdir, "run.png"), os.path.join(outdir, "initial_state.png")]
    figure_run(res, p, out[0])
    figure_initial_state(res, p, ic, out[1])
    if os.path.exists(os.path.join(rundir, "fst.csv")):
        out.append(os.path.join(outdir, "fst.png"))
        figure_fst(res, p, out[-1])
    return out


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", default=None, metavar="RUNDIR",
                    help="simulate nothing: redraw RUNDIR's run.png / initial_state.png (and "
                         "fst.png) from its summary.csv + full_dump.csv, into --outdir "
                         "(default RUNDIR itself)")
    ap.add_argument("--regime", choices=sorted(config.REGIMES), default=None)
    ap.add_argument("--n0", type=float, default=None)
    ap.add_argument("--L", type=float, default=None,
                    help=f"cavity length [um] (default {config.L:g}). N = L/dx, so raising L at "
                         f"fixed dx raises the cell count -- watch the max_cells warning.")
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--T", type=float, default=None)
    seed = ap.add_mutually_exclusive_group()
    seed.add_argument("--cells", type=float, default=None,
                      help="inoculum size in CELLS (total, both strains).  Converted to rho_seed "
                           "via the seed integral and --area/--vcell.  The default rho_seed=1e-6 "
                           "is ~3.6e-4 cells at A=10 um^2, i.e. a two-thousandth of one bacterium; "
                           "--cells 2 seeds one cell per strain.")
    seed.add_argument("--rho-seed", dest="rho_seed", type=float, default=None,
                      help="set the seed amplitude directly instead")
    ap.add_argument("--seed-f1", dest="seed_f1", type=float, default=None,
                    help="strain-1 fraction in the seed (default 0.5)")
    ap.add_argument("--area", type=float, default=1000.0,
                    help="cavity cross-section [um^2], used ONLY to convert density <-> cells.  Default 1000 = 50 x 20 um, the FABRICATED device of Karita, Limmer & Hallatschek (bioRxiv 10.1101/2021.05.13.444017, Methods 1.2: 20 um SU8 mold; Fig 1a: width 50 um, height 20 um, length Ln = n*10 um for n=1..35).")
    ap.add_argument("--vcell", type=float, default=0.65,
                    help="E. coli single-cell volume [um^3].  DEFAULT 0.65, the midpoint of the "
                         "0.6-0.7 range the manuscript's Extinction section cites and uses; the "
                         "code default must match the paper or every cell count in a run that "
                         "forgets this flag is off by 1.54x (this already happened -- the jammed "
                         "summaries were written at 1.0 while the T=12000 dilute run used 0.65). "
                         "Defensible range 0.6-1.3 for GLUCOSE MINIMAL at mu = 0.2-0.8 /h (a 2 um "
                         "spherocylinder of diameter 0.7-1.0 um).  NOTE Volkmer & Heinemann 2011 "
                         "PLoS ONE 6:e23126 report 1.5-4.4 um^3, but that is COMPLEX medium / "
                         "fast growth -- cell volume follows the growth rate (Schaechter et al. "
                         "1958), so it does not apply here.  Diagnostics only: --vcell rescales "
                         "the reported cell counts and changes no dynamics.")
    ap.add_argument("--n-save", dest="n_save", type=int, default=241)
    ap.add_argument("--max-adv-substeps", dest="max_adv_substeps", type=int, default=None,
                    help="raise Numerics.max_advection_substeps (default 2000).  The substep "
                         "count scales as 1/dx (204/514/1105/2235 at dx=2/1/0.5/0.25 -- "
                         "re-confirmed 2026-09-21 on the production runs), so dx<=0.25 exceeds "
                         "the default cap and those steps would then run ABOVE "
                         "advection_cfl=0.45.  The dx=0.25 convergence run was executed with this "
                         "flag raised and reports adv_capped_steps=0, so it is admissible.  A "
                         "cost guard, not a physical parameter -- ALWAYS check adv_capped_steps=0 "
                         "in summary.csv.  NB read those from summary.csv, not from history.csv: "
                         "history is written per SAVED FRAME, so its advection_substeps column "
                         "samples ~1 step in 500 on a long run and understates the max ~10x.")
    ap.add_argument("--outdir", default=None)
    # full_dump.csv is ON BY DEFAULT (2026-09-23).  It is the only file holding the space-time
    # fields behind the heatmaps (run.png panels H/I) and the flux/budget panels; without it those
    # panels exist only as pixels and cannot be redrawn without rerunning -- which is exactly what
    # happened to run_noise_dilute.  Rule: every plotted array is written to CSV.  --no-dump opts
    # out for throwaway runs; --dump is still accepted so existing commands keep working.
    ap.add_argument("--dump", dest="dump", action="store_true", default=True,
                    help="write full_dump.csv (the default)")
    ap.add_argument("--no-dump", dest="dump", action="store_false",
                    help="skip full_dump.csv -- the heatmap/flux panels then cannot be redrawn")
    ap.add_argument("--fst", action="store_true",
                    help="also write fst.csv and fst.png: Wright's F_ST per frame, the spatial "
                         "homogeneity of the composition (delta_f is a two-zone contrast and "
                         "cannot see structure inside a zone)")
    ap.add_argument("--stride", type=int, default=1, help="thin the drawn curves in profiles_all")
    ap.add_argument("--limiter", nargs="?", const="vanleer", default="none",
                    choices=("none", "minmod", "vanleer", "superbee"),
                    help="TVD flux limiter on the strain-label advection (bare --limiter = "
                         "vanleer). Default 'none' is plain donor-cell upwinding and reproduces "
                         "every existing run bit-for-bit. Only the label advection is first "
                         "order, so this is the only transport error a limiter can remove.")
    config.add_model_args(ap)
    a = ap.parse_args()

    if a.replot:
        d = a.replot if os.path.isabs(a.replot) else os.path.join(HERE, a.replot)
        if not os.path.isdir(d):
            d = os.path.abspath(a.replot)
        t0 = time.time()
        for path in replot(d, a.outdir or d):
            print(f"  wrote {path}")
        print(f"  replotted in {time.time() - t0:.1f}s")
        return

    base = config.REGIMES[a.regime] if a.regime else config.REGIMES["dilute"]
    n0 = a.n0 if a.n0 is not None else base["n0"]
    dx = a.dx if a.dx is not None else base["dx"]
    dt = a.dt if a.dt is not None else base["dt"]
    T = a.T if a.T is not None else base["T"]
    L = a.L if a.L is not None else config.L
    if L <= 0:
        raise SystemExit(f"--L must be positive, got {L:g}")

    p = config.params_from_args(a, n0)
    _num_kw = {} if a.max_adv_substeps is None else {"max_advection_substeps": a.max_adv_substeps}
    num = config.numerics(T_end=T, dt=dt, dx=dx, n_save=a.n_save, label_limiter=a.limiter,
                          **_num_kw)

    # Inoculum.  cells = rho_seed * I * A / V, with I the integral of the unit-amplitude seed
    # profile -- so the SHAPE (seed_center_um, seed_width_um) and the grid enter, and the
    # conversion has to be done here rather than with a hard-coded constant.
    ic_kw = {} if a.seed_f1 is None else {"seed_f1": a.seed_f1}
    ic = config.initial_conditions(**ic_kw)
    N_grid = C.choose_grid_size(L, num)
    xg, dxg = C.cell_grid(L, N_grid)
    from dataclasses import replace as _replace
    I_seed = float((C.initial_total_density(xg, L, p, _replace(ic, rho_seed=1.0)) * dxg).sum())
    if a.cells is not None:
        rho_seed = a.cells * a.vcell / (I_seed * a.area)
        ic = _replace(ic, rho_seed=rho_seed)
    elif a.rho_seed is not None:
        ic = _replace(ic, rho_seed=a.rho_seed)
    inoc_cells = ic.rho_seed * I_seed * a.area / a.vcell
    # L is in the directory name so an L sweep does not overwrite itself.  The seed is anchored
    # to the OPENING (seed_center_um=25) and does not scale with L, so runs at different L share
    # the same inoculum -- which is what makes them comparable.
    # tag the directory only when the seed is overridden, so existing default-seed runs keep
    # their names and an inoculum sweep cannot overwrite itself
    tag = ""
    if a.cells is not None:
        tag = f"_cells{a.cells:g}"
    elif a.rho_seed is not None:
        tag = f"_seed{a.rho_seed:g}"
    # tag the limiter too, so a limiter run cannot overwrite the default-scheme run it is
    # meant to be compared against
    if a.limiter != "none":
        tag += f"_lim-{a.limiter}"
    outdir = a.outdir or os.path.join(HERE, f"run_n0-{n0:g}_L{L:g}_dx{dx:g}_T{T:g}{tag}")
    os.makedirs(outdir, exist_ok=True)

    N = N_grid                              # choose_grid_size already warned if dx was clipped
    print(f"n0={n0:g} uM  L={L:g} um  dx={dx:g} um (N={N}, realized dx={L / N:.4g} um)  "
          f"dt={dt:g} h  T={T:g} h  n*={config.NSTAR:.4f} uM")
    print(f"  seed: rho_seed={ic.rho_seed:.4e}  seed_f1={ic.seed_f1:g}  ->  {inoc_cells:.4g} cells "
          f"total ({inoc_cells * ic.seed_f1:.4g} strain 1) at A={a.area:g} um^2, V={a.vcell:g} um^3"
          + ("   [SUB-CELLULAR INOCULUM]" if inoc_cells < 1.0 else ""))
    t0 = time.time()
    res = solver.simulate(L, p, num, ic, progress=True)
    res["cross_section_um2"] = a.area
    res["cell_volume_um3"] = a.vcell
    res["inoculum_cells"] = inoc_cells
    res["rho_seed"] = ic.rho_seed
    print(f"  simulated {res['N']} cells in {time.time()-t0:.1f}s -- {res['message']}")

    val = D.validate(res, p)
    t1 = time.time()
    F = fields(res, p, mask=True)
    figure_run(res, p, os.path.join(outdir, "run.png"))
    figure_initial_state(res, p, ic, os.path.join(outdir, "initial_state.png"))
    figure_flux_bump(res, p, F, os.path.join(outdir, "flux_bump.png"))
    figure_profiles_all(res, p, fields(res, p, mask=False),
                        os.path.join(outdir, "profiles_all.png"), stride=max(1, a.stride))
    dumped = write_csvs(outdir, res, p, val, dump=a.dump,
                        F=fields(res, p, mask=False) if a.dump else None, fst=a.fst, num=num)
    if a.fst:
        figure_fst(res, p, os.path.join(outdir, "fst.png"))
        s = D.fst_summary(res, p)
        print(f"  F_ST peak {s['F_ST_peak']:.4f} at t={s['t_F_ST_peak']:.0f} h -> "
              f"final {s['F_ST_final']:.4f}  "
              f"(slope {s['dF_ST_per_1000h_late']:+.3e}/1000h, "
              f"{s['F_ST_frames_valid']}/{len(res['t'])} frames above the conditioning floor)")
    print(f"  figures + CSVs in {time.time()-t1:.1f}s")
    print(f"  wrote {outdir}")
    if dumped:
        print(f"    full_dump.csv: {os.path.getsize(dumped)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
