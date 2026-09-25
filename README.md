# Nutrient-limited dynamics in confined cavities

Code, run outputs and draft for the bachelor's thesis *Nutrient-Limited Dynamics in Confined
Cavities*. The model is a 1D continuum description of *E. coli* growing in a dead-end cavity that
is fed with glucose only through its opening, where cells are also washed out.

- **One strain** gives the steady-state chapter: the establishment threshold, the threshold length
  `L*`, and when the jamming transition is continuous or a fold.
- **Two strains** competing for the glucose give the sorting chapter: *inverse sorting*, where a
  strain dominates regions in which it is locally disfavoured. An ancestry kernel explains it.

The thesis itself is in [`thesis/`](thesis/): `draft2.tex`, the compiled `draft2.pdf`,
`references2.bib` and the 42 figures it includes. That is everything `draft2.tex` needs to
compile (pdflatex + biber).

## Model

```
d_t rho1 + d_x J1 = q(rho) g1(n) rho1
d_t rho2 + d_x J2 = q(rho) g2(n) rho2
d_t n             = Dn d_xx n - (1/Y) [ q(rho) g1(n) rho1 + q(rho) g2(n) rho2 ]

g_i(n) = mu_i n / (Ks_i + n)
J      = -D_col(rho) d_x rho
J1     = f J - D_self(rho) rho d_x f          f = rho1 / rho
J2     = (1-f) J + D_self(rho) rho d_x f      J1 + J2 = J
```

The cavity is open at `x = 0` and closed at `x = L`:

```
n(0,t) = n0            d_x n(L,t) = 0
rho(0,t) = 0           J(L,t) = 0
D_self rho d_x f = 0   at x = 0 and x = L     (no relative label flux)
```

- **`D_col`** is Slepukhin's Eq. 16 in the additive, continuous form of their simulation code:
  Richardson–Zaki mobility times the Carnahan–Starling modulus below `rho_jam`, and a stiff branch
  above it that pins the density near close packing. Jamming is carried by the transport
  coefficient alone, so there is no pressure projection and no density clip.
- **`D_self`** is the Slepukhin/Karita form (`alpha = 0.95`, floor `1.5e-3`). It is discontinuous
  at `rho_jam`.
- **`q(rho)`** limits growth near close packing:
  - `q = 1` for `rho <= 0.95`;
  - `q` ramps linearly to `0` between `0.95` and `rho = 1`.

  It keeps `rho <= 1` at very high feed. `q` is exactly 1 in all the main thesis runs; it acts
  only in the `n0 = 10^6 µM` growth-factor comparison.

## Numerics

- **Reaction and nutrient:** solved jointly implicit with a damped Newton iteration.
  Consumption is yield-exact (the nutrient drawn is `(rho_new - rho_old)/Y`), so the reaction
  step conserves `n + rho/Y` to round-off.
- **Collective diffusion:** backward Euler, Newton with an Armijo line search.
- **Strain labels:** advected by donor-cell upwinding, with an optional van Leer flux limiter
  (`--limiter vanleer`).
- **Default settings.** Unless a figure says otherwise, every thesis run uses:
  - the van Leer limiter;
  - `dx = 0.5 um` and `dt = 0.05 h`;
  - a seed of `rho_seed = 1e-4`, about 7 cells.
- **Exception:** the sweeps behind the discretisation-error table use the unlimited scheme at
  `rho_seed = 1e-6`.
- **Ancestry kernel** (`adjoint_kernel.py`): solves the adjoint of the label transport, in two
  variants:
  - first-order (`none`);
  - second-order van Leer with an FCT limiter that keeps `K >= 0` (`vanleer`).

  The kernel heatmaps use `vanleer`; the `h_theory` profiles use `none`.

## Parameters

|            | scavenger (strain 1) | glutton (strain 2) |
|------------|----------------------|--------------------|
| `mu`       | 0.21 h⁻¹             | 0.78 h⁻¹           |
| `Ks`       | 0.2 µM               | 15 µM              |

- **Growth rates.** The maximum growth rates are the measured values of Steinsiek & Bettenbrock for
  the *ptsG* mutant and its parent MG1655.
- **Half-saturation constants** were not measured there. They are chosen to place the crossover at
  `n* = (mu1 Ks2 - mu2 Ks1)/(mu2 - mu1) = 5.2526 µM`.
- **Other parameters:**
  - `Y = 2.895e-7` packing fraction per µM, from a yield of 0.45 g/g;
  - `Dn = 1.2e6 µm²/h`, half the free-solution glucose diffusivity;
  - `D0 = 376 µm²/h`, `eta = 5.8`, `rho_jam = 0.64`.

Units are µM, µm and hours throughout.

## Requirements

Python 3.11 with the packages in `requirements.txt` (the versions the thesis was produced with):

```bash
pip install -r requirements.txt
```

Every script puts its own directory on `sys.path`. Run them from the repository root:

```bash
py -3.11 config.py
```

This is a self-test: it prints the parameters, `n*`, the regimes and a resolution estimate.

## Code

| Module | Role |
|---|---|
| `config.py` | Single source of truth for parameters and regimes. |
| `closures.py` | The physics: Monod growth, `q(rho)`, `D_col`, `D_self`, grid and boundary operators. No solver. |
| `solver.py` | The only time-stepping loop. |
| `diagnostics.py` | Everything measured from a run: `F1`, inverse-sorted fractions, the `n*` front, `F_ST`. |
| `figures.py` | Runs the time-dependent model and writes the run figure, `summary.csv`, `timeseries.csv`, the profiles and `full_dump.csv`. `--replot RUNDIR` redraws a run figure from a run's CSVs and its `full_dump.csv`. |
| `replot_run_from_dump.py` | Re-renders `run.png` for an existing run from its `full_dump.csv`. |
| `plot_growth_maps.py`, `plot_fst.py` | Growth-rate maps and `F_ST(t)`. `figures.py` imports them. |
| `steady_single_strain.py` | Direct Newton solve of the single-strain steady cavity, plus the steady-state sweeps (`--budget`, `--verify-lstar`, `--fold`, `--fold-params`). |
| `plot_onset_law.py` | Validation of the onset amplitude law, including the `D_col'(0)` term. |
| `plot_jammed_analytic.py` | Fully jammed analytic nutrient profile against the steady state. |
| `check_cusp_prediction.py` | The cusp of the jamming fold from the pressure-plane quadrature. Gives `n0*`, `rho_c`, `c1`, `c3`, `kappa`. |
| `ic_study.py` | Initial-condition dependence: seed density, seed composition, seeding mode, nutrient. |
| `ancestry.py` | Neutral transport operator and kernel identities. Used by the kernel scripts. |
| `adjoint_kernel.py` | The ancestry kernel by solving the adjoint equation. |
| `plot_kernel_figures.py` | The eight kernel panels: `h` theory vs numerics, exposure criterion, kernel heatmaps. |
| `plot_adjoint_convergence.py` | `h_sim` vs `h_theory` at several `dx`. |
| `plot_production_maps.py` | Where biomass is produced, in space and time. |
| `survival_origin.py`, `component_kernel.py` | Where the biomass still present at the final time was produced. |
| `plot_survival_origin.py` | Plots the output of `survival_origin.py`. |
| `plot_fst_vs_L.py` | Peak `F_ST` against cavity length, dilute vs jammed. |
| `sweep_dx.py`, `sweep_dt.py` | Grid and time-step sweeps. |
| `plot_conv_test.py` | Turns the four sweeps into the discretisation-error table. |
| `plot_grid_convergence_T2000.py` | Grid convergence of the jammed van Leer runs at `t = 2000 h`. |
| `plot_dnum_ratio.py` | Numerical label diffusivity of the unlimited scheme against `D_self`. |

## Thesis figures and where they come from

Output folders are named by their parameters. For example,
`run_n0-15_L1000_dx0.5_T500_seed0.0001_lim-vanleer` is a run at:
- `n0 = 15 µM`
- `L = 1000 µm`
- `dx = 0.5 µm`
- `T = 500 h`
- `rho_seed = 1e-4`
- van Leer limiter

| Figure (`thesis/`) | Produced by | Data in this repository |
|---|---|---|
| `steady_lstar_general`, `steady_lstar_dilute` | `steady_single_strain.py --verify-lstar` | `steady_sweeps/` |
| `steady_budget` | `steady_single_strain.py --budget` | `steady_sweeps/` |
| `steady_fold`, `steady_fold_params` | `steady_single_strain.py --fold` / `--fold-params` | `steady_sweeps/` |
| `steady_closures`, `steady_density`, `steady_nutrient` | `steady_single_strain.py --n0 15 --L 1000 --dx 5` | `steady_n0-15_L1000_dx5/` |
| `jammed_analytic_nutrient2`, `…3` | `plot_jammed_analytic.py steady_n0-1000_L1000_dx0.5` | `steady_n0-1000_L1000_dx0.5/` |
| `onset_law_validation` | `plot_onset_law.py` | `onset_law_validation_*.csv` |
| cusp constants in the text | `check_cusp_prediction.py` | printed |
| `run_noise_dilute` | `figures.py`, `--n0 15 --T 96` | `run_n0-15_L1000_dx0.5_T96_seed0.0001_lim-vanleer/` ¹ |
| `run` | `figures.py`, `--n0 1000 --T 6000`, re-rendered with `replot_run_from_dump.py` | `run_n0-1000_L1000_dx0.5_T6000_seed0.0001_lim-vanleer/` ¹ |
| `run_dil1000`, `run_dildeath` | `figures.py`, `--n0 15 --T 1000` / `--T 12000` | `run_n0-15_L1000_dx0.5_T{1000,12000}_seed0.0001_lim-vanleer/` ¹ |
| `frame_000000` | `figures.py` (`initial_state.png`) | `run_n0-15_L1000_dx0.5_T500_seed0.0001_lim-vanleer/` ¹ |
| `h_theory_vs_numericdil`, `criteriondilute` | `adjoint_kernel.py --n0 15 --T 96`, then `plot_kernel_figures.py` | `adjoint_n0-15_L1000_dx0.5_T96_m30_d900_seed0.0001_traj-vanleer/` |
| `kerneldilutemouth`, `kerneldilutedeep` | `adjoint_kernel.py --n0 15 --T 500`, then `plot_kernel_figures.py` | `adjoint_n0-15_L1000_dx0.5_T500_m30_d900_seed0.0001_traj-vanleer/` |
| `h_theory_vs_numeric`, `criterioninversejam`, `kerneljam*` | `adjoint_kernel.py --n0 1000 --T 2000`, then `plot_kernel_figures.py` | `adjoint_n0-1000_L1000_dx0.5_T2000_m30_d900_seed0.0001_traj-vanleer/` |
| `adjoint_convergence_overlay` | `plot_adjoint_convergence.py` | `adjoint_n0-1000_L1000_dx{0.5,1,2}_T2000_…/` |
| `production_maps_*` | `plot_production_maps.py <run>` | the `T500` dilute and `T2000` jammed runs ¹ |
| `survival_origindil`, `survival_originjam` | `survival_origin.py`, then `plot_survival_origin.py` | `survival_n0-15_L1000_dx0.5_T500_seed0.0001/`, `survival_n0-1000_L1000_dx0.5_T2000_seed0.0001/` |
| `ic_density`, `ic_f1`, `ic_mode`, `ic_nutrient` | `ic_study.py --regime dilute --axis all --dx 5 --T 2000 --rho-seed 1e-4 --limiter vanleer` | `ic_study_n0-15_dx5_T2000_seed0.0001_lim-vanleer/` |
| `ic_*_jam`, `ic_nutrient_jam2` | `ic_study.py --regime jammed --axis <axis> --dx 0.5 --T 2000 --rho-seed 1e-4 --limiter vanleer` | `ic_study_n0-1000_dx0.5_T2000_<axis>_full/` |
| `fst_peak_vs_L` | `plot_fst_vs_L.py` | the `L100`–`L300` runs and the two `L1000` runs |
| `grid_convergence_T2000` | `plot_grid_convergence_T2000.py` | `run_n0-1000_L1000_dx{0.25,0.5}_T2000_…/`, `dx{1,2}_T6000_…/` ¹ |
| `dnum_over_dself` | `plot_dnum_ratio.py` | `run_n0-1000_L1000_dx2_T6000/`, `run_n0-1000_dx0.5_T10000/` ¹ |
| discretisation-error table | `sweep_dx.py`, `sweep_dt.py`, then `plot_conv_test.py` | `conv_dx_*/`, `conv_dt_*/` |
| `growthfactor` | `figures.py`, `--n0 1e6 --T 1000` (donor-cell, `rho_seed = 1e-6`) | `run_n0-1e+06_dx0.5_T1000/` ¹ |
| `nogrowthfactor` | the earlier `src/` code, not part of this repository | — |

¹ Reads the run's `full_dump.csv`, which is not included (about 110 MB per run). Re-run the run
first. Every other figure regenerates from the repository alone.

Runs are started with `figures.py`, for example:

```bash
py -3.11 figures.py --n0 15 --L 1000 --dx 0.5 --dt 0.05 --T 500 --rho-seed 1e-4 --limiter vanleer
```

Once a run has its `full_dump.csv`, its figure can be redrawn without re-simulating:

```bash
py -3.11 figures.py --replot run_n0-15_L1000_dx0.5_T500_seed0.0001_lim-vanleer
```

## What is not included

Large files are regenerable and stay out of the repository:
- `full_dump.csv`: about 110 MB per run at `dx = 0.5`, above GitHub's file limit.
- Field stores (`*.npz`). The exception is the three behind the kernel figures (5–7 MB each),
  which are included.
- One 12.6 MB mass history.

Rerun the command in the table above to recreate them.

`.gitignore` ignores every output folder by default and re-includes only the runs behind the
thesis figures. To publish a new run, add a `!/<folder>/` line there.
