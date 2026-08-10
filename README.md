# Two-strain cavity model (E. coli / glucose)

A 1D continuum model of two bacterial strains competing for a single diffusing nutrient inside a
cavity that is open at one end (`x = 0`, nutrient source) and closed at the other. The two strains
differ in their Monod parameters so that selection reverses across a crossover concentration `n*`:
the fast, low-affinity strain wins where `n > n*`, the slow, high-affinity strain wins where
`n < n*`. Because the population depletes its own resource, that crossover sweeps through the
cavity, and the final composition reflects the *history* of selection rather than the local
environment at the final time.

This is a clean re-implementation of an earlier `src/` stack, with the chemotaxis machinery
removed and the duplicated code paths collapsed into one. Each module's docstring records what
changed relative to the original and why.

## Model

```
d_t rho1 + d_x J1 = q(rho) g1(n) rho1
d_t rho2 + d_x J2 = q(rho) g2(n) rho2
d_t n             = Dn d_xx n - (1/Y) [ q(rho) g1(n) rho1 + q(rho) g2(n) rho2 ]
```

with Monod growth, a total biomass flux that is collective diffusion down the density gradient,
and a relative label flux that carries no net mass:

```
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

`D_col` follows Slepukhin Eq. 16 in the additive, continuous form their simulation code uses:
Richardson–Zaki mobility times the Carnahan–Starling modulus below `rho_jam`, continuing into a
stiff branch above it that pins the density near close packing. Jamming is therefore carried
entirely by the transport coefficient — there is no pressure projection and no density clip.

### The growth-space factor `q(rho)`

Growth is damped as the cavity approaches close packing:

```
q = 1                                    rho <= rho_c            (rho_c   = 0.95)
  = (rho_max - rho) / (rho_max - rho_c)  rho_c < rho < rho_max   (rho_max = 1.0)
  = 0                                    rho >= rho_max
```

This term is **not** in the older formulation of the model. It exists because the only sink in a
closed cavity is the mouth, so `D_col` redistributes biomass but never removes it: at high feed
concentration the reaction outproduces the mouth outflux and the density accumulates without
bound — the previous stack reached a packing fraction of 199%. Limiting the source is what makes
`rho <= 1` an invariant rather than a hope, and the ramp (rather than a hard cutoff) makes the
density equation logistic near the ceiling so it approaches `rho_max` asymptotically.

`q` is exactly `1.0` for `rho <= 0.95`, which every reference regime here satisfies, so it is
inert in those runs and they remain bit-identical to the stack this replaces. Note that it scales
the *selection source* too: the mean growth rate becomes `q * gbar`, so `g1 - gbar = q (1-f) s(n)`
and the fraction equation reads

```
d_t f + u d_x f = (1/rho) d_x (D_self rho d_x f) + q(rho) f (1-f) s(n)
```

Consumption is yield-exact — the nutrient drawn is `(rho_new - rho_old)/Y` rather than
`dt (g1 rho1 + g2 rho2)/Y` — so `q` damps growth and uptake consistently and `n + rho/Y` is
conserved to round-off at finite `dt`.

## Parameters

The Monod constants are **measured, not designed**: two *E. coli* MG1655 knockouts competing for
glucose, a high-affinity scavenger (`dptsG`, transporting through Mgl) and a fast low-affinity
glutton (`dmglBAC`, transporting through PtsG), with constants from Steinsiek et al. 2012.

| | scavenger (strain 1) | glutton (strain 2) |
|---|---|---|
| `mu` | 0.21 h⁻¹ | 0.78 h⁻¹ |
| `Ks` | 0.2 µM (Mgl Kd) | 15.0 µM (PtsG Km) |

The crossover is therefore **derived rather than targeted**:

```
n* = (mu1 Ks2 - mu2 Ks1) / (mu2 - mu1) = 5.2526 uM
```

Nutrient concentrations are in µM, lengths in µm, time in hours. `Y = 2.895e-7` packing fraction
per µM follows from a yield of 0.45 g/g, and `Dn = 1.2e6 um^2/h` is half the free-solution
glucose diffusivity. Transport uses `D0 = 376 um^2/h`, `eta = 5.8`, `rho_jam = 0.64`.

`D_self` defaults to the discontinuous Slepukhin/Karita form as distributed (`alpha = 0.95`,
floor `1.5e-3`); a smooth variant is available via `dself_shape="continuous"`. The choice is
load-bearing in the jammed phase — it sets whether the graveyard remixes or stays frozen — so it
is a modelling decision, not a tolerance.

## Requirements

Python 3.11 with `numpy`, `scipy`, `matplotlib`, `pandas`.

```bash
pip install numpy scipy matplotlib pandas
```

## Layout

| Module | Role |
|---|---|
| `config.py` | Single source of truth for parameters and regimes. `params(n0)` builds the whole set in one place; regimes carry their own `dx`, because the nutrient boundary layer shrinks as the cavity fills. Run it for a self-test that prints `n*`, the regimes and a resolution estimate. |
| `closures.py` | The physics: Monod growth, the growth-space factor `q(rho)`, collective and self diffusivity, grid and boundary operators. No solver. Jamming is carried by `D_col` alone — no pressure projection, no density clip. |
| `solver.py` | The only time-stepping loop. Reaction and nutrient diffusion are solved **jointly implicit** (damped Newton); collective diffusion is backward Euler with Newton + Armijo. |
| `diagnostics.py` | Everything measured *from* a run, in one implementation — including the `n*` front locator, interpolated in `log n`. |
| `ancestry.py` | The exact deterministic ancestry kernel and the discrete Duhamel decomposition of the final composition into inherited and selection-driven parts. |
| `figures.py` | The standard figure set plus CSV output. |
| `plot_full_dump.py` | Standalone plotting for a `full_dump.csv` produced by `figures.py --dump`. |
| `sweep_dx.py`, `sweep_dt.py` | Grid and timestep convergence sweeps. |
| `test_regression.py` | Bit-identity check against the legacy stack — see the caveat below. |

## Usage

Modules put their own directory on `sys.path`, so they run from anywhere. From inside this
directory:

```bash
py -3.11 config.py
```

Self-test: parameters, `n*`, regimes and a resolution estimate.

```bash
py -3.11 figures.py --regime dilute --dump
```

Run a regime and write the standard figures plus `full_dump.csv`.

```bash
py -3.11 figures.py --n0 1000 --dx 0.5 --T 3000 --outdir jammed_run
```

Run an explicit parameter set into a named output directory.

```bash
py -3.11 ancestry.py --check
```

Verify the ancestry-kernel identities, then `--n0 15 --dx 10 --T 400` for a full kernel run.

```bash
py -3.11 sweep_dx.py
```

Grid convergence; `sweep_dt.py` does the same for the timestep.

## Committed run outputs

The `run_*/` and `sweep_*/` directories hold the outputs of previous runs — figures, summary CSVs
and `full_dump.csv` snapshots — committed so results can be inspected without re-running. They are
regenerable from the scripts above.

## Caveat on `test_regression.py`

This test compares the stack against the **legacy `src/` implementation**, which it locates as a
sibling directory:

```python
OLD = os.path.join(os.path.dirname(HERE), "src")
```

That directory is not part of this repository, so the test will not run in a standalone clone. It
is kept here as a record of how the rewrite was validated: at zero chemotaxis the old code path
reduces to the same arithmetic, so the two are required to agree to exactly `0.0`, not merely to
within a tolerance.
