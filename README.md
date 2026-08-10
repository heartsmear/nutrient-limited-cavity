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

## Requirements

Python 3.11 with `numpy`, `scipy`, `matplotlib`, `pandas`.

```bash
pip install numpy scipy matplotlib pandas
```

## Layout

| Module | Role |
|---|---|
| `config.py` | Single source of truth for parameters and regimes. `params(n0)` builds the whole set in one place; regimes carry their own `dx`, because the nutrient boundary layer shrinks as the cavity fills. |
| `closures.py` | The physics: Monod growth, collective and self diffusivity, grid and boundary operators. No solver. Jamming is carried by `D_col` alone — no pressure projection, no density clip. |
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
