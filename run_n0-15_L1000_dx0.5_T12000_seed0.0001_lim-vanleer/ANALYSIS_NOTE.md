# Dilute T=12000 h — does strain 2 ACTUALLY go extinct?

Written 2026-09-15 while the run was in flight, so the analysis survives a context
compaction. Everything below is either a command, a measured number from an earlier
run, or an extrapolation clearly labelled as such.

## The run

```
py figures.py --n0 15 --L 1000 --dx 0.5 --T 12000 --rho-seed 1e-4 --limiter vanleer \
               --vcell 0.65 --dump
```

Started 2026-09-14 21:25. 240,000 steps at dt = 0.05. Measured cost ~0.16 s/step
=> ~10.7 h of CPU. NOTE: wall-clock is not CPU time — the machine idle-slept for
roughly 10 of the first 16 h (cumulative CPU/wall 0.371, instantaneous 0.972 once
awake). Keep-awake blocks idle sleep only; a closed lid still sleeps.

## The question

The draft claims the dilute cavity reaches Tilman's R* limit, F1 -> 1. That is
**relative** exclusion: strain 2's *fraction* vanishes. It does not establish that
strain 2 is ever actually gone. This run asks whether M2 ever falls below one cell,
i.e. **actual** exclusion, and if so when.

## Conversion

    M_crit = V_cell / A = 0.65 / 1000 = 6.5e-4 um          (one cell)
    N_2    = M_2 * A / V_cell = 1538 * M_2

`--vcell 0.65` was passed, so the `cells1/cells2` columns and `min_cells*_perstep`
are already at the right cell volume. (The default 1.0 would put M_crit at 1e-3 and
move the apparent crossing ~500 h EARLIER — a false positive on exactly this claim.)

## Prediction being tested (extrapolation, not measurement)

Baseline from `run_n0-15_L1000_dx0.5_T1000_seed0.0001_lim-vanleer`:

    M_2(1000 h) = 0.204838 um      (= 315 cells at V=0.65)

Late decay of M_2, fitted on successive windows — decelerating but converging:

    window [h]    d ln M2/dt [1/h]   half-life [h]
    200-400        -1.784e-3          389
    400-600        -1.082e-3          641
    600-800        -8.836e-4          785
    800-1000       -8.302e-4          835
    900-1000       -8.245e-4          841      <- taken as the asymptotic rate

Extrapolating at 841 h half-life:

    crossing of one cell at  t ~ 7980 h   (V=0.65)
                             t ~ 7460 h   (V=1.0)
    M_2(12000 h) ~ 6e-5 um ~ 0.09 cells

MARGIN: T=12000 accommodates any asymptotic half-life up to ~1325 h. If the decay
keeps decelerating past that, M_2 finishes ABOVE one cell and the run yields a lower
bound, not a crossing time. Report whichever you actually got — these are different
claims.

## What to check when it exits

1. `summary.csv` -> `min_cells2_perstep`, `steps_below_one_cell`.
   These are over all 240,000 steps, not 241 frames. That matters: the frame-based
   `min_cells*` fields are a minimum over saved frames only, which is what hid the
   early washout dip in the jammed run.
2. `mass_history.csv` -> M2(t) at every step. Find the first t with M2 < 6.5e-4.
   Compare against the predicted 7980 h.
3. `profile_final.csv` -> where the remnant sits. At T=1000 rho_2 peaked at
   **x = 999.8 um** (the very last cell, against the closed no-flux wall) with 43%
   of M_2 beyond 700 um. If that refuge is still there at 12000 h the erosion
   mechanism is confirmed; if it has been swept, that is the more interesting result.
4. The admissibility line in stdout. A 12,000 h dilute run is far outside anything
   validated before.
5. `F1_final` against the T=1000 value 0.977110.

## Caveats for the write-up

* The continuum has **no absorbing state**: growth is g*rho, so M_2 decays
  exponentially and never reaches zero. "M_2 crosses one cell at t = X" is a
  statement about where the deterministic model stops being physically
  interpretable — the same framing as F1, not a proof of extinction.
* 7980 h ~ **10.9 months**. That collides head-on with F14 (cell death omitted):
  the claim would require starved cells to persist ~11 months, the same
  vulnerability the draft already concedes for jammed permanence. State both
  together rather than letting one look safe.
* The mechanism at T=1000 is a closed-end refuge: n ~ 0 there, so s ~ 0 and neither
  strain grows; erosion is by weak D_self plus slow advection only.

## Provenance

* Decay fit, refuge location and F1 baseline: the T=1000 run above.
* Per-step `mass_history.csv` and the `*_perstep` summary fields were added to
  `solver.py` / `figures.py` in this session. Without them `steps_below_one_cell`
  would be a count over 241 frames and would not answer the question.
* Grid choice: dilute is grid-insensitive (draft convergence table: F1 error
  -0.0038% at dx=0.5; dx/lambda = 0.023), so dx=0.5 is comfortably converged. The
  one honest caveat is that this was measured at T=500 and the crossing time is a
  cumulative long-time quantity; numerical diffusion D_num = |u| dx (1-nu)/2 is
  ~1.4e-3 of D_self here, so it should stay negligible.
