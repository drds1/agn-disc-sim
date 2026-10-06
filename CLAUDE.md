# CLAUDE.md

## What this is

`discsim`: dynamics (N-body + SPH) of AGN discs and their environment, and a
post-processing chain (surface → lamppost irradiation → response functions →
mock light curves) that connects them to continuum reverberation and to the
sibling repo `../echofit` (pycream2).

## Design decisions

1. **Dynamics and radiation are separate layers.** Reverberation acts on frozen
   snapshots, because light-travel delays ≪ orbital times at R ≳ 10 r_g. Any
   particle snapshot (ours, Phantom, REBOUND) can go through `surface` → `reverb`.
2. **Code units G = M_BH = 1, length in light-days** (`Units`). Radiation works
   in light-days/days/cgs.
3. **The reprocessing surface is a single-valued z(R, φ)** for the side facing
   the lamppost. Shadowing is an exact horizon scan, because the lamp is on the
   axis.
4. **The response weighting is the same as pycream2's** (`dB/dT · dT/dL ·
   projected area`, delay `d − o·(x − x_lp)`), so the two codes agree for
   flat discs (`tests/test_reverb.py` checks against the analytic X = 3.36
   radius).
5. **The SPH stays readable numpy.** Optimise with numba/JAX later, but keep
   the plain version as the reference.
6. **Mock light curves use pycream2's text format** (`# t y yerr -- band 'x',
   wavelength W A`), so `scripts/fit_lightcurves.py` there reads them unchanged.

## Conventions

- Commits must not carry a Claude co-author trailer (standing user preference,
  same as echofit).
- Tests: `pytest` (plain functions, no fixtures needed). Keep each under a few
  seconds; long runs belong in `scripts/`.
- Outputs go to `outputs/` (git-ignored). Figures for the docs go in
  `docs/images/`.
- Environment: Poetry (`poetry install -E dev --with remote`); run everything with `poetry run`.
- Heavy runs: Kaggle, as in echofit, or Phantom on a cluster. `remote/build.py` (batches in
  `BATCHES`) builds a private notebook that embeds the working tree's `discsim/` and `scripts/`;
  `remote/runs.py push|wait|usage` pushes, downloads into `outputs/kaggle/` and keeps the ledger
  `remote/usage.json`. The laptop has 2 cores and 8 GB: keep it for tests, figures and short runs.
