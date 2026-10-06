# Roadmap: experiments and code

Each experiment links one open problem (see `literature_review.md`) to a
simulation and an observable you can compare with data or put through pycream2.

## Experiments

| # | Question | Set-up | Observable | Status |
|---|---|---|---|---|
| E1 | Which disc shapes lengthen lags without brightening the disc? | analytic surfaces: flat, flared, rim, ripples, spirals, warps | lag spectrum, lag boost vs flux boost, response tails | `scripts/geometry_responses.py` (first pass) |
| E2 | Do embedded massive bodies (stars, IMBHs, BH clusters) reshape the photosphere enough to matter? | SPH disc + embedded body, q = 1e-4 to 1e-2, R = 1-5 ld, tens of orbits | photosphere map, shadows, lags vs no-body run | `scripts/embedded_body.py` (pipeline demo; needs long, high-N runs) |
| E3 | Does the nuclear star cluster's flux of disc-crossing stars puff up and heat the outer disc? | test-particle stars (`nuclear_star_cluster`) crossing an SPH disc; add a heating/kick term per crossing | T(R) flattening, H(R), lag slope | to do |
| E4 | Can warps (Lense–Thirring / Bardeen–Petterson inside, inclined binary or cluster torque outside) produce Fairall-9-like long response tails? | test-particle or SPH disc with spin (`CentralPotential(spin=...)`) or an inclined secondary | response-function tails, apparent inclination in a flat-disc fit | gravity done (LT verified); experiment to do |
| E5 | Does irradiation feedback create rims by itself (Pringle 1996 radiation warping, thermal puffing)? | make `SPHGas.sound_speed` depend on the irradiated temperature from `reverb.irradiate`, recomputed every few steps | H(R) evolution; self-shadowing instability | to do |
| E6 | How biased are pycream2 fits of non-flat discs? Does the bias look like the disc-size problem? | `scripts/mock_campaign.py` → `fit_with_pycream2.sh`; many realisations | inferred log ṁ, i, α vs truth; coverage | mock generation done; fits to run in echofit |
| E7 | Do disc-scale structures change the X-ray/UV coupling? | add an X-ray driver independent of a slow intrinsic disc process; reprocess both | frequency-resolved lags | to do |

## Code

Near term:
- [ ] `discsim.io`: read Phantom dumps (via `sarracen` or `plonk`) and REBOUND
      simulation archives, so production runs feed the same post-processing.
- [ ] Line-of-sight occultation in `reverb.visibility` (a rim hides the inner disc at high inclination).
- [ ] Underside surface for strongly warped discs (lamppost sees both sides).
- [ ] Irradiation-dependent sound speed for SPH (E5).
- [ ] Gas back-reaction on embedded bodies (migration) and disc self-gravity
      (needed beyond ~10^3 r_g, i.e. the optical-lag region).
- [ ] Speed: numba or JAX kernels for SPH pairs; individual time steps. The
      numpy version does ~0.2 s/step at 6000 particles.
- [ ] A pycream2 bridge: register a tabulated simulated response with
      `pycream2.responses.register_response` for forward-modelling.

Later:
- [ ] Relativistic ray tracing for the inner disc (light bending, Doppler) or
      import transfer functions from KYNXILAG-type codes.
- [ ] Colour-corrected or atmosphere-model emission instead of blackbodies.
- [ ] Diffuse continuum: a BLR cloud distribution (particles!) reprocessing the
      same lamppost, to compare disc and DC contributions per band on equal footing.
- [ ] Simulation-based inference: train on many mock campaigns and fit real ones.
