# Open problems in continuum reverberation mapping of AGN discs

*A working review, October 2026, written alongside the echofit / pycream2
NGC 5548 and Fairall 9 analyses. References were compiled partly from memory
and partly from searches: check each against ADS before citing it in a paper.*

## 1. The standard picture and why it is under strain

The lamppost model: a compact X-ray corona a few to tens of r_g above the hole
irradiates a geometrically thin, optically thick Shakura–Sunyaev disc; each
annulus reprocesses at its local temperature, `T ∝ R^(-3/4)`, so the delay at
wavelength λ follows `τ ∝ (M Ṁ)^(1/3) λ^(4/3)` (Cackett, Horne & Winkler 2007;
Fausnaugh et al. 2016). Intensive campaigns (Swift + HST + ground: NGC 5548,
NGC 4151, NGC 4593, Mrk 509, Fairall 9, Mrk 142, Mrk 817) and survey samples
(Pan-STARRS, DES, SDSS-RM, ZTF) have measured inter-band lags in dozens of
AGN. Five things don't fit.

### 1.1 The disc-size problem (lags too long)

Lags are typically **2–3× (up to ~5×) longer** than the thin-disc prediction for
the AGN's mass and luminosity: Edelson et al. 2015 and Fausnaugh et al. 2016
(NGC 5548); Jiang et al. 2017 (Pan-STARRS); Cackett et al. 2018 (NGC 4593);
Edelson et al. 2019 (four Swift AGN); Guo et al. 2022 (ZTF). The SDSS-RM sample
is milder (Homayouni et al. 2019), which hints at selection or luminosity
dependence. A newer 18-AGN, four-year campaign (arXiv:2511.05669) adds
long-baseline lags.

Two independent checks make it unlikely to be a lag-measurement artefact:

- **Microlensing** of lensed quasars finds optical half-light radii ~3–4× the
  thin-disc value (Morgan et al. 2010), with no reverberation physics involved.
- **The flux test.** A disc big enough to give the lags would be far brighter
  than observed at the known distance. Cackett et al. 2007 cast this as an
  "H0" of 44 km/s/Mpc from 14 AGN. Your echofit analyses push it further:
  equivalent H0 ≈ 15 for Fairall 9 (delays 4.8× too long for its brightness,
  i.e. a 23× too-bright disc) and ≈ 7 for NGC 5548.

So the issue is not just "the disc is bigger" but **"the delays are longer than
the disc's luminosity allows"**. Any solution has to lengthen the delays
*without* adding a matching amount of steady light. That is the constraint
`scripts/geometry_responses.py` in this repo is built to test.

### 1.2 The shape of the lag spectrum

Roughly `λ^(4/3)` overall, but with a clear **excess around the Balmer jump
(U band, ~3600 Å)** and sometimes around the Paschen jump (Edelson et al. 2019;
Cackett et al. 2018; Hernández Santisteban et al. 2020). This is the signature
of **diffuse continuum (DC) from the broad-line region**: free-bound and
free-free emission from BLR clouds, which reverberates on BLR timescales
(Korista & Goad 2001, 2019; Lawther et al. 2018; Netzer 2022). The open
question is how much: anything from a modest U-band bump to the claim that the
BLR dominates the continuum lags at all wavelengths (Netzer et al. 2024 for
Mrk 817, AGN STORM 2 paper X). Your Fairall 9 fits found a second reprocessor
can close the gap only if it carries **63–75% of every band's variable
light**, which needs photoionisation modelling to confirm or rule out.

NGC 5548 adds a twist: your Model 2 lag fit wants α ≈ 1.9 (calibrated to about 1.65
by the NGC-like simulations), while the variable spectrum wants α ≈ 1.1. The
lags and the spectrum disagree about the temperature profile.

### 1.3 The X-ray/UV connection is weak

If X-rays drive the UV, the two should correlate tightly and the X-ray power
should be enough to supply the UV variability. Often it isn't:

- X-ray–UV correlations are weaker than UV–optical ones, and the X-ray-to-UV
  lag is longer than the UV-to-optical lags extrapolate to (Edelson et al.
  2015, 2019).
- The variable X-ray luminosity is often too small to power the UV variations
  (the "energy budget" problem; Gardner & Done 2017).
- NGC 5548's **BLR holiday** (Goad et al. 2016): the emission lines
  decorrelated from the continuum for months, probably because an obscuring
  disc wind changed the ionising SED the BLR saw (Kaastra et al. 2014;
  Dehghanian et al. 2019). Mrk 817 has a similar obscurer (Kara et al. 2021).
- Fairall 9 over three years (Hagen et al. 2025): the UV/optical disc
  brightened while the X-ray power stayed flat. The UV and X-rays look like
  largely independent stochastic processes, coupled only weakly by
  light-travel time.

### 1.4 Lags depend on timescale, and there are slow components

Fourier-resolved analyses find lags that change with frequency: slow
variations carry longer lags (or the opposite sign), fast ones the
reprocessing lag (e.g. Cackett et al. 2022; Lewin et al. 2023; Hagen et al.
2025). Fairall 9 has a slow, non-reverberating component that has to be
removed (Hernández Santisteban et al. 2020) or fitted jointly. Your joint
quadratic background improves ln Z by ~340, and without it the fit is driven to
an unphysical high inclination. Lags also grow with campaign baseline and with
fainter AGN (Li et al. 2021). That points to intrinsic disc fluctuations
(e.g. the **CHAR** model: magnetic coupling between corona and disc,
Sun et al. 2020) as well as reprocessing.

### 1.5 The disc itself doesn't behave like a thin disc

- UV/optical variability is too large and too fast for viscous or thermal
  timescales in a standard disc. Changing-look AGN make this worse
  (Lawrence 2018).
- Proposed fixes change the disc: magnetically elevated, thicker discs
  (Dexter & Begelman 2019); iron-opacity-bump convection and turbulence
  (Jiang & Blaes 2020); strongly inhomogeneous, flickering discs (Dexter & Agol
  2011, which also enlarges the effective size).
- Beyond ~10^3 r_g (a few light-days for 10^8 M_⊙, right where the optical
  lags come from) the disc should be **self-gravitating** and form stars
  (Goodman 2003; Sirko & Goodman 2003). Stars and compact objects embedded in
  or crossing the disc (McKernan et al. 2012; Cantiello et al. 2021; Tagawa et
  al. 2020) add heating that flattens T(R), and can open gaps, launch spiral
  waves and puff the disc up.

## 2. Candidate explanations, and what each predicts

| Idea | Lengthens lags? | Adds steady flux? | Key test |
|---|---|---|---|
| Higher corona / relativistic lamppost (Kammoun et al. 2019, 2021) | yes, modestly | some | X-ray spectral constraints on height; needs large L_X |
| Non-flat geometry: rims, ripples, flaring, warps (Starkey et al. 2023; Pringle 1996 radiation warping) | yes, at red λ | very little (this repo: rim ×1.8 lag at ×1.01 flux) | lag-spectrum curvature; response-function tails; inclination dependence |
| BLR diffuse continuum (Korista & Goad; Netzer) | yes, everywhere, with a Balmer bump | yes (DC flux) | BLR-scale delays; spectral decomposition; photoionisation fractions |
| Intrinsic disc fluctuations, CHAR (Sun et al. 2020) | yes, timescale-dependent | no | frequency-resolved lags; baseline dependence |
| Inhomogeneous disc (Dexter & Agol 2011) | yes | changes SED shape | variability amplitude vs λ |
| X-ray heating of an optically thin skin (Secunda et al. 2025, RMHD) | yes (T up 2–5×) | some | needs enough L_X and low albedo; non-linear response |
| Different temperature profile (α ≠ 3/4: winds, star heating, torques) | depends | depends | lag slope vs variable-spectrum slope: they must agree |

No single idea passes every test. The most likely answer is a combination:
a disc reprocessing component (with non-trivial geometry at its outer edge),
plus a BLR DC component, plus intrinsic fluctuations at low frequencies.
The problem is no longer to find one mechanism but to **separate the
components quantitatively**.

## 3. What has to happen

### Theory
- **A self-consistent outer disc.** The optical lags come from the region where
  the disc turns self-gravitating, forms stars and meets the dust sublimation
  radius and the inner BLR. Theory has to predict its thickness profile H(R),
  i.e. whether rims and puffing arise naturally (irradiation feedback, embedded
  stars, warps), rather than putting them in by hand.
- **A physical source of intrinsic UV variability** coupled to the corona, with
  predicted transfer functions *per Fourier frequency*, not just a mean lag.
- **Predicted DC fractions** per wavelength from photoionisation models tied to
  each AGN's own BLR, so the 60–75% that Fairall 9 needs can be tested.
- **Non-linear reprocessing.** RMHD shows the reprocessing response changes with
  the disc's state (Secunda et al. 2025). The linear transfer-function
  assumption behind CREAM, JAVELIN and PyROA is a model choice, not a law.

### Code and inference
- **Joint multi-component models** (disc, DC, slow component, intrinsic
  fluctuations) in one likelihood. pycream2 already does disc + DC + background.
  Next: frequency-dependent coupling and an X-ray driver with its own
  stochastic process.
- **Simulation-calibrated inference.** Your NGC-like synthetics found α biased by
  +0.2–0.25 with under-coverage. Bias like that only shows up if fits are
  calibrated on data from a *different* model than the one being fitted
  (simulation-based inference / coverage tests). Mock campaigns from physical
  simulations (this repo) are the obvious supply.
- **Arbitrary-geometry response functions** that are fast and differentiable
  (pycream2's rippled disc is the template), so geometry parameters can be
  inferred rather than fixed.
- **Fourier-domain likelihoods** to fit frequency-resolved lags directly.

### Data
- Long (multi-year), dense, simultaneous X-ray/UV/optical campaigns, for
  frequency-resolved lags at low frequency.
- Spectroscopic RM at the same time, to measure the DC directly
  (AGN STORM 2 is the template).
- Samples spanning mass and Eddington ratio: Rubin/LSST deep drilling fields,
  ULTRASAT and UVEX for the UV, SDSS-V BHM.
- Microlensing sizes for the same objects, where possible.

## 4. Where simulations fit in

Reverberation happens on light-crossing times (days). Disc structure changes on
dynamical times: an orbit at 1 light-day around 10^8 M_⊙ takes ~80 days, at 5
light-days ~2.5 years. So simulations provide the **geometry and thermal
state** that the light echoes off, as a sequence of frozen snapshots. The
questions they can answer:

1. **Does the outer disc naturally develop the structures that lengthen lags?**
   Rims at the self-gravity or dust radius; irradiation-driven warping
   (Pringle 1996); puffing from embedded stars; spiral waves and gaps from
   embedded massive bodies; warps from an inclined secondary or the nuclear
   star cluster's torque.
2. **How large is the lag boost per unit flux boost?** (Figure from
   `scripts/geometry_responses.py`.)
3. **How badly do flat-disc fits misread these discs?** Mock campaigns →
   pycream2 → compare the inferred log ṁ, inclination and α with the truth.
4. **Are the response functions' long tails** (which in Fairall 9 pushed fits to
   high inclination) a sign of a second reprocessor, or of a warped or
   self-shadowed disc?

N-body (stars, compact objects, binaries) + SPH (gas) is the right tool for (1),
with the caveat that particle discs under-resolve the vertical structure that
sets the photosphere unless N ≳ 10^6. The established codes for that scale are
Phantom (SPH, with Lense–Thirring, sinks and self-gravity; Price et al. 2018)
and REBOUND (N-body; Rein & Liu 2012). This repo's in-house integrator is for
fast prototyping and teaching. Its post-processing layer (surface → irradiation
→ response → mock light curves) is written to take snapshots from any of them.

## References (abbreviated)

Bardeen & Petterson 1975, ApJ 195, L65 · Cackett, Horne & Winkler 2007, MNRAS 380, 669 ·
Cackett et al. 2018, ApJ 857, 53 · Cackett, Bentz & Kara 2021, iScience (review, arXiv:2105.06926) ·
Cantiello, Jermyn & Lin 2021, ApJ 910, 94 · Dexter & Agol 2011, ApJ 727, L24 ·
Dexter & Begelman 2019, MNRAS 483, L17 · Edelson et al. 2015, ApJ 806, 129 ·
Edelson et al. 2019, ApJ 870, 123 · Fausnaugh et al. 2016, ApJ 821, 56 ·
Gardner & Done 2017, MNRAS 470, 3591 · Goad et al. 2016, ApJ 824, 11 · Goodman 2003, MNRAS 339, 937 ·
Guo et al. 2022, ApJ 929, 19 · Hagen et al. 2025, arXiv:2509.25324 ·
Hernández Santisteban et al. 2020, MNRAS 498, 5399 · Homayouni et al. 2019, ApJ 880, 126 ·
Jiang & Blaes 2020, ApJ 900, 25 · Jiang et al. 2017, ApJ 836, 186 · Kaastra et al. 2014, Science 345, 64 ·
Kammoun et al. 2019, ApJ 879, L24; 2021, MNRAS 503, 4163 · Kara et al. 2021, ApJ 922, 151 ·
Korista & Goad 2001, ApJ 553, 695; 2019, MNRAS 489, 5284 · Lawrence 2018, Nat. Astron. 2, 102 ·
Lawther et al. 2018, MNRAS 481, 533 · Li et al. 2021, ApJ 912, 118 (faint AGN, longer lags) ·
Lodato & Price 2010, MNRAS 405, 1212 · McKernan et al. 2012, MNRAS 425, 460 ·
Morgan et al. 2010, ApJ 712, 1129 · Nelson & Papaloizou 2000, MNRAS 315, 570 ·
Netzer 2022, MNRAS 509, 2637 · Netzer et al. 2024, arXiv:2410.02652 (AGN STORM 2 X) ·
Neustadt & Kochanek 2022, MNRAS 513, 1046 · Price et al. 2018, PASA 35, e031 (Phantom) ·
Pringle 1996, MNRAS 281, 357 · Rein & Liu 2012, A&A 537, A128 (REBOUND) ·
Secunda, Jiang & Greene 2025, arXiv:2501.06304 · Sirko & Goodman 2003, MNRAS 341, 501 ·
Starkey et al. 2017, ApJ 835, 65 · Starkey, Huang, Horne & Lin 2023, MNRAS 519, 2754 ·
Sun et al. 2020, ApJ 891, 178 (CHAR) · Tagawa, Haiman & Kocsis 2020, ApJ 898, 25
