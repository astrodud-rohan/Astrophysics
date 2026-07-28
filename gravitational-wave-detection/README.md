# Gravitational Wave Detection & Parameter Estimation (GW150914-like)
 
## Data disclosure (read this first)
 
**The strain data is simulated**
 
**What is real and research-grade:** the waveform model (`IMRPhenomD` via `pycbc`, the actual waveform
family used in real LIGO/Virgo parameter estimation) and the noise PSD (`aLIGOZeroDetHighPower`,
PyCBC's built-in analytic aLIGO design-sensitivity curve). The signal is injected with masses matching
the real published GW150914 values (36.2 + 29.1 Msun, Abbott et al. 2016) into a colored Gaussian noise
realization drawn from that PSD.
 
**One documented calibration choice:** `aLIGOZeroDetHighPower` is quieter than the real O1-era detector
noise present when GW150914 was actually observed. Injecting at the literal published distance (410
Mpc) against this PSD gives an unrealistic SNR (~96). To keep results comparable to the real event, the
signal is injected at an **effective distance of ~1639 Mpc**, calibrated via `pycbc.filter.sigma` to
reproduce the real reported network SNR of ~24. This is a deliberate, stated modeling choice, not hidden
tuning — see `src/reference_events.py` for the full explanation and the calibration code.
 
## What this project demonstrates
 
- Real gravitational-wave signal processing tools (`pycbc`): waveform generation, PSD estimation,
  matched filtering
- Realistic detection-pipeline structure: search stage (template bank, unknown true parameters) kept
  separate from the "known template" validation stage
- Bayesian parameter estimation via MCMC (`emcee`), with a methodologically sound simplification
  (time/phase-marginalized matched-filter likelihood; analytically derived distance posterior)
- Honest, explicit handling of a PSD/distance calibration choice rather than silently absorbing it

## Pipeline
 
```
reference_events.py       Real GW150914 parameters + documented effective-distance calibration
build_injection.py        IMRPhenomD signal injected into aLIGO-PSD colored noise
matched_filter_search.py  Template-bank matched-filter search over (mass1, mass2)
run_pe.py                 MCMC (emcee) posterior over masses + analytically derived distance
make_figures.py           All plots
notebooks/gw_detection_pe.ipynb   Full executed walkthrough
```
 
## Headline results
 
| Quantity | Recovered | True / Real published |
|---|---|---|
| Matched-filter SNR (true-parameter template) | 23.7 | ~23.7 (real GW150914 network SNR) |
| Merger time | within 5 ms | exact |
| Best-fit chirp mass (template-bank search) | 28.19 Msun | 28.22 Msun (true injected) |
| MCMC mass1 posterior | 36.5 +4.7/-2.8 Msun | 36.2 Msun (real: 36 +5/-4) |
| MCMC mass2 posterior | 28.8 +2.5/-3.7 Msun | 29.1 Msun (real: 29 +4/-4) |
| Derived distance posterior | 1657 +13/-42 Mpc | 1639 Mpc (effective, by construction) |
 
**The core finding:** the full detection → search → inference pipeline recovers the true injected
parameters well within its own credible intervals, and — notably — the *shape* of the mass posterior
uncertainty (mass1 skewed wider upward, mass2 skewed wider downward, reflecting the chirp-mass/mass-ratio
degeneracy) closely mirrors the real published GW150914 posterior shape, even though this is a
simplified 2-parameter (rather than full 15-parameter) analysis.

## Honest limitations
 
- Strain is simulated, not downloaded from GWOSC (see disclosure above).
- Only 2 of the ~15 real GW parameters (mass1, mass2) are sampled via MCMC; spins, inclination, sky
  location, and polarization are fixed/simplified — a full analysis would use nested sampling (e.g.
  `bilby`, `dynesty`) over the full parameter set.
- Single "detector" (no real two-detector coincidence/triangulation), so no sky localization is
  produced here.
- The effective-distance calibration (see disclosure) means the "true distance" being recovered is a
  documented construct, not GW150914's literal 410 Mpc.