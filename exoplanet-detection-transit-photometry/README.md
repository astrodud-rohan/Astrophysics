# Exoplanet Transit Detection & Vetting via BLS + ML Classification
 
## Data disclosure (read this first)
 
**All light curves in this project are synthetic.**
 
To keep the project scientifically meaningful anyway, every injected signal is **anchored to real,
published system parameters** — periods, radius ratios, impact parameters, and stellar properties for
Kepler-10b, Kepler-90b, Kepler-90i, TRAPPIST-1b, TRAPPIST-1e, Kepler-452b, and HAT-P-7b (sourced from
Batalha et al. 2011, Shallue & Vanderburg 2018, Gillon et al. 2017/Agol et al. 2021, Jenkins et al.
2015, and Pál et al. 2008 respectively — see `src/reference_systems.py` for the exact numbers and
citations). False positives are built from realistic eclipsing-binary archetypes (grazing, background-
diluted, secondary-eclipse, aliased contact binary) that are the actual dominant failure modes in real
transit surveys. Transit shapes use `batman-package` (Mandel & Agol 2002), the same physics library
used in real published pipelines. Noise is scaled from real Kepler CDPP tables by magnitude.
 
This is also methodologically legitimate on its own terms: **injection-recovery testing** — injecting
known synthetic signals into (real or simulated) data to measure a pipeline's detection efficiency and
false-alarm rate — is a standard technique the Kepler and TESS teams use to validate their own
pipelines.
 
## What this project demonstrates
 
- Time-series signal processing (Savitzky-Golay detrending, BLS period search)
- Use of a real domain physics package (`batman`) rather than ad hoc sinusoidal fakes
- Feature engineering directly from astrophysical vetting diagnostics (odd/even depth mismatch, SNR)
- Comparative ML: classical tabular classifiers vs. a CNN, with a rigorous **no-ML baseline**
- Honest empirical reporting of *where* each method wins and loses, not just a headline accuracy number
## Pipeline
 
```
reference_systems.py      Real system parameter anchors (documented sourcing)
simulate_lightcurves.py   batman transit injection + stellar variability + noise
build_dataset.py          Generates 1,200 labeled light curves (400/400/400 split)
detect_and_extract.py     Savitzky-Golay detrend -> BLS search -> feature extraction
train_classifier.py       Random Forest + Gradient Boosting on BLS features
train_cnn.py              1D CNN on phase-folded, binned light curves (AstroNet-style)
make_figures.py           All plots
notebooks/exoplanet_detection.ipynb   Full executed walkthrough
```
 
## Headline results
 
| Model | ROC-AUC |
|---|---|
| BLS power alone (no ML) | **0.49** (~random) |
| 1D CNN on folded curve | 0.83 |
| Gradient Boosting on BLS features | 0.96 |
| **Random Forest on BLS features** | **0.96** |
 
**The core finding:** raw BLS detection strength does not separate real planets from false positives
in this dataset — grazing/diluted eclipsing binaries produce box-shaped dips just as strong as real
transits. What actually discriminates is a small set of physically-motivated vetting features, above
all **odd/even transit depth mismatch** (true planets: symmetric; blended eclipsing binaries: often
not) and **depth-to-photometric-noise ratio**. This mirrors why NASA's real production vetting system
(the Kepler "Robovetter") is a feature-based classifier rather than a raw-pixel deep model — and why
the CNN here, despite having access to the full folded light curve shape, underperforms the simpler
feature-based models.
 
## Honest limitations
 
- Synthetic data (see disclosure above) — real archive light curves have systematics (spacecraft
  pointing jitter, safe modes, quarter-to-quarter discontinuities) not modeled here.
- 1,200 light curves is small by modern deep-learning standards; the CNN result should be read as
  indicative, not definitive, at this sample size.
- Real vetting pipelines incorporate centroid/pixel-level diagnostics (to catch background blends) that
  aren't reproduced here since this project works from flux time series only.
 