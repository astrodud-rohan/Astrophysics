# Astrophysics Projects
 
Five projects demonstrating time-series signal processing, statistical inference, Bayesian
parameter estimation, computer vision, and imbalanced classification, applied to real
astrophysics problems (exoplanets, stellar populations, gravitational waves, galaxy
morphology, and pulsar surveys).
 
Each project's own folder contains a full README with data-source disclosure, detailed
methodology, and honest limitations. This document is a one-page index.
 
---
 
## 1. Exoplanet Transit Detection & Vetting
 
**What is it about**
A pipeline that detects candidate exoplanet transits in Kepler-style light curves via Box Least
Squares (BLS) search, then vets those candidates with machine learning to separate real planets
from astrophysical false positives (eclipsing binaries) — the same two-stage structure real
Kepler/TESS pipelines use.
 
**Core concepts**
- Time-series detrending (Savitzky-Golay filtering)
- Box Least Squares (BLS) period search
- Transit injection-recovery methodology
- Odd/even transit depth mismatch (real vetting diagnostic)
- Classical ML vs. CNN comparison on the same task
**Tech stack**
`batman-package` (Mandel-Agol transit models) · `astropy.timeseries.BoxLeastSquares` ·
`scikit-learn` (Random Forest, Gradient Boosting) · `TensorFlow/Keras` (1D CNN) · `pandas`/`numpy`
 
**Result / Findings**
Raw BLS detection strength alone is essentially useless for separating planets from false
positives (ROC-AUC ≈ 0.49, random). Adding vetting features (depth-to-noise ratio, odd/even
mismatch) lifts ROC-AUC to 0.96 — a CNN on raw folded curves reaches only 0.83.
 
**What this project achieved**
Demonstrated that detection and vetting are distinct pipeline problems, and produced a
defensible, literature-grounded explanation for *why* feature-based classical ML beats a CNN
here — mirroring the real reason NASA's production Kepler vetting system is feature-based.
 
---
 
## 2. Stellar Classification & HR Diagram from a Gaia-like Catalog
 
**What is it about**
Builds a multi-population stellar catalog (three real open clusters + field stars), constructs
the observational HR diagram, and recovers cluster membership purely from proper motion +
parallax via unsupervised clustering — without ever looking at color or magnitude.
 
**Core concepts**
- Apparent-to-absolute magnitude conversion via parallax
- Gaia-realistic photometric/astrometric noise scaling
- Gaussian Mixture Model clustering in kinematic space
- Parallax quality cuts (S/N-based, standard real-survey practice)
- Purity/completeness evaluation of recovered cluster membership
**Tech stack**
`scikit-learn` (`GaussianMixture`, `StandardScaler`) · `numpy`/`pandas` · `matplotlib`
 
**Result / Findings**
Two of three clusters (Pleiades, Hyades) were recovered with 100% purity and completeness,
because their proper motions are 5.6σ and 13σ from the field mean. The third (NGC 752) was
completely missed — its proper motion is only 1.5σ from the field, a real, documented limitation
of naive kinematic clustering.
 
**What this project achieved**
Delivered a genuine success *and* a genuine, well-explained failure case in the same analysis —
a stronger story than a project that only shows results that work, and it explains
why real cluster-finding pipelines combine kinematics with photometric/isochrone consistency.
 
---
 
## 3. Gravitational Wave Detection & Parameter Estimation (GW150914-like)
 
**What is it about**
A full matched-filtering detection pipeline for a GW150914-like binary black hole merger: signal
injection, matched-filter search over a template bank, and Bayesian parameter estimation (MCMC)
recovering the component masses and distance.
 
**Core concepts**
- Matched filtering against a bank of gravitational waveform templates
- Chirp mass / mass-ratio degeneracy
- MCMC (`emcee`) Bayesian parameter estimation
- SNR-distance scaling for analytically derived distance posteriors
- Detector noise PSD modeling
**Tech stack**
`pycbc` (`IMRPhenomD` waveform generation, `aLIGOZeroDetHighPower` PSD, matched filtering) ·
`emcee` (MCMC) · `corner` (posterior visualization) · `numpy`/`pandas`
 
**Result / Findings**
Recovered a matched-filter SNR of 23.7 (matching GW150914's real reported network SNR of ~23.7),
localized the merger time to within 5 ms, and produced mass/distance posteriors whose shapes
closely mirror the real published GW150914 posterior (mass1 skewed wide upward, mass2 skewed
wide downward).
 
**What this project achieved**
Used the same research-grade waveform physics and noise-PSD tooling as real LIGO/Virgo analyses
(only the strain data itself was simulated, with the calibration choice explicitly disclosed),
and kept detection, search, and inference as clearly separated pipeline stages.
 
---
 
## 4. Galaxy Morphology Classification (Elliptical / Spiral / Merger)
 
**What is it about**
Compares the real pre-deep-learning quantitative morphology feature set — CAS (Concentration,
Asymmetry, Smoothness) and Gini/M20 — against a CNN trained directly on pixel images, for
classifying galaxies into elliptical, spiral, and merger categories.
 
**Core concepts**
- Sersic surface-brightness profile modeling
- CAS statistics (Conselice 2003) and Gini/M20 (Lotz et al. 2004)
- Flux-centroid-based, aperture-restricted feature computation
- PSF convolution and survey-realistic imaging noise
- Classical feature engineering vs. CNN comparison
**Tech stack**
`scipy.ndimage` (Sersic/PSF/CAS computation) · `scikit-learn` (Random Forest) ·
`TensorFlow/Keras` (2D CNN) · `numpy`/`pandas`/`matplotlib`
 
**Result / Findings**
The CNN (96.6% accuracy) outperformed the classical CAS/Gini-M20 Random Forest (93.2%) — the
opposite result from Project 1, where classical features won. M20 specifically drove merger
classification, exactly matching its literature-documented purpose (detecting multi-nucleus
systems).
 
**What this project achieved**
Produced a genuine methodological contrast with Project 1 (features win vs. CNN wins, depending
on the problem) and included an honest mid-project debugging story: an initial version of the
CAS code let background noise swamp the asymmetry signal, fixed by switching to centroid-based
apertures matching real methodology.
 
---
 
## 5. Pulsar Candidate Identification (Real HTRU2 Survey Data)
 
**What is it about**
Classifies 17,898 real pulsar candidates from the actual HTRU-South survey (Parkes radio
telescope) as real pulsars or RFI/noise, handling the genuine ~9.9:1 class imbalance with
appropriate evaluation methodology.
 
**Core concepts**
- Imbalanced classification (class weighting, PR-AUC over raw accuracy)
- Pulse-profile and DM-SNR curve statistical features
- Precision/recall trade-off with real operational meaning (telescope follow-up cost)
- Feature-set richness comparison (8 vs. 30 features)
**Tech stack**
`scikit-learn` (Logistic Regression, Random Forest, Gradient Boosting) · `pandas`/`numpy`/`matplotlib`
 
**Result / Findings**
All models reached ROC-AUC > 0.97 despite the severe imbalance. Feature importance independently
reproduced the real published result (Lyon et al. 2016): pulse-profile kurtosis is the single
most discriminating feature. A richer 30-feature set improved ROC-AUC from 0.978 to 0.987.
 
**What this project achieved**
Sourced directly from the dataset creator's own GitHub repository — with results that independently validate against the real published literature rather than merely being internally consistent.
 
---