# Galaxy Morphology Classification — Classical Features vs. CNN
 
## Data disclosure (read this first)
 
**These are synthetic galaxy images.**
 
Images are procedurally generated from real, documented structural models: **Sersic light profiles**
(n=4 de Vaucouleurs for ellipticals, n=1 exponential disks for spirals, matching real SDSS bulge-disk
decomposition conventions), **logarithmic spiral arms** with literature-typical pitch angles, **SDSS-
realistic PSF and pixel scale** (1.4" seeing, 0.396"/px), and a simplified tidal-shear proxy for
mergers. See `src/reference_morphology.py` and `src/simulate_galaxies.py` for full sourcing. This
reproduces realistic morphological *structure*, not pixel-for-pixel fidelity to real SDSS imaging.

 
## What this project demonstrates
 
- The actual pre-deep-learning quantitative morphology feature set: **Concentration, Asymmetry,
  Smoothness** (CAS, Conselice 2003) and **Gini/M20** (Lotz et al. 2004) — not invented features
- Correct methodology: flux-centroid-based, aperture-restricted feature computation, matching real
  practice (an earlier naive version using the geometric image center and full frame is documented
  as a mistake that was caught and fixed)
- Honest comparison between classical ML and a CNN, with the CNN winning here — the *opposite* result
  from Project 1, useful for discussing when each approach is preferable
- Real published context: M20 was specifically designed to catch multi-nucleus/merger signatures, and
  that's exactly where it does the most discriminating work in this simulation

## Pipeline
 
```
reference_morphology.py     Real Sersic indices, SDSS imaging params, Galaxy Zoo class fractions
simulate_galaxies.py        Procedural Sersic + spiral-arm + merger image generator, PSF + noise
compute_cas_features.py     CAS (Conselice 2003) + Gini/M20 (Lotz et al. 2004) feature extraction
train_classifiers.py        Random Forest (CAS features) + CNN (raw pixels)
make_figures.py             All plots
notebooks/galaxy_morphology.ipynb   Full executed walkthrough
```
 
## Headline results
 
| Model | Accuracy |
|---|---|
| Random Forest on CAS/Gini-M20 features | 93.2% |
| **CNN on raw pixel images** | **96.6%** |
 
| Feature | RF importance |
|---|---|
| Concentration | 39.6% |
| M20 | 33.0% |
| Gini | 17.3% |
| Smoothness | 6.1% |
| Asymmetry | 4.0% |
 
**The core finding:** the CNN outperforms classical features here — consistent with the real published
result in Dieleman et al. 2015, where a CNN beat hand-engineered morphology features on actual Galaxy
Zoo data. This is the *opposite* conclusion from Project 1 (exoplanet vetting), where classical features
beat a CNN — a genuinely useful contrast for discussing *when* each approach wins rather than defaulting
to "deep learning is always better" or "always engineer features first."

## Honest limitations
 
- Synthetic images (see disclosure above); no real SDSS pixel-level systematics (cosmic rays, chip
  gaps, real PSF variation across the field) are modeled.
- The merger simulation is a simplified shear proxy, not an N-body merger simulation — real tidal tails
  have more complex, extended morphology than reproduced here.
- Only 3 broad classes (elliptical/spiral/merger); real Galaxy Zoo has a much finer taxonomy (bar
  presence, arm number, edge-on disks, etc.) not attempted here.