# Stellar Classification & HR Diagram from a Gaia-like Catalog
 
## Data disclosure (read this first)
 
**This catalog is synthetic.**
 
The catalog is anchored to real, published values: distances, ages, and proper motions for the
**Pleiades** (~136 pc, ~125 Myr), **Hyades** (~47 pc, ~680 Myr), and **NGC 752** (~440 pc, ~1.4 Gyr)
open clusters, plus Gaia DR3's actual published photometric and astrometric precision-vs-magnitude
curves for realistic noise. Stellar physics uses simplified analytic mass–luminosity–temperature
relations (documented in `src/simulate_catalog.py`), not a full stellar-isochrone grid — this
reproduces realistic HR-diagram *morphology* and kinematic *separability*, which is what the analysis
pipeline actually needs, without overclaiming precision stellar modeling.
 
## What this project demonstrates
 
- Realistic simulation of a multi-population astronomical catalog with correct noise scaling
- Standard real-world data quality practices (parallax S/N cuts — Gaia analyses always need these)
- HR diagram construction from apparent magnitude + parallax → absolute magnitude
- **Unsupervised** cluster membership recovery via Gaussian Mixture clustering on kinematics
  (proper motion + parallax), deliberately avoiding circular validation against the HRD itself
- Honest diagnosis of *when and why* the method fails, not just a headline success number

## Pipeline
 
```
reference_data.py           Real cluster parameters + Gaia DR3 precision tables (documented sourcing)
simulate_catalog.py         IMF sampling, simplified stellar physics, cluster + field simulation
hr_diagram_clustering.py    Quality cuts, absolute magnitude, GMM clustering, recovery evaluation
make_figures.py             All plots
notebooks/gaia_hr_diagram.ipynb   Full executed walkthrough
```
 
## Headline results
 
| Cluster | True N | PM separation from field | GMM completeness | GMM purity |
|---|---|---|---|---|
| Pleiades | 250 | 5.6σ | 100% | 100% |
| Hyades | 180 | 13.0σ | 100% | 100% |
| NGC 752 | 93 | **1.5σ** | **0%** | n/a (absorbed into field) |
 
**The core finding:** kinematic clustering (proper motion + parallax, no color/magnitude information)
recovers cluster membership essentially perfectly when a cluster's motion is kinematically distinct
from the field — but fails completely for a cluster whose proper motion happens to sit close to the
field's own distribution. This isn't an implementation bug; it's a genuine, well-documented limitation
of naive kinematic clustering, and it's exactly why real production cluster-finding pipelines (e.g.
UPMASK, or the methodology behind Gaia DR3's own published cluster catalogs) combine kinematics with
photometric isochrone consistency and radial velocities rather than relying on proper motion + parallax
alone.

## Honest limitations
 
- Synthetic data (see disclosure above); no real systematics (Gaia scanning-law-dependent errors,
  crowding in dense fields, extinction/reddening) are modeled.
- Simplified stellar-evolution proxy (luminosity boost + fixed Teff range for "evolved" stars) rather
  than a real isochrone grid — adequate for demonstrating pipeline mechanics, not for precision age
  dating.
- No radial velocity dimension was simulated; a real analysis facing NGC 752's kinematic overlap would
  likely add RV as a fourth clustering dimension, which is a natural next step.