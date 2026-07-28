import nbformat as nbf
 
nb = nbf.v4.new_notebook()
cells = []
 
def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))
 
def code(s):
    cells.append(nbf.v4.new_code_cell(s))
 
md("""# Exoplanet Transit Detection & Vetting: BLS Search + ML Classification
 
**Project 1 of 5 — Astrophysics Portfolio**
 
## Data disclosure
 
All light curves in this notebook are **synthetic**, generated in this sandbox because live access
to MAST / the NASA Exoplanet Archive is not available here. They are not arbitrary noise, though:
every injected signal is **anchored to real, published system parameters** (Kepler-10b, Kepler-90b/i,
TRAPPIST-1b/e, Kepler-452b, HAT-P-7b for planets; realistic grazing/diluted/background eclipsing-binary
archetypes for false positives — see `src/reference_systems.py` for exact sourcing). Noise is scaled to
real Kepler long-cadence CDPP tables, and transit shapes are generated with `batman-package`
(Mandel & Agol 2002), the same physics library used in real research pipelines.
 
This "injection-recovery" approach is itself a standard technique in the field — it's how the
Kepler/TESS teams validate their own pipelines' detection efficiency and false-alarm rates — so the
methodology is industry-standard even though the specific light curves are simulated rather than
downloaded.
 
## Pipeline overview
 
1. Simulate 1,200 light curves (400 planet / 400 false-positive / 400 no-signal), Kepler long-cadence sampling, 90-day baseline
2. Detrend with Savitzky-Golay filtering to remove stellar variability
3. Run Box Least Squares (BLS) period search (`astropy.timeseries.BoxLeastSquares`)
4. Extract vetting features (period, depth, SNR, odd/even mismatch, transit count)
5. Train two model families: tabular classifiers (Random Forest / Gradient Boosting) on BLS features, and a 1D CNN on phase-folded curves (AstroNet-style)
6. Compare against a **BLS-power-only baseline** (no ML) to quantify the value the ML layer adds
""")
 
code("""import sys
sys.path.insert(0, '../src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
 
DATA_DIR = '../data'
FIG_DIR = '../figures'
plt.rcParams.update({'figure.dpi': 100})
""")
 
md("## 1. Reference system parameters (real, published anchors)")
code("""from reference_systems import CONFIRMED_PLANET_ANCHORS, FALSE_POSITIVE_ANCHORS
pd.DataFrame(CONFIRMED_PLANET_ANCHORS).T
""")
 
code("""pd.DataFrame(FALSE_POSITIVE_ANCHORS).T
""")
 
md("""## 2. Simulated light curve dataset
 
Built via `build_dataset.py`. Class balance (1 : 1 : 1 planet : false-positive : no-signal here, for
balanced training) intentionally undersamples the real-world false-positive-heavy ratio seen in raw
Kepler candidate lists — in production, false positives outnumber real planets several-fold, which is
precisely why automated vetting classifiers exist: human eyeballing doesn't scale to thousands of
candidates.""")
 
code("""meta = pd.read_csv(f'{DATA_DIR}/metadata.csv')
print(meta['label'].value_counts())
meta.head()
""")
 
md("### Example raw light curves, one per class")
code("""display(Image(f'{FIG_DIR}/01_example_lightcurves.png'))
""")
 
md("""## 3. Detrending + BLS period search
 
Savitzky-Golay detrending removes slow stellar variability (rotation, granulation) while preserving
transit-shaped dips, which are short relative to the smoothing window. BLS then searches a grid of
trial periods/durations for the box-shaped dip that best matches a transit.""")
 
code("""features = pd.read_csv(f'{DATA_DIR}/features.csv')
features[['label', 'bls_period', 'bls_duration', 'bls_power', 'bls_depth', 'bls_odd_even_mismatch', 'n_transits_found']].groupby('label').median()
""")
 
md("### Phase-folded light curves at the best BLS period")
code("""display(Image(f'{FIG_DIR}/02_folded_lightcurves.png'))
""")
 
md("""## 4. Why BLS detection alone isn't enough
 
BLS power/SNR ranks *how box-shaped* a dip is — but grazing and diluted eclipsing binaries also
produce strong box-shaped dips. This is exactly the false-positive problem the real Kepler/TESS
pipelines ran into, motivating the Robovetter/Autovetter classifiers. Let's quantify that gap here.""")
 
code("""baseline_auc = pd.read_csv(f'{DATA_DIR}/model_comparison.csv')
baseline_auc
""")
 
md("""## 5. ML vetting models
 
Two model families, mirroring the two real approaches used in the literature:
 
- **Tabular classifiers on BLS-derived features** (Random Forest, Gradient Boosting) — mirrors NASA's
  actual production Robovetter/Autovetter (Coughlin et al. 2016; McCauliff et al. 2015)
- **1D CNN on phase-folded, binned light curves** — mirrors the AstroNet global-view architecture
  (Shallue & Vanderburg 2018)""")
 
code("""test_pred = pd.read_csv(f'{DATA_DIR}/test_predictions.csv')
test_pred.describe()
""")
 
md("### ROC comparison: classical BLS vs. ML vetting models")
code("""display(Image(f'{FIG_DIR}/03_roc_comparison.png'))
""")
 
md("""**Key result:** thresholding on raw BLS power alone is essentially random (ROC-AUC ≈ 0.49) at
separating real planet-like transits from false positives in this dataset — both classes produce
strong box-shaped dips. Adding a small set of vetting features (depth-to-noise ratio, odd/even depth
mismatch, transit count, duration/period ratio) and training a classifier on them lifts ROC-AUC to
**~0.96**. The CNN on raw folded curves reaches ~0.83 — better than the BLS-only baseline, but behind
the feature-based classifiers, consistent with why NASA's actual production vetting system is
feature-based rather than a raw-pixel deep net.""")
 
md("### Feature importance (Random Forest)")
code("""display(Image(f'{FIG_DIR}/04_feature_importance.png'))
""")
 
md("""The two dominant features are `bls_power` and `bls_depth` — as expected, transit shape strength
still matters — but `depth_over_sigma` (a physically motivated SNR) and `bls_odd_even_mismatch` (the
classic real-vs-eclipsing-binary vetting diagnostic used by the actual Kepler team, since true planets
show equal odd/even transit depths while blended eclipsing binaries often don't) contribute
meaningfully on top of raw BLS power — this is the actual mechanism by which the classifier
out-performs the naive baseline.""")
 
md("### Confusion matrices")
code("""display(Image(f'{FIG_DIR}/05_confusion_matrices.png'))
""")
 
md("""## 6. Summary & interview talking points
 
1. **BLS detection and ML vetting are two separate stages** — BLS finds candidate periodic dips; ML
   vetting decides which candidates are astrophysically real planets vs. false positives. Conflating
   the two (i.e., trusting BLS power/SNR as a final answer) is a common naive mistake.
2. **Odd/even depth mismatch and depth-to-noise ratio are the features that do the real
   discriminating work**, not just raw detection strength — this mirrors real published vetting
   metrics (Coughlin et al. 2016 Robovetter).
3. **Classical ML (Random Forest/GB) on domain-engineered features (ROC-AUC 0.96) beats a CNN on raw
   folded curves (ROC-AUC 0.83) here** — a legitimate, defensible empirical finding for this dataset
   size, and consistent with why NASA's actual production system (Robovetter) is feature-based.
4. **Injection-recovery methodology**: the simulation-based approach used throughout this notebook is
   the same technique real pipelines use to validate detection efficiency, even though this
   notebook's data is synthetic due to sandbox network constraints (no live MAST/NASA Exoplanet
   Archive access here).
""")
 
nb['cells'] = cells
nbf.write(nb, '../notebooks/exoplanet_detection.ipynb')
print("Notebook written.")
