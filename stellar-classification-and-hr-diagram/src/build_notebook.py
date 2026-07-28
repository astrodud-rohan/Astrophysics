import nbformat as nbf
 
nb = nbf.v4.new_notebook()
cells = []
 
def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))
 
def code(s):
    cells.append(nbf.v4.new_code_cell(s))
 
md("""# Stellar Classification & HR Diagram from a Gaia-like Catalog
 
**Project 2 of 5 — Astrophysics Portfolio**
 
## Data disclosure
 
This notebook uses a **synthetic Gaia-like catalog**. To keep the analysis scientifically
meaningful, the catalog is anchored to real published values: cluster distances, ages, and proper
motions for the **Pleiades, Hyades, and NGC 752** open clusters (Cantat-Gaudin et al. 2018; Perryman et
al. 1998; standard open-cluster catalogs), and Gaia DR3's actual published photometric/astrometric
precision-vs-magnitude curves. Stellar physics uses simplified analytic mass–luminosity–temperature
relations rather than a full isochrone grid — documented in `src/simulate_catalog.py`. The goal is
realistic HRD **morphology** and realistic **kinematic separability** between clusters and field stars,
for demonstrating the actual Gaia analysis workflow (ADQL-style catalog structure, HRD construction,
kinematic cluster-membership recovery), not precision stellar modeling.
 
## Pipeline
 
1. Simulate a ~2,600-star catalog: 3 real open clusters + a field population, with Gaia-DR3-realistic
   photometric and astrometric noise
2. Apply standard Gaia analysis quality cuts (positive parallax, parallax S/N > 5)
3. Compute absolute magnitudes and build the observational HR diagram
4. Recover cluster membership via **unsupervised GMM clustering on proper motion + parallax only**
   (not on color-magnitude, to avoid circularity)
5. Validate recovered clusters against ground truth and diagnose *why* one cluster (NGC 752) is harder
   to recover than the other two
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
 
md("## 1. Reference cluster parameters (real, published anchors)")
code("""from reference_data import CLUSTERS, FIELD_POPULATION
pd.DataFrame(CLUSTERS).T
""")
 
md("## 2. Simulated catalog")
code("""catalog = pd.read_csv(f'{DATA_DIR}/gaia_like_catalog.csv')
print(f'Total stars: {len(catalog)}')
catalog['true_cluster'].value_counts()
""")
 
code("""catalog.head()
""")
 
md("""## 3. Quality cuts
 
Standard real-world Gaia analysis practice: reject negative parallaxes (a real, well-documented noise
artifact at the faint end of Gaia data) and require parallax signal-to-noise > 5.""")
 
code("""from hr_diagram_clustering import apply_quality_cuts, compute_abs_mag
n_before = len(catalog)
df = apply_quality_cuts(catalog)
df = compute_abs_mag(df)
print(f'{n_before} -> {len(df)} stars retained ({100*len(df)/n_before:.1f}%)')
""")
 
md("## 4. Observational HR diagram (true population labels)")
code("""display(Image(f'{FIG_DIR}/01_hr_diagram_true_labels.png'))
""")
 
md("""Note the main-sequence turnoff point: NGC 752 (oldest, ~1.4 Gyr) turns off at a lower luminosity
than Pleiades (~125 Myr) — the classic age diagnostic used throughout real cluster astrophysics. Field
stars show a much broader, noisier spread since they span a huge range of distances and ages along the
line of sight.""")
 
md("""## 5. Kinematic cluster membership recovery (GMM on proper motion + parallax)
 
Crucially, we cluster on **kinematics only** (proper motion + parallax), not on color-magnitude — this
mirrors the actual technique used in real cluster-finding, and avoids the circularity of using the HRD
to find clusters that are then validated on the HRD.""")
 
code("""from hr_diagram_clustering import run_gmm_clustering, match_clusters_to_truth
from sklearn.metrics import adjusted_rand_score, confusion_matrix
 
n_true_clusters = df['true_cluster'].nunique()
labels, probs, gmm = run_gmm_clustering(df, n_components=n_true_clusters)
df, mapping = match_clusters_to_truth(df, labels)
 
ari = adjusted_rand_score(df['true_cluster'], df['gmm_label'])
print(f'Adjusted Rand Index: {ari:.4f}')
mapping
""")
 
md("### Proper motion diagram: why some clusters separate easily and others don't")
code("""display(Image(f'{FIG_DIR}/03_proper_motion_diagram.png'))
""")
 
code("""from reference_data import CLUSTERS, FIELD_POPULATION
f = FIELD_POPULATION
for name, c in CLUSTERS.items():
    d_ra = abs(c['pm_ra'] - f['pm_ra_center']) / f['pm_ra_spread']
    d_dec = abs(c['pm_dec'] - f['pm_dec_center']) / f['pm_dec_spread']
    sep = np.hypot(d_ra, d_dec)
    print(f'{name}: proper-motion separation from field mean = {sep:.2f} sigma')
""")
 
md("""**Key finding:** Pleiades (5.6σ) and Hyades (13σ) are kinematically distinct enough from the field
population that GMM recovers them with 100% purity and completeness. NGC 752, however, sits only ~1.5σ
from the field's proper-motion mean — its motion across the sky isn't distinctive enough on its own —
so pure kinematic clustering absorbs it into the field component almost entirely. This is a real,
well-documented limitation of naive kinematic clustering for more distant, less kinematically-distinct
clusters, and is exactly why real cluster-membership pipelines (e.g., UPMASK, or Gaia DR3's own
cluster catalogs) combine kinematics with photometric/isochrone consistency and radial velocities rather
than relying on proper motion + parallax alone.""")
 
md("### Parallax distribution: cluster spikes over the smooth field background")
code("""display(Image(f'{FIG_DIR}/04_parallax_distribution.png'))
""")
 
md("### HR diagram colored by GMM-recovered membership")
code("""display(Image(f'{FIG_DIR}/02_hr_diagram_gmm_recovered.png'))
""")
 
md("""Pleiades and Hyades recover clean, tight sequences matching their true isochrone shapes — a good
internal-consistency check, since the clustering never saw color/magnitude information. NGC 752's true
members are scattered into the field-labeled population instead of forming their own recovered group.""")
 
md("### Confusion matrix: true population vs. GMM-recovered cluster")
code("""display(Image(f'{FIG_DIR}/05_confusion_matrix.png'))
""")
 
md("""## 6. Summary & interview talking points
 
1. **Kinematic clustering (proper motion + parallax) can recover open cluster membership with
   essentially perfect purity/completeness — but only when the cluster's motion is kinematically
   distinct from the field.** Pleiades and Hyades: 100%/100%. NGC 752: ~0%, because its proper motion
   is only ~1.5σ from the field mean.
2. **This is a real, well-known limitation, not a bug** — it's the reason production cluster-finding
   pipelines (UPMASK, Gaia DR3 cluster catalogs) combine kinematics with photometric isochrone
   consistency and, where available, radial velocities, rather than relying on proper motion +
   parallax in isolation.
3. **Quality cuts matter and are standard practice** — filtering on `parallax > 0` and
   `parallax/parallax_error > 5` removed ~20% of sources here, mirroring how real Gaia analyses handle
   noisy faint-end astrometry.
4. **Clustering on kinematics, not color-magnitude, avoids circular validation** — the fact that
   recovered Pleiades/Hyades members trace clean, tight HRD sequences is an independent consistency
   check on the clustering result, not something built into it.
""")
 
nb['cells'] = cells
nbf.write(nb, '../notebooks/gaia_hr_diagram.ipynb')
print("Notebook written.")