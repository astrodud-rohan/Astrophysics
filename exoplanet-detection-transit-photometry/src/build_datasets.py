"""
Build the full labeled dataset of light curves.
"""
import numpy as np
import pandas as pd
from simulate_lightcurves import make_lightcurve
from reference_systems import CONFIRMED_PLANET_ANCHORS, FALSE_POSITIVE_ANCHORS
 
N_PLANET = 400
N_FALSE_POSITIVE = 400
N_NO_SIGNAL = 400
BASELINE_DAYS = 90.0
OUT_PATH = "../data/lightcurves.npz"
META_PATH = "../data/metadata.csv"
 
planet_anchors = list(CONFIRMED_PLANET_ANCHORS.keys())
fp_anchors = list(FALSE_POSITIVE_ANCHORS.keys())
 
rng = np.random.default_rng(2026)
 
records = []
flux_matrix = []
time_ref = None
 
seed_counter = 0
for i in range(N_PLANET):
    anchor = planet_anchors[i % len(planet_anchors)]
    t, flux, meta = make_lightcurve("planet", anchor, BASELINE_DAYS, seed=seed_counter)
    seed_counter += 10
    if time_ref is None:
        time_ref = t
    flux_matrix.append(flux)
    meta["id"] = f"SIM-PL-{i:04d}"
    records.append(meta)
 
for i in range(N_FALSE_POSITIVE):
    anchor = fp_anchors[i % len(fp_anchors)]
    t, flux, meta = make_lightcurve("false_positive", anchor, BASELINE_DAYS, seed=seed_counter)
    seed_counter += 10
    flux_matrix.append(flux)
    meta["id"] = f"SIM-FP-{i:04d}"
    records.append(meta)
 
for i in range(N_NO_SIGNAL):
    t, flux, meta = make_lightcurve("no_signal", None, BASELINE_DAYS, seed=seed_counter)
    seed_counter += 10
    flux_matrix.append(flux)
    meta["id"] = f"SIM-NS-{i:04d}"
    records.append(meta)
 
flux_matrix = np.array(flux_matrix)
df = pd.DataFrame(records)
 
np.savez_compressed(OUT_PATH, time=time_ref, flux=flux_matrix, ids=df["id"].values)
df.to_csv(META_PATH, index=False)
 
print(f"Saved {flux_matrix.shape[0]} light curves x {flux_matrix.shape[1]} points -> {OUT_PATH}")
print(df["label"].value_counts())
print(df.head())