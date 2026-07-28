import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
 
DATA_DIR = "../data"
FIG_DIR = "../figures"
 
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
 
# ---------- 1. Example raw light curves per class ----------
npz = np.load(f"{DATA_DIR}/lightcurves.npz", allow_pickle=True)
t = npz["time"]
flux = npz["flux"]
ids = npz["ids"]
meta = pd.read_csv(f"{DATA_DIR}/metadata.csv").set_index("id")
 
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
examples = {"planet": "SIM-PL-0003", "false_positive": "SIM-FP-0000", "no_signal": "SIM-NS-0000"}
titles = {
    "planet": "Confirmed-planet-like (anchored: Kepler-10b system params)",
    "false_positive": "False positive (grazing eclipsing binary)",
    "no_signal": "No signal (variability + noise only)",
}
for ax, (label, sim_id) in zip(axes, examples.items()):
    idx = np.where(ids == sim_id)[0][0]
    ax.plot(t, flux[idx], lw=0.4, color="#2b6cb0")
    ax.set_title(titles[label], fontsize=10)
    ax.set_ylabel("Normalized flux")
axes[-1].set_xlabel("Time (days)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_example_lightcurves.png")
plt.close()
 
# ---------- 2. Phase-folded curves per class ----------
folded_npz = np.load(f"{DATA_DIR}/folded_curves.npz", allow_pickle=True)
folded = folded_npz["folded"]
fids = folded_npz["ids"]
 
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, (label, sim_id) in zip(axes, examples.items()):
    idx = np.where(fids == sim_id)[0][0]
    ax.plot(np.linspace(-0.15, 0.15, len(folded[idx])), folded[idx], "o-", ms=3, lw=1, color="#c05621")
    ax.set_title(label)
    ax.set_xlabel("Phase")
axes[0].set_ylabel("Folded flux")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_folded_lightcurves.png")
plt.close()
 
# ---------- 3. ROC curve comparison ----------
test_pred = pd.read_csv(f"{DATA_DIR}/test_predictions.csv")
cnn_res = np.load(f"{DATA_DIR}/cnn_test_results.npz")
 
fig, ax = plt.subplots(figsize=(6, 6))
for col, name in [("rf_proba", "Random Forest"), ("gb_proba", "Gradient Boosting")]:
    fpr, tpr, _ = roc_curve(test_pred["is_planet"], test_pred[col])
    ax.plot(fpr, tpr, label=name, lw=2)
 
fpr_bls, tpr_bls, _ = roc_curve(test_pred["is_planet"], test_pred["bls_power_baseline"])
ax.plot(fpr_bls, tpr_bls, label="BLS power alone (no ML)", lw=2, linestyle="--", color="gray")
 
fpr_cnn, tpr_cnn, _ = roc_curve(cnn_res["y_test"], cnn_res["proba"])
ax.plot(fpr_cnn, tpr_cnn, label="1D CNN (folded curve)", lw=2, linestyle=":", color="green")
 
ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Comparison: Classical BLS vs. ML Vetting Models")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_roc_comparison.png")
plt.close()
 
# ---------- 4. Feature importance ----------
import joblib
rf = joblib.load(f"{DATA_DIR}/random_forest.joblib")
FEATURE_COLS = [
    "bls_period", "bls_duration", "bls_power", "bls_depth",
    "bls_odd_even_mismatch", "n_transits_found", "depth_over_sigma",
    "duration_over_period",
]
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
importances.plot.barh(ax=ax, color="#38a169")
ax.set_xlabel("Feature importance (Random Forest)")
ax.set_title("Which BLS-derived features drive the vetting decision")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_feature_importance.png")
plt.close()
 
# ---------- 5. Confusion matrices side by side ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cms = {
    "Random Forest": confusion_matrix(test_pred["is_planet"], (test_pred["rf_proba"] > 0.5).astype(int)),
    "Gradient Boosting": confusion_matrix(test_pred["is_planet"], (test_pred["gb_proba"] > 0.5).astype(int)),
    "1D CNN": confusion_matrix(cnn_res["y_test"], (cnn_res["proba"] > 0.5).astype(int)),
}
for ax, (name, cm) in zip(axes, cms.items()):
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not_planet", "planet"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not_planet", "planet"])
    ax.set_title(name)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_confusion_matrices.png")
plt.close()
 
print("All figures saved to", FIG_DIR)
