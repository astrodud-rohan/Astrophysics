import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
 
DATA_DIR = "data"
FIG_DIR = "figures"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
 
CLASS_COLORS = {"elliptical": "#c53030", "spiral": "#2b6cb0", "merger": "#38a169"}
 
# ---------- 1. Example images grid ----------
d = np.load(f"{DATA_DIR}/galaxy_images.npz", allow_pickle=True)
images, labels = d["images"], d["labels"]
 
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
for row, cls in enumerate(["elliptical", "spiral", "merger"]):
    idx = np.where(labels == cls)[0][:6]
    for col, i in enumerate(idx):
        axes[row, col].imshow(images[i], cmap="inferno", origin="lower")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(cls)
    axes[row, 0].text(-15, 32, cls, rotation=90, va="center", fontsize=12)
plt.suptitle("Example simulated galaxy images (Sersic profiles + PSF + SDSS-like noise)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_example_galaxies.png")
plt.close()
 
# ---------- 2. CAS/Gini-M20 feature space ----------
cas = pd.read_csv(f"{DATA_DIR}/cas_features.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for cls, color in CLASS_COLORS.items():
    sub = cas[cas.label == cls]
    axes[0].scatter(sub.concentration, sub.m20, s=12, alpha=0.6, color=color, label=cls)
    axes[1].scatter(sub.gini, sub.m20, s=12, alpha=0.6, color=color, label=cls)
axes[0].set_xlabel("Concentration"); axes[0].set_ylabel("M20")
axes[0].set_title("Concentration vs. M20")
axes[0].legend()
axes[1].set_xlabel("Gini"); axes[1].set_ylabel("M20")
axes[1].set_title("Gini vs. M20 (Lotz et al. 2004 classification plane)")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_cas_feature_space.png")
plt.close()
 
# ---------- 3. Feature importance ----------
import joblib
rf = joblib.load(f"{DATA_DIR}/rf_cas_model.joblib")
feature_cols = ["concentration", "asymmetry", "smoothness", "gini", "m20"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.5))
importances.plot.barh(ax=ax, color="#805ad5")
ax.set_xlabel("Feature importance (Random Forest)")
ax.set_title("Which classical morphology features drive classification")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_feature_importance.png")
plt.close()
 
# ---------- 4. Confusion matrices: RF vs CNN ----------
rf_pred = pd.read_csv(f"{DATA_DIR}/rf_test_predictions.csv")
cnn_pred = pd.read_csv(f"{DATA_DIR}/cnn_test_predictions.csv")
labels_order = ["elliptical", "spiral", "merger"]
 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, df) in zip(axes, [("Random Forest (CAS/Gini-M20)", rf_pred), ("CNN (raw pixels)", cnn_pred)]):
    cm = confusion_matrix(df.true_label, df.pred_label, labels=labels_order)
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels_order, rotation=45)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels_order)
    ax.set_title(name)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_confusion_matrices.png")
plt.close()
 
# ---------- 5. CNN training curve ----------
hist = pd.read_csv(f"{DATA_DIR}/cnn_history.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(hist["accuracy"], label="train")
axes[0].plot(hist["val_accuracy"], label="val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()
axes[0].set_title("CNN training accuracy")
axes[1].plot(hist["loss"], label="train")
axes[1].plot(hist["val_loss"], label="val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()
axes[1].set_title("CNN training loss")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_cnn_training_curve.png")
plt.close()
 
print("All figures saved to", FIG_DIR)