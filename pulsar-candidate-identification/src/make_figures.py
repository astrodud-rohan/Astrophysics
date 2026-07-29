import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
 
DATA_DIR = "data"
FIG_DIR = "figures"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
 
df = pd.read_csv(f"{DATA_DIR}/htru2_clean.csv")
 
# ---------- 1. Class imbalance ----------
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["class"].value_counts().sort_index()
ax.bar(["Non-pulsar\n(RFI/noise)", "Real pulsar"], counts.values, color=["#a0aec0", "#c53030"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f"{v:,}\n({100*v/len(df):.1f}%)", ha="center")
ax.set_ylabel("Number of candidates")
ax.set_title(f"Real HTRU2 class distribution (n={len(df):,})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_imbalance.png")
plt.close()
 
# ---------- 2. Key feature distributions by class ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
feat_pairs = [("mean_ip", "Mean of integrated profile"),
              ("kurtosis_ip", "Excess kurtosis of integrated profile"),
              ("mean_dmsnr", "Mean of DM-SNR curve"),
              ("std_dmsnr", "Std of DM-SNR curve")]
for ax, (col, title) in zip(axes.ravel(), feat_pairs):
    for cls, color, label in [(0, "#a0aec0", "non-pulsar"), (1, "#c53030", "pulsar")]:
        sub = df[df["class"] == cls][col]
        ax.hist(sub, bins=50, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(title)
    ax.legend()
plt.suptitle("Real HTRU2 feature distributions by class")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_feature_distributions.png")
plt.close()
 
# ---------- 3. ROC + PR curves ----------
test_pred = pd.read_csv(f"{DATA_DIR}/test_predictions.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for col, name, color in [
    ("logistic_regression_proba", "Logistic Regression", "#2b6cb0"),
    ("random_forest_proba", "Random Forest", "#38a169"),
    ("gradient_boosting_proba", "Gradient Boosting", "#c53030"),
]:
    fpr, tpr, _ = roc_curve(test_pred["true"], test_pred[col])
    axes[0].plot(fpr, tpr, label=name, color=color, lw=2)
    prec, rec, _ = precision_recall_curve(test_pred["true"], test_pred[col])
    axes[1].plot(rec, prec, label=name, color=color, lw=2)
 
axes[0].plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend()
 
baseline_prevalence = test_pred["true"].mean()
axes[1].axhline(baseline_prevalence, color="gray", linestyle=":", lw=1,
                 label=f"baseline prevalence ({baseline_prevalence:.3f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve (more informative given 9.9:1 imbalance)")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_roc_pr_curves.png")
plt.close()
 
# ---------- 4. Feature importance ----------
imp = pd.read_csv(f"{DATA_DIR}/feature_importances.csv", index_col=0)
imp = imp.sort_values("importance")
fig, ax = plt.subplots(figsize=(7, 5))
imp["importance"].plot.barh(ax=ax, color="#805ad5")
ax.set_xlabel("Feature importance (Random Forest)")
ax.set_title("Which real pulsar-vetting features matter most")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_feature_importance.png")
plt.close()
 
# ---------- 5. Confusion matrices ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_cols = [("logistic_regression_proba", "Logistic Regression"),
              ("random_forest_proba", "Random Forest"),
              ("gradient_boosting_proba", "Gradient Boosting")]
for ax, (col, name) in zip(axes, model_cols):
    preds = (test_pred[col] > 0.5).astype(int)
    cm = confusion_matrix(test_pred["true"], preds)
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["non_pulsar", "pulsar"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["non_pulsar", "pulsar"])
    ax.set_title(name)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_confusion_matrices.png")
plt.close()
 
# ---------- 6. Feature set comparison (8 vs 30) ----------
fs = pd.read_csv(f"{DATA_DIR}/feature_set_comparison.csv")
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(fs))
width = 0.25
for i, metric in enumerate(["roc_auc", "pr_auc", "f1"]):
    ax.bar(x + i * width, fs[metric], width, label=metric.upper())
ax.set_xticks(x + width)
ax.set_xticklabels([f"{r.feature_set}\n({r.n_features} features)" for r in fs.itertuples()])
ax.set_ylim(0.85, 1.0)
ax.set_ylabel("Score")
ax.set_title("Richer feature set (Lyon+Thornton) vs. standard 8 features")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_feature_set_comparison.png")
plt.close()
 
print("All figures saved to", FIG_DIR)