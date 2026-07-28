import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
DATA_DIR = "../data"
FIG_DIR = "../figures"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
 
df = pd.read_csv(f"{DATA_DIR}/catalog_with_clusters.csv")
 
CLUSTER_COLORS = {"Pleiades": "#3182ce", "Hyades": "#dd6b20", "NGC752": "#38a169", "field": "#a0aec0"}
 
# ---------- 1. Overall observational HR diagram, true labels ----------
fig, ax = plt.subplots(figsize=(7, 8))
for name, color in CLUSTER_COLORS.items():
    sub = df[df["true_cluster"] == name]
    ax.scatter(sub["bp_rp"], sub["abs_G"], s=6 if name == "field" else 14,
               alpha=0.35 if name == "field" else 0.85, color=color, label=name)
ax.invert_yaxis()
ax.set_xlabel("BP - RP (color)")
ax.set_ylabel("Absolute G magnitude")
ax.set_title("Observational HR Diagram (true population labels)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_hr_diagram_true_labels.png")
plt.close()
 
# ---------- 2. HR diagram colored by GMM-recovered cluster ----------
fig, ax = plt.subplots(figsize=(7, 8))
for name, color in CLUSTER_COLORS.items():
    sub = df[df["gmm_cluster_name"] == name]
    ax.scatter(sub["bp_rp"], sub["abs_G"], s=6 if name == "field" else 14,
               alpha=0.35 if name == "field" else 0.85, color=color, label=name)
ax.invert_yaxis()
ax.set_xlabel("BP - RP (color)")
ax.set_ylabel("Absolute G magnitude")
ax.set_title("HR Diagram colored by GMM-recovered cluster\n(clustering used ONLY proper motion + parallax, not color/magnitude)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_hr_diagram_gmm_recovered.png")
plt.close()
 
# ---------- 3. Proper motion diagram: why NGC752 is hard ----------
fig, ax = plt.subplots(figsize=(7, 7))
for name, color in CLUSTER_COLORS.items():
    sub = df[df["true_cluster"] == name]
    ax.scatter(sub["pmra"], sub["pmdec"], s=6 if name == "field" else 18,
               alpha=0.3 if name == "field" else 0.85, color=color, label=name)
ax.set_xlabel("Proper motion RA (mas/yr)")
ax.set_ylabel("Proper motion Dec (mas/yr)")
ax.set_title("Proper motion space: Pleiades/Hyades stand out from field,\nNGC752 overlaps heavily (only ~1.5σ separation)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_proper_motion_diagram.png")
plt.close()
 
# ---------- 4. Parallax distribution: cluster spikes vs. field ----------
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 25, 100)
for name, color in CLUSTER_COLORS.items():
    sub = df[df["true_cluster"] == name]
    ax.hist(sub["parallax"], bins=bins, alpha=0.6 if name != "field" else 0.3,
            color=color, label=name, density=True)
ax.set_xlabel("Parallax (mas)")
ax.set_ylabel("Density")
ax.set_title("Parallax distribution: cluster spikes above the smooth field distribution")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_parallax_distribution.png")
plt.close()
 
# ---------- 5. Confusion matrix heatmap ----------
from sklearn.metrics import confusion_matrix
labels_order = ["Pleiades", "Hyades", "NGC752", "field"]
cm = confusion_matrix(df["true_cluster"], df["gmm_cluster_name"], labels=labels_order)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center",
            color="white" if v > cm.max() / 2 else "black", fontsize=11)
ax.set_xticks(range(4)); ax.set_xticklabels(labels_order, rotation=45)
ax.set_yticks(range(4)); ax.set_yticklabels(labels_order)
ax.set_xlabel("GMM-assigned cluster"); ax.set_ylabel("True population")
ax.set_title("Cluster membership recovery: confusion matrix")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_confusion_matrix.png")
plt.close()
 
print("All figures saved to", FIG_DIR)
