"""
HR diagram construction + unsupervised cluster membership recovery.
 
Standard real-world Gaia analysis workflow:
    1. Apply parallax quality cuts (positive parallax, parallax S/N > 5)
       -- this is a standard cut in real Gaia cluster papers, not
       something invented for this project
    2. Compute absolute magnitude from apparent magnitude + parallax
    3. Build the observational HR diagram (BP-RP vs absolute G)
    4. Cluster on proper-motion + parallax space (NOT on the HRD itself
       -- this mirrors the real technique: kinematic clustering finds
       physically associated stars, which THEN reveals a clean
       isochrone-like sequence in the HRD as a validation, rather than
       clustering directly on color-magnitude which would be circular)
    5. Compare recovered clusters against ground truth (available here
       because this is simulated data; in real analysis this step would
       be validated against independently known cluster catalogs)
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, confusion_matrix
 
DATA_PATH = "../data/gaia_like_catalog.csv"
OUT_PATH = "../data/catalog_with_clusters.csv"
 
 
def apply_quality_cuts(df):
    df = df.copy()
    df["parallax_over_error"] = df["parallax"] / df["parallax_error"]
    mask = (df["parallax"] > 0) & (df["parallax_over_error"] > 5)
    return df[mask].reset_index(drop=True)
 
 
def compute_abs_mag(df):
    df = df.copy()
    df["abs_G"] = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"] / 1000) + 5
    return df
 
 
def run_gmm_clustering(df, n_components=4, seed=42):
    """Cluster on proper motion + parallax (kinematic space), NOT on
    color-magnitude, to avoid circularity."""
    features = df[["pmra", "pmdec", "parallax"]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
 
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                           random_state=seed, n_init=5)
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)
    return labels, probs, gmm
 
 
def match_clusters_to_truth(df, labels):
    """Map each GMM component to its majority-vote true cluster label
    (for evaluation purposes only -- in a real unsupervised analysis
    you wouldn't have ground truth available)."""
    df = df.copy()
    df["gmm_label"] = labels
    mapping = {}
    for lab in np.unique(labels):
        subset = df[df["gmm_label"] == lab]
        majority = subset["true_cluster"].mode()
        mapping[lab] = majority.iloc[0] if len(majority) else "unknown"
    df["gmm_cluster_name"] = df["gmm_label"].map(mapping)
    return df, mapping
 
 
def main():
    df = pd.read_csv(DATA_PATH)
    n_before = len(df)
    df = apply_quality_cuts(df)
    n_after = len(df)
    print(f"Quality cuts: {n_before} -> {n_after} stars retained "
          f"({100*n_after/n_before:.1f}%)")
 
    df = compute_abs_mag(df)
 
    n_true_clusters = df["true_cluster"].nunique()  # includes 'field'
    labels, probs, gmm = run_gmm_clustering(df, n_components=n_true_clusters)
    df, mapping = match_clusters_to_truth(df, labels)
 
    ari = adjusted_rand_score(df["true_cluster"], df["gmm_label"])
    print(f"\nAdjusted Rand Index (GMM labels vs. true cluster membership): {ari:.4f}")
    print("\nGMM component -> majority true-cluster mapping:")
    for k, v in mapping.items():
        print(f"  component {k} -> {v}")
 
    cm = confusion_matrix(df["true_cluster"], df["gmm_cluster_name"],
                           labels=sorted(df["true_cluster"].unique()))
    cm_df = pd.DataFrame(cm, index=sorted(df["true_cluster"].unique()),
                          columns=sorted(df["true_cluster"].unique()))
    print("\nConfusion matrix (rows=true, cols=GMM-assigned):")
    print(cm_df)
 
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
 
    # Per-cluster purity/completeness for the actual physical clusters
    # (excluding field, which is intentionally a diffuse catch-all)
    print("\nPer-cluster recovery (excluding field):")
    for true_name in ["Pleiades", "Hyades", "NGC752"]:
        true_members = df[df["true_cluster"] == true_name]
        recovered_as = true_members["gmm_cluster_name"].value_counts()
        completeness = (recovered_as.get(true_name, 0)) / len(true_members)
        assigned_group = df[df["gmm_cluster_name"] == true_name]
        purity = (assigned_group["true_cluster"] == true_name).mean() if len(assigned_group) else np.nan
        print(f"  {true_name}: completeness={completeness:.2f}, purity={purity:.2f}, n_true={len(true_members)}")
 
 
if __name__ == "__main__":
    main()
