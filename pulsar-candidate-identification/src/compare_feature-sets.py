"""
Secondary comparison: does the richer 30-feature set (Lyon et al. 2015's
8 features + Thornton et al. 2013's additional 22 features) improve
pulsar classification over the standard 8-feature set? This mirrors a
real methodological question addressed in the pulsar-classification
literature (does additional DM-curve/profile feature engineering pay
off, or do the original 8 features already capture most of the signal).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
 
COMBINED_PATH = "data/raw/HTRU_2_Combined_30.csv"
OUT_PATH = "data/feature_set_comparison.csv"
 
 
def main():
    df30 = pd.read_csv(COMBINED_PATH, header=None)
    n_features = df30.shape[1] - 1
    print(f"Combined feature set: {n_features} features, {len(df30)} rows")
 
    X30 = df30.iloc[:, :n_features].values
    y = df30.iloc[:, n_features].values
    X8 = df30.iloc[:, :8].values  # first 8 columns are the Lyon 2015 features
 
    results = []
    for name, X in [("lyon_8_features", X8), ("combined_30_features", X30)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        preds = rf.predict(X_test)
 
        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        f1 = f1_score(y_test, preds)
        print(f"{name}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}")
        results.append(dict(feature_set=name, n_features=X.shape[1],
                             roc_auc=roc_auc, pr_auc=pr_auc, f1=f1))
 
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
 
 
if __name__ == "__main__":
    main()