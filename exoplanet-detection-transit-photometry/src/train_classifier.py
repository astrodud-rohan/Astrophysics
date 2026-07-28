"""
Train tabular classifiers (Random Forest, Gradient Boosting) on
BLS-derived vetting features to separate real transit-like planet
signals from false positives and no-signal light curves.
 
This mirrors NASA's actual Kepler "Robovetter" / "Autovetter" approach
(Coughlin et al. 2016, McCauliff et al. 2015, Armstrong et al. 2018):

The CNN-on-folded-lightcurve (AstroNet-style) model is trained
separately as a comparison / secondary model.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
)
import joblib
 
FEATURES_PATH = "../data/features.csv"
MODEL_DIR = "../data"
 
FEATURE_COLS = [
    "bls_period", "bls_duration", "bls_power", "bls_depth",
    "bls_odd_even_mismatch", "n_transits_found", "depth_over_sigma",
    "duration_over_period",
]
 
 
def load_and_engineer():
    df = pd.read_csv(FEATURES_PATH)
    df["depth_over_sigma"] = df["bls_depth"].abs() / df["sigma"].replace(0, np.nan)
    df["duration_over_period"] = df["bls_duration"] / df["bls_period"]
    df["is_planet"] = (df["label"] == "planet").astype(int)
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    return df
 
 
def main():
    df = load_and_engineer()
    X = df[FEATURE_COLS].values
    y = df["is_planet"].values
 
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=42, stratify=y
    )
 
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }
 
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, target_names=["not_planet", "planet"])
        cm = confusion_matrix(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"ROC-AUC: {auc:.4f}")
        print(report)
        print("Confusion matrix:\n", cm)
        results[name] = dict(model=model, auc=auc, proba=proba, preds=preds)
        joblib.dump(model, f"{MODEL_DIR}/{name}.joblib")
 
    # BLS-power-only baseline: what you'd get from classical detection alone,
    # thresholding on bls_power with no ML -- the key "why ML adds value" comparison
    baseline_score = df_test["bls_power"].values
    baseline_auc = roc_auc_score(y_test, baseline_score)
    print(f"\n=== Baseline: BLS power alone (no ML) ===")
    print(f"ROC-AUC: {baseline_auc:.4f}")
 
    # feature importances from RF
    rf = results["random_forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nRandom Forest feature importances:")
    print(importances)
 
    # Save test predictions + baseline for plotting later
    out = df_test[["id", "label", "is_planet"]].copy()
    out["rf_proba"] = results["random_forest"]["proba"]
    out["gb_proba"] = results["gradient_boosting"]["proba"]
    out["bls_power_baseline"] = baseline_score
    out.to_csv(f"{MODEL_DIR}/test_predictions.csv", index=False)
 
    summary = pd.DataFrame({
        "model": ["random_forest", "gradient_boosting", "bls_power_baseline"],
        "roc_auc": [results["random_forest"]["auc"], results["gradient_boosting"]["auc"], baseline_auc],
    })
    summary.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)
    print("\n", summary)
 
 
if __name__ == "__main__":
    main()
