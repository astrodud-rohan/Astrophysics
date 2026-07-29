"""
Train and evaluate pulsar candidate classifiers on the real HTRU2
dataset. Given the genuine ~9.9:1 class imbalance, evaluation uses
precision/recall/F1/ROC-AUC/PR-AUC rather than raw accuracy (accuracy
alone is meaningless here -- a trivial "always non-pulsar" classifier
already scores ~91%).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve, f1_score,
)
 
DATA_PATH = "data/htru2_clean.csv"
OUT_DIR = "data"
 
FEATURE_COLS = [
    "mean_ip", "std_ip", "kurtosis_ip", "skew_ip",
    "mean_dmsnr", "std_dmsnr", "kurtosis_dmsnr", "skew_dmsnr",
]
 
 
def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df["class"].values
 
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} ({y_train.sum()} pulsars), "
          f"Test: {len(X_test)} ({y_test.sum()} pulsars)")
 
    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }
 
    results = {}
    test_pred_rows = {"true": y_test}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
 
        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        f1 = f1_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=["non_pulsar", "pulsar"])
        cm = confusion_matrix(y_test, preds)
 
        print(f"\n=== {name} ===")
        print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}  |  F1 (pulsar): {f1:.4f}")
        print(report)
        print("Confusion matrix:\n", cm)
 
        results[name] = dict(model=model, roc_auc=roc_auc, pr_auc=pr_auc, f1=f1, proba=proba, preds=preds)
        test_pred_rows[f"{name}_proba"] = proba
        joblib.dump(model, f"{OUT_DIR}/{name}.joblib")
 
    # Feature importance from Random Forest
    rf = results["random_forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nRandom Forest feature importances:")
    print(importances)
    importances.to_csv(f"{OUT_DIR}/feature_importances.csv", header=["importance"])
 
    pd.DataFrame(test_pred_rows).to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)
 
    summary = pd.DataFrame({
        "model": list(results.keys()),
        "roc_auc": [r["roc_auc"] for r in results.values()],
        "pr_auc": [r["pr_auc"] for r in results.values()],
        "f1_pulsar": [r["f1"] for r in results.values()],
    })
    summary.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    print("\n", summary)
 
 
if __name__ == "__main__":
    main()