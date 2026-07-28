"""
Train and compare two galaxy morphology classifiers:
  1. Random Forest on classical CAS + Gini/M20 features (the real
     pre-deep-learning morphology approach)
  2. A CNN on raw pixel images (mirroring Dieleman et al. 2015's actual
     Galaxy Zoo CNN approach)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
 
DATA_DIR = "data"
 
tf.random.set_seed(42)
np.random.seed(42)
 
 
def train_rf():
    df = pd.read_csv(f"{DATA_DIR}/cas_features.csv")
    feature_cols = ["concentration", "asymmetry", "smoothness", "gini", "m20"]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
 
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
    )
 
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                 class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("=== Random Forest on CAS/Gini-M20 features ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\n", cm)
 
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importances:\n", importances)
 
    joblib.dump(rf, f"{DATA_DIR}/rf_cas_model.joblib")
    joblib.dump(le, f"{DATA_DIR}/label_encoder.joblib")
 
    out = pd.DataFrame({
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(preds),
    })
    out.to_csv(f"{DATA_DIR}/rf_test_predictions.csv", index=False)
 
    return acc, cm, le.classes_
 
 
def train_cnn():
    d = np.load(f"{DATA_DIR}/galaxy_images.npz", allow_pickle=True)
    images, labels = d["images"], d["labels"]
 
    le = joblib.load(f"{DATA_DIR}/label_encoder.joblib")
    y = le.transform(labels)
    X = images[..., np.newaxis].astype(np.float32)
    # per-image normalization
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-8)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
 
    model = keras.Sequential([
        keras.layers.Input(shape=X.shape[1:]),
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(le.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
 
    early_stop = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=50, batch_size=32,
        callbacks=[early_stop], verbose=2,
    )
 
    proba = model.predict(X_test)
    preds = np.argmax(proba, axis=1)
    acc = accuracy_score(y_test, preds)
    print("\n=== CNN on raw pixel images ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\n", cm)
 
    model.save(f"{DATA_DIR}/cnn_model.keras")
    pd.DataFrame(history.history).to_csv(f"{DATA_DIR}/cnn_history.csv", index=False)
 
    out = pd.DataFrame({
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(preds),
    })
    out.to_csv(f"{DATA_DIR}/cnn_test_predictions.csv", index=False)
 
    return acc, cm, le.classes_
 
 
if __name__ == "__main__":
    rf_acc, rf_cm, classes = train_rf()
    cnn_acc, cnn_cm, _ = train_cnn()
 
    summary = pd.DataFrame({
        "model": ["random_forest_cas", "cnn_raw_pixels"],
        "accuracy": [rf_acc, cnn_acc],
    })
    summary.to_csv(f"{DATA_DIR}/model_comparison.csv", index=False)
    print("\n", summary)
