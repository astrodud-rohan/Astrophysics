"""
1D CNN classifier on folded, binned light curves -- the "global view"
architecture from Shallue & Vanderburg 2018 (AstroNet), simplified for
this dataset size. Trained as a comparison/ensemble candidate against
the tabular BLS-feature classifiers in train_classifier.py.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
 
DATA_DIR = "../data"
 
tf.random.set_seed(42)
np.random.seed(42)
 
 
def main():
    npz = np.load(f"{DATA_DIR}/folded_curves.npz", allow_pickle=True)
    folded = npz["folded"]
    ids = npz["ids"]
    meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    meta = meta.set_index("id").loc[ids].reset_index()
    y = (meta["label"] == "planet").astype(int).values
 
    # normalize each folded curve to zero-mean/unit-std (standard practice)
    X = (folded - folded.mean(axis=1, keepdims=True)) / (folded.std(axis=1, keepdims=True) + 1e-8)
    X = X[..., np.newaxis]  # (N, n_bins, 1) for Conv1D
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
 
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1], 1)),
        keras.layers.Conv1D(16, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
 
    early_stop = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=60, batch_size=32,
        callbacks=[early_stop], verbose=2,
    )
 
    proba = model.predict(X_test).ravel()
    preds = (proba > 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    print(f"\nCNN Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["not_planet", "planet"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
 
    model.save(f"{DATA_DIR}/cnn_model.keras")
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f"{DATA_DIR}/cnn_history.csv", index=False)
 
    np.savez(f"{DATA_DIR}/cnn_test_results.npz", y_test=y_test, proba=proba)
    print(f"\nSaved CNN model + history + test results to {DATA_DIR}")
 
 
if __name__ == "__main__":
    main()
 