"""
Exploratory data analysis and validation of the real HTRU2 dataset
against documented values.
"""
import pandas as pd
from reference_data import (
    RAW_LYON8_PATH, LYON8_COLUMNS,
    N_TOTAL_DOCUMENTED, N_PULSAR_DOCUMENTED, N_NONPULSAR_DOCUMENTED,
)
 
OUT_PATH = "data/htru2_clean.csv"
 
 
def load_and_validate():
    df = pd.read_csv(RAW_LYON8_PATH, header=None, names=LYON8_COLUMNS)
 
    assert len(df) == N_TOTAL_DOCUMENTED, f"Row count mismatch: {len(df)} vs {N_TOTAL_DOCUMENTED}"
    n_pulsar = (df["class"] == 1).sum()
    n_nonpulsar = (df["class"] == 0).sum()
    assert n_pulsar == N_PULSAR_DOCUMENTED, f"Pulsar count mismatch: {n_pulsar} vs {N_PULSAR_DOCUMENTED}"
    assert n_nonpulsar == N_NONPULSAR_DOCUMENTED, f"Non-pulsar count mismatch: {n_nonpulsar} vs {N_NONPULSAR_DOCUMENTED}"
    assert df.isna().sum().sum() == 0, "Unexpected NaNs in dataset"
 
    print(f"Validated: {len(df)} total candidates")
    print(f"  Pulsars: {n_pulsar} ({100*n_pulsar/len(df):.2f}%)")
    print(f"  Non-pulsars: {n_nonpulsar} ({100*n_nonpulsar/len(df):.2f}%)")
    print(f"  Class imbalance ratio: {n_nonpulsar/n_pulsar:.2f}:1")
 
    return df
 
 
def summarize(df):
    print("\nFeature summary by class:")
    print(df.groupby("class").mean().T)
 
 
if __name__ == "__main__":
    df = load_and_validate()
    summarize(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved validated dataset -> {OUT_PATH}")