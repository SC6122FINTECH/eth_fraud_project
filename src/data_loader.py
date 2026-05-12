"""Load the raw Ethereum fraud detection CSV into (X, y).

Used by `notebooks/00_eda_and_preprocessing.ipynb` only. Every other
notebook reads the already-processed splits via `preprocess.load_splits()`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "FLAG"

# Identifier / row-number columns — never features.
ID_COLS = ["Unnamed: 0", "Index", "Address"]

# Token-name string columns (~300+ unique values). Dropped for this baseline:
# encoding them is out of scope and they leak very little signal compared
# to the numeric activity counts.
TEXT_COLS = ["ERC20 most sent token type", "ERC20_most_rec_token_type"]


def load_raw(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series, int]:
    """Read the CSV, drop IDs/text columns and duplicates, split into X and y.

    Returns
    -------
    X : pd.DataFrame   numeric features only
    y : pd.Series      binary target (0 = legitimate, 1 = fraud)
    n_dropped_dupes : int   number of duplicate rows removed
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Download from "
            "https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset "
            "and place transaction_dataset.csv in eth_fraud_project/data/."
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    drop_cols = [c for c in ID_COLS + TEXT_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = before - len(df)

    if TARGET not in df.columns:
        raise KeyError(f"Expected target column {TARGET!r} not found in CSV.")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # Coerce any stray object columns to numeric (some versions of the CSV
    # leave a column as object due to a stray space).
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, n_dropped
