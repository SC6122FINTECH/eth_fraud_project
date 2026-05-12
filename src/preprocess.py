"""Train/test split + preprocessing pipeline — Member B owns this module.

Single source of truth for all five model notebooks. Each model notebook
calls `load_splits()` and gets the *exact same* preprocessed train/test
arrays, guaranteeing apples-to-apples comparison and preventing test-set
leakage (the `ColumnTransformer` is fit on training data only).

Imbalance handling is NOT applied here. Each model notebook wires it into
its own training pipeline (class_weight on the estimator, or SMOTE via
imblearn.pipeline.Pipeline so it runs only on training folds inside CV).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2


def signed_log1p(X):
    """log1p that tolerates the occasional negative value.

    Most features in this dataset are non-negative counts and Ether totals,
    but `total ether balance` can drift slightly below zero. Signed log1p
    keeps the sign and damps magnitude the same way log1p does for positives.
    """
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def stratified_split(X: pd.DataFrame, y: pd.Series,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE):
    """80/20 stratified split. Class proportions are preserved in both parts."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Median-impute -> signed log1p -> standard-scale, all numeric.

    Why a single uniform pipeline for every feature:
      * Tree models are scale-invariant, so scaling is harmless for them.
      * LR / KNN / SVM need scaling — including it once avoids per-model
        re-implementation.
      * log1p damps the heavy right skew of transaction-count and Ether-value
        features so distance- and gradient-based learners behave better.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_save(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series,
                       output_dir: str | Path) -> dict:
    """Fit the preprocessor on TRAIN, transform both, persist everything.

    Writes to `output_dir/`:
      splits.npz           X_train, X_test, y_train, y_test (numpy)
      preprocessor.joblib  fitted ColumnTransformer (for inference later)
      feature_names.txt    column names of the transformed matrix

    Returns a dict with the in-memory transformed arrays for inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = X_train.columns.tolist()
    preprocessor = build_preprocessor(numeric_cols)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    np.savez_compressed(
        output_dir / "splits.npz",
        X_train=X_train_proc.astype(np.float32),
        X_test=X_test_proc.astype(np.float32),
        y_train=y_train.to_numpy().astype(np.int8),
        y_test=y_test.to_numpy().astype(np.int8),
    )
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    (output_dir / "feature_names.txt").write_text("\n".join(feature_names))

    return {
        "X_train": X_train_proc,
        "X_test": X_test_proc,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }


def load_splits(input_dir: str | Path) -> dict:
    """Loaded by every model notebook. Returns numpy arrays + feature names."""
    input_dir = Path(input_dir)
    data = np.load(input_dir / "splits.npz")
    feature_names = (input_dir / "feature_names.txt").read_text().splitlines()
    return {
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_train": data["y_train"].astype(int),
        "y_test": data["y_test"].astype(int),
        "feature_names": feature_names,
    }
