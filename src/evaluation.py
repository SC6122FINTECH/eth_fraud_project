"""Shared metric + plotting helpers so all five model notebooks report
the exact same numbers in the same order.

Use `evaluate(model, X_train, y_train, X_test, y_test, name)` at the end
of each model notebook and `save_results(...)` to drop a JSON into
`artifacts/results/`. The comparison notebook reads those JSONs back.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _scores(model, X):
    """Return positive-class scores: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


def evaluate(model, X_train, y_train, X_test, y_test, model_name: str,
             best_params: dict | None = None,
             imbalance_strategy: str | None = None) -> dict:
    """Run the standard evaluation pack on a fitted model."""
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    score_tr = _scores(model, X_train)
    score_te = _scores(model, X_test)

    return {
        "model_name": model_name,
        "best_params": best_params or {},
        "imbalance_strategy": imbalance_strategy,
        "train": compute_metrics(y_train, y_pred_tr, score_tr),
        "test": compute_metrics(y_test, y_pred_te, score_te),
        "confusion_matrix_test": confusion_matrix(y_test, y_pred_te).tolist(),
    }


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def print_metric_table(results: dict) -> None:
    print(f"\n=== {results['model_name']} ({results.get('imbalance_strategy') or 'no rebalancing'}) ===")
    print(f"Best params: {results['best_params']}")
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    print(f"{'metric':<20}{'train':>10}{'test':>10}")
    for k in keys:
        tr = results["train"].get(k)
        te = results["test"].get(k)
        if tr is None or te is None:
            continue
        print(f"{k:<20}{tr:>10.4f}{te:>10.4f}")
    print(f"Confusion matrix (test):\n{np.array(results['confusion_matrix_test'])}")


def plot_confusion(y_true, y_pred, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap="Blues", colorbar=False,
        display_labels=["legit", "fraud"],
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc(y_true, y_score, label="model"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_score, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curve")
    fig.tight_layout()
    return fig


def plot_pr(y_true, y_score, label="model"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, name=label, ax=ax)
    ax.set_title("Precision–Recall curve")
    fig.tight_layout()
    return fig
