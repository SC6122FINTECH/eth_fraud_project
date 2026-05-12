"""Thin wrappers around GridSearchCV / RandomizedSearchCV so every model
notebook uses the same CV strategy and the same scoring config.

The CV is StratifiedKFold(5) — required for imbalanced binary classification.
The refit metric is F1; ROC-AUC and PR-AUC are recorded for the report.
"""

from __future__ import annotations

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

RANDOM_STATE = 42

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
}

REFIT_METRIC = "f1"


def grid_search(estimator, param_grid, n_jobs: int = -1, verbose: int = 1):
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=CV,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=verbose,
    )


def random_search(estimator, param_distributions, n_iter: int = 30,
                  n_jobs: int = -1, verbose: int = 1,
                  random_state: int = RANDOM_STATE):
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=CV,
        n_jobs=n_jobs,
        random_state=random_state,
        return_train_score=True,
        verbose=verbose,
    )
