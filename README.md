# Ethereum Phishing / Scam Address Detection

SC6122 group project. Binary classification on the Kaggle
[Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset).

- **Original:** 9,841 addresses × 51 columns
- **After dedupe & dropping IDs/text:** 9,288 rows × 45 numeric features
- **Target:** `FLAG` (1 = fraud, 0 = legitimate), ~17.8% positive after dedupe

Full plan: [`project_plan.md`](project_plan.md) (or `project_plan.html`).
Conventions and data-discipline rules: [`CLAUDE.md`](CLAUDE.md).

## Setup

```bash
pip install -r requirements.txt
```

## Get the data

```bash
# Option A — Kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d vagifa/ethereum-frauddetection-dataset -p data --unzip
```

Or download `transaction_dataset.csv` manually from the Kaggle page and place
it at `data/transaction_dataset.csv`.

## Run order

| # | Notebook | Owner | Status | Time |
| --- | --- | --- | --- | --- |
| 00 | `notebooks/00_eda_and_preprocessing.ipynb` | A + B | Executed | ~30 s |
| 01 | `notebooks/01_logistic_regression.ipynb` | Member A | Executed (smoke test) | ~30 s |
| 02 | `notebooks/02_knn.ipynb` | Member B | Scaffolded | ~1–2 min |
| 03 | `notebooks/03_decision_tree.ipynb` | Member C | Scaffolded | ~30–60 s |
| 04 | `notebooks/04_random_forest.ipynb` | Member D | Scaffolded | ~2–3 min |
| 05 | `notebooks/05_xgboost.ipynb` | Member E | Scaffolded | ~3–5 min |
| 06 | `notebooks/06_comparison_and_interpretation.ipynb` | all | Not yet written | — |

**Order rule:** run `00` first (produces `artifacts/splits/splits.npz`, shared by all
model notebooks). After that, `01`–`05` are independent and can be developed in
parallel. Run `06` only once `01`–`05` have each written their `results.json`.

Run a notebook from CLI:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/02_knn.ipynb --ExecutePreprocessor.timeout=600
```

## Project structure

```
eth_fraud_project/
├── data/                # transaction_dataset.csv (gitignored)
├── notebooks/           # 00 (shared) + 01..05 (one per member) + 06 (comparison)
├── src/
│   ├── data_loader.py   # load + clean raw CSV (drops IDs, text cols, dupes)
│   ├── preprocess.py    # Member B's module — split + ColumnTransformer
│   ├── evaluation.py    # shared metrics, plots, save_results()
│   └── tuning.py        # shared StratifiedKFold(5), SCORING dict, search wrappers
├── artifacts/
│   ├── splits/          # splits.npz + preprocessor.joblib + feature_names.txt
│   ├── models/          # <model>.joblib (one per model)
│   └── results/         # <model>.json (one per model) — consumed by notebook 06
├── figures/             # EDA + per-model PNGs for the report/slides
├── report/              # final_report.md + slides_outline.md (not yet written)
└── requirements.txt
```

## Data discipline (graded — §7.3, §8 of the rubric)

1. The preprocessing pipeline is **fit on TRAIN only**. Notebook 00 fits the
   `ColumnTransformer` on `X_train`, transforms both splits, and persists them.
   Model notebooks **load** the saved arrays — they never re-split, re-impute,
   or re-scale.
2. **SMOTE goes inside the CV training fold.** Always wrap it in
   `imblearn.pipeline.Pipeline` so it's applied only inside each CV split's
   training data, never on the test set.
3. **Hyperparameter tuning uses `StratifiedKFold(5)` on the training set only.**
   The shared CV object lives in `src/tuning.CV`. Refit metric is `f1`.
4. **The test set is touched exactly once per model**, in
   `evaluation.evaluate(...)` at the end of the notebook.
5. **`random_state=42` everywhere** — split, CV, SMOTE, and every estimator
   that accepts a seed.

## Template for a model notebook

Each of `01`–`05` follows the same 6-section skeleton. To start a new one,
copy the closest sibling and swap the estimator + grid:

```python
# 1. Setup & load shared splits
import sys; from pathlib import Path
PROJECT = Path.cwd().parent
sys.path.insert(0, str(PROJECT))

from src.preprocess import load_splits
from src.tuning import grid_search, random_search
from src.evaluation import evaluate, save_results, print_metric_table

data = load_splits(PROJECT / "artifacts" / "splits")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# 2. Variant A — class_weight / native imbalance knob
search_cw = grid_search(estimator_cw, grid_cw).fit(X_train, y_train)

# 3. Variant B — SMOTE inside imblearn Pipeline
search_smote = grid_search(pipe_smote, grid_smote).fit(X_train, y_train)

# 4. Pick the variant with the better CV F1, refit, evaluate on test
# 5. save_results(...) + joblib.dump(...)
# 6. Confusion / ROC / PR plots
```

The exact grids per model are documented in `project_plan.md` §6 and are
already wired into the scaffolded notebooks.

## Current results

Only `01_logistic_regression` has been executed end-to-end (as a smoke test of
the scaffold). For reference / baseline comparison:

| Metric | Train | Test |
| --- | ---: | ---: |
| Accuracy | 0.887 | 0.884 |
| Balanced accuracy | 0.897 | 0.880 |
| Precision (fraud) | 0.625 | 0.626 |
| **Recall (fraud)** | **0.912** | **0.873** |
| F1 (fraud) | 0.742 | 0.729 |
| ROC AUC | 0.960 | 0.941 |
| PR AUC | 0.830 | 0.828 |

Winning variant: **SMOTE**. Best params: `C=10, penalty='l1'`.
Test confusion: 1354 TN / 173 FP / 42 FN / 289 TP — the recall-favouring
profile we want for fraud detection.

The four remaining models (KNN, Decision Tree, Random Forest, XGBoost)
should match or beat these numbers, especially on F1 and PR-AUC.

## Reproducibility

Every random source is seeded with `42`:
- the train/test split (`preprocess.stratified_split`)
- the CV folds (`tuning.CV`)
- the SMOTE oversampler (inside each model notebook)
- estimators that accept a seed

Re-running `00` then any single model notebook should reproduce results to
4 decimal places.
