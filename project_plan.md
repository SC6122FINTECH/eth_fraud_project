# Ethereum Phishing / Scam Address Detection — Project Plan

**Course:** SC6122 — Group Project
**Project dir:** `<...>/NTU/fintech/eth_fraud_project` (this directory)
**Language:** Python 3.13 (already installed: pandas, numpy, scikit-learn, xgboost, lightgbm, imblearn)

---

## 0. Status (live)

| Phase | State | Owner |
| --- | --- | --- |
| Dataset downloaded → `data/transaction_dataset.csv` | Done | — |
| `src/data_loader.py`, `preprocess.py`, `evaluation.py`, `tuning.py` | Done | shared |
| `00_eda_and_preprocessing.ipynb` — executed, splits saved | Done | A + B |
| `01_logistic_regression.ipynb` — executed (smoke test) | Done | Member A |
| `02_knn.ipynb` — scaffolded, not yet run | Pending | Member B |
| `03_decision_tree.ipynb` — scaffolded, not yet run | Pending | Member C |
| `04_random_forest.ipynb` — scaffolded, not yet run | Pending | Member D |
| `05_xgboost.ipynb` — scaffolded, not yet run | Pending | Member E |
| `06_comparison_and_interpretation.ipynb` | Not started | Member D |
| `report/final_report.md`, `slides_outline.md` | Not started | all |

**LR baseline (test set):** Accuracy 0.884 · Precision 0.626 · Recall 0.873 · F1 0.729 · ROC-AUC 0.941 · PR-AUC 0.828. Winning variant: SMOTE. Use these numbers as the floor each tree model should reach or beat.

---

## 1. Context

**Why this project.** The course requires a supervised learning project on a real dataset with a real prediction/detection problem (see [../group_project_requirements.md](../group_project_requirements.md)). Ethereum phishing/scam detection is a strong fit because:

- The Kaggle dataset (9,841 labelled addresses, 51 columns) is real, public, and large enough to train multiple models meaningfully.
- It is a **binary classification + detection** problem with a clear business stakeholder (exchanges, wallets, on-chain analytics firms blocking risky counterparties).
- It is **class-imbalanced** (~17.8% fraud after dedupe), which forces the team to demonstrate stratified splitting, imbalance handling, and metrics beyond accuracy — all items the requirements doc explicitly grades on (§7.2, §8.4, §11.4).
- The cost asymmetry (false negative = victim loses funds; false positive = legitimate user blocked) gives the interpretation section (§12.1) a real story to tell.

**Decisions taken at planning time, now implemented:**

- **5 models**, one per group member: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost.
- **Code layout:** one notebook per model + a comparison notebook (clean group-work division).
- **Imbalance:** run **both** `class_weight='balanced'` (or `scale_pos_weight` / distance-weighted KNN) and **SMOTE** (training set only) per model and report which works better.

---

## 2. Problem Definition

| Item | Value |
| --- | --- |
| **Real-world issue** | Phishing/scam Ethereum addresses defraud users of millions of dollars; exchanges and wallets need to flag risky counterparties before transfers settle. |
| **Task type** | Supervised **binary classification** (detection). |
| **`Y` (target)** | `FLAG` ∈ {0 = legitimate, 1 = phishing/scam}. |
| **`X` (features)** | 45 behavioural features per address: transaction counts, average/min/max sent & received values, time gaps, unique counterparties, contract-creation counts, ERC20 token activity, balance. |
| **Decision supported** | Whether to block / warn / require step-up review for an outbound transfer to a given address. |
| **Cost of FN** | A user sends ETH to a scammer — direct financial loss, irreversible on-chain. |
| **Cost of FP** | A legitimate user is blocked — friction, customer-support load, churn. |
| **Primary metric** | **Recall on the fraud class** and **PR AUC** (because of imbalance + asymmetric costs); accuracy reported but not used to pick the winner. |

---

## 3. Dataset

- **Source:** Kaggle — [Ethereum Fraud Detection Dataset (vagifa)](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset).
- **Raw size:** 9,841 rows × 51 columns.
- **After load-time cleanup:** 9,288 rows × 45 numeric features (dropped 3 ID columns, 2 high-cardinality token-name string columns, and 553 duplicate rows — disproportionately fraud-labelled).
- **Target distribution after dedupe:** 7,632 legitimate (82.2%) / 1,656 fraud (17.8%).
- **Train/test split (stratified 80/20, `random_state=42`):**
  - Train: 7,430 rows / 1,325 fraud (17.83%)
  - Test:  1,858 rows / 331 fraud (17.82%)
- **Feature families:**
  - Transaction timing: `Avg min between sent tnx`, `Avg min between received tnx`, `Time Diff between first and last (Mins)`.
  - Transaction counts: `Sent tnx`, `Received Tnx`, `Number of Created Contracts`, `total transactions (including tnx to create contract)`.
  - Counterparties: `Unique Received From Addresses`, `Unique Sent To Addresses`.
  - Ether values: `min/max/avg value received`, `min/max/avg value sent`, `min/max/avg val sent to contract`, `total Ether sent`, `total ether received`, `total ether balance`.
  - ERC20 token activity (~20 cols): `Total ERC20 tnxs`, `ERC20 total Ether received/sent`, `ERC20 uniq sent/rec token name`, etc. Many NaNs (addresses that never touched ERC20 tokens) — median-imputed in §5.4.
- **Dropped at load:** `Index`, `Unnamed: 0` (row id), `Address` (string id, no signal); `ERC20 most sent token type`, `ERC20_most_rec_token_type` (300+ unique values, out of scope to encode).
- **Constant-after-preprocessing columns (7):** all-zero ERC20 contract-related columns (e.g. `ERC20 avg time between sent tnx`, `ERC20 min/max/avg val sent contract`). Kept for schema stability; tree models and L1-regularized LR will down-weight them automatically.
- **Known limitations for the report:**
  - Class labels come from a single curated list — selection bias possible.
  - Features are point-in-time aggregates, not time series — no concept of "when did the scam start".
  - Only on-chain behaviour; no off-chain context (social engineering, app context).
  - The duplicate fraud rows suggest the source list may have been concatenated from multiple feeds; we kept only one copy of each address.

---

## 4. Project File Layout (built)

```
eth_fraud_project/
├── data/
│   └── transaction_dataset.csv               # downloaded
├── notebooks/
│   ├── 00_eda_and_preprocessing.ipynb        # done, executed
│   ├── 01_logistic_regression.ipynb          # Member A — executed
│   ├── 02_knn.ipynb                          # Member B — scaffolded
│   ├── 03_decision_tree.ipynb                # Member C — scaffolded
│   ├── 04_random_forest.ipynb                # Member D — scaffolded
│   ├── 05_xgboost.ipynb                      # Member E — scaffolded
│   └── 06_comparison_and_interpretation.ipynb # TODO — Member D
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # load CSV, drop IDs/text/dupes → X, y
│   ├── preprocess.py                         # stratified split + ColumnTransformer
│   ├── evaluation.py                         # compute_metrics, evaluate, save_results, plots
│   └── tuning.py                             # shared StratifiedKFold(5), grid_search / random_search
├── artifacts/
│   ├── splits/
│   │   ├── splits.npz                        # X_train, X_test, y_train, y_test
│   │   ├── preprocessor.joblib               # fitted ColumnTransformer
│   │   └── feature_names.txt                 # 45 names, post-transform
│   ├── models/
│   │   └── logistic_regression.joblib        # (others appear as each member runs theirs)
│   └── results/
│       └── logistic_regression.json          # (others appear as each member runs theirs)
├── figures/
│   ├── eda_target_distribution.png
│   ├── eda_missing_values.png
│   ├── eda_feature_distributions_by_class.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_outlier_boxplots.png
│   ├── logistic_regression_confusion.png
│   ├── logistic_regression_roc.png
│   └── logistic_regression_pr.png
├── report/                                    # empty — TODO
├── requirements.txt
├── README.md
├── CLAUDE.md (at repo root)
└── .gitignore
```

**Critical shared files:**

1. `src/preprocess.py` — single source of truth for the split + preprocessing pipeline. Every model notebook imports it. Prevents test-set leakage (§7.3, §8 of requirements).
2. `notebooks/00_eda_and_preprocessing.ipynb` — produced `artifacts/splits/splits.npz`. All model notebooks load this — guarantees identical splits.
3. `src/evaluation.py` — every model notebook calls the same `evaluate()` so the comparison table is apples-to-apples.
4. `notebooks/06_comparison_and_interpretation.ipynb` — TODO. Will glob `artifacts/results/*.json` to build the comparison table, ROC/PR overlays, and feature-importance interpretation.

---

## 5. Pipeline (end-to-end)

### 5.1 Load & sanity check (`00_eda...ipynb`) ✓ done
- Read CSV, drop `Index`, `Unnamed: 0`, `Address`, and two ERC20 token-name string columns.
- Drop duplicate rows (553 removed).
- `df.info()`, `df.describe()`, NaN counts, target distribution bar chart.

### 5.2 Exploratory data analysis ✓ done
Saved to `figures/`:
- `eda_target_distribution.png` — class counts + percentages.
- `eda_missing_values.png` — NaN counts per column (ERC20 columns dominate, as expected).
- `eda_feature_distributions_by_class.png` — log1p histograms of 8 high-signal features, legit vs fraud overlay.
- `eda_correlation_heatmap.png` — 45×45 numeric correlation.
- `eda_outlier_boxplots.png` — top-6 most-skewed features.

### 5.3 Train/test split (§7) ✓ done
- Stratified, 80/20, `random_state=42`.
- Verified: train fraud rate 17.83% vs test 17.82% (within 0.02pp).

### 5.4 Preprocessing — fit on TRAIN ONLY (§8) ✓ done
Single `ColumnTransformer` in `src/preprocess.build_preprocessor`:

- **Median impute** missing values (fit on train).
- **Signed log1p** (`numpy.sign(x) * np.log1p(np.abs(x))`) — robust to the occasional small-negative balance value.
- **Standard scaling** (fit on train).
- **No categorical encoding** — all surviving columns are numeric.
- **No imbalance handling here** — it's applied per-model inside the training pipeline so it lives inside CV folds.

Sanity-checked: train max |mean| = 2.3e-16, non-constant std mean = 1.0000, test std mean = 0.97 (close to but not exactly 1, confirming the scaler was *not* refit on test). 7 constant-in-train columns identified and kept.

### 5.5 Model training & tuning (§9, §10) — 1 of 5 done

Each model notebook does the same 5 steps:

1. Load the saved processed splits via `preprocess.load_splits()`.
2. Define the estimator with a hyperparameter grid (table in §6).
3. Run `GridSearchCV` (small grids) or `RandomizedSearchCV` (RF, XGBoost) with `StratifiedKFold(n_splits=5)` on `X_train` only. Scoring: `{f1, roc_auc, pr_auc, recall, precision}`; refit on `f1`.
4. Run **two variants** — once with the native imbalance knob, once with SMOTE wrapped in `imblearn.pipeline.Pipeline`. Pick the better variant by validation F1.
5. Refit the winning estimator on the full training set, predict on test, save `<model>.joblib` + `<model>.json` (metrics + best params + variant chosen).

### 5.6 Comparison & interpretation (`06_comparison_...ipynb`) — TODO
- Load all 5 `results.json` files → combined dataframe.
- Render the two required tables (§11.2): tuned-hyperparam summary table + the metrics table (Accuracy / Precision / Recall / F1 / ROC AUC / PR AUC).
- Overlay all 5 ROC curves on one axis; same for PR curves.
- Permutation importance on the top model; bar chart of top-15 features.
- Confusion matrix for the chosen "best for objective" model.
- Discuss: which model wins on F1 vs PR-AUC vs recall; tie-break by stakeholder objective (high recall preferred — losing user funds is worse than minor friction).

---

## 6. Models & Hyperparameter Grids

All grids below are wired verbatim into the corresponding scaffolded notebook.

| Owner | Model | Library | Hyperparameters tuned | Search | Status |
| --- | --- | --- | --- | --- | --- |
| A | **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | `C` ∈ {0.01, 0.1, 1, 10}; `penalty` ∈ {l1, l2}; `solver='liblinear'` | GridSearchCV | ✓ executed |
| B | **K-Nearest Neighbors** | `sklearn.neighbors.KNeighborsClassifier` | `n_neighbors` ∈ {3, 5, 7, 11, 21}; `weights` ∈ {uniform, distance}; `metric` ∈ {euclidean, manhattan} | GridSearchCV | scaffolded |
| C | **Decision Tree** | `sklearn.tree.DecisionTreeClassifier` | `max_depth` ∈ {None, 5, 10, 20}; `min_samples_split` ∈ {2, 10, 50}; `min_samples_leaf` ∈ {1, 5, 20}; `criterion` ∈ {gini, entropy} | GridSearchCV | scaffolded |
| D | **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | `n_estimators` ∈ {200, 500}; `max_depth` ∈ {None, 10, 20}; `max_features` ∈ {sqrt, log2}; `min_samples_leaf` ∈ {1, 5} | RandomizedSearchCV(n_iter=20) | scaffolded |
| E | **XGBoost** | `xgboost.XGBClassifier` | `n_estimators` ∈ {200, 500, 1000}; `learning_rate` ∈ {0.05, 0.1, 0.2}; `max_depth` ∈ {3, 6, 10}; `subsample` ∈ {0.7, 1.0}; `colsample_bytree` ∈ {0.7, 1.0} | RandomizedSearchCV(n_iter=30) | scaffolded |

CV strategy for all: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — exported as `src.tuning.CV`.

**Imbalance handling per model (Variant A vs Variant B):**

| Model | Variant A (native knob) | Variant B (SMOTE) |
| --- | --- | --- |
| LR | `class_weight='balanced'` | `imblearn.SMOTE` then LR |
| KNN | distance-weighted neighbour voting | `imblearn.SMOTE` then KNN |
| DT | `class_weight='balanced'` | `imblearn.SMOTE` then DT |
| RF | `class_weight='balanced'` | `imblearn.SMOTE` then RF |
| XGB | `scale_pos_weight = neg/pos` | `imblearn.SMOTE` then XGB |

---

## 7. Evaluation Metrics (§11)

Reported per model on **the held-out test set only** (also reported on train, to surface overfitting per §11.3):

- **Accuracy** (required by professor, but not the decision metric).
- **Balanced accuracy**.
- **Precision** (fraud class).
- **Recall** (fraud class) — primary, because missing fraud is the costlier error.
- **F1** (fraud class).
- **ROC AUC**.
- **PR AUC / Average Precision** — most informative under class imbalance.
- **Confusion matrix** (counts).

All seven metrics + the confusion matrix are written to `artifacts/results/<model>.json` by `evaluation.evaluate()`.

---

## 8. Interpretation (§12)

The comparison notebook must answer, in prose:

1. Which model wins on F1 vs on PR-AUC vs on recall? (Expectation: XGBoost / Random Forest on F1 & PR-AUC; Logistic Regression as interpretable baseline; KNN / Decision Tree weakest.)
2. Why is recall the metric we ultimately prioritise? (User funds are irreversible.)
3. Top features driving the decision — via permutation importance and XGBoost's `feature_importances_`. Expected high-signal: short time gaps, high transaction frequency, low ratio of unique counterparties, sparse ERC20 activity.
4. Overfitting check — gap between train and test F1 per model. (LR train F1 = 0.742, test F1 = 0.729 → ~0.01 gap, healthy.)
5. Limitations: label noise, point-in-time features (no time series), behaviour may have shifted post-2022.
6. Future improvements: graph features (counterparty network), temporal features, transformer-based wallet embeddings.

---

## 9. Group Division of Labour

| Member | Model | Also responsible for |
| --- | --- | --- |
| A | Logistic Regression | EDA section in `00_eda...` (target/feature distributions) |
| B | KNN | Preprocessing pipeline (`src/preprocess.py`) and imbalance comparison |
| C | Decision Tree | Evaluation helpers (`src/evaluation.py`) |
| D | Random Forest | Comparison tables in `06_comparison...` |
| E | XGBoost | Feature-importance interpretation + slides |

All members co-author Sections 1, 2, 11, 12 of the report; the contribution table goes in the appendix (§13.1).

---

## 10. Reusable Code (already in `src/`)

These are standard scikit-learn / imblearn pieces wired into the shared modules:

- `sklearn.model_selection.train_test_split(..., stratify=y, random_state=42)` — `preprocess.stratified_split`.
- `sklearn.compose.ColumnTransformer` + `sklearn.pipeline.Pipeline` — `preprocess.build_preprocessor`.
- `imblearn.pipeline.Pipeline` + `SMOTE` — used inline in each model notebook's Variant B (keeps SMOTE inside CV folds).
- `sklearn.model_selection.GridSearchCV` / `RandomizedSearchCV` with `StratifiedKFold` — `tuning.grid_search`, `tuning.random_search`.
- `sklearn.metrics.{classification_report, confusion_matrix, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay}` — wrapped in `evaluation.{compute_metrics, evaluate, plot_*}`.
- `sklearn.inspection.permutation_importance` — to be used in `06_comparison...` for the feature-importance section.

---

## 11. Deliverables (matches §13 + §14 of requirements)

| Deliverable | Path | Status |
| --- | --- | --- |
| Code (notebooks + `src/` modules) | `eth_fraud_project/` | Built; 4 model notebooks await execution |
| Comparison notebook | `notebooks/06_comparison_and_interpretation.ipynb` | TODO (Member D) |
| Final report (≤7 pages) | `report/final_report.md` | TODO (all) |
| Presentation slides | `report/slides_outline.md` → PPTX/PDF | TODO (E owns deck; every member speaks per §14.2) |
| Member contribution statement | appendix of final report | TODO |

---

## 12. Verification (how we'll know it works end-to-end)

| Check | Status |
| --- | --- |
| 1. **Smoke test:** `00_eda...ipynb` runs top-to-bottom; splits saved; |train_pos_rate − test_pos_rate| < 0.005. | ✓ passed (0.0002 gap) |
| 2. **No-leakage check:** `ColumnTransformer` fit only on `X_train`; train max |mean| ≈ 0, non-constant std ≈ 1, test stats close-but-not-equal to train. | ✓ passed |
| 3. **Per-model test:** each model produces a `results.json` with all 7 metrics + best params + variant chosen. Sanity floor: test F1 ≥ 0.7 for tree models, ≥ 0.5 for LR/KNN. | LR ✓ (F1 = 0.729); 4 pending |
| 4. **Comparison notebook:** runs end-to-end, produces the comparison table + ROC/PR overlay figures with 5 models. | Pending |
| 5. **Reproducibility:** every notebook sets `random_state=42`; re-running should produce identical test metrics to 4 decimal places. | Verified for LR |
