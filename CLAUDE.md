# CLAUDE.md

Active project directory for **SC6122**. All code, notebooks, artifacts, the
plan, and a snapshot of the requirements doc live here. The parent folder
(`../`) holds the canonical course materials (lecture PDFs, sample report,
sample slides, original requirements markdown).

## What lives here

| Path | Purpose |
| --- | --- |
| `../group_project_requirements.md` | **Authoritative** grading rubric — every coding decision must comply. (A snapshot copy exists at `./group_project_requirements.md` too.) |
| `../Project Overview.pdf`, `../Group Project Guideline.pdf` | Source PDFs the markdown above was distilled from. |
| `../Sample Final Report.pdf`, `../Sample Presentation Slides.pdf` | Reference for the deliverable's expected depth and tone. |
| `project_plan.md` / `project_plan.html` | The agreed implementation plan. Edit the .md, regenerate HTML if it changes. |
| `../1. Introduction…` / `../2. …` / `../3. …` / `../6. Association Analysis.pdf` | Lecture slides — useful when explaining model choices in the report. |

## The project at a glance

**Task.** Binary classification: detect phishing/scam Ethereum addresses.

**Dataset.** Kaggle `vagifa/ethereum-frauddetection-dataset` — 9,841 rows × 51
columns; after deduplication 9,288 rows × 45 numeric features.
Target = `FLAG` (1 = fraud, 0 = legitimate). Imbalanced ~18% positive
post-dedupe.

**Models.** 5 models, one per group member:
LogisticRegression, KNN, DecisionTree, RandomForest, XGBoost.

**Primary decision metric.** Recall (fraud class) and PR-AUC.
Accuracy is reported but does not pick the winner — losing user funds
is the costlier error than blocking a legit transfer.

## Project structure (this directory)

```
data/                # transaction_dataset.csv (gitignored)
notebooks/
  00_eda_and_preprocessing.ipynb   # run first, run once
  01_logistic_regression.ipynb     # Member A
  02_knn.ipynb                     # Member B
  03_decision_tree.ipynb           # Member C
  04_random_forest.ipynb           # Member D
  05_xgboost.ipynb                 # Member E
  06_comparison_and_interpretation.ipynb   # not yet scaffolded
src/
  data_loader.py    # load + clean CSV, drops IDs / text cols / dupes
  preprocess.py     # Member B's module — stratified split + ColumnTransformer
  evaluation.py     # shared metrics + plots + save_results()
  tuning.py         # shared StratifiedKFold(5), SCORING dict, search wrappers
artifacts/
  splits/           # splits.npz, preprocessor.joblib, feature_names.txt
  models/           # one .joblib per model
  results/          # one .json per model — consumed by notebook 06
figures/            # PNGs for report & slides
report/             # final_report.md + slides_outline.md (not yet written)
```

## Non-negotiable data discipline (graded items §7.3, §8 of the rubric)

1. **The preprocessing pipeline is fit on TRAIN only.** Notebook 00 fits the
   `ColumnTransformer` on `X_train`, transforms both splits, and writes the
   arrays to `artifacts/splits/splits.npz`. Model notebooks **load** these
   arrays — they never re-split, re-impute, or re-scale.
2. **SMOTE goes inside the CV training fold.** Always wrap it in
   `imblearn.pipeline.Pipeline`, never apply it before the split.
3. **Hyperparameter tuning uses StratifiedKFold(5) on training data only.**
   `src/tuning.CV` is the single shared CV object. Refit metric is `f1`.
4. **The test set is touched once, at the end of each model notebook**,
   via `evaluation.evaluate(...)`. Never use it for selection.
5. **`random_state=42` everywhere.** Split, CV, SMOTE, estimators that
   take a seed.

## Common commands

All commands assume cwd = `eth_fraud_project/`.

```bash
# install deps (Python 3.13, anaconda env)
pip install -r requirements.txt

# (Re)download the dataset
kaggle datasets download \
  -d vagifa/ethereum-frauddetection-dataset -p data --unzip

# Run a notebook in-place from CLI
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/02_knn.ipynb \
  --ExecutePreprocessor.timeout=600

# Regenerate the HTML plan after editing project_plan.md
python3 /tmp/render_plan.py   # script is updated to point at this directory
```

## Conventions for new code

- **One source of truth per concern.** Splits live in `splits.npz`; metrics
  computation lives in `src/evaluation.compute_metrics`; CV strategy lives in
  `src/tuning.CV`. Don't duplicate these in notebooks.
- **Model notebooks all follow the same 6-section template** (see any of
  01–05). The comparison notebook will glob `artifacts/results/*.json` —
  do not rename the `model_name`, `train`, `test`, or `confusion_matrix_test`
  keys produced by `evaluation.evaluate()`.
- **Imbalance is handled per model in two variants** (`class_weight`/native
  knob vs `SMOTE`). The notebook picks the variant with the higher CV F1 and
  records the choice in `results.imbalance_strategy`.
- **Figures are saved to `figures/` with the model_key as prefix** — e.g.
  `figures/random_forest_roc.png`. The report pulls from there.

## When you make changes

- Edit `project_plan.md` first if scope changes; the plan is the contract
  with the human group members.
- If you change `src/preprocess.py`, re-run `00_eda_and_preprocessing.ipynb`
  so every model notebook picks up the new splits next time it runs.
- Do **not** edit notebook outputs by hand — re-execute the cell instead.
- Do **not** commit `data/transaction_dataset.csv`, `artifacts/`, or
  `figures/` (already covered by `.gitignore`).

## Out of scope / explicitly not doing

- No graph features (counterparty network embeddings) — mentioned as
  future work in the report only.
- No neural network — the 5-model line-up is fixed.
- No deployment / inference server — this is a coursework analysis.

## Pointers for cold starts

- "Why this dataset / problem?" → `project_plan.md` §1–3.
- "Which hyperparameters?" → `project_plan.md` §6 (matches the grids in
  notebooks 01–05 verbatim).
- "How do I write member 06 (comparison)?" → read any of `results.json`
  to see the schema, then glob the directory and build a DataFrame.
- "Which lecture introduced X?" → the four lecture PDFs in `../` (parent
  folder).
