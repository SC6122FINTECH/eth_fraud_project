# Presentation Slides Outline
## Ethereum Phishing Address Detection — SC6122 Group Project

---

### Slide 1 — Title
- **Ethereum Phishing Address Detection using Machine Learning**
- SC6122 — Group Project
- Member names & student IDs
- May 2026

---

### Slide 2 — Contents
1. Problem Statement
2. Dataset Overview
3. Exploratory Data Analysis
4. Data Preprocessing
5. Machine Learning Models
6. Results & Comparison
7. Interpretation & Feature Importance
8. Limitations & Future Work
9. Conclusion
10. Q&A

---

### Slide 3 — Problem Statement
- Ethereum phishing scams drain millions of dollars from users every year
- Exchanges and wallet providers need to flag risky counterparties **before** transfers settle
- **Task:** Binary classification — is this address a phishing/scam address?
- **Y (target):** `FLAG` = 1 (fraud), 0 (legitimate)
- **X (features):** 45 on-chain behavioural aggregates per address
- **Who benefits:** exchanges, wallets, on-chain analytics firms
- Key visual: brief diagram showing wallet → transfer → risk flag

---

### Slide 4 — Why This Problem Matters
- **False Negative (miss a scam):** Victim sends ETH → irreversible loss
- **False Positive (block legitimate):** Friction, customer-support load, churn — recoverable
- Therefore: **recall is the primary metric**, not accuracy
- Accuracy baseline: labelling every address "legitimate" = 82.2% accuracy → detects zero fraud!

---

### Slide 5 — Dataset Overview
- Source: Kaggle "Ethereum Fraud Detection Dataset" (vagifa)
- 9,841 raw rows → 9,288 after deduplication (553 duplicate rows removed)
- 51 raw columns → 45 numeric features (dropped 3 ID cols + 2 high-cardinality text cols)
- **Target:** 7,632 legitimate (82.2%) / 1,656 fraud (17.8%) — imbalanced
- Feature families: transaction timing, counts, Ether values, counterparty diversity, ERC20 activity
- Include: `eda_target_distribution.png`

---

### Slide 6 — Exploratory Data Analysis
- **Missing values:** 23 ERC20 columns × 549 NaNs each — addresses that never used token contracts (domain-meaningful, not random)
- **Skewness:** Ether-value features skewed up to 93.8× — log-transform required
- **Feature distributions by class:** Fraud addresses show shorter inter-transaction gaps, higher send counts
- **Correlation:** Several ERC20 columns are highly correlated (redundant) — trees auto-down-weight them
- Include: `eda_missing_values.png`, `eda_feature_distributions_by_class.png`

---

### Slide 7 — Data Preprocessing
| Step | Detail |
| --- | --- |
| Train/test split | 80/20 **stratified** on FLAG, `random_state=42` |
| Median imputation | Fitted on train only; applied to both |
| Signed log1p | `sign(x)·log1p(|x|)` — handles skew + rare negatives |
| Standard scaling | Mean=0, std=1 on train; test stats close but ≠ train → no leakage |
| Imbalance | SMOTE **inside CV folds** only — never applied to test |

- All steps in `src/preprocess.py` — single source of truth, prevents test-set leakage

---

### Slide 8 — Machine Learning Models
| Member | Model | Key idea |
| --- | --- | --- |
| A | Logistic Regression | Linear interpretable baseline; L1/L2 regularised |
| B | KNN | Non-parametric; sensitive to scale (handled in preprocessing) |
| C | Decision Tree | Rule-based; human-readable splits; prone to overfitting |
| D | Random Forest | Bagging ensemble; 200 de-correlated trees; feature importances |
| E | ANN (MLP) | Non-linear; 2 hidden layers (128→64); Adam + early stopping |

- All models: **StratifiedKFold(5)**, two imbalance variants, refit on F1

---

### Slide 9 — Results — Metrics Comparison

| Model | Accuracy | Recall | F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.884 | 0.873 | 0.729 | 0.941 | 0.828 |
| KNN | 0.949 | 0.834 | 0.853 | 0.948 | 0.878 |
| Decision Tree | 0.937 | 0.828 | 0.824 | 0.894 | 0.710 |
| Random Forest | 0.956 | 0.846 | **0.874** | 0.985 | **0.949** |
| **ANN (MLP)** | 0.951 | **0.891** | 0.866 | **0.977** | 0.945 |

- Include: `comparison_roc_overlay.png`, `comparison_pr_overlay.png`

---

### Slide 10 — Results — Overfitting Check

| Model | Train F1 | Test F1 | Gap |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.742 | 0.729 | **0.013** ✓ |
| ANN | 0.925 | 0.866 | 0.059 |
| KNN | 1.000 | 0.853 | 0.147 |
| Random Forest | 1.000 | 0.874 | 0.126 |
| Decision Tree | 0.999 | 0.824 | **0.175** ✗ |

- Tree models and KNN memorise training data
- ANN generalises best among expressive models (L2 regularisation + early stopping)
- Include: `comparison_confusion_top3.png`

---

### Slide 11 — Model Interpretation & Feature Importance
- **Best model for our objective:** ANN — highest recall (0.891), catches 295/331 fraud test addresses
- **Runner-up:** Random Forest — best F1 and PR-AUC; faster inference; more interpretable
- **Key features driving fraud detection:**
  1. Short average time between sent transactions (rapid burst activity)
  2. High sent transaction count (many victims)
  3. Near-zero ether balance (drained after attack)
  4. High number of unique sent-to addresses (broadcast attacks)
- Include: `ann_permutation_importance.png`, `rf_feature_importances.png`

---

### Slide 12 — Limitations
- Labels from a single blocklist → selection bias; unknown scam addresses exist
- Features are static aggregates — cannot flag a wallet that has not yet acted
- Temporal inconsistency in label collection (addresses added at different times)
- ANN: no built-in `class_weight` → SMOTE required; longer tuning time

---

### Slide 13 — Future Work
- **Graph features:** counterparty network embeddings (GNN) to capture cluster behaviour
- **Temporal features:** rolling-window counts, time-to-first-suspicious-event
- **Threshold calibration:** fix recall ≥ 0.95 and minimise false positives for a given business SLA
- **Larger MLP search:** dropout regularisation, batch normalisation
- **Online learning:** periodic retraining as new fraud patterns emerge

---

### Slide 14 — Conclusion
- Five ML models trained and compared on Ethereum phishing detection
- **ANN (MLP + SMOTE)** recommended for maximum recall (0.891) — fewest missed fraudsters
- **Random Forest** recommended when F1/PR-AUC trade-off or interpretability is key (0.874 F1)
- Logistic Regression is a strong, explainable baseline with near-zero overfitting
- Code, notebooks, and results at `eth_fraud_project/` — fully reproducible with `random_state=42`

---

### Slide 15 — Q&A
- Thank you!
- Questions welcome

---

*Speaker notes:*
- Member A presents slides 3–4 (problem + dataset)
- Member B presents slides 5–7 (EDA + preprocessing)
- Member C presents slides 8, 9 (models + metrics table)
- Member D presents slides 10–11 (overfitting + feature importance)
- Member E presents slides 12–15 (limitations + conclusion)
