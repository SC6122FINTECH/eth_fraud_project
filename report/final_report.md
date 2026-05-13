# Ethereum Phishing Address Detection using Machine Learning

**Course:** SC6122 — Group Project  
**Date:** May 2026

---

## Abstract

We apply five supervised machine learning models to detect phishing and scam Ethereum addresses using on-chain behavioural features. On a 9,288-address dataset with 17.8% fraud prevalence, an ANN (MLP) trained with SMOTE oversampling achieves the highest test recall of 0.891 with PR-AUC of 0.945, making it the recommended model for a deployment scenario where missing a fraudster is more costly than a false alarm. Random Forest achieves the best F1 (0.874) and PR-AUC (0.949), confirming that ensemble methods generalise well on this tabular detection task.

---

## 1. Introduction

Ethereum phishing attacks cost users hundreds of millions of dollars annually. Scam addresses impersonate legitimate services or exploit social engineering to steal ETH and ERC20 tokens. Exchanges, wallet providers, and on-chain analytics firms need automated tools to flag risky counterparties before transfers settle — a task well-suited to supervised binary classification on historical transaction behaviour.

This project predicts whether an Ethereum address is fraudulent (phishing/scam) given aggregated on-chain features. We train five models — Logistic Regression, KNN, Decision Tree, Random Forest, and ANN — and evaluate them on a held-out test set using recall, F1, and PR-AUC as primary metrics.

---

## 2. Problem Statement

| Item | Value |
| --- | --- |
| **Task** | Binary classification / fraud detection |
| **Target Y** | `FLAG` — 1 = phishing/scam address, 0 = legitimate |
| **Features X** | 45 aggregated on-chain behavioural features per address |
| **Decision supported** | Block or step-up review for an outbound transfer to a given address |
| **Cost of false negative** | User sends ETH to a scammer — irreversible financial loss |
| **Cost of false positive** | Legitimate user is blocked — recoverable friction |
| **Primary metric** | Recall (fraud class) and PR-AUC; accuracy reported but not used for model selection |

---

## 3. Dataset

- **Source:** Kaggle — [Ethereum Fraud Detection Dataset (vagifa)](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
- **Raw size:** 9,841 rows × 51 columns
- **After deduplication:** 9,288 rows × 45 numeric features (553 duplicate rows removed — disproportionately fraud-labelled, likely from overlapping source lists)
- **Dropped columns:** `Index`, `Unnamed: 0`, `Address` (identifier columns); `ERC20 most sent token type`, `ERC20_most_rec_token_type` (300+ unique text values, out of scope to encode)
- **Target distribution:** 7,632 legitimate (82.2%) / 1,656 fraud (17.8%) — moderately imbalanced
- **Feature families:** transaction timing, transaction counts, Ether values (sent/received/balance), counterparty counts, ERC20 token activity (~20 columns, many NaN for addresses that never used tokens)
- **Missing values:** 23 ERC20 columns, each with 549 NaNs (addresses with no token activity) — handled by median imputation on training data

**Known limitations:** Labels come from a single curated blocklist — selection bias possible. Features are point-in-time aggregates (no time series). Behaviour patterns may have shifted post-2022.

---

## 4. Feature Engineering & Preprocessing

All preprocessing decisions comply with §7 and §8 of the requirements (no test-set leakage):

1. **Train/test split:** Stratified 80/20 (`random_state=42`). Train: 7,430 rows (17.83% fraud); Test: 1,858 rows (17.82% fraud) — stratification preserved within 0.02 pp.

2. **Preprocessing pipeline** (fitted on training data only, then applied to test):
   - **Median imputation** — replaces ERC20 NaNs with training-set medians.
   - **Signed log1p transform** — `sign(x) × log1p(|x|)` — damps the extreme right skew of Ether-value features (skewness up to 93.8) while tolerating the occasional slightly negative balance.
   - **Standard scaling** — zero-mean, unit-variance on training data. Test mean ≈ 0.014 (not exactly zero, confirming the scaler was not refit on test).

3. **Imbalance handling:** Applied per-model inside CV folds only. Two variants compared for each model:
   - **Variant A:** native imbalance knob (`class_weight='balanced'` for LR/DT/RF; distance-weighted voting for KNN; plain MLP for ANN — MLP has no class_weight).
   - **Variant B:** SMOTE inside `imblearn.pipeline.Pipeline` so synthetic samples are generated only on the CV training fold and never contaminate validation or test data.
   - The variant with higher CV F1 is selected; the test set is touched only once at the end.

---

## 5. Machine Learning Methods

All models share: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, refit metric = F1, `random_state=42` everywhere applicable.

| Model | Library | Key hyperparameters tuned | Search | Imbalance (winner) |
| --- | --- | --- | --- | --- |
| Logistic Regression | sklearn | `C` ∈ {0.01, 0.1, 1, 10}; `penalty` ∈ {l1, l2} | GridSearchCV | SMOTE |
| KNN | sklearn | `n_neighbors` ∈ {3,5,7,11,21}; `weights` ∈ {uniform, distance}; `metric` ∈ {euclidean, manhattan} | GridSearchCV | distance-weighted |
| Decision Tree | sklearn | `max_depth` ∈ {None,5,10,20}; `min_samples_split` ∈ {2,10,50}; `min_samples_leaf` ∈ {1,5,20}; `criterion` ∈ {gini, entropy} | GridSearchCV | SMOTE |
| Random Forest | sklearn | `n_estimators` ∈ {200,500}; `max_depth` ∈ {None,10,20}; `max_features` ∈ {sqrt, log2}; `min_samples_leaf` ∈ {1,5} | RandomizedSearchCV (20 iter) | SMOTE |
| ANN (MLP) | sklearn | `hidden_layer_sizes` ∈ {(64,),(128,),(64,32),(128,64)}; `activation` ∈ {relu, tanh}; `alpha` ∈ {0.001, 0.01} | GridSearchCV | SMOTE |

**ANN architecture:** 2 hidden layers (128 → 64 units), ReLU activation, L2 regularisation α = 0.001, Adam optimiser, early stopping (patience = 20 epochs on 10% validation split).

---

## 6. Results

### 6.1 Test-set metrics

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.884 | 0.626 | **0.873** | 0.729 | 0.941 | 0.828 |
| KNN | 0.949 | 0.873 | 0.834 | 0.853 | 0.948 | 0.878 |
| Decision Tree | 0.937 | 0.820 | 0.828 | 0.824 | 0.894 | 0.710 |
| Random Forest | 0.956 | **0.903** | 0.846 | **0.874** | 0.985 | **0.949** |
| **ANN (MLP)** | **0.951** | 0.843 | **0.891** | 0.866 | **0.977** | 0.945 |

*Precision, Recall, F1 reported for the positive (fraud) class.*

### 6.2 Overfitting analysis

| Model | Train F1 | Test F1 | Gap |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.742 | 0.729 | **0.013** |
| ANN | 0.925 | 0.866 | 0.059 |
| KNN | 1.000 | 0.853 | 0.147 |
| Random Forest | 1.000 | 0.874 | 0.126 |
| Decision Tree | 0.999 | 0.824 | **0.175** |

Logistic Regression generalises best (near-zero gap), confirming it is well-regularised. KNN and tree models memorise training data; the Decision Tree is worst despite depth pruning. The ANN achieves a moderate gap thanks to L2 regularisation and early stopping.

---

## 7. Interpretation

### 7.1 Model selection

For this problem the primary objective is **maximising recall** (catch as many fraudsters as possible) because the cost of a missed fraud is irreversible. The ANN achieves the highest test recall (0.891) — it misses only 36 of 331 fraud addresses in the test set. Random Forest is the preferred alternative when deployment latency or interpretability is important: it achieves the best F1 (0.874) and PR-AUC (0.949) with a well-understood decision process.

Logistic Regression serves as a transparent baseline. Its recall (0.873) is surprisingly close to the ANN, suggesting that linear separability with proper regularisation and SMOTE captures most of the signal. Decision Tree is the weakest model — its high train F1 and low test PR-AUC (0.710) indicate it has overfit to spurious splits.

### 7.2 Feature importance

Permutation importance on the ANN (shuffling features and measuring recall drop) and Random Forest Gini importances both highlight the same behavioural patterns:

1. **Transaction timing** — `Avg min between sent tnx`, `Time Diff between first and last (Mins)`: fraudulent addresses operate in rapid, concentrated bursts; the short inter-transaction interval is the single strongest signal.
2. **Transaction counts** — `Sent tnx`, `total transactions (including tnx to create contract)`: scam wallets send to many victims in a short window.
3. **Ether flow** — `total Ether sent`, `total ether balance`: phishing wallets drain collected ETH quickly, leaving near-zero balances after the attack.
4. **Counterparty diversity** — `Unique Sent To Addresses`: legitimate users interact with a stable set of counterparties; scammers broadcast to many victims.

ERC20 token features contribute comparatively less — most Ethereum phishing involves direct ETH transfers rather than token contract interactions.

### 7.3 Business implication

Deploying the ANN with a default 0.5 decision threshold would catch 89.1% of fraudsters at the cost of flagging 3.7% of legitimate addresses (55 false positives out of 1,527 legitimate test addresses). The threshold can be lowered to increase recall at the cost of more false alarms, depending on the risk tolerance of the deploying exchange or wallet.

---

## 8. Conclusion

Five supervised learning models were applied to Ethereum phishing address detection. The ANN (MLP with SMOTE) achieves the best recall (0.891) and generalises well thanks to regularisation and early stopping. Random Forest is the runner-up by F1 and PR-AUC. Logistic Regression performs surprisingly well as a linear baseline. Decision Tree overfits significantly.

**Recommendations:** Deploy the ANN for maximum fraud recall; monitor for concept drift as Ethereum behaviour patterns evolve. Future improvements include graph-network features (counterparty embeddings), temporal rolling-window statistics, and threshold calibration to a target recall level.

---

## References

1. Kaggle — Ethereum Fraud Detection Dataset, vagifa. https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
2. Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825–2830, 2011.

---

## Appendix: Member Contributions

| Member | Notebook / Module | Section in report |
| --- | --- | --- |
| A | `01_logistic_regression.ipynb`, EDA section of `00_eda_and_preprocessing.ipynb` | §3 (Dataset), §6 |
| B | `02_knn.ipynb`, `src/preprocess.py` | §4 (Preprocessing), §6 |
| C | `03_decision_tree.ipynb`, `src/evaluation.py` | §5 (Methods), §6 |
| D | `04_random_forest.ipynb`, `06_comparison_and_interpretation.ipynb` | §6, §7 |
| E | `05_ann.ipynb` | §5 (ANN), §7 (feature importance), slides |

All members contributed to Sections 1, 2, 8 and attended all team meetings.
