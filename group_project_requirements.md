# SC6122 Group Project Requirements

This file consolidates the requirements from the professor's verbal/project notes, `Project Overview.pdf`, `Group Project Guideline.pdf`, `Sample Final Report.pdf`, and `Sample Presentation Slides.pdf`.

## 1. Project Purpose

The group project is a supervised learning / data analysis project.

The goal is to apply machine learning methods to a real dataset and a real-life prediction or detection problem. The project should not only train models and report numbers; it should explain the data, justify the methods, evaluate results properly, and interpret what the results mean.

The programming language or software is not restricted. You may use Python, R, MATLAB, Excel add-ons, or other tools, as long as the analysis is valid and reproducible enough to explain in the report and presentation.

## 2. Core Workflow Expected

The project should follow this basic structure:

1. Identify or download a real dataset.
2. Define a real prediction or detection problem.
3. Define `X`, the input features.
4. Define `Y`, the target/output variable.
5. Split the dataset into training and test data.
6. Perform exploratory data analysis.
7. Preprocess and engineer features.
8. Train several machine learning models.
9. Calibrate/tune the hyperparameters for each method.
10. Evaluate and compare models using proper test metrics.
11. Interpret the results and explain what they mean.
12. Prepare a final report and presentation slides.

## 3. Dataset Requirements

### 3.1 Dataset Must Be Real

The dataset must be a real dataset. It should represent a real-world problem rather than a fully artificial example.

If the data is sensitive or confidential, the data may be:

- re-labeled,
- re-coded,
- anonymized,
- transformed,
- or otherwise modified so that confidential identities are not disclosed.

The dataset does not need to be shared if it is sensitive, as long as the report accurately describes it. However, if the dataset is shareable, it should be submitted if applicable.

### 3.2 Dataset Does Not Need to Be Blockchain Data

Blockchain or on-chain datasets are acceptable, but not required.

The professor encouraged groups to explore datasets beyond the few sample datasets, because many groups may otherwise choose the same dataset.

Possible dataset sources include:

| Source | Suitable Use |
| --- | --- |
| UCI Machine Learning Repository | Standard machine learning datasets |
| Kaggle | Data science and applied ML datasets |
| Professor-provided sample datasets | Acceptable, but may be used by many groups |
| Public finance / fintech / business datasets | Suitable if they support a clear prediction task |
| Blockchain / on-chain datasets | Acceptable if the target and features are well defined |

### 3.3 Dataset Description Should Be Clear

The report and slides should describe:

- where the dataset came from,
- what the observations/rows represent,
- the number of rows and columns,
- the target variable,
- the input features,
- whether features are numerical, categorical, text-based, time-series, or mixed,
- whether the dataset has missing values,
- whether duplicate rows exist,
- whether the target classes are balanced or imbalanced,
- any limitations in the dataset,
- and whether the original dataset was modified, filtered, or transformed.

## 4. Problem Definition Requirements

The group must define a concrete prediction or detection problem.

A good project problem statement should answer:

- What real-life issue is being studied?
- Why does the problem matter?
- What decision could the model support?
- What is the prediction target `Y`?
- What are the input variables `X`?
- Is the task classification, regression, detection, ranking, or another supervised learning task?
- Who would use the model output?
- What would be the cost of wrong predictions?

Examples of acceptable supervised learning tasks:

- Predict whether a customer defaults on a loan or credit card.
- Detect fraudulent transactions.
- Predict whether a blockchain address is risky.
- Predict asset returns or volatility, if framed carefully as a supervised problem.
- Classify customers into risk groups.
- Predict whether an event will happen based on historical features.

The problem should not be only descriptive statistics. It needs a prediction or detection objective.

## 7. Train/Test Split Requirements

### 7.1 Must Split Data

The dataset must be split into training data and test data.

Notation from the project overview:

- `D` = full dataset
- `D_tr` = training dataset
- `D_te` = test dataset

The model development process should use `D_tr`. The final evaluation should use `D_te`.

### 7.2 Random vs Stratified Sampling

Choose the split method based on the target distribution:

| Target/Class Distribution | Recommended Split |
| --- | --- |
| Balanced class distribution | Random sampling |
| Highly imbalanced class distribution | Stratified sampling |

For classification problems, the report should show or discuss the target class distribution before and after splitting.

### 7.3 Do Not Touch the Test Set During Training

This is a key principle from the professor's overview:

Do not touch `D_te` during model training.

The test set should not be used for:

- fitting models,
- selecting features,
- fitting scalers,
- imputing missing values,
- tuning hyperparameters,
- selecting PCA dimension,
- choosing thresholds,
- choosing the "best" model repeatedly,
- or any other training-time decision.

The test set should only be used once the model training and tuning procedure is finalized.

### 7.4 Validation/Cross-Validation Should Happen Inside Training Data

Hyperparameter tuning should be done using only the training data.

Acceptable approaches include:

- train/validation split inside the training set,
- k-fold cross-validation on the training set,
- stratified k-fold cross-validation for imbalanced classification,
- grid search,
- random search,
- Bayesian optimization,
- or another clearly explained tuning method.

## 8. Feature Engineering and Preprocessing Requirements

The report and slides should include a feature engineering and data preprocessing section.

Possible preprocessing steps include:

- removing duplicate rows,
- dropping irrelevant ID/text columns when justified,
- correcting invalid category values,
- handling missing values,
- encoding categorical variables,
- scaling or standardizing numerical variables,
- handling outliers,
- transforming skewed variables,
- creating new features,
- selecting features,
- reducing dimensionality,
- handling class imbalance.

Important rule: preprocessing steps that learn from data should be fitted using training data only, then applied to the test data.

Examples:

- Fit imputation values on training data only.
- Fit encoders on training data only.
- Fit scalers on training data only.
- Fit PCA on training data only.
- Apply the learned transformation to the test set.

### 8.1 Missing Values

If there are missing values, the group should explain:

- which variables contain missing values,
- how many or what percentage are missing,
- whether missingness seems random or systematic,
- and how missing values were handled.

Common approaches:

- drop rows or columns if justified,
- mean/median imputation for numerical variables,
- mode imputation for categorical variables,
- model-based imputation,
- category "Unknown" for missing categories,
- or domain-specific imputation.

### 8.2 Categorical Encoding

If categorical variables are used, explain how they were converted into numerical features.

Possible methods:

- binary encoding,
- one-hot encoding,
- ordinal encoding,
- target encoding, if done carefully within cross-validation,
- embeddings, if using neural networks.

### 8.3 Feature Scaling

Scaling is important for models such as:

- KNN,
- logistic regression,
- lasso/ridge regression,
- support vector machines,
- neural networks,
- PCA-based methods.

Tree-based models such as decision trees, random forests, and XGBoost usually do not require scaling, but the group should still explain the preprocessing pipeline.

### 8.4 Class Imbalance

If the target is imbalanced, the group should address it explicitly.

Possible approaches:

- stratified train/test split,
- class weights,
- oversampling,
- undersampling,
- SMOTE,
- threshold adjustment,
- metrics beyond accuracy, such as precision, recall, F1, ROC AUC, PR AUC, or balanced accuracy.

If using oversampling or SMOTE, apply it only to the training data, not to the test data.

## 9. Machine Learning Model Requirements

The group should apply multiple machine learning models and compare their performance.

Each group member can take responsibility for one or two models.

Possible models include:

- logistic regression,
- lasso / ridge / elastic net,
- K-nearest neighbors,
- decision tree,
- random forest,
- support vector machine,
- naive Bayes,
- gradient boosting,
- XGBoost / LightGBM / CatBoost,
- PCA + regression/classification model,
- neural network,
- other supervised learning models if justified.

The exact choice of models depends on the dataset and problem type.

## 10. Hyperparameter Calibration Requirements

Each method should have its hyperparameters calibrated.

This was explicitly emphasized by the professor.

Examples:

| Method | Hyperparameters to Tune |
| --- | --- |
| Lasso | regularization strength `lambda` |
| Ridge | regularization strength `lambda` |
| Elastic Net | `lambda`, l1/l2 mixing ratio |
| PCA + Lasso | PCA dimension, lasso `lambda` |
| KNN | number of neighbors `k`, distance metric, weighting |
| Decision Tree | max depth, min samples split, min samples leaf, criterion |
| Random Forest | number of trees, max depth, min samples split, min samples leaf, max features |
| SVM | kernel, `C`, gamma, degree for polynomial kernel |
| XGBoost | learning rate, max depth, number of estimators, subsampling, regularization |
| Neural Network | architecture, learning rate, batch size, epochs, optimizer, dropout |
| Naive Bayes | smoothing parameter, distributional assumptions |

The report should say:

- which hyperparameters were tuned,
- what search method was used,
- what validation or cross-validation setup was used,
- which hyperparameters were selected,
- and why the selected model was reasonable.

## 11. Evaluation Requirements

### 11.1 Test Performance Must Be Reported

The project must evaluate model performance on the test data.

The professor specifically emphasized accuracy / test error.

For classification tasks, useful metrics include:

- accuracy,
- test error,
- confusion matrix,
- precision,
- recall,
- F1 score,
- ROC AUC,
- PR AUC, especially for imbalanced data,
- class-wise performance.

For regression tasks, useful metrics include:

- mean squared error,
- root mean squared error,
- mean absolute error,
- R-squared,
- test error,
- residual analysis.

The exact metrics should match the project objective.

### 11.2 Compare Methods Clearly

The deliverables should include a model comparison table.

A good comparison table may include:

| Model | Tuned Hyperparameters | Validation Metric | Test Metric | Strength | Weakness |
| --- | --- | --- | --- | --- | --- |
| Model 1 | ... | ... | ... | ... | ... |
| Model 2 | ... | ... | ... | ... | ... |

For classification, a table like this is also useful:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Model 1 | ... | ... | ... | ... | ... | ... |
| Model 2 | ... | ... | ... | ... | ... | ... |

### 11.3 Train Performance vs Test Performance

The group may report both training and test performance, but the final comparison should rely on test performance.

If training performance is much better than test performance, discuss overfitting.

If both training and test performance are poor, discuss underfitting, weak features, noisy labels, or model limitations.

### 11.4 Accuracy Alone May Not Be Enough

Although accuracy/test error is required, it may be misleading for imbalanced classification.

If the dataset is imbalanced, include metrics such as:

- recall for the minority class,
- precision for the positive class,
- F1 score,
- ROC AUC,
- PR AUC,
- balanced accuracy,
- confusion matrix.

The group should explain which metric matters most for the real-world problem.

## 12. Interpretation Requirements

Interpretation is important.

The professor emphasized that the project should not only show numbers. The group must explain what the model results mean.

The interpretation section should answer:

- Which model performs best?
- Best according to which metric?
- Why is that metric appropriate?
- What are the most important features?
- Are the results economically, financially, or practically meaningful?
- What mistakes does the model make?
- What are the costs of false positives and false negatives?
- Does the model appear to overfit?
- Does the model generalize well?
- What limitations should be considered?
- How could the model be improved?

### 12.1 Model Choice Should Depend on Objective

The sample presentation shows that the "best" model can depend on stakeholder priorities.

For example, in a default prediction problem:

- a risk management objective may prioritize high recall for defaulters,
- a balanced objective may prioritize strong performance across both classes,
- a revenue/customer relationship objective may avoid falsely labeling good customers as defaulters.

Therefore, do not simply say "highest accuracy wins" unless accuracy truly matches the decision objective.

## 13. Required Report Content

The final report should contain:

- introduction to the problem,
- description of the dataset,
- project objectives,
- methods used,
- feature engineering and preprocessing,
- findings,
- results comparison,
- conclusion/discussion,
- references, if any,
- member contribution statement.

The project overview also indicates that report/slides should include the following key components:

1. Data & Problem
2. Analysis Methods
3. Feature Engineering & Data Preprocessing
4. ML Models
5. Results & Interpretation

### 13.1 Suggested Report Structure

A strong report can use this structure:

1. Title page
   - course name,
   - project title,
   - group number/name,
   - member names,
   - contribution statement or reference to contribution appendix.

2. Abstract / Executive Summary
   - one short summary of the problem, dataset, methods, best model, and key finding.

3. Introduction
   - real-world background,
   - motivation,
   - why the prediction problem matters.

4. Problem Statement and Objective
   - define the supervised learning task,
   - define `X` and `Y`,
   - state whether it is classification/regression/detection,
   - state how model success will be judged.

5. Dataset Description
   - source,
   - number of rows and columns,
   - target variable,
   - feature descriptions,
   - data types,
   - class distribution or target distribution,
   - any known limitations.

6. Exploratory Data Analysis
   - summary statistics,
   - missing values,
   - duplicates,
   - outliers,
   - target distribution,
   - important visualizations.

7. Feature Engineering and Data Preprocessing
   - train/test split method,
   - random or stratified sampling justification,
   - missing value handling,
   - categorical encoding,
   - scaling,
   - imbalance handling,
   - feature creation or selection.

8. Machine Learning Methods
   - brief explanation of each model,
   - why each model was selected,
   - hyperparameters tuned,
   - validation/cross-validation method.

9. Results and Model Comparison
   - test metrics,
   - comparison table,
   - confusion matrices or residual plots if relevant,
   - ROC/PR curves if relevant,
   - discussion of overfitting/underfitting.

10. Interpretation
   - best model under chosen objective,
   - feature importance or coefficient interpretation,
   - practical meaning of the results,
   - trade-offs between models.

11. Conclusion and Discussion
   - main findings,
   - limitations,
   - future improvements,
   - practical recommendations.

12. References
   - dataset source,
   - packages or papers if needed.

13. Appendix, if allowed within page limit or separate if permitted
   - extra figures,
   - variable dictionary,
   - hyperparameter grids,
   - member contribution table.

### 13.2 Report Length Limit

The final report, including references if any, should not be more than 7 pages.

Because of the 7-page limit, the main report should be concise. Detailed tables or many figures should be used only if they directly support the analysis.

## 14. Presentation Requirements

### 14.1 Suggested Slide Structure

A strong presentation can follow this structure:

1. Title slide
   - project title,
   - group number,
   - member names.

2. Contents / Agenda
   - introduction,
   - dataset,
   - preprocessing,
   - ML models,
   - results,
   - interpretation/conclusion.

3. Problem Statement
   - real-world problem,
   - why it matters.

4. Objectives
   - what the group wants to predict/detect,
   - who benefits from the prediction.

5. Dataset Overview
   - source,
   - size,
   - features,
   - target variable.

6. Exploratory Data Analysis
   - target distribution,
   - key numerical/categorical patterns,
   - missing data or imbalance.

7. Data Processing and Encoding
   - missing value handling,
   - feature encoding,
   - scaling,
   - train/test split,
   - imbalance handling.

8. Model Overview
   - list of models,
   - team member responsible for each model.

9. Model Results
   - main test metrics,
   - confusion matrices/ROC curves if useful,
   - model comparison table.

10. Model Interpretation and Selection
   - which model is best for the chosen objective,
   - important features,
   - trade-offs.

11. Limitations
   - dataset limitations,
   - modeling limitations,
   - evaluation limitations.

12. Future Work
   - feature improvements,
   - more data,
   - better hyperparameter search,
   - external data sources.

13. Q&A

### 14.2 Presentation Participation

Every member should speak during the presentation.

The professor may use the Q&A session to examine individual contributions.

Each member should be prepared to answer questions about:

- the model(s) they handled,
- the data preprocessing,
- the train/test split,
- hyperparameter tuning,
- evaluation metrics,
- interpretation of results,
- limitations.

## 15. Submission Requirements

The group must submit:

1. Final report.
2. Dataset, if applicable/shareable.

The report should not exceed 7 pages including references.

## 16. Discussion Group Requirement

The professor will assign a discussion group for each presentation.

The discussion group's role is to listen to another group's presentation and ask questions afterward.

The purpose is not to attack classmates. The purpose is to learn how to:

- ask useful questions,
- think critically about ML projects,
- identify strengths and weaknesses,
- understand modeling choices,
- discuss limitations and improvements.

Groups should also be prepared to answer questions from their assigned discussion group.

## 17. Minimum Technical Checklist

Before submitting, the project should satisfy the following checklist:

- [ ] The dataset is real.
- [ ] The problem is a supervised prediction or detection problem.
- [ ] `X` input features are clearly defined.
- [ ] `Y` target/output variable is clearly defined.
- [ ] Dataset source is clearly stated.
- [ ] Dataset size and feature types are described.
- [ ] Target distribution is shown or discussed.
- [ ] Train/test split is performed.
- [ ] Random or stratified sampling choice is justified.
- [ ] Test data is not used during training or tuning.
- [ ] Missing values are handled and explained.
- [ ] Categorical variables are encoded if needed.
- [ ] Numerical variables are scaled if needed.
- [ ] Class imbalance is addressed if relevant.
- [ ] Several ML models are trained.
- [ ] Each model has hyperparameters calibrated.
- [ ] Calibration uses only training data or cross-validation inside training data.
- [ ] Test accuracy or test error is reported.
- [ ] Other suitable metrics are reported where needed.
- [ ] Models are compared clearly.
- [ ] The best model is selected based on a justified objective.
- [ ] Results are interpreted, not only reported.
- [ ] Limitations are discussed.
- [ ] Future improvements are discussed.
- [ ] Final report includes member contribution statement.
- [ ] Every member has a role in the presentation.
- [ ] Report is within 7 pages including references.
- [ ] Report and dataset are submitted one day before presentation.

## 18. Common Mistakes to Avoid

- Using a fake or purely toy dataset.
- Not clearly defining the prediction target.
- Reporting only EDA without a supervised learning task.
- Splitting data after using the full dataset for preprocessing.
- Fitting scalers, PCA, imputation, SMOTE, or feature selection on the full dataset before the split.
- Using the test set repeatedly for model selection.
- Not tuning hyperparameters.
- Comparing models only by training accuracy.
- Reporting only accuracy on an imbalanced dataset.
- Showing many numbers without explaining what they mean.
- Claiming one model is best without explaining the decision objective.
- Ignoring false positive vs false negative trade-offs.
- Omitting the contribution statement.
- Letting only one or two members present.
- Exceeding the 7-page report limit.

## 20. What a Strong Project Should Demonstrate

A strong project should show:

- a meaningful real-life problem,
- a suitable real dataset,
- clear definition of inputs and target,
- correct train/test discipline,
- thoughtful preprocessing,
- multiple calibrated models,
- fair comparison using test data,
- interpretation connected to the real problem,
- awareness of limitations,
- clear division of group work,
- and a concise, well-organized report/presentation.

## 21. Example Project Ideas

The examples below show how to turn a dataset into a supervised learning project. They are not the only possible topics, but they illustrate the expected level of clarity.

### Example 1: Credit Card Default Prediction

| Item | Example |
| --- | --- |
| Dataset | Customer credit card / loan repayment dataset |
| Problem | Predict whether a customer will default |
| Task Type | Binary classification |
| `Y` Target | `default = 1` if the customer defaults, `0` otherwise |
| `X` Features | income, age, credit score, credit limit, previous defaults, debt payments, employment length |
| Split Method | Stratified train/test split if defaults are rare |
| Preprocessing | missing value imputation, categorical encoding, scaling for logistic regression/SVM/KNN, imbalance handling |
| Models | logistic regression, decision tree, random forest, XGBoost, SVM, neural network |
| Metrics | accuracy, recall for default class, precision, F1, ROC AUC, confusion matrix |
| Interpretation | Identify which features increase default risk and discuss the cost of missing actual defaulters |

Possible conclusion:

Random forest may provide the best balance between detecting defaulters and avoiding false alarms, while a model with higher recall may be preferred if the bank wants to minimize missed default cases.

### Example 2: Fraudulent Transaction Detection

| Item | Example |
| --- | --- |
| Dataset | Bank/card/e-commerce transaction dataset |
| Problem | Detect whether a transaction is fraudulent |
| Task Type | Binary classification / anomaly-related detection |
| `Y` Target | `fraud = 1` for fraudulent transactions, `0` for normal transactions |
| `X` Features | transaction amount, merchant category, transaction time, country, device type, customer history |
| Split Method | Stratified split because fraud cases are usually highly imbalanced |
| Preprocessing | encode categorical variables, scale transaction amount, create time-based features, handle imbalance with class weights or SMOTE |
| Models | logistic regression, random forest, XGBoost, SVM, naive Bayes |
| Metrics | recall, precision, F1, PR AUC, ROC AUC, confusion matrix |
| Interpretation | Explain false positive vs false negative trade-off: blocking a normal transaction hurts customers, but missing fraud causes financial loss |

Possible conclusion:

The best model may not be the one with the highest accuracy, because a model can achieve high accuracy by predicting almost all transactions as non-fraud. Recall, precision, F1, and PR AUC are more meaningful.

### Example 3: Loan Approval or Loan Default Risk

| Item | Example |
| --- | --- |
| Dataset | Loan application / lending dataset |
| Problem | Predict whether an applicant is likely to repay or default |
| Task Type | Binary classification |
| `Y` Target | `loan_status`, `default`, or `fully_paid` depending on dataset |
| `X` Features | income, loan amount, interest rate, employment length, debt-to-income ratio, credit history |
| Split Method | Random split if balanced; stratified split if default class is imbalanced |
| Preprocessing | remove leakage variables, impute missing values, encode employment/home ownership categories, scale numeric variables |
| Models | logistic regression, lasso logistic regression, decision tree, random forest, XGBoost |
| Metrics | accuracy, recall, precision, F1, ROC AUC |
| Interpretation | Discuss whether the model supports risk control, fair lending, or profitability objectives |

Important caution:

Avoid using features that reveal the outcome after the loan has already been issued. For example, if predicting default at application time, do not use future repayment behavior as an input feature.

### Example 4: Blockchain Address Risk Classification

| Item | Example |
| --- | --- |
| Dataset | Blockchain wallet/address transaction dataset |
| Problem | Classify whether an address is risky, suspicious, or normal |
| Task Type | Binary or multi-class classification |
| `Y` Target | risk label, scam label, phishing label, illicit/licit label |
| `X` Features | number of transactions, total received, total sent, average transaction value, account age, counterparties, transaction frequency |
| Split Method | Stratified split if risky addresses are rare |
| Preprocessing | aggregate transaction-level data to address-level features, handle skewed numerical variables, scale features if using SVM/logistic regression |
| Models | logistic regression, random forest, XGBoost, graph-based features plus classifier |
| Metrics | accuracy, precision, recall, F1, ROC AUC, confusion matrix |
| Interpretation | Identify transaction behavior patterns associated with risky addresses |

Possible conclusion:

Addresses with unusually high transaction frequency, short account age, or concentrated interactions with suspicious counterparties may be more likely to be classified as risky.

### Example 5: Stock or Crypto Price Direction Prediction

| Item | Example |
| --- | --- |
| Dataset | Historical price, volume, and technical indicator dataset |
| Problem | Predict whether tomorrow's price will go up or down |
| Task Type | Binary classification |
| `Y` Target | `1` if next-day return is positive, `0` otherwise |
| `X` Features | lagged returns, moving averages, volatility, trading volume, RSI, MACD, market index returns |
| Split Method | Time-based split, not random split, because financial time series has chronological order |
| Preprocessing | create lagged features, avoid future information leakage, scale features if needed |
| Models | logistic regression, random forest, XGBoost, SVM, LSTM if justified |
| Metrics | accuracy, precision, recall, F1, AUC, simple trading/backtesting metric if appropriate |
| Interpretation | Discuss whether predictive performance is economically meaningful after considering transaction costs and market noise |

Important caution:

For time-series prediction, do not randomly shuffle observations if doing so would let future information leak into the training process. Use an earlier period for training and a later period for testing.

### Example 6: Customer Churn Prediction

| Item | Example |
| --- | --- |
| Dataset | Telecom, subscription, banking, or platform customer dataset |
| Problem | Predict whether a customer will leave the service |
| Task Type | Binary classification |
| `Y` Target | `churn = 1` if the customer leaves, `0` otherwise |
| `X` Features | tenure, monthly charges, usage frequency, contract type, complaints, payment method, customer service contacts |
| Split Method | Stratified split if churners are a minority |
| Preprocessing | encode categorical variables, scale numeric variables, create tenure/usage features |
| Models | logistic regression, decision tree, random forest, XGBoost, neural network |
| Metrics | accuracy, recall, precision, F1, ROC AUC |
| Interpretation | Explain which customer characteristics are associated with churn and how the company could target retention actions |

Possible conclusion:

A model with high recall may be preferred if the company wants to identify as many likely churners as possible for retention campaigns.

## 22. Example Report Mini-Outline

This is a short example using a credit card default prediction project.

### 22.1 Data & Problem

Financial institutions lose money when customers fail to repay credit card bills. The goal is to predict whether a customer will default based on customer profile, credit history, and repayment behavior.

- `Y`: whether the customer defaults on the credit card bill.
- `X`: age, income, credit score, debt payments, credit limit, previous default history, employment duration, and ownership indicators.
- Task: binary classification.
- Business objective: identify high-risk customers before approving or adjusting credit limits.

### 22.2 Analysis Methods

The group uses several supervised learning methods:

- logistic regression as an interpretable baseline,
- decision tree for rule-based classification,
- random forest for nonlinear relationships and feature importance,
- XGBoost for boosted tree performance,
- SVM for margin-based classification.

Each model is tuned using cross-validation on the training set only.

### 22.3 Feature Engineering & Preprocessing

The dataset is split into 70% training and 30% test data using stratified sampling because default cases are less frequent than non-default cases.

Preprocessing includes:

- dropping customer ID and name because they do not provide predictive meaning,
- imputing missing numerical values with training-set medians,
- imputing missing categorical values with training-set modes,
- one-hot encoding occupation type,
- scaling numerical variables for logistic regression, SVM, and KNN,
- applying SMOTE only to the training data if class imbalance is serious.

### 22.4 ML Models

Example hyperparameter tuning:

| Model | Hyperparameters |
| --- | --- |
| Logistic Regression | penalty type, regularization strength `C` |
| Decision Tree | max depth, min samples split, min samples leaf |
| Random Forest | number of trees, max depth, max features |
| XGBoost | learning rate, max depth, number of estimators |
| SVM | kernel, `C`, gamma |

### 22.5 Results & Interpretation

Example result table:

| Model | Accuracy | Recall for Default | Precision for Default | F1 for Default | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.91 | 0.74 | 0.62 | 0.67 | 0.88 |
| Decision Tree | 0.89 | 0.70 | 0.58 | 0.63 | 0.82 |
| Random Forest | 0.94 | 0.81 | 0.72 | 0.76 | 0.93 |
| XGBoost | 0.95 | 0.84 | 0.75 | 0.79 | 0.95 |
| SVM | 0.92 | 0.77 | 0.66 | 0.71 | 0.90 |

Example interpretation:

XGBoost has the strongest overall test performance, with the highest ROC AUC and F1 score for the default class. However, if the bank's main goal is to avoid missing defaulters, recall for the default class should be prioritized. If the bank wants to avoid rejecting good customers, precision should receive more weight. The final model choice should therefore depend on the business objective, not accuracy alone.

## 23. Example Q&A Preparation

Possible questions from the professor or discussion group:

1. Why did you choose this dataset?
2. What exactly are `X` and `Y`?
3. Is your problem classification or regression?
4. Why did you use random sampling or stratified sampling?
5. Did you use the test set during training or tuning?
6. How did you handle missing values?
7. How did you handle class imbalance?
8. Which hyperparameters did you tune?
9. Why did you choose these models?
10. Which metric is most important for your problem?
11. Why is accuracy not enough for your dataset?
12. Which model is best, and why?
13. What are the most important features?
14. What does a false positive mean in your problem?
15. What does a false negative mean in your problem?
16. What are the limitations of your data?
17. How would you improve the project if you had more time?
18. What did each member contribute?

