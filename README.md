# Customer Transaction Prediction

A binary classification project that predicts whether a bank customer will make a specific future transaction, using an anonymized banking dataset.

## Problem Statement

Banks want to identify which customers are likely to make a transaction in the future, regardless of the amount, so they can personalize offers and improve customer engagement. This project builds a machine learning model to predict that likelihood from anonymized customer data.

## Dataset

- Source: [Santander Customer Transaction Prediction (Kaggle)](https://www.kaggle.com/c/santander-customer-transaction-prediction/data)
- 200,000 rows × 200 anonymized numerical features (`var_0` to `var_199`)
- Target: binary (`1` = will make a transaction, `0` = will not)
- Class distribution: imbalanced (~90% negative, ~10% positive)

## Approach

1. **Exploratory Data Analysis** — checked for missing values, studied feature distributions and target imbalance
2. **Preprocessing** — feature scaling using `StandardScaler`
3. **Handling Class Imbalance** — used `class_weight='balanced'` instead of resampling, to avoid overfitting on a duplicated minority class
4. **Modeling** — trained and compared Logistic Regression and Random Forest classifiers
5. **Evaluation** — compared models using Accuracy, Precision, Recall, F1-score, and ROC-AUC (Accuracy alone is misleading on imbalanced data)

## Results

Compared 4 model variants — default vs. balanced class weights for both algorithms:

- **Logistic Regression (default):** Accuracy 0.91, Precision 0.63, Recall 0.10, F1-Score 0.17, ROC-AUC 0.86
- **Random Forest (default):** Accuracy 0.90, Recall 0.00, F1-Score 0.00, ROC-AUC 0.74
- **Logistic Regression (balanced) — final model:** Accuracy 0.78, Precision 0.29, Recall 0.78, F1-Score 0.42, ROC-AUC 0.86
- **Random Forest (balanced):** Accuracy 0.90, Recall 0.00, F1-Score 0.00, ROC-AUC 0.74

**Final model: Logistic Regression (balanced class weights)**

Even though the default models show higher accuracy, they're misleading — with ~90% of customers labeled "0", a model can score 90%+ accuracy while barely detecting any real transactions (Random Forest's recall is 0.00). Since the bank's real goal is to correctly flag customers who *will* transact, **Recall and ROC-AUC matter more than raw accuracy** here.

Logistic Regression with balanced class weights strikes the best trade-off — it correctly identifies **78% of customers who actually make a transaction** (Recall = 0.78) while maintaining the highest ROC-AUC (0.86), making it the most business-relevant model for this problem.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Repository Contents

- `PRCP-1003-CustTransPred_anuj-checkpoint.ipynb` — full notebook with EDA, preprocessing, model building, and evaluation

*Note: the raw dataset is not included in this repository due to file size — use the Kaggle link above to download it.*
