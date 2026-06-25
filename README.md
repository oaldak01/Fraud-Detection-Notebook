# Fraud-Detection-Notebook

========================================================================
FRAUD DETECTION SYSTEM - PROJECT README
========================================================================

Project Overview
----------------
This project focuses on building an end-to-end Machine Learning pipeline
to detect fraudulent transactions in mobile money transfer activities.
It analyzes a synthetic dataset containing over 6.3 million transaction
records to identify patterns indicative of fraud.

Dataset Details
---------------
- Source file: data.csv
- Shape: 6,362,620 rows, 11 columns
- Class Imbalance: Very high (Fraud Ratio: ~0.0013 / 0.13%)
- Features: 
  - step: Maps a unit of time in the real world. 1 step is 1 hour of time.
  - type: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
  - amount: Amount of the transaction in local currency.
  - nameOrig: Customer who started the transaction.
  - oldbalanceOrg: Initial balance before the transaction.
  - newbalanceOrig: New balance after the transaction.
  - nameDest: Customer who is the recipient of the transaction.
  - oldbalanceDest: Initial recipient balance before the transaction.
  - newbalanceDest: New recipient balance after the transaction.
  - isFraud: The actual fraud status (Target: 1 for fraud, 0 otherwise).
  - isFlaggedFraud: System flags transactions transferring more than 200k.

Models Evaluated & Comparison
-----------------------------
Three classification models were trained and tuned using hyperparameter
optimization, evaluating both ROC-AUC and PR-AUC (AUPRC):

1. XGBoost (Tuned):
   - ROC-AUC: 0.9982
   - PR-AUC:  0.8650
   - Status:  Best performing model (Selected)

2. Random Forest (Tuned):
   - ROC-AUC: 0.9978
   - PR-AUC:  0.8329

3. Logistic Regression (Tuned):
   - ROC-AUC: 0.9775
   - PR-AUC:  0.5504

Best Model Artifact & Threshold
--------------------------------
The best model (XGBoost) was saved along with its optimal decision
threshold and feature list to a joblib artifact file:

- Saved Model File: best_fraud_model_tuned.pkl
- Selected Decision Threshold: 0.9140

Selected Model Performance (On Test Set)
----------------------------------------
At the chosen threshold of 0.9140, the XGBoost model achieved:

- Correctly detected fraud (True Positives): 1,472
- Missed fraud (False Negatives):            171
- False alarms (False Positives):            3,302
- Correct non-fraud (True Negatives):        1,267,579

Metrics:
- Fraud Recall:    89.6% (Detects ~90% of all fraudulent transactions)
- Fraud Precision: 30.8%

File Structure
--------------
- analysis.ipynb:             Jupyter Notebook containing setup, EDA,
                             preprocessing, model training, tuning,
                             evaluation, and exporting logic.
- data.csv:                   Raw transaction dataset.
- best_fraud_model_tuned.pkl: Saved best model artifact (XGBoost, 
                             threshold, features).
- README.txt:                 This documentation file.
- README.md:                  Markdown version of project title.

Usage / How to Load the Model
-----------------------------
To load and predict using the saved model artifact, use the following
Python code:

```python
import joblib

# Load artifact
artifact = joblib.load("best_fraud_model_tuned.pkl")
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# Predict probability
probabilities = model.predict_proba(test_df[features])[:, 1]

# Apply decision threshold
predictions = (probabilities >= threshold).astype(int)
```
========================================================================

