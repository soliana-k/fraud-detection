# Fraud Detection ML Pipeline

A machine learning pipeline for detecting fraudulent transactions using e-commerce and credit card transaction data. This repository currently covers **Task 1: data analysis and preprocessing**, **Task 2: Model Training & Evaluation**, and **Task 3: Model Explainability** .

---

## Project overview

This project builds a fraud detection system across two datasets with different characteristics, a raw e-commerce transaction dataset requiring full preprocessing, and a PCA-preprocessed credit card dataset. The pipeline handles data cleaning, exploratory analysis, geolocation enrichment, feature engineering, and class imbalance correction in preparation for model training.

The end-to-end pipeline includes data cleaning, EDA, feature engineering, class imbalance handling, model training with hyperparameter tuning, and **SHAP-based model explainability** to generate actionable business insights.

---

## Datasets

| Dataset | Description | Source |
|---|---|---|
| `Fraud_Data.csv` | E-commerce transactions with user/device metadata | Internal |
| `creditcard.csv` | PCA-transformed credit card transactions (V1–V28) | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `IpAddress_to_Country.csv` | IP range to country mapping for geolocation | Internal |

> **Note:** Data files are not included in this repository. Place them in the `data/` directory before running.

---

## Repository structure

```
├── data/                  
├── notebooks/
│   ├── eda_fraud.ipynb
│   └── eda_creditcard.ipynb
│   └── shap_explainability.ipynb
│   ├── ml_fraud.ipynb
│   └── ml_credit_card.ipynb
├── src/
│   ├── preprocessing.py           
│   ├── model_train.py 
│   └── eda.py              
├── scripts/
│   └── run_pipeline.py     
├── models/
│   ├── creditcard/                
│   └── fraud/                     
├── tests/
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/soliana-k/fraud-detection
cd fraud-detection
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add data files**

Place `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv` in the `data/` directory.

---

## Usage

### EDA
```python
from src.eda import EdaPipeline

# Fraud_Data.csv — full pipeline including geolocation
fraud_eda = EdaPipeline(
    'data/Fraud_Data.csv',
    'data/IpAddress_to_Country.csv'
)
fraud_eda.run()

# creditcard.csv —
cc_eda = EdaPipeline('data/creditcard.csv')
cc_eda.run()
```
### Preprocessing and Modelling

```python
from src.preprocessing import FraudDetectionPipeline
from src.model_train import ModelTrain

# Preprocessing + Feature Engineering
fraud_pipeline = FraudDetectionPipeline('data/processed/processedFraud_Data.csv') # or could be the processed creditcard data
X_train_f, y_train_f, X_test_f, y_test_f = fraud_pipeline.run()

# Model Training & Evaluation
trainer = ModelTrain(X_train_f, y_train_f, X_test_f, y_test_f)
lg= trainer.train_log_reg() # train baseline logistic regression model
best_models = trainer.train_and_evaluate_models() # train random forest, xgboost and lightgbm 
trainer.compare_all_models(best_models=best_models) # cross validate all the ensemble models and print comparison table
```

---

## Pipeline Steps

1. **Data Ingestion & Validation**
2. **Cleaning + Geolocation (Fraud only)**
3. **EDA & Visualization**
4. **Feature Engineering**
5. **Train/Test Split + Preprocessing**
6. **SMOTE Balancing (Training only)**
7. **Model Training + Hyperparameter Tuning**
8. **Business-Oriented Evaluation** (Precision-Recall tradeoff)

---



## What Was Done

### Task 1: Data Analysis & Preprocessing (Completed)
- **Data Loading & Validation** with strict column checks for both datasets.
- **Cleaning**: Duplicate removal, missing value analysis, datetime conversion, type optimization.
- **Geolocation Enrichment** (Fraud dataset only): IP-to-country mapping using `merge_asof`.
- **EDA**: Univariate, bivariate, multivariate analysis, correlation heatmaps, fraud-by-country analysis, imbalance visualization.
- **Feature Engineering** (Fraud dataset only):
  - `hour_of_day`, `day_of_week`
  - `time_since_signup`
  - `transaction_frequency` per user
  - Target encoding on `country`
- **Preprocessing**: Scaling (StandardScaler), One-Hot Encoding, handling of categorical features.
- **Class Imbalance Handling**: SMOTE / SMOTENC applied **only on training set** (sampling strategy = 0.5).

### Task 2: Model Training & Evaluation (Completed)
- Trained and evaluated four models on both datasets:
  - **Logistic Regression** (baseline with `class_weight='balanced'`)
  - **Random Forest**
  - **XGBoost**
  - **LightGBM**
- Hyperparameter tuning using `GridSearchCV` (scoring = Average Precision).
- Comprehensive evaluation: AUC-ROC, PR-AUC, F1-score, Precision, Recall, Confusion Matrices, Precision-Recall curves.
- Cross-validation (Stratified K-Fold) for robust performance estimation.
- Business-focused analysis: Precision-Recall tradeoff, false positive costs, operational impact.

**Models are saved** in the `models/creditcard/` and `models/fraud/` directories for future deployment/explainability.

### Task 3: Model Explainability (Completed)
SHAP (SHapley Additive exPlanations) was used to interpret the best models (`LightGBM` for Fraud_Data and `XGBoost` for Creditcard).

#### Fraud_Data → LightGBM Explainability Highlights
- **Dominant Feature**: `time_since_signup` — by far the strongest predictor. Very short account age dramatically increases fraud probability.
- Other key drivers: `hour_of_day`, `day_of_week`, `country_target_enc`, `purchase_value`.
- The model excels at catching "velocity" fraud (new accounts acting immediately) but can miss sophisticated/account-takeover fraud on aged accounts.

**Business Recommendations**:
1. **Dynamic Step-Up Authentication** for new accounts: Trigger MFA/ID verification when `time_since_signup` is low.
2. **Automated High-Precision Blocking**: Use LightGBM’s near-perfect precision to auto-block very high-risk scores.
3. **Behavioral Velocity Features**: Add features like device changes, IP distance, or transaction spikes on mature accounts to reduce the ~48% missed fraud (False Negatives).

#### Creditcard → XGBoost Explainability Highlights
- **Top Drivers**: `V14`, `V4`, `V12`, `V11` — strong structural signals of fraudulent patterns.
- `Amount` and `Time` have minimal impact, helping suppress false positives on legitimate large transactions.
- The model requires coordinated risk signals across multiple V-features before flagging.

**Business Recommendations**:
1. **Automated Action for High Scores**: Auto-block or trigger MFA when score > 0.85 (leveraging 0.89 precision).
2. **Velocity & Behavioral Features**: Add `card_velocity_1h`, declined attempts sequence, and country mismatch to close the 20% fraud leakage.
3. **Upstream Hard Rules**: Create gateway rules based on extreme values in `V14`/`V4` for early filtering.

**Full SHAP analysis** (Summary plots, Force plots for TP/FP/FN cases, and detailed interpretations) is available in `notebooks/shap_explainability.ipynb`.


---


## Progress

- [x] Task 1: Data analysis and preprocessing
- [x] Task 2: Model training and evaluation
- [x] Task 3: Model explainability


---

## Requirements

See `requirements.txt`. Key dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`
- `shap`
- `xgboost`
- `lightgbm`
- `category_encoders`


## Model Results Summary

### 1. Credit Card Dataset (Highly Imbalanced)
**Best Model: XGBoost**

- Highest F1-score (0.8444) and excellent precision (0.89) with only **9 false positives**.
- Strong balance between catching fraud and minimizing customer friction.

### 2. Fraud_Data Dataset (Moderately Imbalanced)
**Best Model: LightGBM**

- Highest F1-score (0.6874), near-perfect precision (**1.00**), and only **2 false positives**.
- Most operationally efficient model for this dataset.

**Full detailed comparison reports** are available in the respective notebooks (`ml_fraud.ipynb` and `ml_creditcard.ipynb`).

---


## Precision-Recall Tradeoff & Business Implications

Fraud detection requires careful balancing:
- **High Recall** → Catch more fraud → Higher financial protection but more false alarms (customer frustration + review costs).
- **High Precision** → Fewer false positives → Better user experience and lower operational costs, but risk of missing fraud.

**Recommendations**:
- **Creditcard dataset**: XGBoost offers the best business value (low false positives + solid fraud capture).
- **Fraud_Data dataset**: LightGBM is ideal due to almost zero false positives while maintaining acceptable recall.
- Future work will include threshold tuning based on average fraud amount vs. review cost.

---
