# Fraud Detection ML Pipeline

A machine learning pipeline for detecting fraudulent transactions using e-commerce and credit card transaction data. This repository currently covers **Task 1: data analysis and preprocessing** & **Task 2: Model Training & Evaluation** .

---

## Project overview

This project builds a fraud detection system across two datasets with different characteristics, a raw e-commerce transaction dataset requiring full preprocessing, and a PCA-preprocessed credit card dataset. The pipeline handles data cleaning, exploratory analysis, geolocation enrichment, feature engineering, and class imbalance correction in preparation for model training.

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
├── data/                   # Raw data files (gitignored)
├── notebooks/
│   ├── eda_fraud.ipynb
│   └── eda_creditcard.ipynb
│   ├── ml_fraud.ipynb
│   └── ml_credit_card.ipynb
├── src/
│   ├── preprocessing.py           
│   ├── model_train.py 
│   └── eda.py              # Core EDA and preprocessing class
├── scripts/
│   └── run_pipeline.py     # Entry point to run the full pipeline
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

```python
from src.eda import EdaPipeline

# Fraud_Data.csv — full pipeline including geolocation
fraud_eda = EdaPipeline(
    fraud_filepath='data/Fraud_Data.csv',
    ip_filepath='data/IpAddress_to_Country.csv'
)
X_train, y_train, X_test, y_test = fraud_eda.run()

# creditcard.csv — skips transformation (already PCA-scaled)
cc_eda = EdaPipeline(fraud_filepath='data/creditcard.csv')
X_train, y_train, X_test, y_test = cc_eda.run()
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
8. **Business-Oriented Evaluation** (Precision-Recall tradeoff, cost implications)

---


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

---


## Progress

- [x] Task 1: Data analysis and preprocessing
- [x] Task 2: Model training and evaluation
- [ ] Task 3: Model explainability
<!-- - [ ] Task 4: API deployment -->

---

## Requirements

See `requirements.txt`. Key dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`


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

