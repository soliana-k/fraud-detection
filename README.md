# Fraud Detection ML Pipeline

A machine learning pipeline for detecting fraudulent transactions using e-commerce and credit card transaction data. This repository currently covers **Task 1: data analysis and preprocessing**.

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
│   ├── eda-fraud-data.ipynb
│   └── eda-creditcard.ipynb
├── src/
│   └── eda.py              # Core EDA and preprocessing class
├── scripts/
│   └── run_pipeline.py     # Entry point to run the full pipeline
├── models/                 # Saved model artifacts (gitignored)
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
from src.eda import Eda

# Fraud_Data.csv — full pipeline including geolocation
fraud_eda = Eda(
    fraud_filepath='data/Fraud_Data.csv',
    ip_filepath='data/IpAddress_to_Country.csv'
)
X_train, y_train, X_test, y_test = fraud_eda.run()

# creditcard.csv — skips transformation (already PCA-scaled)
cc_eda = Eda(fraud_filepath='data/creditcard.csv')
X_train, y_train, X_test, y_test = cc_eda.run()
```

---

## Pipeline steps (Task 1)

1. **Data cleaning** — missing value inspection, duplicate removal, dtype correction
2. **EDA** — univariate and bivariate analysis, class imbalance quantification
3. **Geolocation integration** — IP integer conversion, range-based merge with country data, fraud rate by country analysis *(Fraud_Data.csv only)*
4. **Feature engineering** — `hour_of_day`, `day_of_week`, `time_since_signup`, `transaction_frequency` *(Fraud_Data.csv only)*
5. **Data transformation** — `StandardScaler` on numerical features, `OneHotEncoder` on categoricals *(Fraud_Data.csv only; creditcard.csv is already PCA-transformed)*
6. **Class imbalance handling** — SMOTE applied to training set only to avoid data leakage

---

## Progress

- [x] Task 1: Data analysis and preprocessing
- [ ] Task 2: Model training and evaluation
- [ ] Task 3: Model explainability
- [ ] Task 4: API deployment

---

## Requirements

See `requirements.txt`. Key dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`

# P.S for now the model training is run in the respective eda notebooks