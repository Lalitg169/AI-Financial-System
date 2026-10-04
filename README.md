# 🏦 Smart Financial Risk Intelligence System (SFRIS)

> An end-to-end machine learning platform for Fraud Detection, Credit Risk Scoring,
> and Inflation Forecasting — inspired by the Reserve Bank of India.

---

## 📁 Project Structure

```
smart_financial_risk/
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   ├── fraud_data.csv            # Credit card transactions (imbalanced)
│   ├── credit_data.csv           # Loan applications
│   └── inflation_data.csv        # CPI monthly time-series (10 years)
│
├── src/
│   ├── preprocessing.py          # Reusable pipeline (impute/scale/encode)
│   ├── fraud_detection.py        # Isolation Forest + Random Forest
│   ├── credit_risk.py            # Logistic Regression + Decision Tree
│   ├── economic_trend.py         # ARIMA inflation forecasting
│   └── train_all.py              # Master training script
│
├── models/
│   ├── fraud_rf.pkl              # Trained Random Forest (fraud)
│   ├── fraud_iso.pkl             # Trained Isolation Forest
│   ├── credit_lr.pkl             # Trained Logistic Regression
│   ├── credit_dt.pkl             # Trained Decision Tree
│   └── arima_model.pkl           # Trained ARIMA model
│
├── api/
│   └── app.py                    # Flask REST API (2 endpoints)
│
├── dashboard/
│   └── app_dashboard.py          # Streamlit 4-page interactive dashboard
│
├── reports/
│   └── metrics_report.json       # All model metrics
│
├── screenshots/                  # Generated visualization PNGs
│
├── generate_visuals.py           # Script to regenerate all charts
├── requirements.txt
└── README.md
```

---

## 🚀 Step-by-Step Implementation Guide

### STEP 1 — Environment Setup

```bash
# Create virtual environment
python -m venv sfris_env
source sfris_env/bin/activate       # Linux/Mac
# sfris_env\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### STEP 2 — Generate Datasets

```bash
python data/generate_data.py
```

**What it does:**
- `fraud_data.csv` — 5,000 transactions (97% legit, 3% fraud) with 7 features
- `credit_data.csv` — 3,000 loan applications with income, credit score, etc.
- `inflation_data.csv` — 120 months (10 years) of synthetic CPI data

---

### STEP 3 — Train All Models

```bash
python src/train_all.py
```

**Pipeline per module:**

| Module | Preprocessing | Models | Output |
|--------|--------------|--------|--------|
| Fraud | Median impute → StandardScaler | Isolation Forest, Random Forest | fraud_rf.pkl, fraud_iso.pkl |
| Credit | Label encode → StandardScaler | Logistic Regression, Decision Tree | credit_lr.pkl, credit_dt.pkl |
| Inflation | Date index, fill median | ARIMA(2,1,2) | arima_model.pkl |

**Expected metrics:**
- Fraud RF F1: ~0.98
- Credit LR AUC: ~0.97
- ARIMA RMSE: ~0.55

---

### STEP 4 — Start the Flask API

```bash
python api/app.py
# Runs on http://localhost:5000
```

**Endpoints:**

#### `POST /predict-fraud`
```json
{
  "amount": 1200.0,
  "time_of_day": 2,
  "transaction_freq": 18,
  "distance_from_home": 450,
  "v1": 2.5,
  "v2": -1.8,
  "v3": 2.1
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "rf_prediction": 1,
    "rf_probability": 0.94,
    "iso_prediction": 1,
    "label": "FRAUD"
  }
}
```

#### `POST /predict-credit`
```json
{
  "age": 42,
  "income": 75000,
  "loan_amount": 50000,
  "loan_tenure": 36,
  "credit_score": 580,
  "existing_loans": 3,
  "employment_type": "Self-employed",
  "education": "Graduate"
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "lr_probability": 72.3,
    "dt_probability": 68.1,
    "avg_probability": 70.2,
    "risk_category": "High",
    "default_prediction": 1
  }
}
```

---

### STEP 5 — Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app_dashboard.py
# Opens automatically at http://localhost:8501
```

**Dashboard Pages:**

| Page | Features |
|------|---------|
| 📊 Dashboard Overview | KPI cards, model metrics summary, architecture diagram |
| 🔍 Fraud Detection | Transaction input form, risk score, feature importance |
| 💳 Credit Risk | Applicant form, probability score, risk category |
| 📈 Economic Trends | ARIMA forecast chart, 6-month table, ADF test results |

---

### STEP 6 — Generate Visualizations

```bash
python generate_visuals.py
# Saves 6 PNG files to screenshots/
```

---

## 🧠 Module Deep Dive

### Module 1 — Fraud Detection

**Challenge:** Severe class imbalance (97% legit, 3% fraud)

**Solution:**
- `class_weight="balanced"` in Random Forest
- Isolation Forest as unsupervised backup (no labels needed)

**Key features:**
- `amount` — high amounts flag suspicion
- `time_of_day` — frauds spike at night (hours 0–3)
- `transaction_freq` — many txns in short time = suspicious
- `distance_from_home` — unusually far from home address

---

### Module 2 — Credit Risk

**Risk Categories:**
| Probability | Category |
|------------|---------|
| < 35% | Low Risk ✅ |
| 35% – 65% | Medium Risk ⚠️ |
| > 65% | High Risk 🚨 |

**Ensemble approach:** Average of LR + DT probabilities for robustness.

---

### Module 3 — Inflation Forecasting (ARIMA)

**ARIMA(p,d,q) = (2,1,2)**
- `p=2` — 2 autoregressive terms
- `d=1` — 1st order differencing to achieve stationarity
- `q=2` — 2 moving average terms

**ADF Test:** Confirms stationarity (p < 0.05) after differencing.

**Output:** 6-month CPI forecast with 95% confidence intervals.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn, Imbalanced-learn |
| Time Series | Statsmodels (ARIMA) |
| API Backend | Flask |
| Dashboard | Streamlit |
| Visualization | Matplotlib |
| Model Persistence | Joblib |

---

## 📊 Model Performance Summary

| Model | Metric | Value |
|-------|--------|-------|
| Random Forest (Fraud) | F1-Score | ~0.983 |
| Isolation Forest | F1-Score | ~0.915 |
| Logistic Regression (Credit) | AUC-ROC | ~0.969 |
| Decision Tree (Credit) | AUC-ROC | ~0.937 |
| ARIMA (Inflation) | RMSE | ~0.557 |

---

## 📝 Academic / Interview Notes

This project demonstrates:
1. **Imbalanced Learning** — class weights, anomaly detection
2. **Ensemble Methods** — RF, combining model predictions
3. **Explainability** — feature importance, coefficient magnitude
4. **Time Series Forecasting** — stationarity testing, ARIMA
5. **API Design** — RESTful endpoints with JSON I/O
6. **Production Readiness** — modular code, error handling, model persistence

---

*Built for academic and professional portfolio — simulates real-world financial analytics.*
