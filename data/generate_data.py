"""
Generate synthetic datasets for all three modules.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
DATA_DIR = Path(__file__).parent

# ── 1. Fraud Detection Dataset ──────────────────────────────────────────────
def generate_fraud_data(n=5000):
    n_legit = int(n * 0.97)
    n_fraud = n - n_legit

    legit = pd.DataFrame({
        "amount":        np.random.exponential(80,  n_legit),
        "time_of_day":   np.random.randint(0, 24,   n_legit),
        "transaction_freq": np.random.randint(1, 10, n_legit),
        "distance_from_home": np.random.exponential(20, n_legit),
        "v1": np.random.normal(0, 1, n_legit),
        "v2": np.random.normal(0, 1, n_legit),
        "v3": np.random.normal(0, 1, n_legit),
        "is_fraud": 0,
    })
    fraud = pd.DataFrame({
        "amount":        np.random.exponential(500, n_fraud),
        "time_of_day":   np.random.choice([0,1,2,3,23], n_fraud),
        "transaction_freq": np.random.randint(10, 30, n_fraud),
        "distance_from_home": np.random.exponential(200, n_fraud),
        "v1": np.random.normal(3, 2, n_fraud),
        "v2": np.random.normal(-2, 2, n_fraud),
        "v3": np.random.normal(2, 1.5, n_fraud),
        "is_fraud": 1,
    })
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fraud_data.csv", index=False)
    print(f"Fraud data: {len(df)} rows  |  fraud={n_fraud}  ({n_fraud/len(df)*100:.1f}%)")
    return df

# ── 2. Credit Risk Dataset ───────────────────────────────────────────────────
def generate_credit_data(n=3000):
    df = pd.DataFrame({
        "age":            np.random.randint(22, 65, n),
        "income":         np.random.normal(50000, 20000, n).clip(10000),
        "loan_amount":    np.random.normal(15000, 8000, n).clip(1000),
        "loan_tenure":    np.random.choice([12, 24, 36, 48, 60], n),
        "credit_score":   np.random.randint(300, 850, n),
        "existing_loans": np.random.randint(0, 5, n),
        "employment_type":np.random.choice(["Salaried","Self-employed","Business"], n),
        "education":      np.random.choice(["Graduate","Post-Graduate","Under-Graduate"], n),
    })
    # Business logic for default
    risk_score = (
        - (df["credit_score"] - 300) / 550
        + df["loan_amount"] / df["income"]
        + df["existing_loans"] * 0.15
        + np.random.normal(0, 0.1, n)
    )
    df["default"] = (risk_score > 0.5).astype(int)
    df.to_csv(DATA_DIR / "credit_data.csv", index=False)
    print(f"Credit data: {len(df)} rows  |  defaults={df['default'].sum()}  ({df['default'].mean()*100:.1f}%)")
    return df

# ── 3. Inflation / CPI Time-Series ──────────────────────────────────────────
def generate_inflation_data():
    periods = 120   # 10 years monthly
    dates = pd.date_range("2014-01", periods=periods, freq="MS")
    trend   = np.linspace(5.0, 6.5, periods)
    seasonal = 0.8 * np.sin(np.linspace(0, 4*np.pi, periods))
    noise   = np.random.normal(0, 0.3, periods)
    cpi     = trend + seasonal + noise
    df = pd.DataFrame({"date": dates, "cpi": cpi.round(2)})
    df.to_csv(DATA_DIR / "inflation_data.csv", index=False)
    print(f"Inflation data: {len(df)} months  |  CPI range [{cpi.min():.2f}, {cpi.max():.2f}]")
    return df

if __name__ == "__main__":
    generate_fraud_data()
    generate_credit_data()
    generate_inflation_data()
    print("All datasets generated ✓")
