"""
Central configuration: paths and feature lists shared across modules.
"""
from pathlib import Path

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── Fraud ────────────────────────────────────────────────────────────────
FRAUD_FEATURES = ["amount", "time_of_day", "transaction_freq",
                  "distance_from_home", "v1", "v2", "v3"]
FRAUD_TARGET   = "is_fraud"

# ── Credit ───────────────────────────────────────────────────────────────
CREDIT_NUM_FEATURES = ["age", "income", "loan_amount", "loan_tenure",
                       "credit_score", "existing_loans"]
CREDIT_CAT_FEATURES = ["employment_type", "education"]
CREDIT_TARGET       = "default"

# TODO: move model hyperparameters here too
RANDOM_STATE = 42
TEST_SIZE    = 0.2
