"""
Train all three models and save reports.
Run: python src/train_all.py
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from src.config import MODEL_DIR
from data.generate_data import generate_fraud_data, generate_credit_data, generate_inflation_data
from src.preprocessing  import preprocess_fraud, preprocess_credit, preprocess_inflation
from src.fraud_detection import train_fraud_models
from src.credit_risk     import train_credit_models
from src.economic_trend  import train_arima
from src.logger          import get_logger

log = get_logger("train_all")


def main():
    log.info("Smart Financial Risk Intelligence System – Training")

    # ── Generate data ─────────────────────────────────────────────────────
    log.info("[1/3] Generating datasets")
    fraud_df     = generate_fraud_data()
    credit_df    = generate_credit_data()
    inflation_df = generate_inflation_data()

    # ── Fraud ─────────────────────────────────────────────────────────────
    log.info("[2/3a] Training fraud detection models")
    splits_f, scaler_f, feat_f = preprocess_fraud(fraud_df)
    fraud_metrics, rf, iso     = train_fraud_models(splits_f, feat_f)
    joblib.dump(scaler_f, MODEL_DIR / "fraud_scaler.pkl")
    log.info("RF F1=%s | ISO F1=%s", fraud_metrics["random_forest"]["f1_score"],
             fraud_metrics["isolation_forest"]["f1_score"])

    # ── Credit ────────────────────────────────────────────────────────────
    log.info("[2/3b] Training credit risk models")
    splits_c, scaler_c, feat_c, encoders_c = preprocess_credit(credit_df)
    credit_metrics, lr, dt                  = train_credit_models(splits_c, feat_c)
    joblib.dump(scaler_c,   MODEL_DIR / "credit_scaler.pkl")
    joblib.dump(encoders_c, MODEL_DIR / "credit_encoders.pkl")
    log.info("LR AUC=%s | DT AUC=%s", credit_metrics["logistic_regression"]["auc_roc"],
             credit_metrics["decision_tree"]["auc_roc"])

    # ── ARIMA ─────────────────────────────────────────────────────────────
    log.info("[2/3c] Training ARIMA inflation model")
    ts_df           = preprocess_inflation(inflation_df)
    arima_metrics, arima_results = train_arima(ts_df)
    log.info("AIC=%s | RMSE=%s", arima_metrics["aic"], arima_metrics["rmse"])

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "fraud":    {k: v for k, v in fraud_metrics.items() if k != "report"},
        "credit":   {k: v for k, v in credit_metrics.items() if k != "report"},
        "inflation": arima_metrics,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    import numpy as np
    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            return super().default(o)
    with open(ROOT / "reports" / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2, cls=_Encoder)

    log.info("All models trained and saved. Report: reports/metrics_report.json")


if __name__ == "__main__":
    main()
