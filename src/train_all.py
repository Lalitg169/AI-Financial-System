"""
Train all three models and save reports.
Run: python src/train_all.py
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from data.generate_data import generate_fraud_data, generate_credit_data, generate_inflation_data
from src.preprocessing  import preprocess_fraud, preprocess_credit, preprocess_inflation
from src.fraud_detection import train_fraud_models
from src.credit_risk     import train_credit_models
from src.economic_trend  import train_arima


def main():
    print("=" * 60)
    print("  Smart Financial Risk Intelligence System – Training")
    print("=" * 60)

    # ── Generate data ─────────────────────────────────────────────────────
    print("\n[1/3] Generating datasets …")
    fraud_df     = generate_fraud_data()
    credit_df    = generate_credit_data()
    inflation_df = generate_inflation_data()

    # ── Fraud ─────────────────────────────────────────────────────────────
    print("\n[2/3a] Training Fraud Detection models …")
    X_f, y_f, scaler_f, feat_f = preprocess_fraud(fraud_df)
    fraud_metrics, rf, iso      = train_fraud_models(X_f, y_f, feat_f)
    print("  RF  F1:", fraud_metrics["random_forest"]["f1_score"])
    print("  ISO F1:", fraud_metrics["isolation_forest"]["f1_score"])

    # ── Credit ────────────────────────────────────────────────────────────
    print("\n[2/3b] Training Credit Risk models …")
    X_c, y_c, scaler_c, feat_c, _ = preprocess_credit(credit_df)
    credit_metrics, lr, dt         = train_credit_models(X_c, y_c, feat_c)
    print("  LR  AUC:", credit_metrics["logistic_regression"]["auc_roc"])
    print("  DT  AUC:", credit_metrics["decision_tree"]["auc_roc"])

    # ── ARIMA ─────────────────────────────────────────────────────────────
    print("\n[2/3c] Training ARIMA (Inflation) model …")
    ts_df           = preprocess_inflation(inflation_df)
    arima_metrics, arima_results = train_arima(ts_df)
    print("  AIC:", arima_metrics["aic"], " | RMSE:", arima_metrics["rmse"])

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

    print("\n✅  All models trained & saved.  Reports → reports/metrics_report.json")


if __name__ == "__main__":
    main()
