"""
Fraud Detection Module
  - Isolation Forest  (unsupervised)
  - Random Forest     (supervised + class_weight)
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, precision_score,
                             recall_score, f1_score, confusion_matrix)

from src.config import MODEL_DIR, FRAUD_FEATURES, RANDOM_STATE
from src.logger import get_logger

log = get_logger(__name__)


def train_fraud_models(splits, feature_names):
    X_tr, X_te, y_tr, y_te = splits

    # ── Isolation Forest ──────────────────────────────────────────────────
    iso = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_estimators=100)
    iso.fit(X_tr)
    iso_scores = iso.decision_function(X_te)          # anomaly score
    iso_preds  = (iso.predict(X_te) == -1).astype(int)

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_preds = rf.predict(X_te)
    rf_proba = rf.predict_proba(X_te)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = {
        "random_forest": {
            "precision": round(precision_score(y_te, rf_preds), 4),
            "recall":    round(recall_score(y_te, rf_preds), 4),
            "f1_score":  round(f1_score(y_te, rf_preds), 4),
            "report":    classification_report(y_te, rf_preds),
            "confusion_matrix": confusion_matrix(y_te, rf_preds).tolist(),
        },
        "isolation_forest": {
            "precision": round(precision_score(y_te, iso_preds, zero_division=0), 4),
            "recall":    round(recall_score(y_te, iso_preds, zero_division=0), 4),
            "f1_score":  round(f1_score(y_te, iso_preds, zero_division=0), 4),
        },
        "feature_importance": dict(zip(feature_names,
                                       rf.feature_importances_.round(4))),
    }

    # ── Persist ───────────────────────────────────────────────────────────
    joblib.dump(rf,  MODEL_DIR / "fraud_rf.pkl")
    joblib.dump(iso, MODEL_DIR / "fraud_iso.pkl")
    log.info("Fraud models saved to %s", MODEL_DIR)
    return metrics, rf, iso


def predict_fraud(features: dict):
    """Single-record prediction for API."""
    rf  = joblib.load(MODEL_DIR / "fraud_rf.pkl")
    iso = joblib.load(MODEL_DIR / "fraud_iso.pkl")
    scaler = joblib.load(MODEL_DIR / "fraud_scaler.pkl")

    row = np.array([[features.get(k, 0) for k in FRAUD_FEATURES]])
    row = scaler.transform(row)

    rf_pred  = int(rf.predict(row)[0])
    rf_proba = float(rf.predict_proba(row)[0][1])
    iso_pred = int(iso.predict(row)[0] == -1)

    return {
        "rf_prediction":  rf_pred,
        "rf_probability": round(rf_proba, 4),
        "iso_prediction": iso_pred,
        "label": "FRAUD" if rf_pred == 1 else "LEGITIMATE",
    }
