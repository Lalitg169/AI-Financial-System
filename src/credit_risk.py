"""
Credit Risk Prediction Module
  - Logistic Regression
  - Decision Tree
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, precision_score,
                             recall_score, f1_score, roc_auc_score)

from src.config import MODEL_DIR, RANDOM_STATE, CREDIT_NUM_FEATURES, CREDIT_CAT_FEATURES
from src.logger import get_logger
from src.model_store import load_artifact

log = get_logger(__name__)


def risk_category(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def train_credit_models(splits, feature_names):
    X_tr, X_te, y_tr, y_te = splits

    # ── Logistic Regression ───────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_tr, y_tr)
    lr_preds = lr.predict(X_te)
    lr_proba = lr.predict_proba(X_te)[:, 1]

    # ── Decision Tree ─────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)
    dt.fit(X_tr, y_tr)
    dt_preds = dt.predict(X_te)
    dt_proba = dt.predict_proba(X_te)[:, 1]

    metrics = {
        "logistic_regression": {
            "precision": round(precision_score(y_te, lr_preds), 4),
            "recall":    round(recall_score(y_te, lr_preds), 4),
            "f1_score":  round(f1_score(y_te, lr_preds), 4),
            "auc_roc":   round(roc_auc_score(y_te, lr_proba), 4),
            "report":    classification_report(y_te, lr_preds),
        },
        "decision_tree": {
            "precision": round(precision_score(y_te, dt_preds), 4),
            "recall":    round(recall_score(y_te, dt_preds), 4),
            "f1_score":  round(f1_score(y_te, dt_preds), 4),
            "auc_roc":   round(roc_auc_score(y_te, dt_proba), 4),
        },
        "feature_importance_dt": dict(zip(feature_names,
                                          dt.feature_importances_.round(4))),
        "feature_importance_lr": dict(zip(feature_names,
                                          np.abs(lr.coef_[0]).round(4))),
    }

    joblib.dump(lr, MODEL_DIR / "credit_lr.pkl")
    joblib.dump(dt, MODEL_DIR / "credit_dt.pkl")
    log.info("Credit models saved to %s", MODEL_DIR)
    return metrics, lr, dt


def explain_lr(lr, scaled_row, top_n=3):
    """Per-feature contribution to the Logistic Regression score.

    Inputs are standardised, so 0 means "average applicant" and
    coefficient * value is how far this feature pushes the log-odds of
    default away from average. Exact for a linear model; employment_type and
    education are label-encoded, so their contributions are only rough.
    """
    names = CREDIT_NUM_FEATURES + CREDIT_CAT_FEATURES
    contrib = lr.coef_[0] * scaled_row[0]
    order = np.argsort(-np.abs(contrib))[:top_n]
    return [{"feature": names[i],
             "effect": "raises risk" if contrib[i] > 0 else "lowers risk",
             "weight": round(float(contrib[i]), 3)} for i in order]


def predict_credit(features: dict):
    """Single-record prediction for API."""
    lr       = load_artifact("credit_lr.pkl")
    dt       = load_artifact("credit_dt.pkl")
    scaler   = load_artifact("credit_scaler.pkl")
    encoders = load_artifact("credit_encoders.pkl")

    # same label encoding as training (alphabetical order from LabelEncoder)
    emp = encoders["employment_type"].transform(
        [features.get("employment_type", "Salaried")])[0]
    edu = encoders["education"].transform(
        [features.get("education", "Graduate")])[0]

    row = np.array([[
        features.get("age", 30),
        features.get("income", 50000),
        features.get("loan_amount", 15000),
        features.get("loan_tenure", 36),
        features.get("credit_score", 650),
        features.get("existing_loans", 0),
        emp,
        edu,
    ]])
    row = scaler.transform(row)

    lr_prob = float(lr.predict_proba(row)[0][1])
    dt_prob = float(dt.predict_proba(row)[0][1])
    avg_prob = (lr_prob + dt_prob) / 2

    return {
        "lr_probability":  round(lr_prob * 100, 2),
        "dt_probability":  round(dt_prob * 100, 2),
        "avg_probability": round(avg_prob * 100, 2),
        "risk_category":   risk_category(avg_prob),
        "default_prediction": int(avg_prob > 0.5),
        "top_factors":     explain_lr(lr, row),
    }
