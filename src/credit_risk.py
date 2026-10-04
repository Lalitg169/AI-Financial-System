"""
Credit Risk Prediction Module
  - Logistic Regression
  - Decision Tree
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def risk_category(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def train_credit_models(X, y, feature_names):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # ── Logistic Regression ───────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    lr_preds = lr.predict(X_te)
    lr_proba = lr.predict_proba(X_te)[:, 1]

    # ── Decision Tree ─────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
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
    print("Credit models saved ✓")
    return metrics, lr, dt


def predict_credit(features: dict):
    """Single-record prediction for API."""
    lr = joblib.load(MODEL_DIR / "credit_lr.pkl")
    dt = joblib.load(MODEL_DIR / "credit_dt.pkl")

    # encode employment_type and education simply
    emp_map = {"Salaried": 0, "Self-employed": 1, "Business": 2}
    edu_map = {"Graduate": 0, "Post-Graduate": 1, "Under-Graduate": 2}

    row = np.array([[
        features.get("age", 30),
        features.get("income", 50000),
        features.get("loan_amount", 15000),
        features.get("loan_tenure", 36),
        features.get("credit_score", 650),
        features.get("existing_loans", 0),
        emp_map.get(features.get("employment_type", "Salaried"), 0),
        edu_map.get(features.get("education", "Graduate"), 0),
    ]])

    lr_prob = float(lr.predict_proba(row)[0][1])
    dt_prob = float(dt.predict_proba(row)[0][1])
    avg_prob = (lr_prob + dt_prob) / 2

    return {
        "lr_probability":  round(lr_prob * 100, 2),
        "dt_probability":  round(dt_prob * 100, 2),
        "avg_probability": round(avg_prob * 100, 2),
        "risk_category":   risk_category(avg_prob),
        "default_prediction": int(avg_prob > 0.5),
    }
