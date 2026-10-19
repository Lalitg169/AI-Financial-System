"""
Fraud Detection Module
  - Isolation Forest  (unsupervised)
  - Random Forest     (supervised + class_weight, optional SMOTE)
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import MODEL_DIR, FRAUD_FEATURES, RANDOM_STATE
from src.logger import get_logger
from src.model_store import load_artifact

log = get_logger(__name__)


def _scores(y_true, y_pred, y_score):
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_score), 4),
        "pr_auc":    round(average_precision_score(y_true, y_score), 4),
    }


def _tune_threshold(estimator, X_tr, y_tr):
    """Pick the probability cut-off that maximises F1, using out-of-fold
    predictions on the training split only (the test split stays unseen)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(estimator, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_tr, oof)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))          # last point has no threshold
    return float(thresholds[best])


def train_fraud_models(splits, feature_names, use_smote=False):
    X_tr, X_te, y_tr, y_te = splits

    # ── Isolation Forest ──────────────────────────────────────────────────
    iso = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_estimators=100)
    iso.fit(X_tr)
    iso_scores = -iso.decision_function(X_te)        # higher = more anomalous
    iso_preds  = (iso.predict(X_te) == -1).astype(int)

    # ── Random Forest ─────────────────────────────────────────────────────
    # With SMOTE the classes are balanced by oversampling, so class weights
    # are not needed on top of it.
    rf = RandomForestClassifier(
        n_estimators=200, class_weight=None if use_smote else "balanced",
        random_state=RANDOM_STATE, n_jobs=-1)

    if use_smote:
        # SMOTE inside the pipeline so CV folds never see synthetic copies
        # of their own validation rows
        smote = SMOTE(random_state=RANDOM_STATE)
        threshold = _tune_threshold(ImbPipeline([("smote", smote), ("rf", clone(rf))]), X_tr, y_tr)
        X_fit, y_fit = smote.fit_resample(X_tr, y_tr)
    else:
        threshold = _tune_threshold(clone(rf), X_tr, y_tr)
        X_fit, y_fit = X_tr, y_tr

    rf.fit(X_fit, y_fit)
    rf_proba = rf.predict_proba(X_te)[:, 1]
    rf_preds = (rf_proba >= 0.5).astype(int)
    tuned_preds = (rf_proba >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────
    rf_metrics = _scores(y_te, rf_preds, rf_proba)
    rf_metrics.update({
        "report":           classification_report(y_te, rf_preds),
        "confusion_matrix": confusion_matrix(y_te, rf_preds).tolist(),
        "tuned_threshold": {"threshold": round(threshold, 4),
                            **_scores(y_te, tuned_preds, rf_proba)},
    })
    metrics = {
        "smote": use_smote,
        "random_forest":    rf_metrics,
        "isolation_forest": _scores(y_te, iso_preds, iso_scores),
        "feature_importance": dict(zip(feature_names,
                                       rf.feature_importances_.round(4))),
    }

    # ── Persist ───────────────────────────────────────────────────────────
    joblib.dump(rf,        MODEL_DIR / "fraud_rf.pkl")
    joblib.dump(iso,       MODEL_DIR / "fraud_iso.pkl")
    joblib.dump(threshold, MODEL_DIR / "fraud_threshold.pkl")
    log.info("Fraud models saved to %s (threshold=%.3f, smote=%s)", MODEL_DIR, threshold, use_smote)
    return metrics, rf, iso


def predict_fraud(features: dict):
    """Single-record prediction for API."""
    rf        = load_artifact("fraud_rf.pkl")
    iso       = load_artifact("fraud_iso.pkl")
    scaler    = load_artifact("fraud_scaler.pkl")
    threshold = load_artifact("fraud_threshold.pkl")

    # trained with n_jobs=-1; for a single row, starting workers costs more
    # than the prediction itself
    rf.n_jobs = 1

    row = np.array([[features.get(k, 0) for k in FRAUD_FEATURES]])
    row = scaler.transform(row)

    rf_proba = float(rf.predict_proba(row)[0][1])
    rf_pred  = int(rf_proba >= threshold)
    iso_pred = int(iso.predict(row)[0] == -1)

    return {
        "rf_prediction":  rf_pred,
        "rf_probability": round(rf_proba, 4),
        "threshold":      round(threshold, 4),
        "iso_prediction": iso_pred,
        "label": "FRAUD" if rf_pred == 1 else "LEGITIMATE",
    }
