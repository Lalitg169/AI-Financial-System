"""
Reusable preprocessing pipeline for all modules.

Imputers and scalers are fitted on the training split only, then applied
to the test split, so no information from test rows leaks into training.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.config import (FRAUD_FEATURES, FRAUD_TARGET, CREDIT_NUM_FEATURES,
                        CREDIT_CAT_FEATURES, CREDIT_TARGET, RANDOM_STATE, TEST_SIZE)


def _split_and_scale(df, feature_cols, target_col):
    """Split first, then fit imputer + scaler on the training rows only."""
    X = df[feature_cols].values
    y = df[target_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_te = scaler.transform(imputer.transform(X_te))
    return X_tr, X_te, y_tr, y_te, scaler


def preprocess_fraud(df: pd.DataFrame):
    """Returns (X_tr, X_te, y_tr, y_te), fitted scaler and feature names."""
    X_tr, X_te, y_tr, y_te, scaler = _split_and_scale(df, FRAUD_FEATURES, FRAUD_TARGET)
    return (X_tr, X_te, y_tr, y_te), scaler, FRAUD_FEATURES


def preprocess_credit(df: pd.DataFrame):
    """Returns (X_tr, X_te, y_tr, y_te), scaler, feature names and encoders."""
    df = df.copy()

    # Label-encode categoricals (category list is fixed, so no leakage here)
    le_map = {}
    for c in CREDIT_CAT_FEATURES:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_map[c] = le

    num_cols = CREDIT_NUM_FEATURES + CREDIT_CAT_FEATURES
    X_tr, X_te, y_tr, y_te, scaler = _split_and_scale(df, num_cols, CREDIT_TARGET)
    return (X_tr, X_te, y_tr, y_te), scaler, num_cols, le_map


def preprocess_inflation(df: pd.DataFrame):
    """Returns cleaned time-series DataFrame."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("MS")
    df["cpi"] = df["cpi"].fillna(df["cpi"].median())
    return df
