"""
Reusable preprocessing pipeline for all modules.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_fraud(df: pd.DataFrame):
    """Returns X, y and fitted scaler."""
    df = df.copy()
    num_cols = ["amount", "time_of_day", "transaction_freq",
                "distance_from_home", "v1", "v2", "v3"]

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    df[num_cols] = imputer.fit_transform(df[num_cols])
    X = scaler.fit_transform(df[num_cols])
    y = df["is_fraud"].values
    return X, y, scaler, num_cols


def preprocess_credit(df: pd.DataFrame):
    """Returns X, y, scaler and feature names after encoding."""
    df = df.copy()

    # Label-encode categoricals
    cat_cols = ["employment_type", "education"]
    le_map = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_map[c] = le

    num_cols = ["age", "income", "loan_amount", "loan_tenure",
                "credit_score", "existing_loans"] + cat_cols

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    df[num_cols] = imputer.fit_transform(df[num_cols])
    X = scaler.fit_transform(df[num_cols])
    y = df["default"].values
    return X, y, scaler, num_cols, le_map


def preprocess_inflation(df: pd.DataFrame):
    """Returns cleaned time-series DataFrame."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("MS")
    df["cpi"] = df["cpi"].fillna(df["cpi"].median())
    return df
