"""
Economic Trend Analyzer – ARIMA-based inflation / CPI forecasting.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def adf_test(series):
    result = adfuller(series.dropna())
    return {"adf_stat": round(result[0], 4),
            "p_value":  round(result[1], 4),
            "stationary": result[1] < 0.05}


def train_arima(df: pd.DataFrame, order=(2, 1, 2), forecast_steps=6):
    series = df["cpi"]

    model  = ARIMA(series, order=order)
    fitted = model.fit()

    # In-sample predictions
    in_sample = fitted.fittedvalues

    # Out-of-sample forecast
    forecast_result = fitted.get_forecast(steps=forecast_steps)
    forecast_mean   = forecast_result.predicted_mean
    conf_int        = forecast_result.conf_int()

    last_date     = series.index[-1]
    forecast_idx  = pd.date_range(last_date + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq="MS")
    forecast_mean.index = forecast_idx
    conf_int.index      = forecast_idx

    metrics = {
        "aic":  round(fitted.aic, 2),
        "bic":  round(fitted.bic, 2),
        "mae":  round(np.mean(np.abs(series - in_sample)), 4),
        "rmse": round(np.sqrt(np.mean((series - in_sample)**2)), 4),
        "adf":  adf_test(series),
        "order": order,
    }

    results = {
        "actual":      series,
        "fitted":      in_sample,
        "forecast":    forecast_mean,
        "conf_lower":  conf_int.iloc[:, 0],
        "conf_upper":  conf_int.iloc[:, 1],
    }

    joblib.dump({"fitted_model": fitted, "order": order}, MODEL_DIR / "arima_model.pkl")
    print("ARIMA model saved ✓")
    return metrics, results


def forecast_inflation(steps=6):
    data = joblib.load(MODEL_DIR / "arima_model.pkl")
    fitted = data["fitted_model"]

    fc  = fitted.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci   = fc.conf_int()

    return {
        "forecast_values": mean.round(2).tolist(),
        "lower_bound":     ci.iloc[:, 0].round(2).tolist(),
        "upper_bound":     ci.iloc[:, 1].round(2).tolist(),
        "dates":           [str(d.date()) for d in mean.index],
    }
