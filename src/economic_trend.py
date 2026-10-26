"""
Economic Trend Analyzer – ARIMA-based inflation / CPI forecasting.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.config import MODEL_DIR
from src.logger import get_logger
from src.model_store import load_artifact

log = get_logger(__name__)


def adf_test(series):
    result = adfuller(series.dropna())
    return {"adf_stat": round(result[0], 4),
            "p_value":  round(result[1], 4),
            "stationary": result[1] < 0.05}


def backtest_arima(series, order, test_months=12):
    """Fit on everything except the last `test_months`, forecast them, and
    compare with what actually happened. In-sample error only shows how well
    the model fits data it has already seen."""
    train, test = series[:-test_months], series[-test_months:]
    forecast = ARIMA(train, order=order).fit().forecast(steps=test_months)
    errors = test.values - forecast.values
    naive_errors = test.values - train.iloc[-1]       # "CPI stays the same"
    return {
        "test_months": test_months,
        "mae":       round(float(np.mean(np.abs(errors))), 4),
        "rmse":      round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "naive_mae": round(float(np.mean(np.abs(naive_errors))), 4),
    }


CANDIDATE_ORDERS = [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]


def select_order(series, candidates=CANDIDATE_ORDERS, test_months=12):
    """Pick the (p, d, q) with the lowest MAE on a validation year that sits
    just before the final test year, so the reported backtest stays unseen."""
    history = series[:-test_months]
    results = {}
    for order in candidates:
        try:
            results[order] = backtest_arima(history, order, test_months)["mae"]
        except Exception as e:                 # some orders fail to converge
            log.warning("ARIMA%s skipped: %s", order, e)
    best = min(results, key=results.get)
    log.info("Best ARIMA order %s (validation MAE %.4f, %d candidates)",
             best, results[best], len(results))
    return best


def train_arima(df: pd.DataFrame, order=None, forecast_steps=6):
    series = df["cpi"]
    if order is None:
        order = select_order(series)

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
        "backtest": backtest_arima(series, order),
    }

    results = {
        "actual":      series,
        "fitted":      in_sample,
        "forecast":    forecast_mean,
        "conf_lower":  conf_int.iloc[:, 0],
        "conf_upper":  conf_int.iloc[:, 1],
    }

    joblib.dump({"fitted_model": fitted, "order": order}, MODEL_DIR / "arima_model.pkl")
    log.info("ARIMA model saved to %s", MODEL_DIR)
    return metrics, results


def forecast_inflation(steps=6):
    data = load_artifact("arima_model.pkl")
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
