"""Leakage-safe helpers for direct multi-horizon forecasting."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.models.evaluation import evaluate_regression, time_series_train_test_split
from src.transform.calendar_features import add_calendar_features

HORIZONS = (1, 6, 24, 168)
FORECAST_CALENDAR_FEATURES = [
    "is_holiday", "is_workday", "is_weekend", "dayofyear", "weekofyear",
    "quarter", "is_month_start", "is_month_end", "sin_hour", "cos_hour",
    "sin_dayofyear", "cos_dayofyear",
]


def create_direct_forecast_target(df: pd.DataFrame, target_col: str, horizon: int) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    result = df.copy()
    result[f"{target_col}_target_h{horizon}"] = pd.to_numeric(result[target_col], errors="coerce").shift(-horizon)
    return result


def create_multistep_feature_set(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    allow_lag_1: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create origin-known lags and calendar features for the forecast timestamp."""
    if "datetime" not in df.columns:
        raise KeyError("Missing datetime column")
    result = create_direct_forecast_target(df, target_col, horizon)
    origin = pd.to_datetime(result["datetime"], errors="coerce", utc=True)
    forecast_calendar = pd.DataFrame({"forecast_datetime": origin + pd.to_timedelta(horizon, unit="h")})
    forecast_calendar = add_calendar_features(forecast_calendar, "forecast_datetime")
    for column in FORECAST_CALENDAR_FEATURES:
        if column in forecast_calendar.columns:
            result[f"forecast_{column}"] = forecast_calendar[column].to_numpy()

    target = pd.to_numeric(result[target_col], errors="coerce")
    # The latest optional observation is t-1. Direct seasonal predictors are
    # therefore shifted by at least one hour and are known at issue time.
    daily_offset = 24 - horizon if horizon < 24 else 24
    weekly_offset = 168 - horizon if horizon < 168 else 168
    result[f"{target_col}_daily_direct_h{horizon}"] = target.shift(daily_offset)
    result[f"{target_col}_weekly_direct_h{horizon}"] = target.shift(weekly_offset)

    candidates = [
        f"{target_col}_lag_24", f"{target_col}_lag_168",
        f"{target_col}_rollmean_24", f"{target_col}_rollmean_168",
        f"{target_col}_daily_direct_h{horizon}", f"{target_col}_weekly_direct_h{horizon}",
        *(f"forecast_{column}" for column in FORECAST_CALENDAR_FEATURES),
    ]
    if allow_lag_1:
        candidates.insert(0, f"{target_col}_lag_1")
    return result, [column for column in candidates if column in result.columns]


def run_direct_forecast_experiment(
    model_factory: Callable,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
    datetime_col: str,
) -> dict:
    prepared = create_direct_forecast_target(df, target_col, horizon)
    target_h = f"{target_col}_target_h{horizon}"
    available = [column for column in feature_cols if column in prepared.columns and column not in {target_col, target_h}]
    data = prepared[[datetime_col, target_h, *available]].dropna(subset=[datetime_col, target_h]).copy()
    train, test = time_series_train_test_split(data, datetime_col, test_size=0.2)
    model = model_factory(train[available], train[target_h])
    prediction = model.predict(test[available])
    return {
        **evaluate_regression(test[target_h], prediction),
        "n_train": len(train),
        "n_test": len(test),
        "train_start": train[datetime_col].iloc[0],
        "train_end": train[datetime_col].iloc[-1],
        "test_start": test[datetime_col].iloc[0],
        "test_end": test[datetime_col].iloc[-1],
        "features": ",".join(available),
    }
