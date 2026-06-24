"""Defensive, reusable feature engineering placeholders."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    result = df.copy()
    if datetime_col not in result.columns:
        return result
    values = pd.to_datetime(result[datetime_col], errors="coerce", format="mixed")
    result["hour"] = values.dt.hour
    result["dayofweek"] = values.dt.dayofweek
    result["month"] = values.dt.month
    result["is_weekend"] = values.dt.dayofweek.ge(5)
    return result


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: str,
    lags: list[int] | tuple[int, ...] = (1, 24, 168),
) -> pd.DataFrame:
    result = df.copy()
    if target_col not in result.columns or datetime_col not in result.columns:
        return result
    result[datetime_col] = pd.to_datetime(result[datetime_col], errors="coerce", format="mixed")
    result = result.sort_values(datetime_col).reset_index(drop=True)
    # Positive shifts expose only observations available before the current row.
    for lag in lags:
        if isinstance(lag, int) and lag > 0:
            result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: str,
    windows: list[int] | tuple[int, ...] = (24, 168),
) -> pd.DataFrame:
    result = df.copy()
    if target_col not in result.columns or datetime_col not in result.columns:
        return result
    result[datetime_col] = pd.to_datetime(result[datetime_col], errors="coerce", format="mixed")
    result = result.sort_values(datetime_col).reset_index(drop=True)
    numeric_target = pd.to_numeric(result[target_col], errors="coerce")
    for window in windows:
        if isinstance(window, int) and window > 0:
            # Shift first so the current target is never used to predict itself.
            result[f"{target_col}_rollmean_{window}"] = (
                numeric_target.shift(1).rolling(window=window, min_periods=1).mean()
            )
    return result


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add only weather features supported by columns currently present."""
    result = df.copy()
    temperature = next((c for c in ("temperature", "temperatura", "temp") if c in result.columns), None)
    wind = next((c for c in ("intensidadevento", "intensidadeventokm", "wind_speed") if c in result.columns), None)
    rain = next((c for c in ("precacumulada", "precipitation", "rain") if c in result.columns), None)

    if temperature:
        temp_values = pd.to_numeric(result[temperature], errors="coerce")
        result["hdd_18"] = (18 - temp_values).clip(lower=0)
        result["cdd_22"] = (temp_values - 22).clip(lower=0)
    if rain:
        result["heavy_rain_flag"] = pd.to_numeric(result[rain], errors="coerce").ge(10)
    if wind:
        result["strong_wind_flag"] = pd.to_numeric(result[wind], errors="coerce").ge(50)
    return result


def add_injection_total(df: pd.DataFrame) -> pd.DataFrame:
    """Create a total_injection column when the source components exist."""
    result = df.copy()
    components = ["cogeracao", "eolica", "fotovoltaica", "hidrica", "outras_tecnologias", "rede_dist"]
    available = [column for column in components if column in result.columns]
    if not available:
        return result
    numeric = result[available].apply(pd.to_numeric, errors="coerce")
    result["total_injection"] = numeric.sum(axis=1, min_count=1)
    return result


def add_consumption_total(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a canonical total target exists for consumption data."""
    result = df.copy()
    if "total" in result.columns:
        result["total"] = pd.to_numeric(result["total"], errors="coerce")
    return result
