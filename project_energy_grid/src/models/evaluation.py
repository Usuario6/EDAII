"""Regression evaluation helpers designed for chronological energy data."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if actual.size == 0:
        return {name: np.nan for name in ("mae", "rmse", "mape", "r2")}
    # Ignore values that are effectively zero relative to the target scale.
    scale = np.nanmedian(np.abs(actual)) if actual.size else 0.0
    near_zero = max(np.finfo(float).eps, scale * 1e-8)
    nonzero = np.abs(actual) > near_zero
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else np.nan
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(mape),
        "r2": float(r2_score(actual, predicted)) if actual.size > 1 else np.nan,
    }


def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """Return model metrics ranked by RMSE; dictionaries may include model metadata."""
    if not metrics_list:
        return pd.DataFrame(columns=["model", "mae", "rmse", "mape", "r2"])
    return pd.DataFrame(metrics_list).sort_values("rmse", na_position="last").reset_index(drop=True)


def time_series_train_test_split(df: pd.DataFrame, datetime_col: str, test_size: float = 0.2):
    if datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column: {datetime_col}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    ordered = df.copy()
    ordered[datetime_col] = pd.to_datetime(ordered[datetime_col], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)
    split = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_size))))
    if len(ordered) < 2:
        raise ValueError("At least two chronological observations are required")
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_func: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence must be between 0 and 1")
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if actual.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, actual.size, actual.size)
        values.append(float(metric_func(actual[indices], predicted[indices])))
    alpha = (1 - confidence) / 2
    return tuple(float(value) for value in np.quantile(values, [alpha, 1 - alpha]))
