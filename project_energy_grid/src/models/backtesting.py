"""Expanding-window rolling-origin backtesting utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from src.models.evaluation import evaluate_regression


Split = tuple[list[int], list[int]]


def rolling_origin_splits(
    df: pd.DataFrame,
    datetime_col: str,
    initial_train_size: int,
    test_size: int,
    step_size: int,
) -> list[Split]:
    """Return expanding chronological train/test positional indices."""
    if datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column: {datetime_col}")
    if min(initial_train_size, test_size, step_size) < 1:
        raise ValueError("Split sizes must be positive integers")
    if initial_train_size + test_size > len(df):
        raise ValueError("Data is too short for the requested initial train and test sizes")
    timestamps = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Data must have complete, monotonically increasing timestamps")

    splits: list[Split] = []
    train_end = initial_train_size
    while train_end + test_size <= len(df):
        splits.append((list(range(train_end)), list(range(train_end, train_end + test_size))))
        train_end += step_size
    return splits


def run_backtest(
    model_factory: Callable,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    datetime_col: str,
    splits: Iterable[Split],
) -> pd.DataFrame:
    """Fit a fresh estimator at every origin and return fold-level metrics."""
    available = [column for column in feature_cols if column in df.columns and column != target_col]
    if target_col not in df.columns or not available:
        raise ValueError("Target or model features are unavailable")
    rows = []
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        train_window = df.iloc[train_indices]
        test = df.iloc[test_indices]
        train = train_window.dropna(subset=[target_col])
        model = model_factory(train[available], train[target_col])
        prediction = model.predict(test[available])
        valid_test = test[target_col].notna()
        rows.append(
            {
                "fold": fold,
                "train_start": train_window[datetime_col].iloc[0],
                "train_end": train_window[datetime_col].iloc[-1],
                "test_start": test[datetime_col].iloc[0],
                "test_end": test[datetime_col].iloc[-1],
                "n_train": len(train),
                "n_test": int(valid_test.sum()),
                "test_window_size": len(test),
                **evaluate_regression(test[target_col], prediction),
            }
        )
    return pd.DataFrame(rows)


def summarize_backtest_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean and sample standard deviation for every tested model."""
    required = {"model", "mae", "rmse", "mape", "r2"}
    if results_df.empty or not required.issubset(results_df.columns):
        return pd.DataFrame()
    summary = results_df.groupby("model")[["mae", "rmse", "mape", "r2"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    counts = results_df.groupby("model").size().rename("folds").reset_index()
    return summary.merge(counts, on="model").sort_values("rmse_mean").reset_index(drop=True)
