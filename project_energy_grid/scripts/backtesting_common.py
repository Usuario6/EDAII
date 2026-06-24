"""Shared rolling-origin execution and report helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.backtesting import rolling_origin_splits, run_backtest, summarize_backtest_results
from src.models.evaluation import evaluate_regression
from src.models.regularized import train_lasso_model, train_ridge_model
from src.utils.visualization import CONSUMPTION_HOURLY_SOURCE, INJECTION_HOURLY_SOURCE, save_figure_with_source

LOGGER = logging.getLogger(__name__)


def _train_backtest_random_forest(X, y):
    # Fixed, moderate complexity keeps weekly rolling evaluation reproducible and bounded.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=75, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]
    ).fit(X, y)


def _train_backtest_gradient_boosting(X, y):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
        ]
    ).fit(X, y)


MODEL_FACTORIES = {
    "ridge": train_ridge_model,
    "lasso": train_lasso_model,
    "random_forest": _train_backtest_random_forest,
    "gradient_boosting": _train_backtest_gradient_boosting,
}


def choose_split_sizes(row_count: int) -> tuple[int, int, int]:
    """Use a 60% initial window and weekly folds, adapting for short datasets."""
    if row_count < 48:
        raise ValueError("At least 48 hourly rows are required for backtesting")
    initial = max(24, int(row_count * 0.60))
    available = row_count - initial
    test = min(24 * 7, max(12, available // 3))
    step = test
    if initial + test > row_count:
        initial = row_count - test
    return initial, test, step


def _naive_results(df: pd.DataFrame, target: str, lag: int, splits) -> pd.DataFrame:
    rows = []
    lag_column = f"{target}_lag_{lag}"
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        train, test = df.iloc[train_indices], df.iloc[test_indices]
        valid = test[target].notna() & test[lag_column].notna()
        rows.append(
            {
                "fold": fold,
                "train_start": train.datetime.iloc[0],
                "train_end": train.datetime.iloc[-1],
                "test_start": test.datetime.iloc[0],
                "test_end": test.datetime.iloc[-1],
                "n_train": len(train),
                "n_test": int(valid.sum()),
                "test_window_size": len(test),
                **evaluate_regression(test[target], test[lag_column]),
                "model": f"seasonal_naive_lag_{lag}",
            }
        )
    return pd.DataFrame(rows)


def run_dataset_backtest(
    data_path: Path,
    target: str,
    features: list[str],
    output_dir: Path,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, int, int]]:
    df = pd.read_parquet(data_path).sort_values("datetime").reset_index(drop=True)
    available = [column for column in features if column in df.columns and column != target]
    required = ["datetime", target, f"{target}_lag_24", *available]
    data = df[list(dict.fromkeys(required))].reset_index(drop=True)
    initial, test_size, step = choose_split_sizes(len(data))
    splits = rolling_origin_splits(data, "datetime", initial, test_size, step)
    if not splits:
        raise ValueError("No rolling-origin folds could be created")
    LOGGER.info(
        "%s backtest rows=%s initial=%s test=%s step=%s folds=%s",
        prefix, len(data), initial, test_size, step, len(splits),
    )

    frames = [_naive_results(data, target, 24, splits)]
    for model_name, factory in MODEL_FACTORIES.items():
        result = run_backtest(factory, data, target, available, "datetime", splits)
        result["model"] = model_name
        frames.append(result)
        LOGGER.info("Completed %s model=%s folds=%s", prefix, model_name, len(result))
    results = pd.concat(frames, ignore_index=True)
    summary = summarize_backtest_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"{prefix}_backtest_results.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_backtest_summary.csv", index=False)

    best_by_fold = results.loc[results.groupby("fold")["rmse"].idxmin()].sort_values("fold")
    best_by_fold.to_csv(output_dir / f"{prefix}_best_model_by_fold.csv", index=False)
    _plot_metrics(results, output_dir / f"{prefix}_backtest_metrics_plot.png", prefix)
    return results, summary, (initial, test_size, step)


def _plot_metrics(results: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for model, group in results.groupby("model"):
        ordered = group.sort_values("fold")
        axes[0].plot(ordered["fold"], ordered["mae"], marker="o", markersize=2, label=model)
        axes[1].plot(ordered["fold"], ordered["rmse"], marker="o", markersize=2, label=model)
    axes[0].set_ylabel("MAE")
    axes[1].set(ylabel="RMSE", xlabel="Rolling-origin fold")
    axes[0].set_title(f"Rolling-origin metrics: {title}")
    axes[0].legend(ncol=3, fontsize=8)
    source_text = CONSUMPTION_HOURLY_SOURCE if title == "consumption" else INJECTION_HOURLY_SOURCE
    save_figure_with_source(fig, output_path, source_text)
    plt.close(fig)
