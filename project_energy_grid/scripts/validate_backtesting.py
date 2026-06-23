"""Validate rolling-origin outputs and guard against leakage or random splits."""

from __future__ import annotations

import compileall

import numpy as np
import pandas as pd

from scripts.train_baseline_consumption import FEATURES as CONSUMPTION_FEATURES
from scripts.train_baseline_injection import FEATURES as INJECTION_FEATURES
from src.config import GOLD_DATA_DIR, PROJECT_ROOT, REPORTS_DIR

OUTPUT_DIR = REPORTS_DIR / "backtesting"
EXPECTED_MODELS = {"seasonal_naive_lag_24", "ridge", "lasso", "random_forest", "gradient_boosting"}
EXPECTED = [
    "consumption_backtest_results.csv", "injection_backtest_results.csv",
    "consumption_backtest_summary.csv", "injection_backtest_summary.csv",
    "consumption_backtest_metrics_plot.png", "injection_backtest_metrics_plot.png",
    "consumption_best_model_by_fold.csv", "injection_best_model_by_fold.csv",
    "robustness_summary.csv", "error_by_hour_consumption.csv", "error_by_hour_injection.csv",
    "error_by_weekday_consumption.csv", "error_by_weekday_injection.csv",
    "model_selection_report.md",
]


def main() -> None:
    failures = []
    if not compileall.compile_dir(PROJECT_ROOT / "src", quiet=1) or not compileall.compile_dir(PROJECT_ROOT / "scripts", quiet=1):
        failures.append("compileall failed")
    for filename in EXPECTED:
        path = OUTPUT_DIR / filename
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty: reports/backtesting/{filename}")

    for dataset in ("consumption", "injection"):
        path = OUTPUT_DIR / f"{dataset}_backtest_results.csv"
        if not path.exists():
            continue
        results = pd.read_csv(path)
        if results.empty:
            failures.append(f"empty results: {dataset}")
            continue
        if set(results["model"]) != EXPECTED_MODELS:
            failures.append(f"models missing for {dataset}: {EXPECTED_MODELS - set(results['model'])}")
        train_end = pd.to_datetime(results["train_end"], utc=True)
        test_start = pd.to_datetime(results["test_start"], utc=True)
        if not (train_end < test_start).all():
            failures.append(f"non-chronological fold in {dataset}")
        if results[["mae", "rmse", "mape", "r2"]].isna().any().any():
            failures.append(f"missing metrics in {dataset}")

        target = "total" if dataset == "consumption" else "total_injection"
        hourly = pd.read_parquet(GOLD_DATA_DIR / f"gold_{dataset}_hourly.parquet")
        expected_lag = hourly[target].shift(24)
        if not np.allclose(hourly[f"{target}_lag_24"], expected_lag, equal_nan=True):
            failures.append(f"lag-24 alignment/leakage check failed in {dataset}")
        expected_rolling = hourly[target].shift(1).rolling(24, min_periods=1).mean()
        if not np.allclose(hourly[f"{target}_rollmean_24"], expected_rolling, equal_nan=True):
            failures.append(f"rolling feature leakage check failed in {dataset}")

    if "total" in CONSUMPTION_FEATURES:
        failures.append("consumption target leakage")
    forbidden_injection = {"total_injection", "cogeracao", "eolica", "fotovoltaica", "hidrica", "outras_tecnologias", "rede_dist"}
    leakage = forbidden_injection.intersection(INJECTION_FEATURES)
    if leakage:
        failures.append(f"injection target/component leakage: {sorted(leakage)}")
    source_paths = list((PROJECT_ROOT / "scripts").glob("*.py")) + list((PROJECT_ROOT / "src").rglob("*.py"))
    source = "\n".join(path.read_text() for path in source_paths)
    forbidden = "train" + "_test_split"
    if f"from sklearn.model_selection import {forbidden}" in source:
        failures.append("random sklearn split import found")

    if failures:
        raise RuntimeError("Backtesting validation failed:\n- " + "\n- ".join(failures))
    print(f"Backtesting validation passed: {len(EXPECTED)} artifacts, all models, chronological folds, leakage checks, compileall")


if __name__ == "__main__":
    main()
