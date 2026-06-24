"""Evaluate lag dependence, seasonal baselines, and temporal error patterns."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.backtesting_common import MODEL_FACTORIES
from scripts.train_baseline_consumption import FEATURES as CONSUMPTION_FEATURES
from scripts.train_baseline_injection import FEATURES as INJECTION_FEATURES
from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging
from src.models.evaluation import evaluate_regression, time_series_train_test_split

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = REPORTS_DIR / "backtesting"
SPECS = {
    "consumption": (GOLD_DATA_DIR / "gold_consumption_hourly.parquet", "total", CONSUMPTION_FEATURES),
    "injection": (GOLD_DATA_DIR / "gold_injection_hourly.parquet", "total_injection", INJECTION_FEATURES),
}


def _fit_predict(factory, train, test, target, features):
    model = factory(train[features], train[target])
    return model, model.predict(test[features])


def _dataset_checks(name: str, path, target: str, features: list[str]) -> list[dict]:
    df = pd.read_parquet(path).sort_values("datetime").reset_index(drop=True)
    available = [column for column in features if column in df.columns and column != target]
    required = list(dict.fromkeys(["datetime", target, f"{target}_lag_24", f"{target}_lag_168", *available]))
    data = df[required].dropna().reset_index(drop=True)
    train, test = time_series_train_test_split(data, "datetime", test_size=0.2)
    rows = []

    for model_name, factory in MODEL_FACTORIES.items():
        _, with_prediction = _fit_predict(factory, train, test, target, available)
        without_features = [column for column in available if column != f"{target}_lag_1"]
        _, without_prediction = _fit_predict(factory, train, test, target, without_features)
        with_metrics = evaluate_regression(test[target], with_prediction)
        without_metrics = evaluate_regression(test[target], without_prediction)
        rows.extend(
            [
                {"dataset": name, "check": "lag_1_dominance", "model": model_name, "configuration": "with_lag_1", **with_metrics, "mae_change_pct": 0.0},
                {"dataset": name, "check": "lag_1_dominance", "model": model_name, "configuration": "without_lag_1", **without_metrics, "mae_change_pct": (without_metrics["mae"] / with_metrics["mae"] - 1) * 100},
            ]
        )

    for lag in (24, 168):
        metrics = evaluate_regression(test[target], test[f"{target}_lag_{lag}"])
        rows.append({"dataset": name, "check": "seasonal_baseline", "model": f"seasonal_naive_lag_{lag}", "configuration": f"lag_{lag}", **metrics, "mae_change_pct": pd.NA})

    summary = pd.read_csv(OUTPUT_DIR / f"{name}_backtest_summary.csv")
    best_name = summary.iloc[0]["model"]
    if best_name.startswith("seasonal_naive_lag_"):
        lag = int(best_name.rsplit("_", 1)[1])
        prediction = test[f"{target}_lag_{lag}"].to_numpy()
    else:
        _, prediction = _fit_predict(MODEL_FACTORIES[best_name], train, test, target, available)
    errors = test[["datetime"]].copy()
    errors["absolute_error"] = np.abs(test[target].to_numpy() - prediction)
    errors["hour"] = pd.to_datetime(errors["datetime"], utc=True).dt.hour
    errors["dayofweek"] = pd.to_datetime(errors["datetime"], utc=True).dt.dayofweek
    errors.groupby("hour", as_index=False)["absolute_error"].agg(mae="mean", observations="size").to_csv(OUTPUT_DIR / f"error_by_hour_{name}.csv", index=False)
    errors.groupby("dayofweek", as_index=False)["absolute_error"].agg(mae="mean", observations="size").to_csv(OUTPUT_DIR / f"error_by_weekday_{name}.csv", index=False)
    return rows


def _write_model_selection_report() -> None:
    lines = ["# Rolling-origin model selection report", ""]
    for name, (path, target, _) in SPECS.items():
        data = pd.read_parquet(path)
        results = pd.read_csv(OUTPUT_DIR / f"{name}_backtest_results.csv")
        summary = pd.read_csv(OUTPUT_DIR / f"{name}_backtest_summary.csv")
        lines.extend(
            [
                f"## {name.title()}", "",
                f"- Coverage: {data['datetime'].min()} to {data['datetime'].max()}",
                f"- Hourly rows: {len(data):,}",
                f"- Models: {', '.join(summary['model'])}",
                f"- Rolling origins: {results['fold'].nunique()} folds; initial usable train rows {int(results['n_train'].min()):,}; test window {int(results['test_window_size'].median()):,} hours; weekly step.",
                f"- Recommended candidate by average RMSE: **{summary.iloc[0]['model']}**.", "",
                "Average and standard deviation by model:", "",
                "```text", summary.round(3).to_string(index=False), "```", "",
            ]
        )
    lines.extend(
        [
            "## Interpretation and limitations", "",
            "Lag-1 dominance is reported explicitly: strong degradation without it indicates that short-term persistence drives much of the score.",
            "The evaluation uses an expanding training window and non-overlapping chronological test windows; no random split is used.",
            "This rolling-origin baseline excludes weather by design. IPMA provides current/recent operational context; separate Open-Meteo reanalysis is aligned to 2024-2025 for weather-enriched experiments.",
            "Results cover a bounded historical interval and do not prove performance during unseen structural changes.", "",
        ]
    )
    (OUTPUT_DIR / "model_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in SPECS.items():
        rows.extend(_dataset_checks(name, *spec))
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
    _write_model_selection_report()
    LOGGER.info("Robustness checks and model selection report completed")


if __name__ == "__main__":
    main()
