"""Validate enriched datasets and leakage-safe direct forecast outputs."""

from __future__ import annotations

import compileall

import numpy as np
import pandas as pd

from src.config import GOLD_DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from src.models.multistep import HORIZONS, create_direct_forecast_target, create_multistep_feature_set

OUTPUT = REPORTS_DIR / "multistep"
EXPECTED_REPORTS = [
    "consumption_multistep_results.csv", "injection_multistep_results.csv",
    "consumption_multistep_summary.csv", "injection_multistep_summary.csv",
    "consumption_horizon_metric_plot.png", "injection_horizon_metric_plot.png",
    "lag1_dependency_summary.csv", "weather_coverage_impact.md", "multistep_model_report.md",
]
EXPECTED_MODELS = {"seasonal_naive", "ridge", "lasso", "random_forest", "gradient_boosting"}
CALENDAR_COLUMNS = {
    "is_holiday", "is_workday", "is_weekend", "dayofyear", "weekofyear", "quarter", "season",
    "is_month_start", "is_month_end", "sin_hour", "cos_hour", "sin_dayofyear", "cos_dayofyear",
}


def main() -> None:
    failures = []
    if not compileall.compile_dir(PROJECT_ROOT / "src", quiet=1) or not compileall.compile_dir(PROJECT_ROOT / "scripts", quiet=1):
        failures.append("compileall failed")
    for dataset in ("consumption", "injection"):
        enriched_path = GOLD_DATA_DIR / f"gold_{dataset}_enriched.parquet"
        if not enriched_path.exists() or enriched_path.stat().st_size == 0:
            failures.append(f"missing enriched gold: {enriched_path.name}")
            continue
        target = "total" if dataset == "consumption" else "total_injection"
        enriched = pd.read_parquet(enriched_path)
        missing_calendar = CALENDAR_COLUMNS - set(enriched.columns)
        if missing_calendar:
            failures.append(f"missing calendar features in {dataset}: {sorted(missing_calendar)}")
        new_year = enriched[pd.to_datetime(enriched["datetime"], utc=True).eq(pd.Timestamp("2025-01-01", tz="UTC"))]
        if new_year.empty or not bool(new_year.iloc[0]["is_holiday"]) or bool(new_year.iloc[0]["is_workday"]):
            failures.append(f"Portugal holiday check failed in {dataset}")
        for horizon in HORIZONS:
            direct = create_direct_forecast_target(enriched, target, horizon)
            expected = enriched[target].shift(-horizon)
            if not np.allclose(direct[f"{target}_target_h{horizon}"], expected, equal_nan=True):
                failures.append(f"incorrect shifted target: {dataset} h={horizon}")
            prepared, features = create_multistep_feature_set(enriched, target, horizon, allow_lag_1=False)
            if target in features or any("target_h" in feature for feature in features):
                failures.append(f"target leakage in features: {dataset} h={horizon}")
            weekly_offset = 168 - horizon if horizon < 168 else 168
            expected_weekly = enriched[target].shift(weekly_offset)
            if not np.allclose(prepared[f"{target}_weekly_direct_h{horizon}"], expected_weekly, equal_nan=True):
                failures.append(f"future seasonal value leakage: {dataset} h={horizon}")
            if horizon == 24:
                christmas_origin = prepared[pd.to_datetime(prepared["datetime"], utc=True).eq(pd.Timestamp("2024-12-24", tz="UTC"))]
                if christmas_origin.empty or not bool(christmas_origin.iloc[0]["forecast_is_holiday"]):
                    failures.append(f"forecast calendar alignment failed in {dataset}")

        results_path = OUTPUT / f"{dataset}_multistep_results.csv"
        if not results_path.exists() or results_path.stat().st_size == 0:
            failures.append(f"missing results: {dataset}")
            continue
        results = pd.read_csv(results_path)
        if results.empty or set(results["horizon"]) != set(HORIZONS):
            failures.append(f"missing horizons: {dataset}")
        if set(results["model"]) != EXPECTED_MODELS:
            failures.append(f"missing models: {dataset}")
        train_end = pd.to_datetime(results["train_end"], utc=True)
        test_start = pd.to_datetime(results["test_start"], utc=True)
        if not (train_end < test_start).all():
            failures.append(f"non-chronological split: {dataset}")
        if results[["mae", "rmse", "mape", "r2"]].isna().any().any():
            failures.append(f"missing metrics: {dataset}")
        forbidden_components = {"cogeracao", "eolica", "fotovoltaica", "hidrica", "outras_tecnologias", "rede_dist"}
        for feature_text in results["features"].fillna(""):
            tokens = set(feature_text.split(","))
            if target in tokens or forbidden_components.intersection(tokens):
                failures.append(f"target/component leakage in recorded features: {dataset}")
                break

    for filename in EXPECTED_REPORTS:
        path = OUTPUT / filename
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty report: {filename}")
    for path in [
        GOLD_DATA_DIR / "gold_weather_features_hourly.parquet",
        REPORTS_DIR / "weather/weather_alignment_summary.csv",
    ]:
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing weather alignment output: {path.name}")
    source_paths = list((PROJECT_ROOT / "scripts").glob("*.py")) + list((PROJECT_ROOT / "src").rglob("*.py"))
    source = "\n".join(path.read_text() for path in source_paths)
    forbidden = "train" + "_test_split"
    if f"from sklearn.model_selection import {forbidden}" in source:
        failures.append("random sklearn train/test split import found")

    if failures:
        raise RuntimeError("Multi-step validation failed:\n- " + "\n- ".join(failures))
    print(f"Multi-step validation passed: enriched gold, {len(EXPECTED_REPORTS)} reports, four horizons, all models, chronological split, leakage checks, compileall")


if __name__ == "__main__":
    main()
