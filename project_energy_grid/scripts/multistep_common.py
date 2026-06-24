"""Shared direct multi-step experiment and reporting workflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtesting_common import MODEL_FACTORIES
from src.models.evaluation import evaluate_regression, time_series_train_test_split
from src.models.multistep import HORIZONS, create_multistep_feature_set, run_direct_forecast_experiment

LOGGER = logging.getLogger(__name__)
SCENARIOS = ("with_lag1", "without_lag1", "calendar_seasonal")
WEATHER_FEATURES = [
    "temperatura", "radiacao", "intensidadeventokm", "precacumulada",
    "humidade", "pressao", "hdd_18", "cdd_22", "heavy_rain_flag", "strong_wind_flag",
]


def _train_direct_lasso(X, y):
    """Use a scale-appropriate grid for collinear direct-horizon features."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LassoCV(alphas=np.logspace(0, 5, 30), cv=TimeSeriesSplit(5), max_iter=50_000)),
        ]
    ).fit(X, y)


DIRECT_MODEL_FACTORIES = {**MODEL_FACTORIES, "lasso": _train_direct_lasso}


def _seasonal_baseline(prepared: pd.DataFrame, target: str, horizon: int) -> dict:
    target_h = f"{target}_target_h{horizon}"
    predictor = f"{target}_daily_direct_h{horizon}" if horizon < 24 else f"{target}_weekly_direct_h{horizon}"
    data = prepared[["datetime", target_h, predictor]].dropna(subset=["datetime", target_h])
    train, test = time_series_train_test_split(data, "datetime", test_size=0.2)
    return {
        **evaluate_regression(test[target_h], test[predictor]),
        "n_train": len(train), "n_test": len(test),
        "train_start": train.datetime.iloc[0], "train_end": train.datetime.iloc[-1],
        "test_start": test.datetime.iloc[0], "test_end": test.datetime.iloc[-1],
        "features": predictor,
    }


def run_multistep_dataset(data_path: Path, target: str, dataset: str, output_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path).sort_values("datetime").reset_index(drop=True)
    weather_usable = bool(df.get("weather_available", pd.Series(False, index=df.index)).sum() >= 24 * 30)
    rows = []
    for horizon in HORIZONS:
        prepared, with_features = create_multistep_feature_set(df, target, horizon, allow_lag_1=True)
        lag1 = f"{target}_lag_1"
        without_features = [column for column in with_features if column != lag1]
        calendar_seasonal = [
            column for column in without_features
            if column.startswith("forecast_") or "_direct_h" in column or column.endswith("_lag_24") or column.endswith("_lag_168")
        ]
        scenario_features = {
            "with_lag1": with_features,
            "without_lag1": without_features,
            "calendar_seasonal": calendar_seasonal,
        }
        if weather_usable:
            scenario_features["weather_enriched"] = with_features + [column for column in WEATHER_FEATURES if column in prepared.columns]

        baseline = _seasonal_baseline(prepared, target, horizon)
        rows.append({"dataset": dataset, "horizon": horizon, "scenario": "seasonal_baseline", "model": "seasonal_naive", **baseline})
        for scenario, features in scenario_features.items():
            for model_name, factory in DIRECT_MODEL_FACTORIES.items():
                metrics = run_direct_forecast_experiment(factory, prepared, target, features, horizon, "datetime")
                rows.append({"dataset": dataset, "horizon": horizon, "scenario": scenario, "model": model_name, **metrics})
                LOGGER.info("Completed %s horizon=%s scenario=%s model=%s", dataset, horizon, scenario, model_name)

    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"{dataset}_multistep_results.csv", index=False)
    summary = results.sort_values(["horizon", "rmse"]).reset_index(drop=True)
    summary.to_csv(output_dir / f"{dataset}_multistep_summary.csv", index=False)
    _plot_horizons(results, output_dir / f"{dataset}_horizon_metric_plot.png", dataset)
    refresh_shared_reports(output_dir)
    return results


def _plot_horizons(results: pd.DataFrame, output_path: Path, dataset: str) -> None:
    best = results.loc[results.groupby(["horizon", "scenario"])["rmse"].idxmin()]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for scenario, group in best.groupby("scenario"):
        ordered = group.sort_values("horizon")
        axes[0].plot(ordered.horizon, ordered.mae, marker="o", label=scenario)
        axes[1].plot(ordered.horizon, ordered.rmse, marker="o", label=scenario)
    axes[0].set(ylabel="MAE", title=f"Direct multi-step metrics: {dataset}")
    axes[1].set(xlabel="Forecast horizon (hours)", ylabel="RMSE", xticks=list(HORIZONS))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def refresh_shared_reports(output_dir: Path) -> None:
    result_paths = {name: output_dir / f"{name}_multistep_results.csv" for name in ("consumption", "injection")}
    coverage_path = output_dir.parent / "weather/weather_alignment_summary.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        usable = bool(coverage["usable_for_historical_modelling"].astype(str).str.lower().eq("true").any())
        overlap = int(coverage["overlap_hours"].max())
        source = str(coverage.get("weather_source", pd.Series(["historical weather"])).iloc[0])
        source_label = "Open-Meteo reanalysis" if source == "open_meteo_reanalysis" else source
        weather_text = (
            "# Weather coverage impact\n\n"
            f"The aligned {source_label} table contains {int(coverage['weather_hourly_rows'].max())} hourly timestamps, "
            f"with {overlap} overlapping E-REDES hours. Historical weather enrichment was "
            f"{'enabled' if usable else 'not enabled'} for modelling. No long-range filling or timestamp extrapolation was applied.\n"
        )
        (output_dir / "weather_coverage_impact.md").write_text(weather_text, encoding="utf-8")

    if not all(path.exists() for path in result_paths.values()):
        return
    combined = pd.concat([pd.read_csv(path) for path in result_paths.values()], ignore_index=True)
    with_lag = combined[combined.scenario.eq("with_lag1")]
    without_lag = combined[combined.scenario.eq("without_lag1")]
    dependency = with_lag.merge(without_lag, on=["dataset", "horizon", "model"], suffixes=("_with_lag1", "_without_lag1"))
    dependency["mae_change_pct"] = (dependency["mae_without_lag1"] / dependency["mae_with_lag1"] - 1) * 100
    dependency["rmse_change_pct"] = (dependency["rmse_without_lag1"] / dependency["rmse_with_lag1"] - 1) * 100
    dependency.to_csv(output_dir / "lag1_dependency_summary.csv", index=False)
    _write_report(combined, dependency, output_dir / "multistep_model_report.md", coverage_path)


def _write_report(results: pd.DataFrame, dependency: pd.DataFrame, output_path: Path, coverage_path: Path) -> None:
    lines = [
        "# Direct multi-step forecasting report", "",
        "## Purpose", "",
        "Direct models predict each horizon independently. This tests whether the nowcasting results remain useful when the latest lag is unavailable, without recursively feeding predictions back as observations.", "",
        "Horizons: 1, 6, 24, and 168 hours. Models: seasonal naive, Ridge, LASSO, Random Forest, and Gradient Boosting. Evaluation uses a chronological 80/20 holdout; rolling-origin evaluation is deferred because the full scenario grid would be computationally expensive.", "",
        "## Best result per horizon", "",
    ]
    best = results.loc[results.groupby(["dataset", "horizon"])["rmse"].idxmin()].sort_values(["dataset", "horizon"])
    lines.extend(["```text", best[["dataset", "horizon", "scenario", "model", "mae", "rmse", "mape", "r2"]].round(3).to_string(index=False), "```", ""])
    lines.extend(["## Lag-1 dependence", ""])
    lag_summary = dependency.groupby(["dataset", "horizon"])[["mae_change_pct", "rmse_change_pct"]].mean().reset_index()
    lines.extend(["```text", lag_summary.round(2).to_string(index=False), "```", ""])
    for dataset in ("consumption", "injection"):
        subset = best[best.dataset.eq(dataset)].set_index("horizon")
        base = float(subset.loc[1, "rmse"])
        degradation = ((subset["rmse"] / base - 1) * 100).rename("rmse_change_vs_h1_pct")
        lines.extend([f"### {dataset.title()} degradation by horizon", "", "```text", degradation.round(2).to_string(), "```", ""])
    weather_usable = False
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        weather_usable = bool(coverage["usable_for_historical_modelling"].astype(str).str.lower().eq("true").any())
    lines.extend(
        [
            "## Weather and recommendation", "",
            f"Historically aligned weather features were {'usable' if weather_usable else 'not usable'} for this experiment. No weather values were force-filled across the 2024–2025 interval.",
            "Use the strongest with-lag model for next-hour nowcasting. For operational horizons where lag 1 is unavailable, select from the without-lag or calendar/seasonal scenarios and treat the performance loss as the realistic forecast cost.",
            "Injection forecasts are credible primarily at short horizons. The 24-hour result has weak explanatory power, and the 168-hour result is weak and exploratory; neither should be presented as established operational performance.",
            "Limitations: one chronological holdout, a bounded two-year energy window, origin-time rather than target-time weather, and no holiday-locality features beyond national Portuguese holidays.", "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
