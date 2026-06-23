"""Generate exploratory analysis outputs for the project datasets."""

from __future__ import annotations

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import GOLD_DATA_DIR, REPORTS_DIR, SILVER_DATA_DIR, configure_logging, ensure_data_directories
from src.utils.io import save_dataframe

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)
EDA_DIR = REPORTS_DIR / "eda"


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        LOGGER.warning("Missing file: %s", path)
        return None
    return pd.read_parquet(path)


def _save_table(df: pd.DataFrame, stem: str) -> None:
    save_dataframe(df, EDA_DIR / f"{stem}.csv")


def _save_plot(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(EDA_DIR / f"{stem}.png", dpi=150)
    plt.close(fig)


def _time_series_plot(df: pd.DataFrame, datetime_col: str, value_col: str, title: str, stem: str) -> None:
    if df is None or datetime_col not in df.columns or value_col not in df.columns:
        return
    series = df[[datetime_col, value_col]].dropna().sort_values(datetime_col)
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series[datetime_col], series[value_col], linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(datetime_col)
    ax.set_ylabel(value_col)
    _save_plot(fig, stem)


def _grouped_bar_plot(df: pd.DataFrame, group_col: str, value_col: str, title: str, stem: str) -> None:
    if df is None or group_col not in df.columns or value_col not in df.columns:
        return
    grouped = df[[group_col, value_col]].dropna().groupby(group_col)[value_col].mean()
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.sort_index().plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(f"Average {value_col}")
    _save_plot(fig, stem)


def _coverage_summary(name: str, df: pd.DataFrame) -> dict[str, object]:
    datetime_col = next((c for c in ("datetime", "observation_time", "datahora", "starttime", "endtime") if c in df.columns), None)
    coverage: dict[str, object] = {"dataset": name, "rows": len(df), "columns": len(df.columns)}
    if datetime_col:
        parsed = pd.to_datetime(df[datetime_col], errors="coerce", format="mixed")
        coverage["datetime_col"] = datetime_col
        coverage["datetime_min"] = parsed.min()
        coverage["datetime_max"] = parsed.max()
        coverage["datetime_nulls"] = int(parsed.isna().sum())
    return coverage


def _missing_summary(name: str, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": name,
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100),
        }
    )


def _outlier_summary(name: str, df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=["dataset", "column", "method", "outlier_count"])
    records = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        records.append(
            {
                "dataset": name,
                "column": column,
                "method": "iqr",
                "outlier_count": int(((series < lower) | (series > upper)).sum()),
            }
        )
    return pd.DataFrame(records)


def _feature_availability(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = [
        "total_lag_1",
        "total_lag_24",
        "total_lag_168",
        "total_rollmean_24",
        "total_rollmean_168",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "total_injection_lag_1",
        "total_injection_lag_24",
        "total_injection_lag_168",
        "total_injection_rollmean_24",
        "total_injection_rollmean_168",
        "cogeracao",
        "eolica",
        "fotovoltaica",
        "hidrica",
        "outras_tecnologias",
        "rede_dist",
        "temperatura",
        "radiacao",
        "intensidadeventokm",
        "precacumulada",
        "humidade",
        "pressao",
        "hdd_18",
        "cdd_22",
        "heavy_rain_flag",
        "strong_wind_flag",
    ]
    rows = []
    for feature in candidates:
        row = {"feature": feature}
        for name, df in datasets.items():
            row[name] = feature in df.columns
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    ensure_data_directories()
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    files = {
        "gold_consumption": GOLD_DATA_DIR / "gold_consumption.parquet",
        "gold_injection": GOLD_DATA_DIR / "gold_injection.parquet",
        "gold_weather_hourly": GOLD_DATA_DIR / "gold_weather_hourly.parquet",
        "silver_production": SILVER_DATA_DIR / "e_redes_production.parquet",
    }
    for name, path in files.items():
        df = _load_parquet(path)
        if df is not None:
            datasets[name] = df

    coverage = pd.DataFrame([_coverage_summary(name, df) for name, df in datasets.items()])
    if not coverage.empty:
        _save_table(coverage, "coverage_summary")

    missing_frames = [_missing_summary(name, df) for name, df in datasets.items()]
    if missing_frames:
        _save_table(pd.concat(missing_frames, ignore_index=True), "missing_values_summary")

    stats_frames = []
    correlation_targets = {}
    for name, df in datasets.items():
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            stats = numeric.describe().T.reset_index().rename(columns={"index": "column"})
            stats.insert(0, "dataset", name)
            stats_frames.append(stats)
            correlation_targets[name] = numeric
    if stats_frames:
        _save_table(pd.concat(stats_frames, ignore_index=True), "descriptive_statistics")

    for name, numeric in correlation_targets.items():
        corr = numeric.corr(numeric_only=True)
        if not corr.empty:
            _save_table(corr, f"correlation_matrix_{name}")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index, fontsize=7)
            ax.set_title(f"Correlation matrix - {name}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _save_plot(fig, f"correlation_matrix_{name}")

    if "gold_consumption" in datasets:
        df = datasets["gold_consumption"]
        _time_series_plot(df, "datetime", "total", "Consumption time series", "consumption_time_series")
        _grouped_bar_plot(df, "hour", "total", "Hourly average consumption", "consumption_hourly_average")
        _grouped_bar_plot(df, "dayofweek", "total", "Weekday average consumption", "consumption_weekday_average")

    if "gold_injection" in datasets:
        df = datasets["gold_injection"]
        _time_series_plot(df, "datetime", "total_injection", "Injection time series", "injection_time_series")
        _grouped_bar_plot(df, "hour", "total_injection", "Hourly average injection", "injection_hourly_average")
        _grouped_bar_plot(df, "dayofweek", "total_injection", "Weekday average injection", "injection_weekday_average")

    if "silver_production" in datasets:
        df = datasets["silver_production"]
        _time_series_plot(df, "datetime", "total", "Production time series", "production_time_series")

    outlier_frames = [_outlier_summary(name, df) for name, df in datasets.items()]
    outlier_frames = [frame for frame in outlier_frames if not frame.empty]
    if outlier_frames:
        _save_table(pd.concat(outlier_frames, ignore_index=True), "outlier_summary")

    feature_availability = _feature_availability(datasets)
    _save_table(feature_availability, "feature_availability")

    LOGGER.info("EDA report generated in %s", EDA_DIR)


if __name__ == "__main__":
    main()
