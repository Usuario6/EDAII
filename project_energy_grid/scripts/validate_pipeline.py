"""Validate the silver and gold pipeline outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import GOLD_DATA_DIR, SILVER_DATA_DIR, configure_logging, ensure_data_directories

LOGGER = logging.getLogger(__name__)


def _describe_numeric(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    numeric = df.select_dtypes(include="number")
    for column in numeric.columns:
        desc = numeric[column].describe()
        summary[column] = {
            "count": float(desc.get("count", 0)),
            "mean": float(desc.get("mean", 0)),
            "std": float(desc.get("std", 0)) if pd.notna(desc.get("std", pd.NA)) else float("nan"),
            "min": float(desc.get("min", 0)),
            "max": float(desc.get("max", 0)),
        }
    return summary


def _inspect_file(path: Path) -> None:
    if not path.exists():
        print(f"{path.name}: MISSING")
        return
    df = pd.read_parquet(path)
    print(f"\n{path.name}: shape={df.shape}")
    print(f"columns={df.columns.tolist()}")
    if "datetime" in df.columns:
        print(f"datetime_nulls={int(df['datetime'].isna().sum())}")
    print(f"duplicate_rows={int(df.duplicated().sum())}")
    print(f"missing_values={df.isna().sum().to_dict()}")
    print(f"numeric_summary={_describe_numeric(df)}")


def main() -> None:
    configure_logging()
    ensure_data_directories()
    for path in [
        SILVER_DATA_DIR / "e_redes_consumption.parquet",
        SILVER_DATA_DIR / "e_redes_injection.parquet",
        SILVER_DATA_DIR / "e_redes_production.parquet",
        SILVER_DATA_DIR / "ipma_observations.parquet",
        SILVER_DATA_DIR / "ipma_warnings.parquet",
        SILVER_DATA_DIR / "ipma_daily_forecast.parquet",
        GOLD_DATA_DIR / "gold_consumption.parquet",
        GOLD_DATA_DIR / "gold_injection.parquet",
        GOLD_DATA_DIR / "gold_weather_hourly.parquet",
    ]:
        _inspect_file(path)


if __name__ == "__main__":
    main()
