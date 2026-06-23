"""Aggregate quarter-hourly E-REDES observations into leakage-safe hourly gold data."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import GOLD_DATA_DIR, SILVER_DATA_DIR, configure_logging, ensure_data_directories
from src.transform.features import add_lag_features, add_rolling_features, add_time_features
from src.utils.io import save_dataframe

LOGGER = logging.getLogger(__name__)


def _build(source_name: str, output_name: str, target: str, value_cols: list[str]) -> None:
    path = SILVER_DATA_DIR / source_name
    if not path.exists():
        LOGGER.warning("Missing source file: %s", path)
        return
    df = pd.read_parquet(path)
    if "datetime" not in df.columns or target not in df.columns:
        LOGGER.warning("Skipping %s: datetime or %s is missing", source_name, target)
        return
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    available = [column for column in value_cols if column in df.columns]
    for column in available:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # Values are interval-level power/energy indicators; hourly means preserve their scale.
    hourly = (
        df.dropna(subset=["datetime"])
        .set_index("datetime")[available]
        .resample("1h")
        .mean()
        .dropna(subset=[target])
        .reset_index()
    )
    hourly = add_time_features(hourly, "datetime")
    hourly = add_lag_features(hourly, target, "datetime")
    hourly = add_rolling_features(hourly, target, "datetime")
    save_dataframe(hourly, GOLD_DATA_DIR / output_name)
    LOGGER.info("Saved %s: rows=%d coverage=%s to %s", output_name, len(hourly), hourly.datetime.min(), hourly.datetime.max())


def main() -> None:
    configure_logging()
    ensure_data_directories()
    _build(
        "e_redes_consumption.parquet",
        "gold_consumption_hourly.parquet",
        "total",
        ["total", "bt", "mt", "at", "mat"],
    )
    injection_source = SILVER_DATA_DIR / "e_redes_injection.parquet"
    if injection_source.exists():
        raw = pd.read_parquet(injection_source)
        components = ["cogeracao", "eolica", "fotovoltaica", "hidrica", "outras_tecnologias", "rede_dist"]
        available = [column for column in components if column in raw.columns]
        if available and "total_injection" not in raw.columns:
            raw["total_injection"] = raw[available].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            temporary = SILVER_DATA_DIR / ".e_redes_injection_hourly_source.parquet"
            raw.to_parquet(temporary, index=False)
            try:
                _build(temporary.name, "gold_injection_hourly.parquet", "total_injection", ["total_injection", *components])
            finally:
                temporary.unlink(missing_ok=True)
        else:
            _build("e_redes_injection.parquet", "gold_injection_hourly.parquet", "total_injection", ["total_injection", *components])
    else:
        LOGGER.warning("Missing source file: %s", injection_source)


if __name__ == "__main__":
    main()
