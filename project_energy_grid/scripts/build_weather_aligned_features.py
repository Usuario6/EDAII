"""Aggregate IPMA station observations and quantify overlap with E-REDES history."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging
from src.utils.io import save_dataframe

LOGGER = logging.getLogger(__name__)
MIN_USABLE_OVERLAP_HOURS = 24 * 30

CONTINUOUS = [
    "temperatura", "radiacao", "intensidadeventokm", "precacumulada",
    "humidade", "pressao", "hdd_18", "cdd_22",
]
FLAGS = ["heavy_rain_flag", "strong_wind_flag"]


def main() -> None:
    configure_logging()
    source = GOLD_DATA_DIR / "gold_weather_hourly.parquet"
    if not source.exists():
        raise FileNotFoundError(f"Missing weather source: {source}")
    weather = pd.read_parquet(source)
    if "datetime" not in weather.columns:
        raise ValueError("Weather source has no datetime column")
    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce", utc=True).dt.floor("h")
    continuous = [column for column in CONTINUOUS if column in weather.columns]
    flags = [column for column in FLAGS if column in weather.columns]
    aggregations = {column: "mean" for column in continuous} | {column: "max" for column in flags}
    hourly = weather.dropna(subset=["datetime"]).groupby("datetime", as_index=False).agg(aggregations)
    save_dataframe(hourly, GOLD_DATA_DIR / "gold_weather_features_hourly.parquet")

    rows = []
    weather_times = pd.Index(hourly["datetime"])
    for dataset in ("consumption", "injection"):
        path = GOLD_DATA_DIR / f"gold_{dataset}_hourly.parquet"
        energy = pd.read_parquet(path, columns=["datetime"])
        energy_times = pd.DatetimeIndex(pd.to_datetime(energy["datetime"], errors="coerce", utc=True))
        overlap = energy_times.intersection(weather_times)
        rows.append(
            {
                "dataset": dataset,
                "energy_rows": len(energy),
                "energy_start": energy_times.min(),
                "energy_end": energy_times.max(),
                "weather_source_rows": len(weather),
                "weather_hourly_rows": len(hourly),
                "weather_start": hourly["datetime"].min(),
                "weather_end": hourly["datetime"].max(),
                "overlap_hours": len(overlap),
                "overlap_pct": len(overlap) / len(energy) * 100 if len(energy) else 0.0,
                "usable_for_historical_modelling": len(overlap) >= MIN_USABLE_OVERLAP_HOURS,
            }
        )
    output = REPORTS_DIR / "weather"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "weather_alignment_summary.csv", index=False)
    LOGGER.info("Weather aggregation rows=%s coverage=%s..%s", len(hourly), hourly.datetime.min(), hourly.datetime.max())


if __name__ == "__main__":
    main()
