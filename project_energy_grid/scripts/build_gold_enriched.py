"""Add known calendar features and optional aligned weather to hourly gold data."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import GOLD_DATA_DIR, configure_logging
from src.transform.calendar_features import add_calendar_features
from src.utils.io import save_dataframe

LOGGER = logging.getLogger(__name__)


def _build(dataset: str) -> None:
    source = GOLD_DATA_DIR / f"gold_{dataset}_hourly.parquet"
    if not source.exists():
        LOGGER.warning("Missing hourly dataset: %s", source)
        return
    result = pd.read_parquet(source)
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce", utc=True)
    result = add_calendar_features(result, "datetime")

    weather_path = GOLD_DATA_DIR / "gold_weather_features_hourly.parquet"
    weather_columns: list[str] = []
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce", utc=True)
        weather_columns = [column for column in weather.columns if column != "datetime"]
        result = result.merge(weather, on="datetime", how="left", validate="one_to_one")
    result["weather_available"] = result[weather_columns].notna().any(axis=1) if weather_columns else False
    output = GOLD_DATA_DIR / f"gold_{dataset}_enriched.parquet"
    save_dataframe(result, output)
    LOGGER.info(
        "Saved %s rows=%s weather_available=%s",
        output.name, len(result), int(result["weather_available"].sum()),
    )


def main() -> None:
    configure_logging()
    _build("consumption")
    _build("injection")


if __name__ == "__main__":
    main()
