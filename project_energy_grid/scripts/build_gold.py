"""Build initial gold datasets for modelling."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import GOLD_DATA_DIR, SILVER_DATA_DIR, configure_logging, ensure_data_directories
from src.transform.features import (
    add_consumption_total,
    add_injection_total,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    add_weather_features,
)
from src.utils.io import save_dataframe

LOGGER = logging.getLogger(__name__)


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        LOGGER.warning("Missing silver file: %s", path.name)
        return None
    return pd.read_parquet(path)


def _canonicalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "datetime" in result.columns:
        result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce", format="mixed")
        return result
    for candidate in ("datahora", "observation_time", "starttime", "endtime"):
        if candidate in result.columns:
            parsed = pd.to_datetime(result[candidate], errors="coerce", format="mixed")
            if parsed.notna().any():
                result["datetime"] = parsed
                return result
    if {"date", "time"}.issubset(result.columns):
        parsed = pd.to_datetime(
            result["date"].astype(str) + " " + result["time"].astype(str),
            errors="coerce",
            format="mixed",
        )
        if parsed.notna().any():
            result["datetime"] = parsed
    return result


def _build_gold_consumption() -> None:
    df = _read_parquet(SILVER_DATA_DIR / "e_redes_consumption.parquet")
    if df is None:
        return
    df = _canonicalize_datetime(df)
    df = add_consumption_total(df)
    if "datetime" in df.columns:
        df = add_time_features(df, "datetime")
        df = add_lag_features(df, "total", "datetime")
        df = add_rolling_features(df, "total", "datetime")
    wanted = [
        "datetime", "total", "bt", "mt", "at", "mat",
        "hour", "dayofweek", "month", "is_weekend",
        "total_lag_1", "total_lag_24", "total_lag_168",
        "total_rollmean_24", "total_rollmean_168",
    ]
    available = [column for column in wanted if column in df.columns]
    save_dataframe(df[available], GOLD_DATA_DIR / "gold_consumption.parquet")
    LOGGER.info("Saved gold_consumption with shape %s", df[available].shape)


def _build_gold_injection() -> None:
    df = _read_parquet(SILVER_DATA_DIR / "e_redes_injection.parquet")
    if df is None:
        return
    df = _canonicalize_datetime(df)
    df = add_injection_total(df)
    if "datetime" in df.columns:
        df = add_time_features(df, "datetime")
        df = add_lag_features(df, "total_injection", "datetime")
        df = add_rolling_features(df, "total_injection", "datetime")
    wanted = [
        "datetime", "total_injection", "cogeracao", "eolica", "fotovoltaica",
        "hidrica", "outras_tecnologias", "rede_dist", "hour", "dayofweek",
        "month", "is_weekend", "total_injection_lag_1", "total_injection_lag_24",
        "total_injection_lag_168", "total_injection_rollmean_24", "total_injection_rollmean_168",
    ]
    available = [column for column in wanted if column in df.columns]
    save_dataframe(df[available], GOLD_DATA_DIR / "gold_injection.parquet")
    LOGGER.info("Saved gold_injection with shape %s", df[available].shape)


def _build_gold_weather() -> None:
    df = _read_parquet(SILVER_DATA_DIR / "ipma_observations.parquet")
    if df is None:
        return
    df = _canonicalize_datetime(df)
    df = add_weather_features(df)
    if "datetime" not in df.columns:
        LOGGER.warning("Could not create gold_weather_hourly: no alignable datetime column")
        return
    available = [
        column
        for column in [
            "datetime",
            "station_id",
            "temperatura",
            "radiacao",
            "precacumulada",
            "intensidadeventokm",
            "humidade",
            "pressao",
            "hdd_18",
            "cdd_22",
            "heavy_rain_flag",
            "strong_wind_flag",
        ]
        if column in df.columns
    ]
    save_dataframe(df[available], GOLD_DATA_DIR / "gold_weather_hourly.parquet")
    LOGGER.info("Saved gold_weather_hourly with shape %s", df[available].shape)


def main() -> None:
    configure_logging()
    ensure_data_directories()
    GOLD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _build_gold_consumption()
    _build_gold_injection()
    _build_gold_weather()


if __name__ == "__main__":
    main()
