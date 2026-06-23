"""Extract and minimally standardize public IPMA data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import (
    HTTP_TIMEOUT_SECONDS,
    IPMA_DAILY_FORECAST_URL,
    IPMA_OBSERVATIONS_URL,
    IPMA_WARNINGS_URL,
)
from src.utils.io import save_dataframe


def _get_json(url: str) -> Any:
    response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _records_to_dataframe(payload: Any) -> pd.DataFrame:
    """Convert IPMA list responses or their common `data` envelope."""
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Unexpected IPMA response: expected a list or a 'data' list")
    return pd.json_normalize(records)


def _clean_ipma_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace(-99.0, np.nan)
    for column in cleaned.columns:
        name = column.lower()
        if any(token in name for token in ("date", "time", "timestamp", "data")):
            converted = pd.to_datetime(cleaned[column], errors="coerce", utc=False)
            if converted.notna().any():
                cleaned[column] = converted
    return cleaned


def fetch_ipma_observations() -> pd.DataFrame:
    """Fetch current hourly observations and flatten timestamp/station keys."""
    payload = _get_json(IPMA_OBSERVATIONS_URL)
    if not isinstance(payload, dict):
        raise ValueError("Unexpected IPMA observations response")

    rows: list[dict[str, Any]] = []
    for timestamp, stations in payload.items():
        if not isinstance(stations, dict):
            continue
        for station_id, values in stations.items():
            row = {"observation_time": timestamp, "station_id": station_id}
            if isinstance(values, dict):
                row.update(values)
            else:
                row["value"] = values
            rows.append(row)
    return _clean_ipma_dataframe(pd.DataFrame(rows))


def fetch_ipma_warnings() -> pd.DataFrame:
    """Fetch active/published weather warnings."""
    return _clean_ipma_dataframe(_records_to_dataframe(_get_json(IPMA_WARNINGS_URL)))


def fetch_ipma_daily_forecast_by_day(id_day: int) -> pd.DataFrame:
    """Fetch the national city forecast for one horizon (0, 1, or 2)."""
    if id_day not in {0, 1, 2}:
        raise ValueError("id_day must be one of {0, 1, 2}")
    url = IPMA_DAILY_FORECAST_URL.format(id_day=id_day)
    df = _records_to_dataframe(_get_json(url))
    df.insert(0, "forecast_day", id_day)
    return _clean_ipma_dataframe(df)


def fetch_ipma_daily_forecast_3_days() -> pd.DataFrame:
    """Fetch and combine forecast horizons 0, 1, and 2."""
    return pd.concat(
        [fetch_ipma_daily_forecast_by_day(day) for day in range(3)],
        ignore_index=True,
    )
