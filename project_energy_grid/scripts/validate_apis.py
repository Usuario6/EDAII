"""Validate all source APIs with bounded samples and persist raw snapshots."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import E_REDES_DATASETS, RAW_DATA_DIR, configure_logging, ensure_data_directories
from src.utils.io import save_dataframe

LOGGER = logging.getLogger(__name__)
E_REDES_SAMPLE_DIR = RAW_DATA_DIR / "e_redes"
IPMA_SAMPLE_DIR = RAW_DATA_DIR / "ipma"


def _save_sample(df: pd.DataFrame, base_path: Path, stem: str) -> None:
    """Persist a sample as Parquet, falling back to CSV when needed."""
    parquet_path = base_path / f"{stem}.parquet"
    csv_path = base_path / f"{stem}.csv"
    try:
        save_dataframe(df, parquet_path)
        LOGGER.info("Saved %s", parquet_path)
    except Exception as exc:
        LOGGER.warning("Parquet save failed for %s: %s", stem, exc)
        save_dataframe(df, csv_path)
        LOGGER.info("Saved %s", csv_path)
    else:
        save_dataframe(df, csv_path)
        LOGGER.info("Saved %s", csv_path)


def report(name: str, status: int, dataframe: pd.DataFrame) -> None:
    print(f"\n{name}: status={status}")
    print(f"shape={dataframe.shape}")
    print(f"columns={dataframe.columns.tolist()}")
    print(dataframe.head(3).to_string(index=False))


def _request_json(url: str, params: dict[str, object] | None = None) -> tuple[int | None, object | None, str | None]:
    try:
        response = requests.get(url, params=params, timeout=30)
        status = response.status_code
        response.raise_for_status()
        return status, response.json(), None
    except Exception as exc:
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None), None, str(exc)


def _to_dataframe(payload: object) -> pd.DataFrame:
    if isinstance(payload, dict):
        records = payload.get("data", payload.get("results", payload))
    else:
        records = payload

    if isinstance(records, list):
        return pd.json_normalize(records)
    if isinstance(payload, dict):
        rows: list[dict[str, object]] = []
        for timestamp, stations in payload.items():
            if not isinstance(stations, dict):
                continue
            for station_id, values in stations.items():
                row: dict[str, object] = {"observation_time": timestamp, "station_id": station_id}
                if isinstance(values, dict):
                    row.update(values)
                else:
                    row["value"] = values
                rows.append(row)
        return pd.DataFrame(rows)
    raise ValueError("Unsupported payload shape")


def main() -> None:
    configure_logging()
    ensure_data_directories()
    E_REDES_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    IPMA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataset_id in E_REDES_DATASETS.items():
        LOGGER.info("Validating E-REDES dataset %s", dataset_id)
        url = f"https://e-redes.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset_id}/records"
        status, payload, error = _request_json(url, params={"limit": 5, "offset": 0})
        if error is not None:
            print(f"\nE-REDES {dataset_id}: status={status}")
            print(f"error={error}")
            continue
        df = _to_dataframe(payload)
        report(f"E-REDES {dataset_id}", status or 200, df)
        _save_sample(df, E_REDES_SAMPLE_DIR, f"e_redes_{name}_sample")

    endpoints = [
        (
            "IPMA observations",
            "https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json",
            None,
            "ipma_observations_sample",
        ),
        (
            "IPMA warnings",
            "https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json",
            None,
            "ipma_warnings_sample",
        ),
    ]

    for label, url, params, stem in endpoints:
        status, payload, error = _request_json(url, params=params)
        if error is not None:
            print(f"\n{label}: status={status}")
            print(f"error={error}")
            continue
        df = _to_dataframe(payload)
        report(label, status or 200, df)
        _save_sample(df, IPMA_SAMPLE_DIR, stem)

    for day in range(3):
        url = f"https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day{day}.json"
        status, payload, error = _request_json(url)
        if error is not None:
            print(f"\nIPMA daily forecast day {day}: status={status}")
            print(f"error={error}")
            continue
        df = _to_dataframe(payload)
        if "forecast_day" not in df.columns:
            df.insert(0, "forecast_day", day)
        report(f"IPMA daily forecast day {day}", status or 200, df)
        _save_sample(df, IPMA_SAMPLE_DIR, f"ipma_forecast_day_{day}_sample")


if __name__ == "__main__":
    main()
