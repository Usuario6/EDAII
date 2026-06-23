"""Extract historical hourly weather from Open-Meteo for Portugal proxy points."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data/raw/open_meteo"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Representative mainland Portugal points.
# This is a national proxy, not official IPMA station data.
LOCATIONS = {
    "lisboa": (38.7223, -9.1393),
    "porto": (41.1579, -8.6291),
    "coimbra": (40.2033, -8.4103),
    "faro": (37.0194, -7.9304),
    "evora": (38.5714, -7.9135),
    "viseu": (40.6566, -7.9125),
    "braganca": (41.8061, -6.7567),
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "surface_pressure",
    "cloud_cover",
]


def fetch_location(name: str, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError(f"Invalid Open-Meteo response for {name}: missing hourly data")

    df = pd.DataFrame(payload["hourly"])
    df["location"] = name
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["datetime"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, (lat, lon) in LOCATIONS.items():
        print(f"Fetching {name}: {lat}, {lon}")
        frame = fetch_location(name, lat, lon, args.start_date, args.end_date)
        frames.append(frame)

        frame.to_csv(OUTPUT_DIR / f"open_meteo_{name}_{args.start_date}_{args.end_date}.csv", index=False)
        frame.to_parquet(OUTPUT_DIR / f"open_meteo_{name}_{args.start_date}_{args.end_date}.parquet", index=False)

        time.sleep(0.5)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["datetime", "location"]).reset_index(drop=True)

    combined.to_csv(OUTPUT_DIR / "open_meteo_historical_hourly.csv", index=False)
    combined.to_parquet(OUTPUT_DIR / "open_meteo_historical_hourly.parquet", index=False)

    print("Open-Meteo extraction completed")
    print("rows:", len(combined))
    print("locations:", combined["location"].nunique())
    print("datetime_min:", combined["datetime"].min())
    print("datetime_max:", combined["datetime"].max())


if __name__ == "__main__":
    main()
