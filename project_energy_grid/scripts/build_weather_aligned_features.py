"""Build historically aligned hourly weather features.

Priority:
1. Open-Meteo historical hourly reanalysis, if available.
2. IPMA recent observations fallback, if Open-Meteo is missing.

The Open-Meteo layer is used only to solve historical alignment for 2024–2025.
IPMA remains the official Portuguese source for warnings/current observations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OPEN_METEO = PROJECT_ROOT / "data/raw/open_meteo/open_meteo_historical_hourly.parquet"
RAW_OPEN_METEO_CSV = PROJECT_ROOT / "data/raw/open_meteo/open_meteo_historical_hourly.csv"
RAW_IPMA = PROJECT_ROOT / "data/silver/ipma_observations.parquet"

GOLD_DIR = PROJECT_ROOT / "data/gold"
REPORT_DIR = PROJECT_ROOT / "reports/weather"

OUTPUT_WEATHER = GOLD_DIR / "gold_weather_features_hourly.parquet"
OUTPUT_SUMMARY = REPORT_DIR / "weather_alignment_summary.csv"
OUTPUT_OPEN_METEO_SUMMARY = REPORT_DIR / "open_meteo_alignment_summary.csv"


def _load_open_meteo() -> pd.DataFrame | None:
    if RAW_OPEN_METEO.exists():
        return pd.read_parquet(RAW_OPEN_METEO)
    if RAW_OPEN_METEO_CSV.exists():
        return pd.read_csv(RAW_OPEN_METEO_CSV)
    return None


def _build_from_open_meteo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])

    grouped = df.groupby("datetime", as_index=False)

    result = pd.DataFrame({
        "datetime": grouped["datetime"].first()["datetime"],
        "temperatura": grouped["temperature_2m"].mean()["temperature_2m"],
        "humidade": grouped["relative_humidity_2m"].mean()["relative_humidity_2m"],
        "precacumulada": grouped["precipitation"].mean()["precipitation"],
        "rain": grouped["rain"].mean()["rain"],
        "intensidadeventokm": grouped["wind_speed_10m"].mean()["wind_speed_10m"],
        "wind_gusts_10m": grouped["wind_gusts_10m"].mean()["wind_gusts_10m"],
        "radiacao": grouped["shortwave_radiation"].mean()["shortwave_radiation"],
        "pressao": grouped["surface_pressure"].mean()["surface_pressure"],
        "cloud_cover": grouped["cloud_cover"].mean()["cloud_cover"],
        "open_meteo_locations": grouped["location"].nunique()["location"],
    })

    result = result.sort_values("datetime").reset_index(drop=True)

    result["hdd_18"] = (18 - result["temperatura"]).clip(lower=0)
    result["cdd_22"] = (result["temperatura"] - 22).clip(lower=0)

    # Simple explainable operational weather flags.
    result["heavy_rain_flag"] = result["precacumulada"].ge(5.0)
    result["strong_wind_flag"] = result["wind_gusts_10m"].ge(50.0) | result["intensidadeventokm"].ge(40.0)

    result["weather_available"] = True
    result["weather_source"] = "open_meteo_reanalysis"

    return result


def _build_from_ipma(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])

    # Aggregate station-level recent observations to one national proxy hour.
    numeric_cols = [
        "temperatura",
        "radiacao",
        "precacumulada",
        "intensidadeventokm",
        "humidade",
        "pressao",
    ]

    available = [col for col in numeric_cols if col in df.columns]
    result = df.groupby("datetime", as_index=False)[available].mean()

    result["hdd_18"] = (18 - result["temperatura"]).clip(lower=0) if "temperatura" in result else 0
    result["cdd_22"] = (result["temperatura"] - 22).clip(lower=0) if "temperatura" in result else 0
    result["heavy_rain_flag"] = result.get("precacumulada", pd.Series(0, index=result.index)).ge(5.0)
    result["strong_wind_flag"] = result.get("intensidadeventokm", pd.Series(0, index=result.index)).ge(40.0)
    result["weather_available"] = True
    result["weather_source"] = "ipma_recent_observations"

    return result.sort_values("datetime").reset_index(drop=True)


def _overlap_hours(weather: pd.DataFrame, energy_path: Path) -> int:
    if not energy_path.exists():
        return 0
    energy = pd.read_parquet(energy_path, columns=["datetime"])
    weather_dt = set(pd.to_datetime(weather["datetime"], utc=True))
    energy_dt = set(pd.to_datetime(energy["datetime"], utc=True))
    return len(weather_dt.intersection(energy_dt))


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    open_meteo = _load_open_meteo()

    if open_meteo is not None:
        weather = _build_from_open_meteo(open_meteo)
        source = "open_meteo_reanalysis"
    elif RAW_IPMA.exists():
        weather = _build_from_ipma(pd.read_parquet(RAW_IPMA))
        source = "ipma_recent_observations"
    else:
        raise FileNotFoundError("No Open-Meteo or IPMA weather source available")

    weather.to_parquet(OUTPUT_WEATHER, index=False)

    overlap_consumption = _overlap_hours(weather, GOLD_DIR / "gold_consumption_hourly.parquet")
    overlap_injection = _overlap_hours(weather, GOLD_DIR / "gold_injection_hourly.parquet")
    overlap_hours = max(overlap_consumption, overlap_injection)

    summary = pd.DataFrame([{
        "weather_source": source,
        "weather_hourly_rows": len(weather),
        "weather_start": weather["datetime"].min(),
        "weather_end": weather["datetime"].max(),
        "duplicate_timestamps": int(pd.to_datetime(weather["datetime"], utc=True).duplicated().sum()),
        "overlap_hours": overlap_hours,
        "overlap_hours_consumption": overlap_consumption,
        "overlap_hours_injection": overlap_injection,
        "usable_for_historical_modelling": bool(overlap_hours >= 24 * 365),
        "weather_available_true_rows": int(weather["weather_available"].sum()),
        "heavy_rain_hours": int(weather["heavy_rain_flag"].sum()),
        "strong_wind_hours": int(weather["strong_wind_flag"].sum()),
    }])

    summary.to_csv(OUTPUT_SUMMARY, index=False)
    summary.to_csv(OUTPUT_OPEN_METEO_SUMMARY, index=False)

    print("Weather alignment completed")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
