"""Build silver datasets from the raw API samples."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, SILVER_DATA_DIR, configure_logging, ensure_data_directories
from src.transform.cleaning import basic_clean
from src.utils.io import save_dataframe

LOGGER = logging.getLogger(__name__)


def _read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path}")


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_raw_sample(stem: str) -> pd.DataFrame | None:
    candidates = [
        RAW_DATA_DIR / "e_redes" / f"{stem.replace('_sample', '_window')}.parquet",
        RAW_DATA_DIR / "e_redes" / f"{stem.replace('_sample', '_window')}.csv",
        RAW_DATA_DIR / "e_redes" / f"{stem}.parquet",
        RAW_DATA_DIR / "e_redes" / f"{stem}.csv",
        RAW_DATA_DIR / "ipma" / f"{stem}.parquet",
        RAW_DATA_DIR / "ipma" / f"{stem}.csv",
        RAW_DATA_DIR / f"{stem}.parquet",
        RAW_DATA_DIR / f"{stem}.csv",
    ]
    path = _first_existing_path(candidates)
    if path is None:
        LOGGER.warning("Missing raw sample: %s", stem)
        return None
    return _read_dataframe(path)


def _prepare_and_save(stem: str, output_name: str) -> None:
    raw = _load_raw_sample(stem)
    if raw is None:
        return
    cleaned = basic_clean(raw)
    save_dataframe(cleaned, SILVER_DATA_DIR / f"{output_name}.parquet")
    LOGGER.info("Saved silver dataset %s with shape %s", output_name, cleaned.shape)


def _prepare_and_save_many(stems: list[str], output_name: str) -> None:
    frames: list[pd.DataFrame] = []
    for stem in stems:
        raw = _load_raw_sample(stem)
        if raw is None:
            continue
        frames.append(basic_clean(raw))
    if not frames:
        LOGGER.warning("No raw samples found for %s", output_name)
        return
    cleaned = pd.concat(frames, ignore_index=True)
    save_dataframe(cleaned, SILVER_DATA_DIR / f"{output_name}.parquet")
    LOGGER.info("Saved silver dataset %s with shape %s", output_name, cleaned.shape)


def main() -> None:
    configure_logging()
    ensure_data_directories()
    SILVER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    _prepare_and_save("e_redes_consumption_sample", "e_redes_consumption")
    _prepare_and_save("e_redes_grid_injection_sample", "e_redes_injection")
    _prepare_and_save("e_redes_production_sample", "e_redes_production")
    _prepare_and_save("ipma_observations_sample", "ipma_observations")
    _prepare_and_save("ipma_warnings_sample", "ipma_warnings")
    _prepare_and_save_many(
        [
            "ipma_forecast_day_0_sample",
            "ipma_forecast_day_1_sample",
            "ipma_forecast_day_2_sample",
        ],
        "ipma_daily_forecast",
    )


if __name__ == "__main__":
    main()
