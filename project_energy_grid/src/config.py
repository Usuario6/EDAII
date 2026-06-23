"""Central project configuration."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SILVER_DATA_DIR = DATA_DIR / "silver"
GOLD_DATA_DIR = DATA_DIR / "gold"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

E_REDES_API_BASE_URL = "https://e-redes.opendatasoft.com/api/explore/v2.1/catalog/datasets"
IPMA_API_BASE_URL = "https://api.ipma.pt/open-data"
IPMA_OBSERVATIONS_URL = f"{IPMA_API_BASE_URL}/observation/meteorology/stations/observations.json"
IPMA_WARNINGS_URL = f"{IPMA_API_BASE_URL}/forecast/warnings/warnings_www.json"
IPMA_DAILY_FORECAST_URL = (
    f"{IPMA_API_BASE_URL}/forecast/meteorology/cities/daily/"
    "hp-daily-forecast-day{id_day}.json"
)

E_REDES_DATASETS = {
    "consumption": "consumo-total-nacional",
    "grid_injection": "energia-injetada-na-rede-de-distribuicao",
    "production": "energia-produzida-total-nacional",
}

HTTP_TIMEOUT_SECONDS = 30


def ensure_data_directories() -> None:
    """Create runtime data directories when they do not exist."""
    for directory in (RAW_DATA_DIR, SILVER_DATA_DIR, GOLD_DATA_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise application logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
