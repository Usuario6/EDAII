"""Small, reusable E-REDES OpenDataSoft extractions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from src.config import E_REDES_API_BASE_URL, HTTP_TIMEOUT_SECONDS
from src.utils.io import save_dataframe


def fetch_e_redes_dataset(
    dataset_id: str,
    limit: int = 100,
    offset: int = 0,
    where: str | None = None,
    select: str | None = None,
    order_by: str | None = None,
) -> pd.DataFrame:
    """Fetch one bounded page from an E-REDES dataset."""
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    params: dict[str, Any] = {"limit": limit, "offset": offset}
    for key, value in {"where": where, "select": select, "order_by": order_by}.items():
        if value is not None:
            params[key] = value

    url = f"{E_REDES_API_BASE_URL}/{dataset_id}/records"
    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Unexpected E-REDES response for dataset {dataset_id!r}")
    return pd.json_normalize(results)


def fetch_e_redes_sample(dataset_id: str, limit: int = 10) -> pd.DataFrame:
    """Fetch a small validation sample from E-REDES."""
    return fetch_e_redes_dataset(dataset_id=dataset_id, limit=limit)
