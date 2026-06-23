"""Extract bounded historical windows from E-REDES datasets."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import E_REDES_API_BASE_URL, E_REDES_DATASETS, HTTP_TIMEOUT_SECONDS, RAW_DATA_DIR, configure_logging
from src.utils.io import save_dataframe

LOGGER = logging.getLogger(__name__)
WINDOW_DIR = RAW_DATA_DIR / "e_redes"
MAX_API_PAGE_SIZE = 100
CHUNK_DAYS = 31


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _window_where(start_date: date, end_date: date) -> str:
    start = f'{start_date.isoformat()}T00:00:00'
    end = f'{(end_date + timedelta(days=1)).isoformat()}T00:00:00'
    return f'datahora >= "{start}" AND datahora < "{end}"'


def _dataset_name(value: str) -> tuple[str, str]:
    aliases = {
        "consumption": "consumption",
        "consumo-total-nacional": "consumption",
        "injection": "grid_injection",
        "grid_injection": "grid_injection",
        "energia-injetada-na-rede-de-distribuicao": "grid_injection",
        "production": "production",
        "energia-produzida-total-nacional": "production",
    }
    reverse = {dataset_id: key for key, dataset_id in E_REDES_DATASETS.items()}
    if value in aliases:
        key = aliases[value]
        return key, E_REDES_DATASETS[key]
    if value in reverse:
        return reverse[value], value
    raise ValueError(f"Unknown dataset identifier: {value}")


def _fetch_page(session: requests.Session, dataset_id: str, params: dict[str, object]) -> tuple[int, pd.DataFrame]:
    url = f"{E_REDES_API_BASE_URL}/{dataset_id}/records"
    response = session.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    status = response.status_code
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Unexpected response format for dataset {dataset_id}")
    return status, pd.json_normalize(results)


def extract_window(
    dataset_key: str,
    dataset_id: str,
    start_date: date,
    end_date: date,
    page_size: int,
    max_pages: int,
) -> pd.DataFrame:
    session = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    frames: list[pd.DataFrame] = []
    page_size = min(page_size, MAX_API_PAGE_SIZE)
    pages_used = 0
    chunk_start = start_date

    # Monthly-sized chunks avoid OpenDataSoft's deep-offset safeguards while the
    # overall page cap still bounds the complete extraction.
    while chunk_start <= end_date:
        chunk_end = min(end_date, chunk_start + timedelta(days=CHUNK_DAYS - 1))
        where = _window_where(chunk_start, chunk_end)
        page = 0
        while True:
            if pages_used >= max_pages:
                raise RuntimeError(
                    f"max_pages={max_pages} reached before completing {dataset_id}; "
                    "increase the explicit bound and rerun"
                )
            offset = page * page_size
            params = {"limit": page_size, "offset": offset, "where": where, "order_by": "datahora"}
            if page == 0 or (page + 1) % 10 == 0:
                LOGGER.info(
                    "Fetching %s chunk=%s..%s page=%s total_pages=%s",
                    dataset_id, chunk_start, chunk_end, page + 1, pages_used + 1,
                )
            try:
                status, frame = _fetch_page(session, dataset_id, params)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch {dataset_id} chunk={chunk_start}..{chunk_end} page={page + 1}"
                ) from exc
            pages_used += 1
            if frame.empty:
                break
            frames.append(frame)
            if len(frame) < page_size:
                break
            page += 1
        LOGGER.info("Completed %s chunk=%s..%s status=%s", dataset_id, chunk_start, chunk_end, status)
        chunk_start = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    if "datahora" in result.columns:
        timestamps = pd.to_datetime(result["datahora"], errors="coerce", utc=True)
        LOGGER.info(
            "%s actual coverage=%s..%s rows=%s pages=%s",
            dataset_id, timestamps.min(), timestamps.max(), len(result), pages_used,
        )
    output_path = WINDOW_DIR / f"e_redes_{dataset_key}_window.parquet"
    try:
        save_dataframe(result, output_path)
        LOGGER.info("Saved %s rows to %s", len(result), output_path)
    except Exception as exc:
        fallback = output_path.with_suffix(".csv")
        LOGGER.warning("Parquet save failed for %s: %s", output_path.name, exc)
        save_dataframe(result, fallback)
        LOGGER.info("Saved %s rows to %s", len(result), fallback)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument(
        "--datasets",
        default="consumption,injection,production",
        help="Comma-separated E-REDES dataset keys or dataset ids.",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="Maximum pages per dataset across all date chunks.",
    )
    return parser


def main() -> None:
    configure_logging()
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    parser = build_arg_parser()
    args = parser.parse_args()
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if start_date > end_date:
        parser.error("--start-date must be on or before --end-date")
    if not 1 <= args.page_size <= MAX_API_PAGE_SIZE:
        parser.error(f"--page-size must be between 1 and {MAX_API_PAGE_SIZE}")
    if args.max_pages < 1:
        parser.error("--max-pages must be positive")
    dataset_values = [item.strip() for item in args.datasets.split(",") if item.strip()]

    for value in dataset_values:
        dataset_key, dataset_id = _dataset_name(value)
        LOGGER.info(
            "Extracting %s (%s) from %s to %s",
            dataset_key,
            dataset_id,
            start_date,
            end_date,
        )
        frame = extract_window(
            dataset_key=dataset_key,
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
        print(f"{dataset_key}: rows={len(frame)} columns={frame.columns.tolist()}")
        if not frame.empty:
            print(frame.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
