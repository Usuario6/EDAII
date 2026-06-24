"""Generate timestamp-quality and data-provenance audit artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import GOLD_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, REPORTS_DIR, SILVER_DATA_DIR


TIMESTAMP_REPORT_DIR = REPORTS_DIR / "validation"
TIMESTAMP_CSV = TIMESTAMP_REPORT_DIR / "timestamp_quality_report.csv"
TIMESTAMP_MD = TIMESTAMP_REPORT_DIR / "timestamp_quality_report.md"
PROVENANCE_JSON = REPORTS_DIR / "validation/data_provenance.json"

E_REDES_LAYERS = {
    "silver_consumption": SILVER_DATA_DIR / "e_redes_consumption.parquet",
    "silver_injection": SILVER_DATA_DIR / "e_redes_injection.parquet",
    "silver_production": SILVER_DATA_DIR / "e_redes_production.parquet",
    "gold_15min_consumption": GOLD_DATA_DIR / "gold_consumption.parquet",
    "gold_15min_injection": GOLD_DATA_DIR / "gold_injection.parquet",
}

PROVENANCE_DATASETS = {
    "raw_e_redes_consumption": RAW_DATA_DIR / "e_redes/e_redes_consumption_window.parquet",
    "raw_e_redes_injection": RAW_DATA_DIR / "e_redes/e_redes_grid_injection_window.parquet",
    "raw_e_redes_production": RAW_DATA_DIR / "e_redes/e_redes_production_window.parquet",
    "raw_open_meteo_hourly": RAW_DATA_DIR / "open_meteo/open_meteo_historical_hourly.parquet",
    **E_REDES_LAYERS,
    "gold_hourly_consumption": GOLD_DATA_DIR / "gold_consumption_hourly.parquet",
    "gold_hourly_injection": GOLD_DATA_DIR / "gold_injection_hourly.parquet",
    "gold_weather_features_hourly": GOLD_DATA_DIR / "gold_weather_features_hourly.parquet",
    "gold_enriched_consumption": GOLD_DATA_DIR / "gold_consumption_enriched.parquet",
    "gold_enriched_injection": GOLD_DATA_DIR / "gold_injection_enriched.parquet",
}

SOURCE_METADATA = {
    "E-REDES": {
        "api": "https://e-redes.opendatasoft.com/api/explore/v2.1/catalog/datasets",
        "datasets": [
            "consumo-total-nacional",
            "energia-injetada-na-rede-de-distribuicao",
            "energia-produzida-total-nacional",
        ],
        "role": "2024-2025 electricity targets and components",
    },
    "IPMA": {
        "api": "https://api.ipma.pt/open-data",
        "datasets": ["recent station observations", "warnings", "daily forecasts"],
        "role": "current/recent operational meteorological context; not historical model enrichment",
    },
    "Open-Meteo": {
        "api": "https://archive-api.open-meteo.com/v1/archive",
        "datasets": ["historical hourly reanalysis for seven Portuguese locations"],
        "role": "2024-2025 historical weather training and enrichment",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _timestamp_summary(name: str, path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path, columns=["datetime"])
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="coerce", utc=True)).dropna()
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="15min", tz="UTC")
    missing = expected.difference(timestamps.unique())
    duplicate_rows = int(timestamps.duplicated().sum())
    duplicate_dates = sorted({str(value.date()) for value in timestamps[timestamps.duplicated(keep=False)]})
    missing_dates = sorted({str(value.date()) for value in missing})
    dst_dates = {"2024-03-31", "2024-10-27", "2025-03-30", "2025-10-26"}
    dst_affected = sorted(set(duplicate_dates + missing_dates).intersection(dst_dates))
    return {
        "dataset": name,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": len(frame),
        "unique_timestamps": int(timestamps.nunique()),
        "duplicated_timestamps_count": duplicate_rows,
        "missing_quarter_hours_count": len(missing),
        "duplicate_dates": ",".join(duplicate_dates),
        "missing_dates": ",".join(missing_dates),
        "dst_transition_dates_affected": ",".join(dst_affected),
        "decision": (
            "Retain source rows; do not infer offsets or silently deduplicate. "
            "Hourly resampling averages all published records and retains empty hours as missing targets."
        ),
    }


def build_timestamp_report() -> pd.DataFrame:
    rows = [_timestamp_summary(name, path) for name, path in E_REDES_LAYERS.items() if path.exists()]
    report = pd.DataFrame(rows)
    TIMESTAMP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(TIMESTAMP_CSV, index=False)

    lines = [
        "# Timestamp and DST quality report",
        "",
        "## Finding",
        "",
        "All duplicated timestamps occur in the 01:00-01:45 UTC interval on Portugal's "
        "2024 and 2025 daylight-saving transition dates. The raw E-REDES snapshot already "
        "contains repeated UTC values with different measurements, so the ambiguity is "
        "source-level and is not introduced by local timezone conversion.",
        "",
        "Missing injection and production quarter-hours occur on the two autumn transition "
        "dates. Consumption has the same DST-related gaps plus a separate 96-quarter-hour "
        "gap spanning 2025-10-13/14; that additional gap is not a DST transition.",
        "",
        "## Decision",
        "",
        "No source record is silently dropped and no synthetic timezone offset is assigned. "
        "The existing hourly layer averages every record published for an hour. Hours with no "
        "source observations remain in the complete hourly index with missing targets and are "
        "excluded from model fitting. This conservative policy preserves traceability, but the "
        "four DST hours may be biased because repeated source records receive equal weight.",
        "",
        "## Counts",
        "",
        "```text",
        report[[
            "dataset", "rows", "unique_timestamps", "duplicated_timestamps_count",
            "missing_quarter_hours_count", "duplicate_dates", "missing_dates",
        ]].to_string(index=False),
        "```",
        "",
        "Machine-readable details and the decision are in `timestamp_quality_report.csv`.",
    ]
    TIMESTAMP_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_provenance() -> dict[str, object]:
    datasets: dict[str, object] = {}
    for name, path in PROVENANCE_DATASETS.items():
        if not path.exists():
            datasets[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "status": "missing"}
            continue
        frame = pd.read_parquet(path)
        datetime_column = "datetime" if "datetime" in frame else "datahora" if "datahora" in frame else None
        timestamps = pd.to_datetime(frame[datetime_column], errors="coerce", utc=True) if datetime_column else None
        datasets[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "snapshot_file_mtime_utc": _iso_mtime(path),
            "rows": len(frame),
            "columns": len(frame.columns),
            "date_min_utc": timestamps.min().isoformat() if timestamps is not None and timestamps.notna().any() else None,
            "date_max_utc": timestamps.max().isoformat() if timestamps is not None and timestamps.notna().any() else None,
            "sha256": _sha256(path),
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "extraction_timestamp_note": (
            "The original extraction command did not persist an API response timestamp. "
            "snapshot_file_mtime_utc is the best available local extraction/snapshot timestamp."
        ),
        "reproducibility_warning": (
            "Live APIs are mutable. Rebuilding later may change rows and metrics; exact reproduction "
            "requires files matching the recorded SHA-256 hashes."
        ),
        "sources": SOURCE_METADATA,
        "datasets": datasets,
    }
    PROVENANCE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    timestamp_report = build_timestamp_report()
    provenance = build_provenance()
    print(f"Timestamp audit rows: {len(timestamp_report)}")
    print(f"Provenance datasets: {len(provenance['datasets'])}")
    print(f"Saved: {TIMESTAMP_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {TIMESTAMP_MD.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {PROVENANCE_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
