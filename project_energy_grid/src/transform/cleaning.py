"""Generic cleaning functions that do not assume final schemas."""

from __future__ import annotations

from collections.abc import Iterable
import re
import unicodedata

import numpy as np
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase ASCII snake_case column names."""
    result = df.copy()
    normalized = []
    for column in result.columns:
        name = unicodedata.normalize("NFKD", str(column)).encode("ascii", "ignore").decode()
        name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
        normalized.append(name)
    result.columns = normalized
    return result


def infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date-like columns when at least one value parses successfully."""
    result = df.copy()
    for column in result.columns:
        if any(token in column.lower() for token in ("date", "time", "data", "hora")):
            parsed = pd.to_datetime(result[column], errors="coerce", format="mixed")
            if parsed.notna().any():
                result[column] = parsed
    return result


def build_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical datetime column when a suitable source exists."""
    result = df.copy()
    if "datetime" in result.columns:
        result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
        return result

    for candidate in ("datahora", "observation_time", "starttime", "endtime"):
        if candidate in result.columns:
            parsed = pd.to_datetime(result[candidate], errors="coerce", format="mixed")
            if parsed.notna().any():
                result["datetime"] = parsed
                return result

    if {"date", "time"}.issubset(result.columns):
        combined = result["date"].astype(str).str.strip() + " " + result["time"].astype(str).str.strip()
        parsed = pd.to_datetime(combined, errors="coerce", format="mixed")
        if parsed.notna().any():
            result["datetime"] = parsed
            return result

    if "date" in result.columns:
        parsed = pd.to_datetime(result["date"], errors="coerce", format="mixed")
        if parsed.notna().any():
            result["datetime"] = parsed
    return result


def replace_common_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common numeric and textual missing markers with NaN."""
    return df.replace([-99, -99.0, "-99", "-99.0", "", "NA", "N/A", "null", "None"], np.nan)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows while keeping the first occurrence."""
    return df.drop_duplicates().reset_index(drop=True)


def coerce_numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Coerce columns that look numeric to numeric dtype when possible."""
    result = df.copy()
    excluded = set(exclude)
    for column in result.columns:
        if column in excluded:
            continue
        series = result[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        non_null = series.notna().sum()
        if non_null and coerced.notna().sum() >= max(1, int(non_null * 0.6)):
            result[column] = coerced
    return result


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply schema-independent baseline cleaning."""
    result = normalize_column_names(df)
    result = replace_common_missing_values(result)
    result = build_datetime_column(result)
    result = infer_datetime_columns(result)
    result = coerce_numeric_columns(result, exclude={"datetime"})
    return remove_duplicates(result)
