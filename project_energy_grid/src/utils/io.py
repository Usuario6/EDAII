"""Shared IO helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as Parquet or CSV based on the file suffix."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    raise ValueError("Output path must end in .parquet or .csv")

