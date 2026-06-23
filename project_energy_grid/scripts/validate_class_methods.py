"""Validate course-method source code, reports, chronology, and leakage controls."""

from __future__ import annotations

import compileall
from pathlib import Path

import pandas as pd

from scripts.train_baseline_consumption import FEATURES as CONSUMPTION_FEATURES
from scripts.train_baseline_injection import FEATURES as INJECTION_FEATURES
from src.config import GOLD_DATA_DIR, PROJECT_ROOT, REPORTS_DIR


EXPECTED = [
    REPORTS_DIR / "models/consumption_model_comparison.csv",
    REPORTS_DIR / "models/injection_model_comparison.csv",
    REPORTS_DIR / "models/consumption_feature_importance.csv",
    REPORTS_DIR / "models/injection_feature_importance.csv",
    REPORTS_DIR / "outliers/consumption_outliers.csv",
    REPORTS_DIR / "outliers/injection_outliers.csv",
    REPORTS_DIR / "outliers/outlier_summary.csv",
    REPORTS_DIR / "dimensionality/pca_consumption_components.csv",
    REPORTS_DIR / "dimensionality/pca_injection_components.csv",
    REPORTS_DIR / "dimensionality/pca_consumption_plot.png",
    REPORTS_DIR / "dimensionality/pca_injection_plot.png",
    REPORTS_DIR / "dimensionality/pca_explained_variance.csv",
    REPORTS_DIR / "clustering/consumption_clusters.csv",
    REPORTS_DIR / "clustering/injection_clusters.csv",
    REPORTS_DIR / "clustering/clustering_summary.csv",
    REPORTS_DIR / "clustering/consumption_clusters_plot.png",
    REPORTS_DIR / "clustering/injection_clusters_plot.png",
]


def main() -> None:
    failures = []
    if not compileall.compile_dir(PROJECT_ROOT / "src", quiet=1) or not compileall.compile_dir(PROJECT_ROOT / "scripts", quiet=1):
        failures.append("compileall failed")
    for path in EXPECTED:
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty output: {path.relative_to(PROJECT_ROOT)}")

    for filename, target in [("gold_consumption_hourly.parquet", "total"), ("gold_injection_hourly.parquet", "total_injection")]:
        path = GOLD_DATA_DIR / filename
        if not path.exists():
            failures.append(f"missing hourly dataset: {filename}")
            continue
        df = pd.read_parquet(path)
        timestamps = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            failures.append(f"non-chronological or duplicate timestamps: {filename}")
        for window in (24, 168):
            column = f"{target}_rollmean_{window}"
            expected = pd.to_numeric(df[target], errors="coerce").shift(1).rolling(window, min_periods=1).mean()
            if column not in df or not df[column].fillna(-1).equals(expected.fillna(-1)):
                failures.append(f"rolling target leakage check failed: {filename}/{column}")

    source_text = "\n".join(path.read_text() for path in (PROJECT_ROOT / "scripts").glob("*.py"))
    forbidden = "train" + "_test_split"
    if f"from sklearn.model_selection import {forbidden}" in source_text:
        failures.append("random sklearn train/test split import found")

    if "total" in CONSUMPTION_FEATURES:
        failures.append("consumption target leakage found in model features")
    injection_leakage = {
        "total_injection", "cogeracao", "eolica", "fotovoltaica",
        "hidrica", "outras_tecnologias", "rede_dist",
    }.intersection(INJECTION_FEATURES)
    if injection_leakage:
        failures.append(f"injection target/component leakage found: {sorted(injection_leakage)}")

    if failures:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(failures))
    print(f"Validation passed: {len(EXPECTED)} non-empty reports, chronological splits, leakage-safe rolling features, compileall OK")


if __name__ == "__main__":
    main()
