"""Validate operational risk proxy outputs."""

from __future__ import annotations

import compileall
import pandas as pd

from src.config import PROJECT_ROOT, REPORTS_DIR

OUTPUT_DIR = REPORTS_DIR / "risk"

EXPECTED_FILES = [
    "consumption_risk_score.csv",
    "injection_risk_score.csv",
    "risk_score_summary.csv",
    "risk_score_report.md",
    "consumption_risk_score_plot.png",
    "injection_risk_score_plot.png",
]

REQUIRED_COLUMNS = {
    "datetime",
    "dataset",
    "target_column",
    "target_value",
    "risk_score",
    "risk_level",
    "pressure_score",
    "change_score",
    "seasonal_deviation_score",
    "outlier_score",
    "weather_score",
    "missing_target",
    "iqr_outlier",
    "zscore_outlier",
    "isolation_forest_outlier",
    "any_outlier",
}


def validate_score_file(filename: str, expected_dataset: str, failures: list[str]) -> None:
    path = OUTPUT_DIR / filename

    if not path.exists() or path.stat().st_size == 0:
        failures.append(f"missing or empty file: {filename}")
        return

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        failures.append(f"{filename} missing columns: {sorted(missing)}")

    if df.empty:
        failures.append(f"{filename} is empty")
        return

    if set(df["dataset"].dropna().unique()) != {expected_dataset}:
        failures.append(f"{filename} has wrong dataset labels")

    scores = pd.to_numeric(df["risk_score"], errors="coerce").dropna()
    if scores.empty:
        failures.append(f"{filename} has no valid scores")
    elif not scores.between(0, 100).all():
        failures.append(f"{filename} has scores outside [0, 100]")

    allowed_levels = {"low", "medium", "high", "critical", "missing_target"}
    levels = set(df["risk_level"].dropna().unique())
    if not levels.issubset(allowed_levels):
        failures.append(f"{filename} has invalid risk levels: {sorted(levels - allowed_levels)}")

    datetimes = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    if datetimes.isna().any():
        failures.append(f"{filename} has invalid datetimes")

    if datetimes.duplicated().any():
        failures.append(f"{filename} has duplicated datetimes")


def main() -> None:
    failures = []

    if not compileall.compile_dir(PROJECT_ROOT / "src", quiet=1):
        failures.append("compileall failed for src")
    if not compileall.compile_dir(PROJECT_ROOT / "scripts", quiet=1):
        failures.append("compileall failed for scripts")

    for filename in EXPECTED_FILES:
        path = OUTPUT_DIR / filename
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty output: {filename}")

    validate_score_file("consumption_risk_score.csv", "consumption", failures)
    validate_score_file("injection_risk_score.csv", "injection", failures)

    summary_path = OUTPUT_DIR / "risk_score_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if set(summary["dataset"]) != {"consumption", "injection"}:
            failures.append("summary must contain consumption and injection")
    else:
        failures.append("missing risk_score_summary.csv")

    report_path = OUTPUT_DIR / "risk_score_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8").lower()
        if "not represent a real probability of grid failure" not in text:
            failures.append("risk report must clearly say it is not real failure prediction")
    else:
        failures.append("missing risk_score_report.md")

    if failures:
        raise RuntimeError("Risk-score validation failed:\n- " + "\n- ".join(failures))

    print("Risk-score validation passed")


if __name__ == "__main__":
    main()
