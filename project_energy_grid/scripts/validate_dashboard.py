"""Validate Streamlit dashboard files."""

from __future__ import annotations

import compileall
from pathlib import Path

from src.config import PROJECT_ROOT


DASHBOARD_PATH = PROJECT_ROOT / "src/dashboard/app.py"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"


def main() -> None:
    failures = []

    if not DASHBOARD_PATH.exists() or DASHBOARD_PATH.stat().st_size == 0:
        failures.append("missing src/dashboard/app.py")

    if not compileall.compile_file(str(DASHBOARD_PATH), quiet=1):
        failures.append("dashboard app does not compile")

    text = DASHBOARD_PATH.read_text(encoding="utf-8")
    required_terms = [
        "streamlit",
        "risk_score",
        "weather_alignment_summary.csv",
        "consumption_multistep_summary.csv",
        "injection_multistep_summary.csv",
        "gold_consumption_enriched.parquet",
        "gold_injection_enriched.parquet",
    ]

    for term in required_terms:
        if term not in text:
            failures.append(f"dashboard missing expected reference: {term}")

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
    if "streamlit" not in requirements:
        failures.append("requirements.txt missing streamlit")

    if failures:
        raise RuntimeError("Dashboard validation failed:\n- " + "\n- ".join(failures))

    print("Dashboard validation passed")


if __name__ == "__main__":
    main()
