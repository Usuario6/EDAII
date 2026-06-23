"""Validate Streamlit dashboard implementation."""

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
    elif not compileall.compile_file(str(DASHBOARD_PATH), quiet=1):
        failures.append("dashboard app does not compile")

    if not REQUIREMENTS_PATH.exists():
        failures.append("missing requirements.txt")

    if DASHBOARD_PATH.exists():
        text = DASHBOARD_PATH.read_text(encoding="utf-8")
        required_terms = [
            "streamlit",
            "plotly",
            "risk_score",
            "weather_score",
            "weather_alignment_summary.csv",
            "dataset_state_report.csv",
            "consumption_multistep_summary.csv",
            "injection_multistep_summary.csv",
            "gold_consumption_enriched.parquet",
            "gold_injection_enriched.parquet",
            "Operational risk",
            "Open-Meteo",
            "IPMA",
        ]
        for term in required_terms:
            if term not in text:
                failures.append(f"missing dashboard reference: {term}")

    if REQUIREMENTS_PATH.exists():
        requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
        for package in ["streamlit", "plotly"]:
            if package not in requirements:
                failures.append(f"requirements.txt missing {package}")

    if failures:
        raise RuntimeError("Dashboard validation failed:\n- " + "\n- ".join(failures))

    print("Dashboard validation passed")


if __name__ == "__main__":
    main()
