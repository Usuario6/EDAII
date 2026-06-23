"""Build operational risk proxy scores for consumption and injection."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import GOLD_DATA_DIR, REPORTS_DIR
from src.models.risk_score import build_risk_score, summarize_risk_scores
from src.utils.io import save_dataframe


OUTPUT_DIR = REPORTS_DIR / "risk"


def _read_dataset(preferred: Path, fallback: Path) -> pd.DataFrame:
    if preferred.exists():
        return pd.read_parquet(preferred)
    if fallback.exists():
        return pd.read_parquet(fallback)
    raise FileNotFoundError(f"Missing both {preferred} and {fallback}")


def _plot_risk(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_data = frame.dropna(subset=["datetime", "risk_score"]).copy()
    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_data["datetime"], plot_data["risk_score"], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Risk proxy score")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_report(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Operational risk proxy score report",
        "",
        "## Purpose",
        "",
        "This report creates an operational risk proxy score for electricity consumption and energy injection.",
        "",
        "The score does not represent a real probability of grid failure because the current project has no labelled failure events.",
        "",
        "Instead, the score combines observable indicators:",
        "",
        "- current pressure level;",
        "- abnormal change from recent and seasonal references;",
        "- deviation from rolling behaviour;",
        "- outlier detection;",
        "- weather flags, only where historically available.",
        "",
        "## Formula logic",
        "",
        "The score is scaled from 0 to 100:",
        "",
        "```text",
        "risk_score = 100 * (",
        "    0.30 * pressure_score",
        "  + 0.25 * seasonal_deviation_score",
        "  + 0.20 * change_score",
        "  + 0.15 * outlier_score",
        "  + 0.10 * weather_score",
        ")",
        "```",
        "",
        "## Risk levels",
        "",
        "| Score | Level |",
        "|---:|---|",
        "| 0–35 | low |",
        "| 35–60 | medium |",
        "| 60–80 | high |",
        "| 80–100 | critical |",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "The risk score should be interpreted as an explainable operational pressure indicator, not as confirmed outage prediction.",
        "",
        "High values identify timestamps where the system behaviour is unusual, elevated, or unstable relative to recent and historical patterns.",
        "",
        "Weather contribution is currently limited because IPMA observations do not overlap with the 2024–2025 E-REDES modelling period.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    consumption = _read_dataset(
        GOLD_DATA_DIR / "gold_consumption_enriched.parquet",
        GOLD_DATA_DIR / "gold_consumption_hourly.parquet",
    )
    injection = _read_dataset(
        GOLD_DATA_DIR / "gold_injection_enriched.parquet",
        GOLD_DATA_DIR / "gold_injection_hourly.parquet",
    )

    consumption_risk = build_risk_score(consumption, "total", "consumption")
    injection_risk = build_risk_score(injection, "total_injection", "injection")

    save_dataframe(consumption_risk, OUTPUT_DIR / "consumption_risk_score.csv")
    save_dataframe(injection_risk, OUTPUT_DIR / "injection_risk_score.csv")

    summary = summarize_risk_scores(consumption_risk, injection_risk)
    save_dataframe(summary, OUTPUT_DIR / "risk_score_summary.csv")

    _plot_risk(consumption_risk, OUTPUT_DIR / "consumption_risk_score_plot.png", "Consumption operational risk proxy")
    _plot_risk(injection_risk, OUTPUT_DIR / "injection_risk_score_plot.png", "Injection operational risk proxy")

    _write_report(summary, OUTPUT_DIR / "risk_score_report.md")

    print("Risk score generation completed")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
