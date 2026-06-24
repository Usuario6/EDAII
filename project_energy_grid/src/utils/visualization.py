"""Shared source-footnote helpers for static and interactive figures."""

from __future__ import annotations

from pathlib import Path


CONSUMPTION_HOURLY_SOURCE = (
    "Source: E-REDES consumo-total-nacional, hourly gold dataset, 2024-2025."
)
INJECTION_HOURLY_SOURCE = (
    "Source: E-REDES energia-injetada-na-rede-de-distribuicao, hourly gold dataset, 2024-2025."
)
WEATHER_SOURCE = (
    "Source: Open-Meteo historical reanalysis, national proxy from seven Portuguese locations, 2024-2025."
)
WEATHER_ENRICHED_SOURCE = (
    "Source: E-REDES hourly data enriched with Open-Meteo historical reanalysis, 2024-2025."
)
RISK_SOURCE = (
    "Source: Derived operational-risk proxy from E-REDES hourly data, lag/rolling features, "
    "outlier indicators and weather flags."
)
COURSE_METHOD_SOURCE = (
    "Source: E-REDES hourly modelling features used for EDA-II course-method analysis."
)


def add_source_footer(fig, source_text: str) -> None:
    """Reserve bottom space and add a small source note to a Matplotlib figure."""
    fig.text(0.01, 0.012, source_text, ha="left", va="bottom", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 1))


def save_figure_with_source(fig, output_path: Path, source_text: str, dpi: int = 150) -> None:
    """Save a Matplotlib figure with a non-overlapping source footer."""
    add_source_footer(fig, source_text)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def add_plotly_source_footer(fig, source_text: str) -> None:
    """Add a source annotation below a Plotly figure without covering the axes."""
    current_bottom = fig.layout.margin.b or 0
    fig.add_annotation(
        text=source_text,
        x=0,
        y=-0.18,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 10, "color": "#666666"},
    )
    fig.update_layout(margin={"b": max(current_bottom, 75)})
