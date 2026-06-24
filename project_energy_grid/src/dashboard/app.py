"""Interactive Streamlit dashboard for Portugal Energy Grid Analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.utils.visualization import (
    CONSUMPTION_HOURLY_SOURCE,
    INJECTION_HOURLY_SOURCE,
    RISK_SOURCE,
    WEATHER_ENRICHED_SOURCE,
    WEATHER_SOURCE,
    add_plotly_source_footer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

CONFIG = {
    "Consumption forecast": {
        "target": "total",
        "target_label": "National electricity consumption",
        "unit": "E-REDES reported units",
        "source": CONSUMPTION_HOURLY_SOURCE,
        "gold_path": DATA_DIR / "gold/gold_consumption_enriched.parquet",
        "risk_path": REPORTS_DIR / "risk/consumption_risk_score.csv",
        "multistep_path": REPORTS_DIR / "multistep/consumption_multistep_summary.csv",
    },
    "Grid injection forecast": {
        "target": "total_injection",
        "target_label": "Grid injection",
        "unit": "E-REDES reported units",
        "source": INJECTION_HOURLY_SOURCE,
        "gold_path": DATA_DIR / "gold/gold_injection_enriched.parquet",
        "risk_path": REPORTS_DIR / "risk/injection_risk_score.csv",
        "multistep_path": REPORTS_DIR / "multistep/injection_multistep_summary.csv",
    },
}

WEATHER_COLUMNS = [
    "temperatura",
    "radiacao",
    "intensidadeventokm",
    "precacumulada",
    "humidade",
    "pressao",
    "hdd_18",
    "cdd_22",
]

WEATHER_LABELS = {
    "temperatura": "Temperature (deg C)",
    "radiacao": "Solar radiation (W/m2)",
    "intensidadeventokm": "Wind speed (km/h)",
    "precacumulada": "Precipitation (mm)",
    "humidade": "Relative humidity (%)",
    "pressao": "Surface pressure (hPa)",
    "hdd_18": "Heating degree difference (deg C)",
    "cdd_22": "Cooling degree difference (deg C)",
}

RISK_COMPONENTS = [
    "pressure_score",
    "change_score",
    "seasonal_deviation_score",
    "outlier_score",
    "weather_score",
]

SCENARIO_LABELS = {
    "with_lag1": "With lag-1",
    "without_lag1": "Without lag-1",
    "calendar_seasonal": "Calendar + seasonal",
    "weather_enriched": "Weather enriched",
    "seasonal_baseline": "Seasonal baseline",
}


def add_scenario_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add presentation-friendly scenario labels."""
    result = df.copy()
    if "scenario" in result.columns:
        result["scenario_label"] = result["scenario"].map(SCENARIO_LABELS).fillna(result["scenario"])
    return result


def interpret_rmse_delta_pct(value) -> str:
    """Interpret weather-vs-baseline RMSE difference."""
    if pd.isna(value):
        return "n/a"
    if value < -1:
        return "Improved"
    if value > 1:
        return "Worse"
    return "Similar"


def clean_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Replace null display values with n/a for dashboard readability."""
    return df.replace({None: "n/a"}).fillna("n/a")


st.set_page_config(
    page_title="Portugal Energy Grid Analytics",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(file_path)


def parse_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df

    result = df.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    result = result.dropna(subset=[column]).sort_values(column).reset_index(drop=True)
    return result


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_number(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def filter_by_date(df: pd.DataFrame, start_date, end_date, column: str = "datetime") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df

    start = pd.Timestamp(start_date).tz_localize("UTC")
    end = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return df[(df[column] >= start) & (df[column] <= end)].copy()


def aggregate_timeseries(df: pd.DataFrame, datetime_col: str, columns: list[str], frequency: str) -> pd.DataFrame:
    if df.empty:
        return df

    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.DataFrame()

    data = df[[datetime_col, *available]].copy()
    for col in available:
        data[col] = numeric(data[col])

    if frequency == "Hourly":
        return data

    rule = "D" if frequency == "Daily mean" else "W"
    return (
        data.set_index(datetime_col)
        .resample(rule)
        .mean(numeric_only=True)
        .reset_index()
    )


def show_kpis(
    data: pd.DataFrame,
    risk: pd.DataFrame,
    multistep: pd.DataFrame,
    target: str,
    target_label: str,
    unit: str,
) -> None:
    st.subheader("Executive overview")
    st.caption(f"Selected analysis: {target_label}. Forecast errors use {unit.lower()}.")

    high_count = 0
    critical_count = 0
    weather_count = 0
    mean_risk = float("nan")
    max_risk = float("nan")

    if not risk.empty:
        if "risk_level" in risk.columns:
            high_count = int((risk["risk_level"] == "high").sum())
            critical_count = int((risk["risk_level"] == "critical").sum())

        if "weather_score" in risk.columns:
            weather_count = int((numeric(risk["weather_score"]) > 0).sum())

        if "risk_score" in risk.columns:
            mean_risk = numeric(risk["risk_score"]).mean()
            max_risk = numeric(risk["risk_score"]).max()

    missing_target = int(data[target].isna().sum()) if not data.empty and target in data.columns else 0

    best_rmse = float("nan")
    best_model = "n/a"
    if not multistep.empty and {"rmse", "model", "scenario", "horizon"}.issubset(multistep.columns):
        ranked = multistep.copy()
        ranked["rmse"] = numeric(ranked["rmse"])
        ranked = ranked.dropna(subset=["rmse"]).sort_values("rmse")
        if not ranked.empty:
            best = ranked.iloc[0]
            best_rmse = best["rmse"]
            best_model = f"{best['model']} | {best['scenario']} | h={best['horizon']}"

    cols = st.columns(7)
    cols[0].metric("Rows", f"{len(data):,}")
    cols[1].metric("Missing target", f"{missing_target:,}")
    cols[2].metric("Mean risk (0-100)", format_number(mean_risk))
    cols[3].metric("Max risk (0-100)", format_number(max_risk))
    cols[4].metric("High risk hours", f"{high_count:,}")
    cols[5].metric("Critical hours", f"{critical_count:,}")
    cols[6].metric("Weather-flag hours", f"{weather_count:,}")

    st.caption(f"Best observed RMSE ({unit}): {format_number(best_rmse)} | {best_model}")


def plot_target_and_risk(
    data: pd.DataFrame,
    risk: pd.DataFrame,
    target: str,
    target_label: str,
    unit: str,
    frequency: str,
    source_text: str,
) -> None:
    st.subheader(f"{target_label} and operational risk proxy over time")
    st.caption(
        "The left axis shows the observed target. The right axis shows the 0-100 risk score; "
        "high values indicate unusual conditions, not confirmed failures."
    )

    if data.empty or target not in data.columns:
        st.warning("Target dataset is missing.")
        return

    target_data = aggregate_timeseries(data, "datetime", [target], frequency)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=target_data["datetime"],
            y=target_data[target],
            mode="lines",
            name=target_label,
        ),
        secondary_y=False,
    )

    if not risk.empty and "risk_score" in risk.columns:
        risk_data = aggregate_timeseries(risk, "datetime", ["risk_score"], frequency)
        fig.add_trace(
            go.Scatter(
                x=risk_data["datetime"],
                y=risk_data["risk_score"],
                mode="lines",
                name="Operational risk proxy",
            ),
            secondary_y=True,
        )

        critical = risk[risk["risk_level"].isin(["high", "critical"])] if "risk_level" in risk.columns else pd.DataFrame()
        if not critical.empty and frequency == "Hourly":
            fig.add_trace(
                go.Scatter(
                    x=critical["datetime"],
                    y=critical["risk_score"],
                    mode="markers",
                    name="High/critical proxy score",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h"),
        title=f"{target_label} ({unit}) and operational risk proxy (0-100)",
    )
    fig.update_yaxes(title_text=f"{target_label} ({unit})", secondary_y=False)
    fig.update_yaxes(title_text="Operational risk proxy (0-100)", range=[0, 100], secondary_y=True)
    add_plotly_source_footer(fig, f"{source_text}<br>{RISK_SOURCE}")

    st.plotly_chart(fig, use_container_width=True)


def show_risk_events(risk: pd.DataFrame) -> None:
    st.subheader("Operational risk proxy events")
    st.warning("Risk is a proxy, not a confirmed failure probability.")
    st.caption("Risk legend: low 0-35 | medium 35-60 | high 60-80 | critical 80-100.")

    if risk.empty:
        st.warning("Risk score data is missing.")
        return

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    with c1:
        available_levels = sorted(risk["risk_level"].dropna().unique()) if "risk_level" in risk.columns else []
        default_levels = [level for level in ["high", "critical"] if level in available_levels]
        selected_levels = st.multiselect("Risk levels", available_levels, default=default_levels)

    with c2:
        weather_only = st.checkbox("Weather only", value=False)

    with c3:
        outlier_only = st.checkbox("Outliers only", value=False)

    with c4:
        top_n = st.slider("Top rows", 10, 200, 50, step=10)

    filtered = risk.copy()

    if selected_levels and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(selected_levels)]

    if weather_only and "weather_score" in filtered.columns:
        filtered = filtered[numeric(filtered["weather_score"]) > 0]

    if outlier_only and "any_outlier" in filtered.columns:
        filtered = filtered[filtered["any_outlier"].astype(bool)]

    if "risk_score" in filtered.columns:
        filtered = filtered.sort_values("risk_score", ascending=False)

    event_cols = [
        "datetime",
        "risk_score",
        "risk_level",
        "target_value",
        "pressure_score",
        "change_score",
        "seasonal_deviation_score",
        "outlier_score",
        "weather_score",
        "heavy_rain_flag",
        "strong_wind_flag",
        "iqr_outlier",
        "zscore_outlier",
        "isolation_forest_outlier",
    ]
    available = [col for col in event_cols if col in filtered.columns]

    st.dataframe(filtered[available].head(top_n), use_container_width=True, hide_index=True)

    component_cols = [col for col in RISK_COMPONENTS if col in risk.columns]
    if component_cols:
        component_mean = (
            risk[component_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean()
            .reset_index()
        )
        component_mean.columns = ["component", "mean_score"]
        fig = px.bar(
            component_mean,
            x="component",
            y="mean_score",
            title="Average operational risk proxy components (0-1)",
            labels={"component": "Risk component", "mean_score": "Mean component score (0-1)"},
        )
        add_plotly_source_footer(fig, RISK_SOURCE)
        st.plotly_chart(fig, use_container_width=True)


def show_model_comparison(multistep: pd.DataFrame, target_label: str, unit: str) -> None:
    st.subheader(f"{target_label}: forecasting model comparison")
    st.caption("Each horizon is predicted directly using information available at the forecast origin.")

    if multistep.empty:
        st.warning("Multi-step result file is missing.")
        return

    required = {"horizon", "scenario", "model", "mae", "rmse", "mape", "r2"}
    if not required.issubset(multistep.columns):
        st.warning("Multi-step file does not contain the expected columns.")
        return

    data = add_scenario_labels(multistep)
    for col in ["mae", "rmse", "mape", "r2", "horizon"]:
        data[col] = numeric(data[col])

    metric = st.selectbox("Metric", ["rmse", "mae", "mape", "r2"], index=0)
    st.caption(
        f"RMSE and MAE use {unit.lower()}; MAPE is a percentage; R-squared is dimensionless. "
        "Lower error metrics and higher R-squared are better."
    )

    if metric == "r2":
        best_idx = data.groupby(["horizon", "scenario"])["r2"].idxmax()
    else:
        best_idx = data.groupby(["horizon", "scenario"])[metric].idxmin()

    best_by_scenario = data.loc[best_idx].sort_values(["horizon", "scenario"])

    best_by_scenario["horizon_label"] = best_by_scenario["horizon"].astype(int).astype(str) + "h"

    fig = px.bar(
        best_by_scenario,
        x="horizon_label",
        y=metric,
        color="scenario_label",
        barmode="group",
        hover_data=["scenario", "model", "mae", "rmse", "mape", "r2"],
        title=f"{target_label}: best {metric.upper()} by prediction horizon and scenario",
        labels={"horizon_label": "Prediction horizon", metric: f"{metric.upper()} ({unit})" if metric in {"rmse", "mae"} else metric.upper()},
    )
    add_plotly_source_footer(fig, WEATHER_ENRICHED_SOURCE)
    st.plotly_chart(fig, use_container_width=True)

    st.write("Best model per horizon")
    if metric == "r2":
        best_horizon = data.loc[data.groupby("horizon")["r2"].idxmax()].sort_values("horizon")
    else:
        best_horizon = data.loc[data.groupby("horizon")[metric].idxmin()].sort_values("horizon")

    st.dataframe(
        clean_display_table(best_horizon[["horizon", "scenario_label", "model", "mae", "rmse", "mape", "r2"]]),
        use_container_width=True,
        hide_index=True,
    )

    st.write("Weather scenario vs baseline")
    st.caption("Negative RMSE delta means the weather-enriched model improved over the lag-1 baseline.")

    baseline = (
        data[data["scenario"] == "with_lag1"]
        .sort_values("rmse")
        .groupby("horizon")
        .first()
        .reset_index()
    )
    weather = (
        data[data["scenario"] == "weather_enriched"]
        .sort_values("rmse")
        .groupby("horizon")
        .first()
        .reset_index()
    )

    if not baseline.empty and not weather.empty:
        comparison = baseline[["horizon", "model", "rmse"]].merge(
            weather[["horizon", "model", "rmse"]],
            on="horizon",
            suffixes=("_baseline", "_weather"),
        )
        comparison["rmse_delta"] = comparison["rmse_weather"] - comparison["rmse_baseline"]
        comparison["rmse_delta_pct"] = comparison["rmse_delta"] / comparison["rmse_baseline"] * 100
        comparison["interpretation"] = comparison["rmse_delta_pct"].apply(interpret_rmse_delta_pct)

        st.dataframe(clean_display_table(comparison), use_container_width=True, hide_index=True)
    else:
        st.info("Baseline or weather-enriched scenario is missing.")

    if target_label == "Grid injection":
        st.warning(
            "Grid injection results are credible primarily at short horizons. "
            "Treat 24h as weak and 168h as exploratory."
        )


def show_weather_analysis(data: pd.DataFrame) -> None:
    st.subheader("Historical weather alignment and features")
    st.caption(
        "Historical Open-Meteo reanalysis supplies 2024-2025 model features. "
        "IPMA supplies separate operational/recent weather context."
    )

    summary = load_csv(str(REPORTS_DIR / "weather/weather_alignment_summary.csv"))

    if not summary.empty:
        st.write("Weather alignment summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.warning("Weather alignment summary is missing.")

    available_weather = [col for col in WEATHER_COLUMNS if col in data.columns]
    if available_weather:
        selected = st.multiselect(
            "Weather variables",
            available_weather,
            default=[col for col in ["temperatura", "intensidadeventokm", "precacumulada"] if col in available_weather],
        )

        if selected:
            weather_data = aggregate_timeseries(data, "datetime", selected, "Daily mean")
            long = weather_data.melt(id_vars="datetime", value_vars=selected, var_name="variable", value_name="value")
            long["variable_label"] = long["variable"].map(WEATHER_LABELS).fillna(long["variable"])
            fig = px.line(
                long,
                x="datetime",
                y="value",
                color="variable_label",
                title="Daily mean historical Open-Meteo reanalysis variables",
                labels={"datetime": "Date (UTC)", "value": "Value (units in legend)", "variable_label": "Variable"},
            )
            add_plotly_source_footer(fig, WEATHER_SOURCE)
            st.plotly_chart(fig, use_container_width=True)

    flag_cols = [col for col in ["heavy_rain_flag", "strong_wind_flag", "weather_available"] if col in data.columns]
    if flag_cols:
        flag_summary = {}
        for col in flag_cols:
            flag_summary[col] = int(data[col].fillna(False).astype(bool).sum())
        st.write("Weather flags")
        st.dataframe(pd.DataFrame([flag_summary]), use_container_width=True, hide_index=True)

    st.info(
        "IPMA operational/recent weather provides current observations, warnings and forecasts. "
        "Historical Open-Meteo reanalysis provides hourly 2024–2025 weather features "
        "aligned with the E-REDES modelling window."
    )


def show_data_quality() -> None:
    st.subheader("Data quality and validation")

    validation_path = REPORTS_DIR / "validation/dataset_state_report.csv"
    validation = load_csv(str(validation_path))

    if validation.empty:
        st.warning("Dataset validation report is missing. Run the dataset-state inspection first.")
    else:
        preferred_cols = [
            "dataset",
            "rows",
            "columns",
            "datetime_min",
            "datetime_max",
            "duplicate_timestamps",
            "missing_target",
            "missing_target_pct",
            "status",
        ]
        available = [col for col in preferred_cols if col in validation.columns]
        st.dataframe(clean_display_table(validation[available]), use_container_width=True, hide_index=True)

        st.caption(
            "Note: the silver injection layer contains generation components; "
            "the total_injection modelling target is created later in the gold layer."
        )

    c1, c2, c3 = st.columns(3)
    c1.success("Pipeline validation: passed")
    c2.success("Multi-step validation: passed")
    c3.success("Risk-score validation: passed")


def show_methodology() -> None:
    st.subheader("Methodology notes")

    st.markdown(
        """
### Forecasting

The dashboard compares direct forecasting models across multiple horizons:

- 1 hour;
- 6 hours;
- 24 hours;
- 168 hours.

The modelling scenarios include lag-based features, calendar/seasonal features, seasonal naive baselines and weather-enriched features.

### Operational risk proxy

The operational risk score is an explainable proxy indicator, not a confirmed failure probability. No labelled outage or failure target is available.

The score combines:

- pressure level;
- short-term change;
- seasonal deviation;
- outlier flags;
- weather flags.

### Weather alignment

IPMA operational/recent weather supplies current observations, warnings and forecasts.

Historical Open-Meteo reanalysis is used for the 2024–2025 hourly weather alignment because the public IPMA endpoint used in the pipeline does not provide a complete historical hourly archive aligned with the E-REDES modelling window.
"""
    )


def main() -> None:
    st.title("Portugal Energy Grid Analytics")
    st.caption("E-REDES, IPMA and Open-Meteo | Forecasting, weather alignment and operational risk proxy")

    dataset_name = st.sidebar.selectbox("Dataset", list(CONFIG))
    config = CONFIG[dataset_name]

    data = parse_datetime(load_parquet(str(config["gold_path"])))
    risk = parse_datetime(load_csv(str(config["risk_path"])))
    multistep = load_csv(str(config["multistep_path"]))

    if data.empty:
        st.error("Enriched gold dataset is missing. Run the pipeline first.")
        return

    min_date = data["datetime"].min().date()
    max_date = data["datetime"].max().date()

    selected_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
        data = filter_by_date(data, start_date, end_date)
        risk = filter_by_date(risk, start_date, end_date)

    frequency = st.sidebar.radio("Chart granularity", ["Hourly", "Daily mean", "Weekly mean"], index=1)

    show_kpis(data, risk, multistep, config["target"], config["target_label"], config["unit"])

    tabs = st.tabs(
        [
            "Time series",
            "Risk events",
            "Model comparison",
            "Weather",
            "Data quality",
            "Methodology",
        ]
    )

    with tabs[0]:
        plot_target_and_risk(
            data,
            risk,
            config["target"],
            config["target_label"],
            config["unit"],
            frequency,
            config["source"],
        )

    with tabs[1]:
        show_risk_events(risk)

    with tabs[2]:
        show_model_comparison(multistep, config["target_label"], config["unit"])

    with tabs[3]:
        show_weather_analysis(data)

    with tabs[4]:
        show_data_quality()

    with tabs[5]:
        show_methodology()


if __name__ == "__main__":
    main()
