"""Streamlit dashboard for Portugal Energy Grid Analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


DATASET_CONFIG = {
    "Consumption": {
        "target": "total",
        "gold_path": DATA_DIR / "gold/gold_consumption_enriched.parquet",
        "risk_path": REPORTS_DIR / "risk/consumption_risk_score.csv",
        "multistep_path": REPORTS_DIR / "multistep/consumption_multistep_summary.csv",
    },
    "Injection": {
        "target": "total_injection",
        "gold_path": DATA_DIR / "gold/gold_injection_enriched.parquet",
        "risk_path": REPORTS_DIR / "risk/injection_risk_score.csv",
        "multistep_path": REPORTS_DIR / "multistep/injection_multistep_summary.csv",
    },
}


st.set_page_config(
    page_title="Portugal Energy Grid Analytics",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def parse_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    result = df.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    result = result.dropna(subset=[column]).sort_values(column)
    return result


def metric_value(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    return str(value)


def filter_date_range(df: pd.DataFrame, start, end, column: str = "datetime") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df

    start_ts = pd.Timestamp(start).tz_localize("UTC")
    end_ts = pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return df[(df[column] >= start_ts) & (df[column] <= end_ts)].copy()


def show_overview(dataset_name: str, data: pd.DataFrame, risk: pd.DataFrame, target_col: str) -> None:
    st.subheader("Current dataset overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Rows", f"{len(data):,}" if not data.empty else "n/a")
    c2.metric("Risk rows", f"{len(risk):,}" if not risk.empty else "n/a")

    if not data.empty and target_col in data.columns:
        c3.metric("Missing target", f"{int(data[target_col].isna().sum()):,}")
    else:
        c3.metric("Missing target", "n/a")

    if not risk.empty and "risk_score" in risk.columns:
        c4.metric("Mean risk", metric_value(pd.to_numeric(risk["risk_score"], errors="coerce").mean()))
        c5.metric("Max risk", metric_value(pd.to_numeric(risk["risk_score"], errors="coerce").max()))
    else:
        c4.metric("Mean risk", "n/a")
        c5.metric("Max risk", "n/a")


def show_time_series(data: pd.DataFrame, risk: pd.DataFrame, target_col: str) -> None:
    st.subheader("Time-series view")

    if data.empty:
        st.warning("Gold enriched dataset not found. Run the data pipeline first.")
        return

    if target_col not in data.columns:
        st.warning(f"Target column not found: {target_col}")
        return

    series = data[["datetime", target_col]].dropna().set_index("datetime")
    st.line_chart(series, use_container_width=True)

    if not risk.empty and {"datetime", "risk_score"}.issubset(risk.columns):
        risk_series = risk[["datetime", "risk_score"]].dropna().set_index("datetime")
        st.line_chart(risk_series, use_container_width=True)


def show_risk_analysis(risk: pd.DataFrame) -> None:
    st.subheader("Operational risk proxy")

    if risk.empty:
        st.warning("Risk-score files not found. Run `python -m scripts.build_risk_score` first.")
        return

    level_counts = risk["risk_level"].value_counts().reset_index()
    level_counts.columns = ["risk_level", "count"]

    c1, c2 = st.columns([1, 2])

    with c1:
        st.write("Risk-level distribution")
        st.dataframe(level_counts, use_container_width=True, hide_index=True)

    with c2:
        st.write("Highest-risk timestamps")
        cols = [
            "datetime",
            "risk_score",
            "risk_level",
            "target_value",
            "pressure_score",
            "change_score",
            "seasonal_deviation_score",
            "outlier_score",
            "weather_score",
        ]
        available = [col for col in cols if col in risk.columns]
        top = risk.sort_values("risk_score", ascending=False)[available].head(20)
        st.dataframe(top, use_container_width=True, hide_index=True)


def show_model_results(multistep: pd.DataFrame) -> None:
    st.subheader("Direct multi-step forecasting")

    if multistep.empty:
        st.warning("Multi-step reports not found. Run the multi-step scripts first.")
        return

    required = {"horizon", "scenario", "model", "mae", "rmse", "mape", "r2"}
    if not required.issubset(multistep.columns):
        st.warning("Multi-step summary does not contain the expected columns.")
        return

    best = multistep.loc[multistep.groupby("horizon")["rmse"].idxmin()]
    best = best.sort_values("horizon")

    st.write("Best model per horizon")
    st.dataframe(
        best[["horizon", "scenario", "model", "mae", "rmse", "mape", "r2"]],
        use_container_width=True,
        hide_index=True,
    )

    st.write("All multi-step scenarios")
    st.dataframe(
        multistep[["horizon", "scenario", "model", "mae", "rmse", "mape", "r2"]],
        use_container_width=True,
        hide_index=True,
    )


def show_weather_summary() -> None:
    st.subheader("Weather alignment")

    path = REPORTS_DIR / "weather/weather_alignment_summary.csv"
    summary = load_csv(str(path))

    if summary.empty:
        st.warning("Weather alignment summary not found.")
        return

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info(
        "IPMA remains the official Portuguese weather source for observations, warnings and forecasts. "
        "Open-Meteo historical reanalysis is used only to provide hourly 2024–2025 weather features aligned "
        "with the E-REDES modelling window."
    )


def main() -> None:
    st.title("Portugal Energy Grid Analytics")
    st.caption("E-REDES + IPMA + Open-Meteo | Forecasting, risk proxy and weather-aware analysis")

    dataset_name = st.sidebar.selectbox("Dataset", list(DATASET_CONFIG))
    config = DATASET_CONFIG[dataset_name]

    data = parse_datetime(load_parquet(str(config["gold_path"])))
    risk = parse_datetime(load_csv(str(config["risk_path"])))
    multistep = load_csv(str(config["multistep_path"]))

    if not data.empty:
        min_date = data["datetime"].min().date()
        max_date = data["datetime"].max().date()
        selected = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        if isinstance(selected, tuple) and len(selected) == 2:
            start, end = selected
            data = filter_date_range(data, start, end)
            risk = filter_date_range(risk, start, end)

    show_overview(dataset_name, data, risk, config["target"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "Time series",
        "Risk proxy",
        "Models",
        "Weather alignment",
    ])

    with tab1:
        show_time_series(data, risk, config["target"])

    with tab2:
        show_risk_analysis(risk)

    with tab3:
        show_model_results(multistep)

    with tab4:
        show_weather_summary()


if __name__ == "__main__":
    main()
