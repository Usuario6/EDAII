"""Operational risk proxy scoring for energy-grid forecasting outputs.

This module does NOT predict real grid failures. It creates an explainable
proxy score from observable pressure, deviations, outliers, and weather flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def robust_minmax(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Scale a numeric series to [0, 1] using robust quantile clipping."""
    values = _numeric(series)
    valid = values.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)

    lo = valid.quantile(lower_q)
    hi = valid.quantile(upper_q)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=series.index)

    return ((values.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1).fillna(0.0)


def safe_relative_abs_diff(current: pd.Series, reference: pd.Series) -> pd.Series:
    """Absolute relative difference using a scale-safe denominator."""
    current = _numeric(current)
    reference = _numeric(reference)

    scale = np.nanmedian(np.abs(current))
    eps = max(np.finfo(float).eps, scale * 1e-8)

    denominator = reference.abs().clip(lower=eps)
    return ((current - reference).abs() / denominator).replace([np.inf, -np.inf], np.nan)


def detect_outlier_flags(df: pd.DataFrame, target_col: str, contamination: float = 0.02) -> pd.DataFrame:
    """Detect outliers using IQR, z-score and Isolation Forest."""
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    result = pd.DataFrame(index=df.index)
    target = _numeric(df[target_col])

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1

    if pd.notna(iqr) and iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result["iqr_outlier"] = (target < lower) | (target > upper)
    else:
        result["iqr_outlier"] = False

    std = target.std(skipna=True)
    mean = target.mean(skipna=True)
    if pd.notna(std) and std > 0:
        zscore = (target - mean).abs() / std
        result["zscore_outlier"] = zscore > 3.0
    else:
        result["zscore_outlier"] = False

    feature_candidates = [
        target_col,
        f"{target_col}_lag_1",
        f"{target_col}_lag_24",
        f"{target_col}_lag_168",
        f"{target_col}_rollmean_24",
        f"{target_col}_rollmean_168",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
    ]
    feature_cols = [column for column in feature_candidates if column in df.columns]

    if len(feature_cols) >= 2 and len(df) >= 100:
        features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
        model = IsolationForest(contamination=contamination, random_state=42)
        result["isolation_forest_outlier"] = model.fit_predict(features) == -1
    else:
        result["isolation_forest_outlier"] = False

    result["any_outlier"] = result[["iqr_outlier", "zscore_outlier", "isolation_forest_outlier"]].any(axis=1)
    return result


def _risk_level(score: float | int | np.floating) -> str:
    if pd.isna(score):
        return "missing_target"
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def build_risk_score(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Build an explainable operational risk proxy score.

    Components:
    - pressure_score: high current target values relative to historical range;
    - change_score: abnormal short-term or daily change;
    - seasonal_deviation_score: deviation from daily/weekly/rolling references;
    - outlier_score: IQR, z-score or Isolation Forest outlier flag;
    - weather_score: optional strong wind / heavy rain flags if available.

    The score is a proxy, not a labelled failure probability.
    """
    if "datetime" not in df.columns:
        raise KeyError("Missing datetime column")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    result = df.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce", utc=True)
    result = result.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    target = _numeric(result[target_col])
    result["missing_target"] = target.isna()

    lag_1 = _numeric(result.get(f"{target_col}_lag_1", target.shift(1)))
    lag_24 = _numeric(result.get(f"{target_col}_lag_24", target.shift(24)))
    lag_168 = _numeric(result.get(f"{target_col}_lag_168", target.shift(168)))
    roll_24 = _numeric(result.get(f"{target_col}_rollmean_24", target.shift(1).rolling(24, min_periods=1).mean()))
    roll_168 = _numeric(result.get(f"{target_col}_rollmean_168", target.shift(1).rolling(168, min_periods=1).mean()))

    pressure_score = robust_minmax(target)

    change_1 = safe_relative_abs_diff(target, lag_1)
    change_24 = safe_relative_abs_diff(target, lag_24)
    change_score = robust_minmax(pd.concat([change_1, change_24], axis=1).max(axis=1))

    deviation_24 = safe_relative_abs_diff(target, lag_24)
    deviation_168 = safe_relative_abs_diff(target, lag_168)
    deviation_roll_24 = safe_relative_abs_diff(target, roll_24)
    deviation_roll_168 = safe_relative_abs_diff(target, roll_168)
    seasonal_deviation_score = robust_minmax(
        pd.concat([deviation_24, deviation_168, deviation_roll_24, deviation_roll_168], axis=1).max(axis=1)
    )

    outliers = detect_outlier_flags(result, target_col)
    outlier_score = outliers["any_outlier"].astype(float)

    weather_available = result.get("weather_available", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    heavy_rain = result.get("heavy_rain_flag", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    strong_wind = result.get("strong_wind_flag", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    weather_score = (weather_available & (heavy_rain | strong_wind)).astype(float)

    risk_score = 100 * (
        0.30 * pressure_score
        + 0.25 * seasonal_deviation_score
        + 0.20 * change_score
        + 0.15 * outlier_score
        + 0.10 * weather_score
    )

    risk_score = risk_score.clip(0, 100)
    risk_score[target.isna()] = np.nan

    output = pd.DataFrame(
        {
            "datetime": result["datetime"],
            "dataset": dataset_name,
            "target_column": target_col,
            "target_value": target,
            "risk_score": risk_score.round(3),
            "risk_level": risk_score.map(_risk_level),
            "pressure_score": pressure_score.round(4),
            "change_score": change_score.round(4),
            "seasonal_deviation_score": seasonal_deviation_score.round(4),
            "outlier_score": outlier_score.round(4),
            "weather_score": weather_score.round(4),
            "weather_available": weather_available,
            "heavy_rain_flag": heavy_rain,
            "strong_wind_flag": strong_wind,
            "missing_target": result["missing_target"],
            "lag_1_value": lag_1,
            "lag_24_value": lag_24,
            "lag_168_value": lag_168,
            "rollmean_24_value": roll_24,
            "rollmean_168_value": roll_168,
        }
    )

    output = pd.concat([output, outliers.reset_index(drop=True)], axis=1)

    return output


def summarize_risk_scores(*frames: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for one or more risk-score outputs."""
    rows = []
    for frame in frames:
        if frame.empty:
            continue
        dataset = frame["dataset"].iloc[0]
        scored = frame["risk_score"].dropna()

        counts = frame["risk_level"].value_counts().to_dict()
        rows.append(
            {
                "dataset": dataset,
                "rows": len(frame),
                "scored_rows": int(scored.size),
                "missing_target_rows": int(frame["missing_target"].sum()),
                "mean_risk_score": float(scored.mean()) if not scored.empty else np.nan,
                "max_risk_score": float(scored.max()) if not scored.empty else np.nan,
                "low_count": int(counts.get("low", 0)),
                "medium_count": int(counts.get("medium", 0)),
                "high_count": int(counts.get("high", 0)),
                "critical_count": int(counts.get("critical", 0)),
                "missing_target_count": int(counts.get("missing_target", 0)),
                "outlier_count": int(frame["any_outlier"].sum()),
                "weather_flag_count": int((frame["weather_score"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)
