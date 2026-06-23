"""Univariate and multivariate outlier detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def detect_iqr_outliers(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, name="iqr_outlier")
    values = pd.to_numeric(df[column], errors="coerce")
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).fillna(False).rename("iqr_outlier")


def detect_zscore_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, name="zscore_outlier")
    values = pd.to_numeric(df[column], errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(False, index=df.index, name="zscore_outlier")
    return (((values - values.mean()) / std).abs() > threshold).fillna(False).rename("zscore_outlier")


def detect_isolation_forest_outliers(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    available = [column for column in feature_cols if column in df.columns]
    if not available or df.empty:
        return pd.Series(False, index=df.index, name="isolation_forest_outlier")
    X = SimpleImputer(strategy="median").fit_transform(df[available])
    X = StandardScaler().fit_transform(X)
    # A fixed, conservative rate makes the report interpretable on this bounded window.
    labels = IsolationForest(contamination=0.02, random_state=42).fit_predict(X)
    return pd.Series(labels == -1, index=df.index, name="isolation_forest_outlier")
