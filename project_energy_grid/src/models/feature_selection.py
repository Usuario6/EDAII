"""Defensive filter and model-based feature selection."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


def correlation_feature_filter(df: pd.DataFrame, target_col: str, threshold: float = 0.05) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Missing target: {target_col}")
    numeric = df.select_dtypes(include="number")
    if target_col not in numeric:
        return pd.DataFrame(columns=["feature", "correlation", "absolute_correlation"])
    correlations = numeric.corr()[target_col].drop(labels=[target_col]).dropna()
    result = pd.DataFrame({"feature": correlations.index, "correlation": correlations.values})
    result["absolute_correlation"] = result["correlation"].abs()
    return result[result["absolute_correlation"] >= threshold].sort_values("absolute_correlation", ascending=False).reset_index(drop=True)


def random_forest_feature_importance(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    available = [column for column in feature_cols if column in df.columns and column != target_col]
    data = df[available + [target_col]].dropna(subset=[target_col])
    if not available or data.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    X = SimpleImputer(strategy="median").fit_transform(data[available])
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(X, data[target_col])
    return pd.DataFrame({"feature": available, "importance": model.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)


def select_top_features(importance_df: pd.DataFrame, top_n: int = 10) -> list[str]:
    if "feature" not in importance_df.columns or top_n < 1:
        return []
    return importance_df.head(top_n)["feature"].tolist()
