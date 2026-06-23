"""Shared chronological baseline modelling workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.models.ensemble import get_tree_feature_importance, train_gradient_boosting_regressor, train_random_forest_regressor
from src.models.evaluation import bootstrap_metric_ci, compare_models, evaluate_regression, time_series_train_test_split
from src.models.regularized import train_lasso_model, train_ridge_model


def run_target_baselines(
    data_path: Path,
    target: str,
    feature_cols: list[str],
    comparison_path: Path,
    importance_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(data_path)
    available = [column for column in feature_cols if column in df.columns and column != target]
    required = ["datetime", target, f"{target}_lag_24", *available]
    data = df[list(dict.fromkeys(required))].dropna(subset=[target, f"{target}_lag_24"])
    train, test = time_series_train_test_split(data, "datetime", test_size=0.2)
    X_train, y_train = train[available], train[target]
    X_test, y_test = test[available], test[target]

    rows = []
    naive_pred = test[f"{target}_lag_24"].to_numpy()
    naive_metrics = evaluate_regression(y_test, naive_pred)
    naive_ci = bootstrap_metric_ci(y_test.to_numpy(), naive_pred, mean_absolute_error)
    rows.append({"model": "seasonal_naive_lag_24", **naive_metrics, "mae_ci_lower": naive_ci[0], "mae_ci_upper": naive_ci[1], "n_train": len(train), "n_test": len(test)})

    models = {
        "ridge": train_ridge_model(X_train, y_train),
        "lasso": train_lasso_model(X_train, y_train),
        "random_forest": train_random_forest_regressor(X_train, y_train),
        "gradient_boosting": train_gradient_boosting_regressor(X_train, y_train),
    }
    for name, model in models.items():
        prediction = model.predict(X_test)
        metrics = evaluate_regression(y_test, prediction)
        ci = bootstrap_metric_ci(y_test.to_numpy(), prediction, mean_absolute_error)
        rows.append({"model": name, **metrics, "mae_ci_lower": ci[0], "mae_ci_upper": ci[1], "n_train": len(train), "n_test": len(test)})

    comparison = compare_models(rows)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    importance = get_tree_feature_importance(models["random_forest"], available)
    importance.to_csv(importance_path, index=False)
    return comparison, importance
