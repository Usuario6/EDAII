"""Tree ensemble regressors."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def train_random_forest_regressor(X_train, y_train):
    estimator = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)
    return _pipeline(estimator).fit(X_train, y_train)


def train_gradient_boosting_regressor(X_train, y_train):
    estimator = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    return _pipeline(estimator).fit(X_train, y_train)


def get_tree_feature_importance(model, feature_names) -> pd.DataFrame:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    return pd.DataFrame({"feature": list(feature_names), "importance": estimator.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)
