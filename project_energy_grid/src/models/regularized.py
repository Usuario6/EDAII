"""Interpretable regularized linear regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])


def train_ridge_model(X_train, y_train):
    return _pipeline(RidgeCV(alphas=np.logspace(-3, 3, 25), cv=TimeSeriesSplit(5))).fit(X_train, y_train)


def train_lasso_model(X_train, y_train):
    return _pipeline(LassoCV(alphas=np.logspace(-4, 2, 40), cv=TimeSeriesSplit(5), max_iter=20_000, random_state=42)).fit(X_train, y_train)


def get_linear_feature_importance(model, feature_names) -> pd.DataFrame:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    coefficients = np.ravel(estimator.coef_)
    return pd.DataFrame({"feature": list(feature_names), "coefficient": coefficients, "importance": np.abs(coefficients)}).sort_values("importance", ascending=False).reset_index(drop=True)
