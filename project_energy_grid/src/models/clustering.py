"""Clustering helpers for standardized hourly profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _matrix(df: pd.DataFrame, feature_cols: list[str]):
    available = [column for column in feature_cols if column in df.columns]
    if not available:
        raise ValueError("No clustering features are available")
    X = SimpleImputer(strategy="median").fit_transform(df[available])
    return StandardScaler().fit_transform(X), available


def run_kmeans(df: pd.DataFrame, feature_cols: list[str], n_clusters: int = 3) -> dict:
    X, available = _matrix(df, feature_cols)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(X)
    return {"labels": model.labels_, "X": X, "model": model, "features": available}


def run_dbscan(df: pd.DataFrame, feature_cols: list[str], eps: float = 0.5, min_samples: int = 5) -> dict:
    X, available = _matrix(df, feature_cols)
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return {"labels": model.labels_, "X": X, "model": model, "features": available}


def evaluate_clustering_silhouette(X, labels) -> float:
    labels = np.asarray(labels)
    clusters = set(labels) - {-1}
    valid = labels != -1
    if len(clusters) < 2 or valid.sum() <= len(clusters):
        return np.nan
    return float(silhouette_score(np.asarray(X)[valid], labels[valid]))
