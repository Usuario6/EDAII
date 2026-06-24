"""Clustering helpers for standardized hourly profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.visualization import COURSE_METHOD_SOURCE, save_figure_with_source


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


def save_cluster_projection_plot(X, labels, output_path: Path, title: str) -> None:
    """Save a two-dimensional cluster projection with categorical colors."""
    coordinates = PCA(n_components=2, random_state=42).fit_transform(X)
    labels = np.asarray(labels)
    cluster_labels = sorted(label for label in np.unique(labels) if label != -1)
    ordered_labels = [*cluster_labels, *([-1] if -1 in labels else [])]
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for color_index, cluster_label in enumerate(ordered_labels):
        mask = labels == cluster_label
        display_label = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}"
        color = "#7f7f7f" if cluster_label == -1 else colors(color_index % colors.N)
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=color,
            label=display_label,
            s=11,
            alpha=0.55,
        )

    ax.set(xlabel="Profile PC1", ylabel="Profile PC2", title=title)
    ax.legend(title="Cluster")
    save_figure_with_source(fig, output_path, COURSE_METHOD_SOURCE)
    plt.close(fig)
