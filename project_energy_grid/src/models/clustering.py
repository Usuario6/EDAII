"""Clustering helpers for standardized hourly profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
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


def evaluate_kmeans_k_range(X, k_values=range(2, 9), silhouette_sample_size: int = 5_000) -> pd.DataFrame:
    """Evaluate candidate counts, sampling silhouette deterministically on large matrices."""
    matrix = np.asarray(X)
    sample_size = min(silhouette_sample_size, len(matrix))
    rows = []
    for k in k_values:
        if k < 2 or k >= len(matrix):
            raise ValueError(f"k must be between 2 and n_samples - 1; received {k}")
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(matrix)
        labels = model.labels_
        rows.append(
            {
                "k": k,
                "silhouette": float(
                    silhouette_score(matrix, labels, sample_size=sample_size, random_state=42)
                ),
                "inertia": float(model.inertia_),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            }
        )
    return pd.DataFrame(rows)


def save_kmeans_diagnostic_plot(diagnostics: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save silhouette and elbow diagnostics for a K-means candidate range."""
    required = {"k", "silhouette", "inertia"}
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        raise ValueError("K-means diagnostics are empty or incomplete")

    ordered = diagnostics.sort_values("k")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ordered["k"], ordered["silhouette"], marker="o", color="#2b6f77")
    axes[0].set(xlabel="Number of clusters (k)", ylabel="Silhouette score", title="Silhouette diagnostic")
    axes[1].plot(ordered["k"], ordered["inertia"], marker="o", color="#cf6a32")
    axes[1].set(xlabel="Number of clusters (k)", ylabel="Inertia", title="Elbow diagnostic")
    for ax in axes:
        ax.set_xticks(ordered["k"])
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    save_figure_with_source(fig, output_path, COURSE_METHOD_SOURCE)
    plt.close(fig)


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
