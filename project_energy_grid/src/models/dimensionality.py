"""PCA utilities for standardized numeric energy features."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils.visualization import COURSE_METHOD_SOURCE, save_figure_with_source


def run_pca(df: pd.DataFrame, feature_cols: list[str], n_components: int = 2) -> dict:
    available = [column for column in feature_cols if column in df.columns]
    if not available:
        raise ValueError("No PCA features are available")
    X = SimpleImputer(strategy="median").fit_transform(df[available])
    X = StandardScaler().fit_transform(X)
    component_count = min(n_components, len(available), len(df))
    model = PCA(n_components=component_count, random_state=42)
    scores = model.fit_transform(X)
    score_df = pd.DataFrame(scores, index=df.index, columns=[f"pc{i + 1}" for i in range(component_count)])
    loadings = pd.DataFrame(model.components_.T, index=available, columns=score_df.columns).reset_index(names="feature")
    variance = pd.DataFrame({"component": score_df.columns, "explained_variance_ratio": model.explained_variance_ratio_, "cumulative_variance": model.explained_variance_ratio_.cumsum()})
    return {"scores": score_df, "loadings": loadings, "variance": variance, "model": model, "features": available}


def save_pca_outputs(result: dict, components_path: Path, plot_path: Path, title: str) -> None:
    components_path.parent.mkdir(parents=True, exist_ok=True)
    result["loadings"].to_csv(components_path, index=False)
    save_pca_plot(result, plot_path, title)


def save_pca_plot(result: dict, plot_path: Path, title: str) -> None:
    """Save the PCA score plot without rewriting tabular PCA outputs."""
    scores = result["scores"]
    if {"pc1", "pc2"}.issubset(scores.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(scores["pc1"], scores["pc2"], s=10, alpha=0.45)
        ax.set(xlabel="PC1", ylabel="PC2", title=title)
        save_figure_with_source(fig, plot_path, COURSE_METHOD_SOURCE)
        plt.close(fig)
