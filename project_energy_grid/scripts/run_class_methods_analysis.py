"""Run the EDL II course methods on leakage-safe hourly energy datasets."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from scripts.train_baseline_consumption import FEATURES as CONSUMPTION_FEATURES
from scripts.train_baseline_consumption import main as train_consumption
from scripts.train_baseline_injection import FEATURES as INJECTION_FEATURES
from scripts.train_baseline_injection import main as train_injection
from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging
from src.models.clustering import evaluate_clustering_silhouette, run_dbscan, run_kmeans
from src.models.dimensionality import run_pca, save_pca_outputs
from src.models.feature_selection import correlation_feature_filter, random_forest_feature_importance
from src.models.outliers import detect_iqr_outliers, detect_isolation_forest_outliers, detect_zscore_outliers

LOGGER = logging.getLogger(__name__)


DATASETS = {
    "consumption": {
        "path": GOLD_DATA_DIR / "gold_consumption_hourly.parquet",
        "target": "total",
        "features": CONSUMPTION_FEATURES,
    },
    "injection": {
        "path": GOLD_DATA_DIR / "gold_injection_hourly.parquet",
        "target": "total_injection",
        "features": INJECTION_FEATURES,
    },
}


def _feature_selection(name: str, df: pd.DataFrame, target: str, features: list[str]) -> None:
    output = REPORTS_DIR / "models/feature_selection"
    output.mkdir(parents=True, exist_ok=True)
    correlation_feature_filter(df[[target, *[c for c in features if c in df]]], target).to_csv(output / f"{name}_correlation_filter.csv", index=False)
    random_forest_feature_importance(df, target, features).to_csv(output / f"{name}_rf_importance.csv", index=False)


def _outliers(name: str, df: pd.DataFrame, target: str, features: list[str]) -> dict:
    result = df[[column for column in ["datetime", target] if column in df]].copy()
    result["iqr_outlier"] = detect_iqr_outliers(df, target)
    result["zscore_outlier"] = detect_zscore_outliers(df, target)
    result["isolation_forest_outlier"] = detect_isolation_forest_outliers(df, [target, *features])
    result["any_outlier"] = result[["iqr_outlier", "zscore_outlier", "isolation_forest_outlier"]].any(axis=1)
    output = REPORTS_DIR / "outliers"
    output.mkdir(parents=True, exist_ok=True)
    result[result["any_outlier"]].to_csv(output / f"{name}_outliers.csv", index=False)
    return {"dataset": name, "rows": len(df), **{column: int(result[column].sum()) for column in ["iqr_outlier", "zscore_outlier", "isolation_forest_outlier", "any_outlier"]}}


def _pca(name: str, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    output = REPORTS_DIR / "dimensionality"
    result = run_pca(df, features, n_components=2)
    save_pca_outputs(result, output / f"pca_{name}_components.csv", output / f"pca_{name}_plot.png", f"PCA: {name} hourly profiles")
    variance = result["variance"].copy()
    variance.insert(0, "dataset", name)
    return variance


def _cluster(name: str, df: pd.DataFrame, target: str, features: list[str]) -> list[dict]:
    clustering_features = [column for column in ["hour", "dayofweek", "month", target, *features[-2:]] if column in df.columns]
    kmeans = run_kmeans(df, clustering_features, n_clusters=3)
    dbscan = run_dbscan(df, clustering_features, eps=0.8, min_samples=8)
    output = REPORTS_DIR / "clustering"
    output.mkdir(parents=True, exist_ok=True)
    rows = df[[column for column in ["datetime", target] if column in df]].copy()
    rows["kmeans_cluster"] = kmeans["labels"]
    rows["dbscan_cluster"] = dbscan["labels"]
    rows.to_csv(output / f"{name}_clusters.csv", index=False)

    coordinates = PCA(n_components=2, random_state=42).fit_transform(kmeans["X"])
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=kmeans["labels"], cmap="tab10", s=11, alpha=0.55)
    ax.set(xlabel="Profile PC1", ylabel="Profile PC2", title=f"K-means hourly clusters: {name}")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output / f"{name}_clusters_plot.png", dpi=150)
    plt.close(fig)
    dbscan_clusters = len(set(dbscan["labels"]) - {-1})
    return [
        {"dataset": name, "method": "kmeans", "clusters": len(set(kmeans["labels"])), "noise_points": 0, "silhouette": evaluate_clustering_silhouette(kmeans["X"], kmeans["labels"]), "features": ",".join(clustering_features)},
        {"dataset": name, "method": "dbscan", "clusters": dbscan_clusters, "noise_points": int((dbscan["labels"] == -1).sum()), "silhouette": evaluate_clustering_silhouette(dbscan["X"], dbscan["labels"]), "features": ",".join(clustering_features)},
    ]


def main() -> None:
    configure_logging()
    missing = [str(spec["path"]) for spec in DATASETS.values() if not spec["path"].exists()]
    if missing:
        raise FileNotFoundError(f"Build hourly gold datasets first: {missing}")

    outlier_rows, pca_rows, cluster_rows = [], [], []
    for name, spec in DATASETS.items():
        df = pd.read_parquet(spec["path"])
        _feature_selection(name, df, spec["target"], spec["features"])
        outlier_rows.append(_outliers(name, df, spec["target"], spec["features"]))
        pca_rows.append(_pca(name, df, spec["features"]))
        cluster_rows.extend(_cluster(name, df, spec["target"], spec["features"]))

    pd.DataFrame(outlier_rows).to_csv(REPORTS_DIR / "outliers/outlier_summary.csv", index=False)
    pd.concat(pca_rows, ignore_index=True).to_csv(REPORTS_DIR / "dimensionality/pca_explained_variance.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(REPORTS_DIR / "clustering/clustering_summary.csv", index=False)
    train_consumption()
    train_injection()
    LOGGER.info("Course-method analysis completed")


if __name__ == "__main__":
    main()
