# K-means Cluster-Number Diagnostics

The final clustering plots retain `k=3` as an interpretable exploratory
configuration. It is not presented as an objectively optimal cluster count.

Silhouette and elbow diagnostics were evaluated separately for consumption and
grid injection over `k=2` to `k=8`. Calinski-Harabasz and Davies-Bouldin scores
provide complementary checks. The diagnostics use all 17,544 hourly rows for
K-means and inertia; silhouette is estimated from a deterministic 5,000-row
sample to keep full-period evaluation computationally bounded.

`k=2` has the highest silhouette and Calinski-Harabasz score for both datasets.
For consumption, `k=6` has a similar silhouette score, while `k=3` is lower.
For injection, `k=3` is materially weaker than `k=2`. Across all configurations,
the low silhouette scores indicate weak-to-moderate clustering structure.

There is no absolute ideal number of clusters: inertia decreases monotonically,
and different validity indices favor different levels of granularity. Retaining
`k=3` prioritizes interpretability and continuity with the submitted exploratory
analysis, not metric optimality.

See `kmeans_k_diagnostics.csv`, `consumption_k_diagnostics.png`, and
`injection_k_diagnostics.png` for the complete results.
