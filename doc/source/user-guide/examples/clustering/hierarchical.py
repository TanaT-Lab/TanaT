"""
Clustering: Hierarchical Clustering
====================================

This example demonstrates hierarchical clustering on a pool of sequences.
"""

import polars as pl

from tanat import build_states
from tanat.clustering import HierarchicalClusterer
from tanat.dataset import simulate_states
from tanat.metric.entity import HammingEntityMetric
from tanat.metric.sequence import EditSequenceMetric

# %%
# Generate synthetic data
# -----------------------

N_IDS = 50
SEED = 42

raw_df = simulate_states(
    n_ids=N_IDS,
    seq_length_range=(3, 8),
    features=["score", "status"],
    seed=SEED,
)

pool = build_states(
    temporal_data=raw_df,
    id_column="id",
    start_column="start",
    end_column="end",
)

# %%

# Cast features to categorical
pool.cast_features({"status": pl.Categorical})
print(pool)

# %%
# Define the metric used by the clusterer
# ---------------------------------------

hamming = HammingEntityMetric(entity_feature="status")
metric = EditSequenceMetric(entity_metric=hamming, normalize=True)

# %%
# Perform hierarchical clustering
# --------------------------------

clusterer = HierarchicalClusterer(
    metric=metric,
    n_clusters=4,
)

clusterer.fit(pool)

# %%

# Clustering results
print(clusterer)

# %%
# Inspect cluster assignments
# ----------------------------

print("\nCluster assignments injected as static features:")
print(pool.static_data().head())
