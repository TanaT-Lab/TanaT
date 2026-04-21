"""
Clustering: Partitioning Around Medoids (PAM)
==============================================

This example demonstrates Partitioning Around Medoids (PAM) clustering,
which finds k representative objects (medoids) that minimize total distance
to all assigned objects.
"""

# %%
# Setup
# -----

import polars as pl

from tanat import build_states
from tanat.clustering import PAMClusterer
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
# Perform PAM clustering
# ----------------------

n_clusters = 5

clusterer = PAMClusterer(metric=metric, n_clusters=n_clusters)
clusterer.fit(pool)

# %%

# Clustering results
print(clusterer)

# %%
# Inspect cluster assignments and medoids
# ----------------------------------------

print("\nMedoids (representative sequences):")
for i, medoid_id in enumerate(clusterer.medoids):
    print(f"  Cluster {i}: {medoid_id}")

print("\nCluster assignments injected as static features:")
print(pool.static_data().head())
