"""
Clustering: CLARA (Clustering LARge Applications)
==================================================

This example demonstrates CLARA clustering, a scalable variant of PAM
that works on large datasets by sampling subsets of the data for medoid
selection.
"""

# %%
# Setup
# -----

import polars as pl

from tanat import build_states
from tanat.clustering import CLARAClusterer
from tanat.dataset import simulate_states
from tanat.metric.entity import HammingEntityMetric
from tanat.metric.sequence import EditSequenceMetric

# %%
# Generate synthetic data
# -----------------------

N_IDS = 100
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
# Perform CLARA clustering
# -------------------------

n_clusters = 5
n_samples = 40  # subset size per PAM instance
n_iterations = 3  # number of PAM instances

clusterer = CLARAClusterer(
    metric=metric,
    n_clusters=n_clusters,
    sampling_ratio=n_samples / N_IDS,
    nb_pam_instances=n_iterations,
    random_state=SEED,
)
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
