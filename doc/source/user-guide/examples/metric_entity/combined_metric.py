"""
Entity Metric: Combined distance
================================

This example demonstrates how a distance involving several features can be defined.
For that, a combined entity metric proposes to combine several entity metrics.
The proposed implementation makes a linear combinaison of several metric values
computed between two entity features.

.. note::

    It is also possible to implement your custom metric which can involve several
    features. This is also to be considered to improve the computational efficiency
    of your workflow.

"""

# %%
# Setup
# -----

import polars as pl
from tanat import build_states
from tanat.dataset import simulate_states
from tanat.metric.entity import (
    HammingEntityMetric,
    L2EntityMetric,
    CombinedEntityMetric,
)

# %%
# Generate synthetic data
# -----------------------
#
# We generate a sequence with entities described by two
# features: `score` is a numerical attribute, while `status` is
# a categorical attribute.

N_IDS = 10
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

# HammingEntityMetric requires Categorical features
pool.cast_features({"status": pl.Categorical})

print(pool)

# %%
# Create the entity metrics
# ----------------------------
#
# In our scenario, we want to compare entities using both
# the `score` and the `status`.
# Then, we will first define metrics between entities for
# each feature, and then combine them.

metric_score = HammingEntityMetric(entity_feature="status")
metric_status = L2EntityMetric(entity_feature="score")

metric = CombinedEntityMetric(
    metrics_config=[
        metric_score.to_config(),
        metric_status.to_config(),
    ],
    agg="mean",
)

# .. warning::
#
#       Note that the CombinedEntityMetric waits for configurations
#       in the metrics configs, but not the metric itself.

# %%
# Compute distance between individual entities
# ---------------------------------------------

ids = pool.unique_ids
seq_a = pool[ids[0]]
seq_b = pool[ids[1]]

# Extract first entity from each sequence
ent_a, ent_b = seq_a[0], seq_b[0]

# %%

# Entity A
print(ent_a)

# %%

# Entity B
print(ent_b)

# %%

# Compute Hamming distance
dist = metric(ent_a, ent_b)
print(f"\nCombined distance: {dist}")

# %%
# Try multiple pairs
# ------------------

print("\nDistances between random entity pairs:")
print("-" * 40)

for i in range(5):
    seq_1 = pool[ids[i]]
    seq_2 = pool[ids[i + 1]]

    # Compare first entities from each sequence
    e1, e2 = seq_1[0], seq_2[0]
    d = metric(e1, e2)

    print(
        f"Pair {i+1}: {e1['score']!r:5}/{e1['status']!r:5} vs {e2['score']!r:5}/{e2['status']!r:5} → {d:.1f}"
    )
