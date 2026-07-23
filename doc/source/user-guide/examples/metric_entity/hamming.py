"""
Entity Metric: Hamming Distance
================================

This example demonstrates the Hamming entity metric, which measures
the distance between two individual entities (single point-in-time observations)
based on categorical feature equality.

.. note::
   Most of sequence-level metrics require an entity metric as a building block.
   Hamming is the most common choice for categorical features.
"""

# %%
# Setup
# -----

import polars as pl
from tanat import build_states
from tanat.dataset import simulate_states
from tanat.metric.entity import HammingEntityMetric

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

# HammingEntityMetric requires Categorical features
pool.cast_features({"status": pl.Categorical})

print(pool)

# %%
# Create Hamming entity metric
# ----------------------------

hamming = HammingEntityMetric(entity_feature="status")
print(hamming)

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
dist = hamming(ent_a, ent_b)
print(f"\nHamming distance: {dist}")
print("  Same categories → 0.0")
print("  Different categories → 1.0 (default mismatch_cost)")

# %%
# Try multiple pairs
# ------------------

print("\nDistances between random entity pairs:")
print("-" * 50)

for i in range(5):
    seq_1 = pool[ids[i]]
    seq_2 = pool[ids[i + 1]]

    # Compare first entities from each sequence
    e1, e2 = seq_1[0], seq_2[0]
    d = hamming(e1, e2)

    print(f"Pair {i+1}: {e1['status']!r:10} vs {e2['status']!r:10} → {d:.1f}")


# %%
# Create Hamming entity metric without entity feature
# -----------------------------------------------------
#
# In case no entity feature is provided while an `EntityMetric` is defined,
# then the first time there is an attempt to compute a metric, the metric
# self-define the entity feature as the "first" categorical feature that
# is found.
# An error is raised if no categorical feature is found.


# no entity feature defined
hamming = HammingEntityMetric()
print(hamming)

# Compute Hamming distance between two entity features
dist = hamming(ent_a, ent_b)

# the "status" feature has been identified as a categorical feature
# compatible with the hamming metric
print(hamming)
