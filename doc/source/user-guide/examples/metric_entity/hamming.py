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
# Use a cost matrix
# ---------------------------------------------
#
# We first define manually a cost matrix. We know that the `status` feature takes values
# in {A,B,C,D,E}, then we have to evaluate how close are each pair of these elements
# according to our problem (it depends on your data!).
#
# Hamming distance assume a symmetric cost, and is definite (distance between similar objects
# is null), thus, it is only necessary to compute the upper-triangle cost matrix.
#
# In addition, it is not mandatory to define a value for all pairs as a
# defaut value can be defined.

cost_matrix = {
    ("A", "B"): 0.1,
    ("A", "C"): 0.3,
    ("A", "D"): 0.1,
    ("A", "E"): 0.3,
    ("B", "C"): 0.3,
    ("B", "D"): 0.1,
    ("B", "E"): 0.3,
    ("C", "D"): 0.1,
    ("C", "E"): 0.1,
    ("D", "E"): 0.5,
}

# %%
# and now, the Hamming metric definition becomes;

hamming_cost = HammingEntityMetric(entity_feature="status", cost=cost_matrix)

# Compute Hamming distance
dist = hamming_cost(ent_a, ent_b)
print(f"\nHamming distance (with costs): {dist}")

# .. note ::
#
#       Be careful when defining a non-standard metrics, its mathematiccal properties
#       may not be suitable for ensuring the quality of the output of clustering algorithms.


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
    dc = hamming_cost(e1, e2)

    print(f"Pair {i+1}: {e1['status']!r:5} vs {e2['status']!r:5} → {d:.1f} / {dc:.1f}")


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
