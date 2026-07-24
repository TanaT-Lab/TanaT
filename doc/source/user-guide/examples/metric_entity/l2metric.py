"""
Entity Metric: L2 Distance
================================

This example demonstrates the L2 entity metric, which measures
the distance between two individual entities (single point-in-time observations)
based on numerical feature.

.. note::
    The L2 metric computed the squared difference of values. When the value is
    normalized, it computes twice the squared difference over the sum of square.
    This value does not belongs in [0,1].

"""

# %%
# Setup
# -----

from tanat import build_states
from tanat.dataset import simulate_states
from tanat.metric.entity import L2EntityMetric

# %%
# Generate synthetic data
# -----------------------

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

print(pool)

# %%
# Create L2 entity metric
# ----------------------------

metric_norm = L2EntityMetric(entity_feature="score")
metric = L2EntityMetric(entity_feature="score", normalize=False)
print(metric)

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
print(f"\nL2 distance: {dist}")

# %%
# Try multiple pairs
# ------------------

print("\nDistances between random entity pairs ('normalized' or not):")
print("-" * 50)

for i in range(5):
    seq_1 = pool[ids[i]]
    seq_2 = pool[ids[i + 1]]

    # Compare first entities from each sequence
    e1, e2 = seq_1[0], seq_2[0]
    d = metric(e1, e2)
    d_n = metric_norm(e1, e2)

    print(f"Pair {i+1}: {e1['score']!r:5} vs {e2['score']!r:5} → {d_n:.1f} / {d:.1f}")


# %%
# Create L2 entity metric without entity feature
# -----------------------------------------------------
#
# In case no entity feature is provided while an `EntityMetric` is defined,
# then the first time there is an attempt to compute a metric, the metric
# self-define the entity feature as the "first" numerical feature that
# is found.
# An error is raised if no numerical feature is found.


# no entity feature defined
metric = L2EntityMetric()
print(metric)

# Compute L2 distance between two entity features
dist = metric(ent_a, ent_b)

# the "score" feature has been identified as a numerical feature
# compatible with the L2 metric
print(metric)
