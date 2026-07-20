"""
Trajectory Metrics: Aggregation Trajectory Metric
=================================================

This example demonstrates trajectory-level metrics using
:class:`~tanat.metric.AggregationTrajectoryMetric`, which computes
distances between trajectories that contain multiple sequence types
(e.g., states, events, or intervals).
"""

# %%
# Setup
# -----

import polars as pl
import matplotlib.pyplot as plt

from tanat import build_states, build_events, build_trajectories
from tanat.dataset import simulate_trajectories
from tanat.metric.entity import HammingEntityMetric
from tanat.metric.sequence import (
    EditSequenceMetric,
    LCSSequenceMetric,
)
from tanat.metric import AggregationTrajectoryMetric

# %%
# Generate synthetic trajectory data
# -----------------------------------

N_TRAJ = 100
SEED = 42

raw = simulate_trajectories(
    sequences={
        "states": {
            "type": "state",
            "n_ids": N_TRAJ,
            "seq_length_range": (3, 8),
            "features": ["score", "status"],
        },
        "events": {
            "type": "event",
            "n_ids": N_TRAJ,
            "seq_length_range": (2, 6),
            "features": ["score", "status"],
        },
    },
    shared_ids=True,
    seed=SEED,
)

# Build pools for each sequence type
states_pool = build_states(
    temporal_data=raw["states"],
    id_column="id",
    start_column="start",
    end_column="end",
)
events_pool = build_events(
    temporal_data=raw["events"],
    id_column="id",
    time_column="time",
)

# Build trajectory pool
traj_pool = build_trajectories(pools={"states": states_pool, "events": events_pool})

# %%

# Cast features to categorical
for sp in traj_pool.sequence_pools.values():
    sp.cast_features({"status": pl.Categorical})

print(traj_pool)

# %%
# Define trajectory metric
# -------------------------

hamming = HammingEntityMetric(entity_feature="status")

# Use different metrics per alias (sequence type)
agg = AggregationTrajectoryMetric(
    sequence_metrics={
        "events": LCSSequenceMetric(entity_metric=hamming, mode="normalized"),
        "states": EditSequenceMetric(entity_metric=hamming, normalize=True),
    },
    agg_fun="mean",
)
print(agg)

# %%
# Compute distance between a single pair
# ----------------------------------------

traj_ids = traj_pool.unique_ids
traj_a = traj_pool[traj_ids[0]]
traj_b = traj_pool[traj_ids[1]]

dist = agg(traj_a, traj_b)
print(f"\nDistance between {traj_ids[0]} and {traj_ids[1]}: {dist:.4f}")

# %%
# Compute full pairwise distance matrix
# ----------------------------------------

matrix = agg.compute_matrix(traj_pool)
print(f"\nDistance matrix shape: {matrix.shape}")
print(f"Mean distance: {matrix.to_numpy()[matrix.to_numpy() > 0].mean():.4f}")

# %%
# Visualize trajectory distances
# --------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
arr = matrix.to_numpy()
im = ax.imshow(arr, cmap="viridis", aspect="auto")
ax.set_title(
    "Trajectory distances\n(Edit distance for states, LCS for events)",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Trajectory index")
ax.set_ylabel("Trajectory index")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Distance")
plt.tight_layout()
plt.show()
