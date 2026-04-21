"""
Sequence Metrics: DTW
======================

This example demonstrates :class:`~tanat.metric.sequence.DTWSequenceMetric`,
which computes Dynamic Time Warping distance allowing flexible temporal
alignment between sequences of different lengths.
"""

# %%
# Setup
# -----

import matplotlib.pyplot as plt
import polars as pl

from tanat import build_states
from tanat.dataset import simulate_states
from tanat.metric.entity import HammingEntityMetric
from tanat.metric.sequence import DTWSequenceMetric

# %%
# Generate synthetic data
# -----------------------

SEED = 42
N_IDS = 80

raw_df = simulate_states(
    n_ids=N_IDS,
    seq_length_range=(3, 8),
    features=["score", "status"],
    seed=SEED,
)
pool = build_states(raw_df, id_column="id", start_column="start", end_column="end")

# %%

# Cast features to categorical
pool.cast_features({"status": pl.Categorical})
print(pool)

# %%
# Define metric
# -------------

hamming = HammingEntityMetric(entity_feature="status")
metric = DTWSequenceMetric(entity_metric=hamming, normalize=True)
print(metric)

# %%
# Compute distance between a single pair
# ---------------------------------------

ids = pool.unique_ids
dist = metric(pool[ids[0]], pool[ids[1]])
print(f"Distance between {ids[0]} and {ids[1]}: {dist:.4f}")

# %%
# Compute full pairwise distance matrix
# --------------------------------------

dm = metric.compute_matrix(pool)
print(f"Distance matrix shape: {dm.shape}")

# %%
# Visualize distances
# -------------------

arr = dm.to_numpy()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(arr, cmap="viridis_r", vmin=0, vmax=1)
ax.set_title("DTW distance matrix", fontsize=12, fontweight="bold")
ax.set_xlabel("Sequence index")
ax.set_ylabel("Sequence index")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Distance")
plt.tight_layout()
plt.show()
