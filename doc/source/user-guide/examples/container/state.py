"""
State Sequences
================

Build a :class:`~tanat.sequence.StateSequencePool` from synthetic data
and navigate all the way down to individual entities.

A **state sequence** partitions an individual's timeline into contiguous,
non-overlapping periods. Each entity is in exactly one state at any given
point in time, and ``end[i] == start[i+1]`` within every sequence.

- ``id`` → sequence identifier
- ``start`` → state start timestamp
- ``end`` → state end timestamp (auto-derived when omitted)
- other columns → entity features (auto-inferred by the builder)
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
from tanat import build_states
from tanat.dataset import simulate_states, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# :func:`~tanat.dataset.simulation.states.simulate_states` produces strictly
# contiguous states: ``end[i] == start[i+1]`` within every ID by construction.

# %%
temporal = simulate_states(
    n_ids=50,
    features=["score", "status"],
    seed=42,
)
print(temporal.shape, temporal.columns.tolist())

# %%
temporal.head()

# %% [markdown]
# Build the pool
# ~~~~~~~~~~~~~~
#
# Two variants are available for :func:`~tanat.sequence.shortcuts.build_states`:
#
# - **Without ``end_column``**: the end of each state is auto-derived
#   from the next state's start (last state stays open-ended).
# - **With ``end_column``**: the explicit end column is used as-is
#   (data must already be contiguous).
#
# Here we pass ``end_column`` directly since ``simulate_states``
# already guarantees contiguity.

# %%
pool = build_states(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
)

# %%
print(pool)

# %% [markdown]
# Explore the pool
# ~~~~~~~~~~~~~~~~

# %%
# All temporal data stacked, one row per entity across all sequences
pool.temporal_data().head()

# %%
print(f"Sequences : {len(pool)}")
print(f"First IDs : {pool.unique_ids[:5]}")

# %% [markdown]
# Navigate sequences
# ~~~~~~~~~~~~~~~~~~
#
# Indexing the pool by ID returns a
# :class:`~tanat.sequence.type.state.sequence.StateSequence`.

# %%
seq = pool[pool.unique_ids[0]]
print(seq)

# %%
print(f"ID {seq.id_value}, {len(seq)} states")

# %%
# Temporal data scoped to this sequence only
seq.temporal_data().head()

# %% [markdown]
# Access entities
# ~~~~~~~~~~~~~~~
#
# Indexing a sequence returns a
# :class:`~tanat.sequence.type.state.entity.StateEntity`.
# Positive and negative indices are both supported.

# %%
entity = seq[0]  # first state
last = seq[-1]  # last state

print(entity)

# %%
print("features      :", entity.data())
print("temporal span :", entity.temporal_extent)  # [start, end]

# %% [markdown]
# Static features
# ~~~~~~~~~~~~~~~
#
# Per-sequence static data (age, group, ...) can be attached at build time
# or added to an existing pool with ``add_static_features``.

# %%
# Generate a static DataFrame: one row per sequence ID
static_df = simulate_static(n_ids=50, features=["age", "group"], seed=0)
static_df.head()

# %%
# Option A: pass static_data at build time
pool_with_static = build_states(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
    static_data=static_df,
)
pool_with_static.static_data().head()

# %%
# Option B: attach static features to an existing pool after construction
pool.add_static_features(static_df)
pool.static_data().head()

# %%
# Static data is also accessible per-sequence (single row)
pool[pool.unique_ids[0]].static_data()

# %% [markdown]
# Iteration
# ~~~~~~~~~
#
# A pool is iterable: it yields one
# :class:`~tanat.sequence.type.state.sequence.StateSequence` per visible ID.
# A sequence is also iterable: it yields one
# :class:`~tanat.sequence.type.state.entity.StateEntity` per row.

# %%
# Pool → Sequence : one sequence per visible ID
for seq in pool.subset(pool.unique_ids[:3]):
    print(f"  {seq.id_value}: {len(seq)} states")

# %%
# Sequence → Entity : one entity per row
seq = pool[pool.unique_ids[0]]
for entity in seq:
    print(f"  {entity.temporal_extent}  data={entity.data()}")
