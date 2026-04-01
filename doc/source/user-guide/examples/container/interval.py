"""
Interval Sequences
===================

Build an :class:`~tanat.sequence.IntervalSequencePool` from synthetic data
and navigate all the way down to individual entities.

An **interval sequence** records duration-based events defined by a start
and an end timestamp. Unlike states, intervals may **overlap**: two
concurrent periods for the same individual are perfectly valid.

- ``id`` → sequence identifier
- ``start`` → interval start timestamp
- ``end`` → interval end timestamp
- other columns → entity features (auto-inferred by the builder)
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
from tanat import build_intervals
from tanat.dataset import simulate_intervals, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# :func:`~tanat.dataset.simulation.intervals.simulate_intervals` returns a ``pd.DataFrame``
# with columns ``id``, ``start``, ``end``, and one column per named entity feature.
# Setting ``allow_overlaps=True`` (the default) authorises concurrent
# intervals within the same sequence.

# %%
temporal = simulate_intervals(
    n_ids=50,
    features=["duration_days", "label"],
    allow_overlaps=True,
    seed=42,
)
print(temporal.shape, temporal.columns.tolist())

# %%
temporal.head()

# %% [markdown]
# Build the pool
# ~~~~~~~~~~~~~~
#
# :func:`~tanat.sequence.shortcuts.build_intervals` infers every column that is not
# ``id``, ``start`` or ``end`` as an entity feature; no explicit feature list needed.

# %%
pool = build_intervals(
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
# Indexing the pool by ID returns an
# :class:`~tanat.sequence.type.interval.sequence.IntervalSequence`.

# %%
seq = pool[pool.unique_ids[0]]
print(seq)

# %%
print(f"ID {seq.id_value}, {len(seq)} intervals")

# %%
# Temporal data scoped to this sequence only
seq.temporal_data().head()

# %% [markdown]
# Access entities
# ~~~~~~~~~~~~~~~
#
# Indexing a sequence returns an
# :class:`~tanat.sequence.type.interval.entity.IntervalEntity`.
# Positive and negative indices are both supported.

# %%
entity = seq[0]  # first interval
last = seq[-1]  # last interval

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
pool_with_static = build_intervals(
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
# :class:`~tanat.sequence.type.interval.sequence.IntervalSequence` per visible ID.
# A sequence is also iterable: it yields one
# :class:`~tanat.sequence.type.interval.entity.IntervalEntity` per row.

# %%
# Pool → Sequence : one sequence per visible ID
for seq in pool.subset(pool.unique_ids[:3]):
    print(f"  {seq.id_value}: {len(seq)} intervals")

# %%
# Sequence → Entity : one entity per row
seq = pool[pool.unique_ids[0]]
for entity in seq:
    print(f"  {entity.temporal_extent}  data={entity.data()}")
