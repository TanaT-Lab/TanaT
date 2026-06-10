"""
The Three Types of Sequences
=============================

*TanaT* supports three types of temporal sequences depending on how each
entity's temporal extent is defined:

.. list-table::
   :header-rows: 1
   :widths: 15 30 35 20

   * - Type
     - Temporal extent
     - Key constraint
     - Builder
   * - **Event**
     - Single timestamp
     - None
     - :func:`~tanat.sequence.shortcuts.build_events`
   * - **Interval**
     - ``[start, end]``
     - Overlaps and gaps are **allowed**
     - :func:`~tanat.sequence.shortcuts.build_intervals`
   * - **State**
     - ``[start, end]``
     - **Contiguous**, no overlap, no gap
     - :func:`~tanat.sequence.shortcuts.build_states`

This example walks through each type step by step: data simulation, pool
construction, pool-level exploration, navigation down to individual
sequences and entities, and a comparison of the ``temporal_extent`` at
the entity level.

For a broader conceptual introduction see :doc:`/getting-started/concepts`.
"""

# %% [markdown]
# Imports
# ~~~~~~~
#
# Each type has its own shortcut builder.  All three live in the same
# :mod:`tanat` namespace.

# %%
from tanat import build_events, build_intervals, build_states
from tanat.dataset import (
    simulate_events,
    simulate_intervals,
    simulate_states,
    simulate_static,
)

# %% [markdown]
# -------------------------
# 1. Event Sequences
# -------------------------
#
# An **event** is a point-in-time observation: it has *one* timestamp and
# no duration.  Think of medical visits, user clicks, or purchase records.

# %% [markdown]
# Simulate data
# ^^^^^^^^^^^^^
#
# :func:`~tanat.dataset.simulation.events.simulate_events` returns a
# ``DataFrame`` with columns ``id``, ``time``, and one column per feature.

# %%
events_data = simulate_events(
    n_ids=10,
    features=["value", "category"],
    seed=42,
)
events_data.head()

# %% [markdown]
# Build the pool
# ^^^^^^^^^^^^^^
#
# :func:`~tanat.sequence.shortcuts.build_events` needs at minimum:
#
# - ``id_column``: the column that identifies each sequence
# - ``time_column``: the column containing the event timestamp
#
# All remaining columns are automatically inferred as entity features.

# %%
events_pool = build_events(
    temporal_data=events_data,
    id_column="id",
    time_column="time",
)

# %%
print(events_pool)

# %% [markdown]
# Explore the pool
# ^^^^^^^^^^^^^^^^

# %%
print(f"Number of sequences : {len(events_pool)}")
print(f"First IDs           : {events_pool.unique_ids[:5]}")

# %%
# Temporal data of the pool in tabular form (one row = one entity)
events_pool.temporal_data().head()

# %% [markdown]
# Navigate to a sequence then to an entity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Index the pool by ID to get an EventSequence
event_seq = events_pool[events_pool.unique_ids[0]]
print(event_seq)
print(f"→ {len(event_seq)} events for ID {event_seq.id_value!r}")

# %%
# Index the sequence by integer to get an EventEntity
event_entity = event_seq[0]  # first event
print(event_entity)

# %%
# At the entity level the temporal extent is a **single timestamp**
print("features      :", event_entity.data())
print("temporal span :", event_entity.temporal_extent)  # single date/time value
print("feature value :", event_entity["value"])

# %% [markdown]
# Iterate
# ^^^^^^^

# %%
# Pool → one sequence per ID
for seq in events_pool.subset(events_pool.unique_ids[:3]):
    print(f"  ID {seq.id_value!r}: {len(seq)} events")

# %%
# Sequence → one entity per row
for entity in event_seq:
    print(f"  t={entity.temporal_extent}  value={entity['value']}")

# %% [markdown]
# Static features
# ^^^^^^^^^^^^^^^
#
# Per-sequence static data (age, group, …) can be attached at build time
# via ``static_data``, or added later with
# :func:`~tanat.sequence.base.pool.SequencePool.add_static_features`.
# Static features are shared by all entities of a given sequence.

# %%
# Generate one row of static attributes per sequence ID
static_df = simulate_static(n_ids=10, features=["age", "group"], seed=0)
static_df.head()

# %%
# Option 1: attach at build time
events_pool_with_static = build_events(
    temporal_data=events_data,
    id_column="id",
    time_column="time",
    static_data=static_df,
)
events_pool_with_static.static_data().head()

# %%
# Option 2: add to an existing pool in place
events_pool.add_static_features(static_df)
events_pool.static_data().head()

# %%
# Static data is also accessible per-sequence (single row)
events_pool[events_pool.unique_ids[0]].static_data()

# %% [markdown]
# -------------------------
# 2. Interval Sequences
# -------------------------
#
# An **interval** spans a period of time with a ``start`` and an ``end``.
# Unlike states, intervals are **not** required to be contiguous:
# two intervals can **overlap** and **gaps** between them are allowed.
# Think of overlapping treatments, project assignments, or sensor readings.

# %% [markdown]
# Simulate data
# ^^^^^^^^^^^^^
#
# :func:`~tanat.dataset.simulation.intervals.simulate_intervals` produces a
# ``DataFrame`` with ``id``, ``start``, ``end``, and feature columns.

# %%
intervals_data = simulate_intervals(
    n_ids=10,
    features=["value", "category"],
    seed=42,
)
intervals_data.head()

# %% [markdown]
# Build the pool
# ^^^^^^^^^^^^^^
#
# :func:`~tanat.sequence.shortcuts.build_intervals` needs:
#
# - ``id_column``: sequence identifier
# - ``start_column``: interval start
# - ``end_column``: interval end

# %%
intervals_pool = build_intervals(
    temporal_data=intervals_data,
    id_column="id",
    start_column="start",
    end_column="end",
)

# %%
print(intervals_pool)

# %% [markdown]
# Navigate to a sequence then to an entity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
interval_seq = intervals_pool[intervals_pool.unique_ids[0]]
print(f"→ {len(interval_seq)} intervals for ID {interval_seq.id_value!r}")

# %%
interval_entity = interval_seq[0]
print(interval_entity)

# %%
# The temporal extent is now a **(start, end) pair**
print("features      :", interval_entity.data())
print("temporal span :", interval_entity.temporal_extent)  # (start, end)
print("feature value :", interval_entity["value"])

# %% [markdown]
# -------------------------
# 3. State Sequences
# -------------------------
#
# A **state sequence** partitions the timeline into **contiguous,
# non-overlapping** periods: ``end[i] == start[i+1]`` within every sequence.
# The individual is in *exactly one state* at any point in time.
# Think of disease stages, employment status, or device modes.

# %% [markdown]
# Simulate data
# ^^^^^^^^^^^^^
#
# :func:`~tanat.dataset.simulation.states.simulate_states` guarantees
# strict continuity: ``end[i] == start[i+1]`` by construction.

# %%
states_data = simulate_states(
    n_ids=10,
    features=["value", "category"],
    seed=42,
)
states_data.head()

# %% [markdown]
# Build the pool
# ^^^^^^^^^^^^^^
#
# :func:`~tanat.sequence.shortcuts.build_states` accepts the same
# ``start_column`` / ``end_column`` pair as :func:`~tanat.sequence.shortcuts.build_intervals`.
#
# .. note::
#   ``end_column`` is **optional** for state sequences.  When omitted, the
#   end of state *i* is automatically derived from the start of state *i+1*.
#   The last state per sequence will have ``end = null`` unless you supply
#   an explicit sentinel value to the builder.

# %%
states_pool = build_states(
    temporal_data=states_data,
    id_column="id",
    start_column="start",
    end_column="end",  # optional: omit to let TanaT infer it
)

# %%
print(states_pool)

# %% [markdown]
# Navigate to a sequence then to an entity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
state_seq = states_pool[states_pool.unique_ids[0]]
print(f"→ {len(state_seq)} states for ID {state_seq.id_value!r}")

# %%
state_entity = state_seq[0]
print(state_entity)

# %%
# Like intervals, the temporal extent is a **(start, end) pair**
print("features      :", state_entity.data())
print("temporal span :", state_entity.temporal_extent)  # (start, end)
print("feature value :", state_entity["value"])

# %% [markdown]
# --------------------------
# 4. Side-by-side comparison
# --------------------------
#
# To summarise the differences, we build all three pools from the
# *same* underlying dataset (states data, which contains both ``start``
# and ``end`` columns) and compare the ``temporal_extent`` of the first
# entity of the first sequence.

# %%
# Re-use states_data for all three types so the raw data is identical
common_events = build_events(
    temporal_data=states_data,
    id_column="id",
    time_column="start",  # use start as the single event timestamp
)
common_intervals = build_intervals(
    temporal_data=states_data,
    id_column="id",
    start_column="start",
    end_column="end",
)
common_states = build_states(
    temporal_data=states_data,
    id_column="id",
    start_column="start",
    end_column="end",
)

first_id = common_events.unique_ids[0]

for label, pool in [
    ("Event   ", common_events),
    ("Interval", common_intervals),
    ("State   ", common_states),
]:
    entity = pool[first_id][0]
    print(f"{label} → temporal_extent: {entity.temporal_extent}")

# %%
# .. note::
#   - ``Event``    : a single timestamp (no duration)
#   - ``Interval`` : a ``(start, end)`` pair; gaps and overlaps are allowed
#   - ``State``    : a ``(start, end)`` pair; ``end[i] == start[i+1]``
#     is guaranteed by construction
