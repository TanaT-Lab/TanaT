"""
Trajectories
=============

Build a :class:`~tanat.trajectory.TrajectoryPool` by composing multiple
sequence pools, then navigate from the pool down to an individual
trajectory, its sub-sequences, and finally to individual entities.

A **trajectory** groups all sequences belonging to the same individual
across multiple temporal dimensions (e.g. visits, treatments, lab results).
A **trajectory pool** aggregates trajectories across an entire cohort.

Each sequence pool is registered under an **alias** that acts as the key
for retrieval::

    tpool["events"]             → EventSequencePool   (full pool)
    tpool[id]                   → Trajectory          (one individual)
    tpool[id]["events"]         → EventSequence       (one sequence)
    tpool[id]["events"][0]      → EventEntity         (one entity)
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
from tanat import build_events, build_intervals, build_states, build_trajectories
from tanat.dataset import simulate_trajectories, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# :func:`~tanat.dataset.simulation.trajectories.simulate_trajectories` is a
# convenience wrapper that calls each ``simulate_*`` function in one shot
# and guarantees a **shared ID space** across all sequence types.

# %%
data = simulate_trajectories(
    sequences={
        "events": {"type": "event", "n_ids": 50, "features": ["value", "category"]},
        "intervals": {
            "type": "interval",
            "n_ids": 50,
            "features": ["duration_days", "label"],
        },
        "states": {"type": "state", "n_ids": 50, "features": ["score", "status"]},
    },
    shared_ids=True,
    seed=42,
)

# Each value is a plain DataFrame.
print("events   :", data["events"].shape)
print("intervals:", data["intervals"].shape)
print("states   :", data["states"].shape)

# %% [markdown]
# Build the sequence pools
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Each pool is built independently with its own ``build_*`` shortcut
# (:func:`~tanat.sequence.shortcuts.build_events`,
# :func:`~tanat.sequence.shortcuts.build_intervals`,
# :func:`~tanat.sequence.shortcuts.build_states`).

# %%
event_pool = build_events(
    temporal_data=data["events"],
    id_column="id",
    time_column="time",
)

interval_pool = build_intervals(
    temporal_data=data["intervals"],
    id_column="id",
    start_column="start",
    end_column="end",
)

state_pool = build_states(
    temporal_data=data["states"],
    id_column="id",
    start_column="start",
    end_column="end",
)

# %% [markdown]
# Build the trajectory pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :func:`~tanat.trajectory.shortcuts.build_trajectories` composes the pools
# under their aliases. The alias becomes the key used to retrieve a
# sub-sequence from a trajectory (``traj["events"]``).

# %%
tpool = build_trajectories(
    pools={
        "events": event_pool,
        "intervals": interval_pool,
        "states": state_pool,
    },
)

# %%
print(tpool)

# %% [markdown]
# Explore the trajectory pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
print(f"Trajectories : {len(tpool)}")
print(f"First IDs    : {tpool.unique_ids[:5]}")

# %%
# The underlying sequence pools are accessible as a read-only mapping
tpool.sequence_pools

# %% [markdown]
# Access a trajectory
# ~~~~~~~~~~~~~~~~~~~
#
# ``tpool[id]`` returns a :class:`~tanat.trajectory.Trajectory`, a
# lightweight view over all sub-sequences for that individual.

# %%
traj = tpool[tpool.unique_ids[0]]
print(traj)

# %% [markdown]
# Sub-sequences
# ~~~~~~~~~~~~~
#
# Use the alias as the key to retrieve the sequence scoped to this individual:
# :class:`~tanat.sequence.type.event.sequence.EventSequence`,
# :class:`~tanat.sequence.type.interval.sequence.IntervalSequence`,
# or :class:`~tanat.sequence.type.state.sequence.StateSequence`.

# %%
event_seq = traj["events"]
interval_seq = traj["intervals"]
state_seq = traj["states"]

print(f"events    : {len(event_seq)} events")
print(f"intervals : {len(interval_seq)} intervals")
print(f"states    : {len(state_seq)} states")

# %%
print(event_seq)

# %%
print(interval_seq)

# %%
print(state_seq)

# %%
# Temporal data for the event sub-sequence of this trajectory
event_seq.temporal_data().head()

# %% [markdown]
# Navigate the sequence pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``tpool.sequence_pools`` gives direct access to the full pool for
# each alias, useful for cohort-level operations without going through
# a trajectory first.

# %%
# Browse the interval pool directly, all individuals and all intervals
tpool.sequence_pools["intervals"].temporal_data().head()

# %% [markdown]
# Access entities
# ~~~~~~~~~~~~~~~
#
# Indexing a sequence returns an :class:`~tanat.sequence.type.event.entity.EventEntity`
# (or its interval/state equivalent). Positive and negative indices are both supported.

# %%
entity = event_seq[0]  # first event of this individual's event sequence
last = event_seq[-1]  # last event

print(entity)

# %%
print("features      :", entity.data())
print("temporal span :", entity.temporal_extent)

# %% [markdown]
# Static features
# ~~~~~~~~~~~~~~~
#
# Per-trajectory static data (age, group, ...) is passed at build time
# via :func:`~tanat.trajectory.shortcuts.build_trajectories`. It is then
# accessible on the pool and on individual trajectories.

# %%
# Generate a static DataFrame matching the shared ID space
static_df = simulate_static(n_ids=50, features=["age", "group"], seed=0)
static_df.head()

# %%
tpool_with_static = build_trajectories(
    pools={
        "events": event_pool,
        "intervals": interval_pool,
        "states": state_pool,
    },
    static_data=static_df,
    id_column="id",
)
tpool_with_static.static_data().head()

# %%
# Static data is also accessible per-trajectory (single row)
tpool_with_static[tpool_with_static.unique_ids[0]].static_data()

# %% [markdown]
# Iteration
# ~~~~~~~~~
#
# All pool and trajectory objects are iterable.
#
# - :class:`~tanat.trajectory.pool.TrajectoryPool` yields
#   :class:`~tanat.trajectory.trajectory.Trajectory` objects;
#   ``.items()`` gives ``(id, trajectory)`` pairs.
# - :class:`~tanat.trajectory.trajectory.Trajectory` yields its aliases
#   (string keys); ``.items()`` gives ``(alias, sequence)`` pairs.
# - A sequence yields its entities.

# %%
# TrajectoryPool → Trajectory
for t in tpool.subset(tpool.unique_ids[:3]):
    print(f"  {t.id_value}: sequences={list(t)}")

# %%
# TrajectoryPool.items() → (id, Trajectory) pairs
for tid, t in tpool.subset(tpool.unique_ids[:3]).items():
    print(f"  {tid}: {type(t).__name__}")

# %%
# Trajectory.items() → (alias, Sequence) pairs
traj = tpool[tpool.unique_ids[0]]
for alias, seq in traj.items():
    print(f"  {alias}: {len(seq)} entities")

# %%
# Sequence → Entity : one entity per row
for entity in event_seq:
    print(f"  {entity.temporal_extent}  data={entity.data()}")
