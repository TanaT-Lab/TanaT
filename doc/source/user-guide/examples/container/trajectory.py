"""
Trajectories
=============

We illustrate here how to build a :class:`~tanat.trajectory.TrajectoryPool`
by composing several sequence pools, then navigate from the pool down
to an individual trajectory, its sub-sequences.

A **trajectory** groups all sequences belonging to the same individual
across multiple temporal dimensions (e.g. visits, treatments, lab results).
A **trajectory pool** aggregates trajectories across an entire cohort.

Each sequence pool is registered under an **alias** that acts as the key
for retrieval::

    tpool.sequence_pools["events"]  → EventSequencePool   (full pool)
    tpool[id]                       → Trajectory          (one individual)
    tpool[id]["events"]             → EventSequence       (one sequence)
    tpool[id]["events"][0]          → EventEntity         (one entity)
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
from tanat import build_events, build_intervals, build_states, build_trajectories

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# :func:`~tanat.dataset.simulation.trajectories.simulate_trajectories` is a
# convenience wrapper that calls each ``simulate_*`` function in one shot
# and guarantees a **shared ID space** across all sequence types.

# %%
from tanat.dataset import simulate_trajectories, simulate_static

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


# %% [markdown]
# Access one of the sequence pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The underlying sequence pools are accessible as a read-only mapping `tpool.sequence_pools`.

# %%
# To access the pool with the alias `states`:
print(tpool.sequence_pools["states"])

# %% [markdown]
# Access a trajectory of the trajectory pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``tpool[id]`` returns a :class:`~tanat.trajectory.Trajectory`, a
# lightweight view over all sub-sequences for that individual.

# %%
traj = tpool[tpool.unique_ids[0]]
print(traj)

# %% [markdown]
# Sequences of a trajectory
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Use the alias as the key to retrieve the sequence of an individual trajectory.

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


# %%
# Access to static data is similar for trajectory pools than for sequences pools.

tpool_with_static.static_data().head()

# %%
# Static data is also accessible per-trajectory (single row)
tpool_with_static[tpool_with_static.unique_ids[0]].static_data()

# %%
# .. note::
#   If a sequence pool combined to create a trajectory pool contains static features
#   they are kept in the sequence pool but not visible at the trajectiry level.

# %% [markdown]
# Iteration
# ~~~~~~~~~
#
# All pool and trajectory objects are iterable.
#
# - :func:`~tanat.trajectory.pool.TrajectoryPool.sequence_pools` yields
#   :class:`~tanat.sequence.pool.SequencePool`
# - :class:`~tanat.trajectory.pool.TrajectoryPool` yields
#   :class:`~tanat.trajectory.trajectory.Trajectory` objects;
#   ``.items()`` gives ``(id, trajectory)`` pairs.
# - :class:`~tanat.trajectory.trajectory.Trajectory` yields its aliases
#   (string keys); ``.items()`` gives ``(alias, sequence)`` pairs.


# %%

# TrajectoryPool → SequencePool
for seq_pool in tpool.sequence_pools:
    print(f"  {len(seq_pool)}")

# %%

# TrajectoryPool → Trajectory
for t in tpool:
    print(f"  {t.id_value}: sequences={list(t)}")

# %%

# TrajectoryPool.items() → (id, Trajectory) pairs
for tid, t in tpool.items():
    print(f"  {tid}: {type(t).__name__}")

# %%

# Trajectory.items() → (alias, Sequence) pairs
traj = tpool[tpool.unique_ids[0]]
for alias, seq in traj.items():
    print(f"  {alias}: {len(seq)} entities")
