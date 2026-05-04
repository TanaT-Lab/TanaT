"""
StaticCriterion
===============

Select sequences or trajectories using a **Polars expression evaluated against
the static (per-ID) data**.  Static features do not vary over time; typical
examples are age, group membership, or a baseline score.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Level
     - Behaviour
   * - ``which()`` on a SequencePool
     - Returns IDs whose static row satisfies the expression.
   * - ``which()`` on a TrajectoryPool
     - Same, at trajectory level.
   * - ``match()``
     - Returns ``True`` iff this sequence / trajectory's static row matches.
   * - ``filter_entities()``
     - **Not supported** — static data has no entity rows to prune.

See :doc:`../../../reference/criterion` for the full reference.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl

from tanat import build_events, build_intervals, build_trajectories
from tanat.criterion import StaticCriterion
from tanat.dataset import simulate_events, simulate_intervals, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# :class:`~tanat.criterion.StaticCriterion` requires the pool to have static
# features attached.  Pass ``static_data`` to the builder (or call
# ``pool.add_static_features()`` later).

# %%
temporal = simulate_intervals(
    n_ids=50,
    features=["value", "status"],
    seed=42,
)
static = simulate_static(n_ids=50, features=["age", "group"], seed=0)
static.head()

# %%
pool = build_intervals(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
    static_data=static,
)

# %%
print(pool)

# %% [markdown]
# ``which()``: sequence-level selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The expression is evaluated once per ID against the static table.
# IDs that lack a static row (e.g. IDs not present in ``static_data``) do
# not appear in the result.

# %%

# Numeric threshold.
ids_old = pool.which(StaticCriterion(query=pl.col("age") > 50))

# %%

# Categorical filter.
target_group = "A"
ids_group = pool.which(StaticCriterion(query=pl.col("group") == target_group))

# %%

# Combine conditions.
ids_combined = pool.which(
    StaticCriterion(query=(pl.col("age") > 50) & (pl.col("group") == target_group))
)

# %%

# Use the result to subset the pool.
pool_old = pool.subset(ids_old)
print(pool_old)

# %% [markdown]
# Complement and partitioning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The two complementary age filters partition the IDs that have a non-null age.

# %%
ids_young = pool.which(StaticCriterion(query=pl.col("age") <= 50))
ids_null_age = pool.which(StaticCriterion(query=pl.col("age").is_null()))

# %% [markdown]
# Trajectory pool
# ~~~~~~~~~~~~~~~
#
# :class:`~tanat.criterion.StaticCriterion` works identically on a
# :class:`~tanat.trajectory.pool.TrajectoryPool` because trajectories share
# the same static-data concept.

# %%
temporal_events = simulate_events(n_ids=50, features=["value", "status"], seed=1)

event_pool = build_events(
    temporal_data=temporal_events,
    id_column="id",
    time_column="time",
)
tpool = build_trajectories(
    pools={"admissions": pool, "labs": event_pool},
    static_data=static,
    id_column="id",
)


# %%
print(tpool)

# %%

# Query on static features to get trajectory IDs.
traj_ids = tpool.which(StaticCriterion(query=pl.col("age") > 50))


# %% [markdown]
# ``match()``: single-trajectory evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%

# Iterate to find the first trajectory that matches.
criterion = StaticCriterion(query=pl.col("age") > 50)
first_match = next((t for t in tpool if t.match(criterion)), None)
if first_match:
    print(f"First matching trajectory: id={first_match.id_value}")
