"""
Trajectory-Level Zeroing
========================

Align a :class:`~tanat.trajectory.pool.TrajectoryPool` to a common reference
date (T0) using any of the four built-in strategies.

At the trajectory level, T0 is computed **once** (from a reference sub-pool
when ``on=`` is provided) then shared across every child object: sub-pools,
trajectories, and individual sequences all return the same ``t0`` value.
Each object still computes its own ``t0_nearest_rank`` from its own temporal grid.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Strategy
     - ``on=`` required?
     - Description
   * - ``position``
     - **yes**
     - T0 = temporal value at row index ``N`` in the reference sub-pool
   * - ``direct``
     - no
     - T0 = scalar or per-id ``dict`` (no sub-pool lookup needed)
   * - ``feature``
     - no
     - T0 = trajectory-level static feature column
   * - ``query``
     - **yes**
     - T0 = first/last row matching a Polars expression in the reference sub-pool

After ``set_t0``, inspect the results with ``tpool.t0_data()``, ``traj.t0``,
and ``traj.t0_nearest_rank``.

See :doc:`../../../reference/zeroing` for the complete reference and
:doc:`sequence_t0` for the sequence-level equivalent.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl
import warnings
from datetime import datetime, timedelta

from tanat import build_events, build_intervals, build_states, build_trajectories
from tanat.dataset import simulate_trajectories, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# Generate three temporal pools (interval, event, state) sharing the same IDs,
# plus a trajectory-level static DataFrame.

# %%
N_IDS = 40
SEED = 42

data = simulate_trajectories(
    sequences={
        "admissions": {
            "type": "interval",
            "n_ids": N_IDS,
            "features": ["value", "status"],
        },
        "labs": {"type": "event", "n_ids": N_IDS, "features": ["value", "status"]},
        "phases": {
            "type": "state",
            "n_ids": N_IDS,
            "features": ["phase_name", "score"],
        },
    },
    seed=SEED,
)
static = simulate_static(n_ids=N_IDS, features=["age", "group"], seed=SEED)

for alias, df in data.items():
    print(f"{alias:12s}: {df.shape[0]:4d} rows, columns={df.columns}")
print(f"{'static':12s}: {static.shape[0]:4d} rows, columns={static.columns}")

# %% [markdown]
# Build the TrajectoryPool
# ~~~~~~~~~~~~~~~~~~~~~~~~

# %%
admissions_pool = build_intervals(
    temporal_data=data["admissions"],
    id_column="id",
    start_column="start",
    end_column="end",
)
labs_pool = build_events(
    temporal_data=data["labs"],
    id_column="id",
    time_column="time",
)
phases_pool = build_states(
    temporal_data=data["phases"],
    id_column="id",
    start_column="start",
    end_column="end",
)

tpool = build_trajectories(
    pools={
        "admissions": admissions_pool,
        "labs": labs_pool,
        "phases": phases_pool,
    },
    static_data=static,
    id_column="id",
)
print(tpool)

# %% [markdown]
# Strategy 1 - position
# ~~~~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(position=N, on="alias")`` selects row ``N`` in the reference
# sub-pool.
#
# - ``on=`` is **required**: the row index depends on the target sub-pool.
# - ``anchor=`` controls which end of the interval is used (interval/state
#   pools only).

# %%

# First row of admissions, start of interval
tpool.set_t0(position=0, anchor="start", on="admissions")
print("position=0, anchor='start', on='admissions'")
tpool.t0_data().head()

# %%
# Last lab event (event pool: anchor is ignored)
tpool.set_t0(position=-1, on="labs")
print("position=-1, on='labs'")
tpool.t0_data().head()

# %% [markdown]
# Strategy 2 - direct
# ~~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(direct=value)`` assigns the **same** T0 to every trajectory.
# ``set_t0(direct={id: value, ...})`` assigns a per-ID T0; absent IDs
# receive ``_T0_ = null``.
#
# ``on=`` is **not** required: the value is provided directly.

# %%
# Scalar: same T0 for everyone
tpool.set_t0(direct=datetime(2010, 6, 1))
print("direct scalar")
tpool.t0_data().head()

# %%
# Per-ID dict (only 3 IDs mapped, rest get null)
first_ids = tpool.unique_ids[:3]
per_id_map = {
    first_ids[0]: datetime(2010, 1, 15),
    first_ids[1]: datetime(2011, 3, 20),
    first_ids[2]: datetime(2012, 7, 10),
}
tpool.set_t0(direct=per_id_map)
print(f"direct per-id (3 IDs mapped, {len(tpool) - 3} get null)")
tpool.t0_data().head(6)

# %% [markdown]
# Strategy 3 - feature
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(feature="col")`` reads T0 from a **trajectory-level static
# feature**.  The dtype must match the pool's temporal dtype.
#
# ``on=`` is **not** required.

# %%
# Add a custom datetime static feature
ids = tpool.unique_ids
admission_dates = [
    datetime(2008, 1, 1) + timedelta(days=i * 45) for i in range(len(ids))
]
date_df = pl.DataFrame(
    {
        "id": ids,
        "admission_date": pl.Series(admission_dates).cast(pl.Datetime("us")),
    }
)
tpool.add_static_features(date_df)

tpool.set_t0(feature="admission_date")
print("feature='admission_date'")
tpool.t0_data().head()

# %% [markdown]
# Strategy 4 - query
# ~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(query=expr, on="alias")`` picks the first (or last) row matching
# the expression in the reference sub-pool.
#
# ``on=`` is **required**: the expression refers to columns in the target
# sub-pool.

# %%

# First lab with status matching a known category
status_values = labs_pool.temporal_data(fmt="polars")["status"].unique().to_list()
ref_status = status_values[0]

tpool.set_t0(
    query=pl.col("status") == ref_status,
    use_first=True,
    on="labs",
)
print(f"query: first lab with status=={ref_status!r}, on='labs'")
tpool.t0_data().head()

# %%

# Last admission matching a status (interval pool, anchor='end')
adm_status_values = (
    admissions_pool.temporal_data(fmt="polars")["status"].unique().to_list()
)
ref_adm_status = adm_status_values[0]

tpool.set_t0(
    query=pl.col("status") == ref_adm_status,
    anchor="end",
    use_first=False,
    on="admissions",
)
print(f"query: last admission with status=={ref_adm_status!r}, anchor='end'")
tpool.t0_data().head()

# %% [markdown]
# Trajectory-level properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# After ``set_t0``, every :class:`~tanat.trajectory.trajectory.Trajectory`
# exposes two read-only properties:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 20 50
#
#    * - Property
#      - Type
#      - Description
#    * - ``traj.t0``
#      - scalar | ``None``
#      - T0 for this trajectory
#    * - ``traj.t0_nearest_rank``
#      - ``dict[str, int | None]``
#      - Per-alias floor index: ``{"admissions": 0, "labs": 2, ...}``

# %%
tpool.set_t0(position=0, anchor="start", on="admissions")

traj = tpool[tpool.unique_ids[0]]
print(f"id               : {traj.id_value}")
print(f"t0               : {traj.t0}")
print(f"t0_nearest_rank  : {traj.t0_nearest_rank}")

# %%

# Iterate over the first few trajectories
for traj in list(tpool)[:6]:
    ranks = ", ".join(f"{alias}={rank}" for alias, rank in traj.t0_nearest_rank.items())
    print(f"  id={traj.id_value!r:<6}  t0={str(traj.t0):<30}  ranks=[{ranks}]")

# %% [markdown]
# ``t0_data()`` column structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``tpool.t0_data()`` returns one row per trajectory with:
#
# - ``_T0_``: the T0 value (identical across all sub-pools)
# - ``<alias>_T0_NEAREST_RANK_``: per-alias floor index (each sub-pool gets
#   its own column because the temporal grids differ)

# %%
df = tpool.t0_data(fmt="polars")
print("Columns:", df.columns)
print(f"Rows   : {len(df)} (one per trajectory)")
df.head()

# %% [markdown]
# Null handling
# ~~~~~~~~~~~~~
#
# Trajectories with ``_T0_ = null`` arise in the same situations as for
# sequence pools (out-of-range position, missing dict key, null feature,
# no query match).  The trajectory is **not** dropped; ``traj.t0`` returns
# ``None`` and all nearest-rank values are ``None``.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tpool.set_t0(query=pl.col("value") > 1e9, on="labs")  # no match

null_trajs = [t.id_value for t in tpool if t.t0 is None]
print(f"{len(null_trajs)}/{len(tpool)} trajectory(ies) with t0 = None")

# %% [markdown]
# T0 is shared across all children
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A single ``tpool.set_t0(...)`` call is enough.  Every object you retrieve
# from the pool - a sub-pool, a trajectory, or an individual sequence -
# automatically returns the same ``t0`` value.  ``t0_nearest_rank`` still
# varies: each pool computes its floor index on its own temporal grid.

# %%
tpool.set_t0(position=0, anchor="start", on="admissions")

uid = tpool.unique_ids[0]
traj = tpool[uid]
seq = traj["labs"]

print(f"traj.t0                        : {traj.t0}")
print(f"traj['labs'].t0                : {seq.t0}")
print(f"same value                     : {seq.t0 == traj.t0}")
print(f"seq.t0_nearest_rank (labs grid): {seq.t0_nearest_rank}")

# %%
# Sub-pools expose the same T0 via t0_data()
labs_t0 = tpool.sequence_pools["labs"].t0_data(fmt="polars")
print("Columns:", labs_t0.columns)
labs_t0.head()
