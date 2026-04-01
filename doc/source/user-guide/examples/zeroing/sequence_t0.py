"""
Sequence Level Zeroing
======================

Align an :class:`~tanat.sequence.IntervalSequencePool` to a reference date (T0)
using each of the four built-in strategies.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Description
   * - ``position``
     - T0 = temporal value at a given row index (``0`` = first, ``-1`` = last)
   * - ``direct``
     - T0 = fixed scalar or per-id ``dict``
   * - ``feature``
     - T0 = value of a static feature column
   * - ``query``
     - T0 = first/last row matching a Polars expression

After calling ``set_t0``, inspect the results with ``pool.t0_data()``,
``seq.t0``, and ``seq.t0_nearest_rank``.

See :doc:`../../../reference/zeroing` for the complete reference.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl
import pandas as pd
from datetime import datetime, timedelta

from tanat import build_intervals
from tanat.dataset import simulate_intervals, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# Generate a small :class:`~tanat.sequence.IntervalSequencePool` with both
# temporal and static features so we can exercise all four strategies.

# %%
temporal = simulate_intervals(
    n_ids=40,
    features=["value", "status"],
    seed=42,
)
temporal.head()

# %%
static = simulate_static(n_ids=40, features=["age"], seed=0)
static.head()

# %% [markdown]
# Build the pool
# ~~~~~~~~~~~~~~

# %%
pool = build_intervals(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
    static_data=static,
)
print(pool)

# %% [markdown]
# Strategy 1 - position
# ~~~~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(position=N)`` selects the temporal value at row index ``N``
# (0-based; negative indices count from the end).
# For interval and state pools the ``anchor`` parameter controls which end
# of the interval is used: ``"start"`` (default) or ``"end"``.

# %%
# First row, start of interval
pool.set_t0(position=0, anchor="start")
print("position=0, anchor='start'")
pool.t0_data().head()

# %%
# Last row, end of interval
pool.set_t0(position=-1, anchor="end")
print("position=-1, anchor='end'")
pool.t0_data().head()

# %% [markdown]
# Strategy 2 - direct
# ~~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(direct=value)`` assigns the **same** timestamp to every sequence.
# ``set_t0(direct={id: value, ...})`` assigns a **per-id** timestamp;
# IDs absent from the dict receive ``_T0_ = null``.

# %%
# Scalar: same T0 for all sequences
pool.set_t0(direct=datetime(2020, 1, 1))
print("direct scalar")
pool.t0_data().head()

# %%
# Dict: per-id mapping
first_ids = pool.unique_ids[:3]
per_id_map = {
    first_ids[0]: datetime(2020, 1, 10),
    first_ids[1]: datetime(2020, 2, 20),
    first_ids[2]: datetime(2020, 3, 15),
}
pool.set_t0(direct=per_id_map)
print("direct per-id (only 3 IDs in dict → remaining get null)")
pool.t0_data().head(6)

# %% [markdown]
# Strategy 3 - feature
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(feature="col")`` reads T0 from a **static feature column**.
# The feature dtype must match the pool's temporal dtype; cast it with
# :py:meth:`~tanat.sequence.base.pool.SequencePool.cast_features` if needed.
#
# We first attach a custom static column ``index_date`` whose dtype already
# matches the pool's temporal dtype (``Datetime[us]``).

# %%
# Build a per-id index_date column (already Datetime[us] in pandas)
n = len(pool)
index_dates = pd.DataFrame(
    {
        "id": pool.unique_ids,
        "index_date": [
            datetime(2020, 1, 1) + timedelta(days=int(i * 7)) for i in range(n)
        ],
    }
)
pool.add_static_features(index_dates)

pool.set_t0(feature="index_date")
print("feature='index_date'")
pool.t0_data().head()

# %%
# All IDs have an index_date so no nulls for this strategy
null_count = pool.t0_data()["_T0_"].isnull().sum()
print(f"Sequences with _T0_ = null: {null_count}/{len(pool)}")

# %% [markdown]
# Strategy 4 - query
# ~~~~~~~~~~~~~~~~~~~
#
# ``set_t0(query=expr)`` scans entity rows and picks the **first** (or last
# with ``use_first=False``) row where the Polars expression is ``True``.
# The ``anchor`` parameter controls which end of the interval becomes T0.
# Sequences with no matching row receive ``_T0_ = null``.

# %%
# T0 = start of the first row where status == "D"
pool.set_t0(query=pl.col("status") == "D", anchor="start", use_first=True)
print("First 'D' row (start)")
pool.t0_data().head()

# %%
# T0 = end of the last row where value > 0.8
pool.set_t0(query=pl.col("value") > 0.8, anchor="end", use_first=False)
print("Last row with value > 0.8 (end)")
pool.t0_data().head()

# %% [markdown]
# Sequence-level properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# After any ``set_t0`` call, every :class:`~tanat.sequence.base.sequence.Sequence`
# exposes ``seq.t0`` and ``seq.t0_nearest_rank``.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 20 50
#
#    * - Property
#      - Type
#      - Description
#    * - ``seq.t0``
#      - scalar | ``None``
#      - T0 for this sequence; ``None`` when no T0 could be computed
#    * - ``seq.t0_nearest_rank``
#      - ``int`` | ``None``
#      - 0-based index of the entity at or just before T0

# %% [markdown]
# T0 is always set at the **pool level** via ``pool.set_t0(...)`` and
# propagated to every sequence automatically. There is no
# ``seq.set_t0()``: the pool is the single source of truth, which
# prevents desynchronisation between sequences after filtering or
# iterating.

# %%
pool.set_t0(position=0, anchor="start")

seq = pool[pool.unique_ids[0]]
print(f"id              : {seq.id_value}")
print(f"t0              : {seq.t0}")
print(f"t0_nearest_rank : {seq.t0_nearest_rank}")

# %%
# Null case: highly selective query → some sequences have no matching row
pool.set_t0(query=pl.col("value") > 0.999, anchor="start")

null_seqs = [seq.id_value for seq in pool if seq.t0 is None]
print(
    f"{len(null_seqs)}/{len(pool)} sequence(s) with t0 = None  "
    f"(no row matched value > 0.999)"
)
