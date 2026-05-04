"""
TimeCriterion
=============

Filter entities or select sequences based on **temporal bounds** applied to
the start and/or end time columns.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``start_ge`` / ``start_le``
     - Inclusive bounds on the **start** column.
   * - ``end_ge`` / ``end_le``
     - Inclusive bounds on the **end** column (interval/state pools only).
   * - ``duration_within``
     - ``False`` (default): any overlap with the window suffices.
       ``True``: the entity interval must be **fully contained** in the window.
   * - ``all_entities``
     - ``False`` (default): at least one row must satisfy the bounds.
       ``True``: **every** row must satisfy the bounds.

All bounds are inclusive.  At least one bound must be supplied.

See :doc:`../../../reference/criterion` for the full reference.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import datetime as dt

from tanat import build_intervals, build_events
from tanat.criterion import TimeCriterion
from tanat.dataset import simulate_intervals, simulate_events

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~

# %%
temporal = simulate_intervals(n_ids=50, features=["value", "status"], seed=42)

pool = build_intervals(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
)

# %%
print(pool)

# %%
# Inspect the time range covered by the data.
df = pool.temporal_data()
print(
    f"start range: {df['start'].min()} → {df['start'].max()}\n"
    f"end range  : {df['end'].min()} → {df['end'].max()}"
)

# %% [markdown]
# ``which()``: sequence-level selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%

# Any entity starts on or after a given date (default: all_entities=False).
cutoff = dt.datetime(2000, 7, 1)
ids_after = pool.which(TimeCriterion(start_ge=cutoff))

# %%

# All entities must start after the cutoff (stricter: all_entities=True).
ids_all_after = pool.which(TimeCriterion(start_ge=cutoff, all_entities=True))

# %%

# Two-sided window: sequences with at least one entity that starts in [t0, t1].
t0 = dt.datetime(2000, 3, 1)
t1 = dt.datetime(2000, 9, 1)
ids_window = pool.which(TimeCriterion(start_ge=t0, start_le=t1))

# %% [markdown]
# Overlap vs containment
# ~~~~~~~~~~~~~~~~~~~~~~
#
# For duration-based sequences (Interval/State) two modes control how entity relate to the query window:
#
# * **Overlap** (``duration_within=False``, default): entity touches the window
#   → start ≤ window_end **and** end ≥ window_start.
# * **Containment** (``duration_within=True``): entity lies fully inside
#   → start ≥ window_start **and** end ≤ window_end.

# %%
window_start = dt.datetime(2007, 1, 1)
window_end = dt.datetime(2008, 1, 1)

filtered_overlap = pool.filter_entities(
    TimeCriterion(start_ge=window_start, end_le=window_end, duration_within=False)
)

# %%
filtered_within = pool.filter_entities(
    TimeCriterion(start_ge=window_start, end_le=window_end, duration_within=True)
)

# %%
ids_overlap = pool.which(TimeCriterion(start_ge=window_start, end_le=window_end))

# %%
ids_within = pool.which(
    TimeCriterion(start_ge=window_start, end_le=window_end, duration_within=True)
)


# %% [markdown]
# Event pools (single time column)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For event sequences only ``start_ge`` / ``start_le`` apply; ``end_ge`` /
# ``end_le`` and ``duration_within`` are unavailable.

# %%
raw_events = simulate_events(n_ids=50, features=["value", "status"], seed=1)
event_pool = build_events(
    temporal_data=raw_events,
    id_column="id",
    time_column="time",
)

# Inspect time range.
ev_df = event_pool.temporal_data()
print(f"event time range: {ev_df['time'].min()} → {ev_df['time'].max()}")

# %%
ev_cutoff = dt.datetime(2000, 6, 1)
ids_ev = event_pool.which(TimeCriterion(start_ge=ev_cutoff))

# %% [markdown]
# ``match()``: single-sequence evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
criterion = TimeCriterion(start_ge=cutoff)
# Iterate to find the first sequence that matches.
first_match = next((s for s in pool if s.match(criterion)), None)
if first_match:
    print(f"First matching sequence: id={first_match.id_value}")
