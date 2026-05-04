"""
RankCriterion
=============

Prune entity rows by their **0-based positional rank** within each sequence.
Ranks can be absolute (from the first entity) or relative to T0 (the nearest
entity to the reference date set via ``pool.set_t0()``).

Exactly **one** parameter group must be specified:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Description
   * - ``first=N``
     - Keep first N rows (``N < 0`` → all except last ``|N|``).
   * - ``last=N``
     - Keep last N rows (``N < 0`` → all except first ``|N|``).
   * - ``start`` / ``end`` / ``step``
     - Python-slice semantics (negative indices supported).
   * - ``ranks=[…]``
     - Explicit list of 0-based positions (negative = from end).

Pass ``relative=True`` to interpret ranks relative to T0 rather than the
start of the sequence.

:class:`~tanat.criterion.RankCriterion` supports **ENTITY** level only
(``filter_entities()``); ``which()`` and ``match()`` are not available.

See :doc:`../../../reference/criterion` for the full reference.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
from tanat import build_intervals
from tanat.criterion import RankCriterion
from tanat.dataset import simulate_intervals, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~

# %%
temporal = simulate_intervals(n_ids=50, features=["value", "status"], seed=42)
static = simulate_static(n_ids=50, features=["age"], seed=0)

pool = build_intervals(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
    static_data=static,
)

# %%
print(pool)

# %%

# Inspect length distribution or other summary statistics.
pool.describe(by_id=False)

# %% [markdown]
# ``first`` and ``last``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Positive ``N``: keep the first (or last) N entities per sequence.
# Negative ``N``: drop the last (or first) ``|N|`` entities per sequence.

# %%
# Keep the first 2 entities.
pool_first2 = pool.filter_entities(RankCriterion(first=2))

# %%

# Inspect length of filtered sequences.
pool_first2.describe(by_id=False)

# %%
# Keep the last 3 entities.
pool_last3 = pool.filter_entities(RankCriterion(last=3))

# %%

# Inspect length of filtered sequences.
pool_last3.describe(by_id=False)

# %%
# Drop the last entity: first=-1 keeps all except the final row.
pool_drop_last = pool.filter_entities(RankCriterion(first=-1))

# %%

# Inspect length of filtered sequences.
pool_drop_last.describe(by_id=False)

# %% [markdown]
# Slice: ``start`` / ``end`` / ``step``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Python-slice semantics.  Negative indices count from the end of each
# sequence.

# %%

# Entities at absolute ranks 1, 2, 3 (0-based → second to fourth row).
pool_slice = pool.filter_entities(RankCriterion(start=1, end=4))

# %%

# Inspect length of filtered sequences.
pool_slice.describe(by_id=False)

# %%

# Every other entity (even-ranked rows).
pool_step = pool.filter_entities(RankCriterion(step=2))

# %%

# Inspect length of filtered sequences.
pool_step.describe(by_id=False)

# %% [markdown]
# Explicit ``ranks``
# ~~~~~~~~~~~~~~~~~~
#
# Pass a list of 0-based positions.  Negative values index from the end.

# %%

# First and last entity of each sequence.
pool_ends = pool.filter_entities(RankCriterion(ranks=[0, -1]))

# %%

# Inspect length of filtered sequences.
pool_ends.describe(by_id=False)

# %% [markdown]
# Relative mode: ranks relative to T0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Set a reference date with ``pool.set_t0()`` first.  Then
# ``relative=True`` interprets ranks relative to the nearest entity to T0:
# rank 0 = that entity, rank -1 = one entity before, rank +1 = one after.

# %%
pool.set_t0(position=-1, anchor="start")  # T0 = start of last entity

# Keep the entity at T0 and the 2 entities before it: [T-2, T-1, T0].
# NOTE: relative=True, end is exclusive.
pool_t0 = pool.filter_entities(RankCriterion(start=-2, end=1, relative=True))

# Inspect length of filtered sequences.
pool_t0.describe(by_id=False)

# %%
# Rank 0 alone: a single "anchor" entity per sequence.
pool_anchor = pool.filter_entities(RankCriterion(ranks=0, relative=True))

# Inspect T0 anchor entities.
pool_anchor.temporal_data().head()
