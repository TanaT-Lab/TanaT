"""
LengthCriterion
===============

Select sequences by their **number of entity rows** (sequence length).

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``gt`` / ``ge``
     - Strictly greater than / greater than or equal to.
   * - ``lt`` / ``le``
     - Strictly less than / less than or equal to.

At least one bound must be supplied.  Contradictory bounds (e.g. ``gt=5,
lt=3``) are rejected at construction time.

:class:`~tanat.criterion.LengthCriterion` supports **SEQUENCE** level only
(``which()``, ``match()``); ``filter_entities()`` is not available.

See :doc:`../../../reference/criterion` for the full reference.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
from tanat import build_intervals
from tanat.criterion import LengthCriterion
from tanat.dataset import simulate_intervals

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
# Inspect length distribution or other summary statistics.
pool.describe(by_id=False)

# %% [markdown]
# ``which()``: single-bound selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%

# Long sequences: more than 6 entities.
ids_long = pool.which(LengthCriterion(gt=6))

# %%

# Short sequences: at most 3 entities.
ids_short = pool.which(LengthCriterion(le=3))
print(f"Length ≤ 3 : {len(ids_short)} / {len(pool)} IDs")


# %% [markdown]
# Range selection
# ~~~~~~~~~~~~~~~
#
# Combine bounds to select sequences whose length falls in a range.

# %%

# Length = ]3, 6]
ids_medium = pool.which(LengthCriterion(gt=3, le=6))


# %% [markdown]
# Subset the pool
# ~~~~~~~~~~~~~~~
#
# Use :py:meth:`~tanat.sequence.base.pool.SequencePool.subset` to obtain a
# restricted pool from the selected IDs.

# %%
pool_long = pool.subset(ids_long)

# %%
print(pool_long)

# %%

# Inspect the length distribution in the subset.
pool_long.describe(by_id=False)

# %% [markdown]
# ``match()``: single-sequence evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
seq = pool[pool.unique_ids[0]]
seq_len = len(seq)
print(
    f"Sequence {seq.id_value}: length={seq_len}  "
    f"gt=6? {seq.match(LengthCriterion(gt=6))}  "
    f"le=3? {seq.match(LengthCriterion(le=3))}"
)
