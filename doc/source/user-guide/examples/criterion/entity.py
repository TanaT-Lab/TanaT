"""
EntityCriterion
===============

Select sequences or prune entity rows using any **Polars expression** evaluated
against the temporal data.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Level
     - Behaviour
   * - ``which()``
     - Returns IDs that have **at least one** row satisfying the expression.
   * - ``filter_entities()``
     - Keeps only the rows where the expression is ``True``; sequences with
       zero matching rows disappear from the filtered view.
   * - ``match()``
     - Returns ``True`` iff the sequence has at least one matching row.

See :doc:`../../../reference/criterion` for the full reference.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl

from tanat import build_intervals
from tanat.criterion import EntityCriterion
from tanat.dataset import simulate_intervals, simulate_static

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~

# %%
temporal = simulate_intervals(
    n_ids=50,
    features=["value", "status"],
    seed=42,
)
static = simulate_static(n_ids=50, features=["age", "group"], seed=0)

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

# Inspect the unique status values present in the data.
pool.temporal_data()["status"].unique()

# %% [markdown]
# ``which()`` : sequence-level selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Return the IDs of all sequences that have **at least one** entity row
# satisfying the expression.  The original pool is left unchanged.

# %%

# Pick a status value that exists in the data
target_status = "A"
# Select sequences that have at least one entity with that status.
ids_with_status = pool.which(EntityCriterion(query=pl.col("status") == target_status))

# %%
# Numeric threshold: sequences with at least one high-value entity.
ids_high_value = pool.which(EntityCriterion(query=pl.col("value") > 80))

# %%
# Combine conditions with a Polars expression.
ids_combined = pool.which(
    EntityCriterion(query=(pl.col("status") == target_status) & (pl.col("value") > 80))
)

# %% [markdown]
# ``filter_entities()``: entity-level pruning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Return a **new pool view** that contains only the rows satisfying the
# expression.  The original pool is unchanged.  Sequences with zero surviving
# rows no longer appear in the filtered pool.

# %%
filtered = pool.filter_entities(
    EntityCriterion(query=pl.col("status") == target_status)
)

# %%

# Combine two conditions in a single criterion to narrow further.
filtered2 = pool.filter_entities(
    EntityCriterion(query=(pl.col("status") == target_status) & (pl.col("value") > 80))
)

# %% [markdown]
# ``match()``: single-sequence evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
criterion = EntityCriterion(query=pl.col("status") == target_status)
# Iterate to find the first sequence that matches.
first_match = next((s for s in pool if s.match(criterion)), None)
if first_match:
    print(f"First matching sequence: id={first_match.id_value}")
