"""
PatternCriterion
================

Select sequences or extract witness rows based on an **ordered pattern** of
string values in a feature column.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Sentinel
     - Meaning
   * - ``ANY`` (``"..."``)
     - Zero or more elements: free gap between adjacent sub-patterns.
   * - ``WILDCARD`` (``"*"``)
     - Exactly **one** element of any value at that position.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Level
     - Behaviour
   * - ``which()``
     - IDs whose temporal sequence contains (``present=True``) or does not
       contain (``present=False``) the ordered pattern.
   * - ``filter_entities()``
     - Keeps the "witness" rows of the greedy first match
       (``present=True``), or all non-witness rows (``present=False``).
   * - ``match()``
     - Returns ``True`` iff the pattern is found (resp. absent).

See :doc:`../../../reference/criterion` for the full reference.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
from tanat import build_intervals
from tanat.criterion import ANY, WILDCARD, PatternCriterion
from tanat.dataset import simulate_intervals

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~

# %%
temporal = simulate_intervals(n_ids=50, features=["score", "status"], seed=42)

pool = build_intervals(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
)

# %%
print(pool)

# %%

# Pick 2 status values existing in the temporal data.
A = "A"
B = "B"

# %% [markdown]
# Single-element pattern
# ~~~~~~~~~~~~~~~~~~~~~~
#
# A plain string (or single-element list) selects sequences that contain **at
# least one** entity with that value.

# %%
ids_has_A = pool.which(PatternCriterion(feature="status", pattern=A))

# %%

# Exclusion: sequences that never show status A.
ids_no_A = pool.which(PatternCriterion(feature="status", pattern=A, present=False))

# %% [markdown]
# Adjacent pattern: A directly followed by B
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``[A, B]`` matches only if B appears **immediately after** A in the ordered
# sequence of entities.

# %%
ids_adj = pool.which(PatternCriterion(feature="status", pattern=[A, B]))

# %% [markdown]
# Free gap: A anywhere before B
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Insert :data:`~tanat.criterion.ANY` between elements to allow an arbitrary
# number of rows in between.

# %%
ids_gap = pool.which(PatternCriterion(feature="status", pattern=[A, ANY, B]))

# %% [markdown]
# Wildcard: exactly one element between A and B
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :data:`~tanat.criterion.WILDCARD` matches **exactly one** entity of any value.

# %%
ids_wildcard = pool.which(PatternCriterion(feature="status", pattern=[A, WILDCARD, B]))

# %% [markdown]
# Combining sentinels
# ~~~~~~~~~~~~~~~~~~~~
#
# You can mix :data:`ANY` and :data:`WILDCARD` freely.
# Here: A, then any gap, then exactly two consecutive B's.

# %%
ids_double_B = pool.which(PatternCriterion(feature="status", pattern=[A, ANY, B, B]))

# %% [markdown]
# Regex and case options
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# By default elements are treated as **regular expressions** (``regex=True``).
# Use ``regex=False`` for literal substring matching.
# Add ``case_sensitive=False`` for case-insensitive matching.

# %%

# Literal, case-insensitive: same result as the exact match above.
a_lower = A.lower()
ids_ci = pool.which(
    PatternCriterion(
        feature="status", pattern=a_lower, regex=False, case_sensitive=False
    )
)

# %% [markdown]
# ``filter_entities()``: witness rows
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With ``present=True`` (default), only the **greedy first-match witness rows**
# are kept.  Each ID contributes at most ``len(pattern)`` rows.

# %%
pattern = [A, B]
filtered = pool.filter_entities(PatternCriterion(feature="status", pattern=pattern))

# %%

# inspect length of filtered sequences
filtered.describe(by_id=False)

# %% [markdown]
# ``match()``: single-sequence evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
criterion = PatternCriterion(feature="status", pattern=[A, B])
# Find all matching sequences by iterating.
matching_seqs = [s for s in pool if s.match(criterion)]
print(f"{len(matching_seqs)} sequence(s) contain A→B")
