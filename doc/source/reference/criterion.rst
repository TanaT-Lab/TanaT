.. _criterion_reference:

=========
Criteria
=========

**Criteria** are composable filtering objects that evaluate temporal or static
properties of sequences and entities.  They expose a uniform three-operation API:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - ``pool.which(criterion)``
     - Returns a ``set`` of IDs whose sequences satisfy the criterion at
       **sequence level**.
   * - ``pool.filter_entities(criterion)``
     - Returns a new pool view where only the entity rows satisfying the
       criterion are kept (**entity level**).  The original pool is unchanged.
   * - ``seq.match(criterion)``
     - Returns ``True`` if the single sequence satisfies the criterion.

Each criterion declares which **levels** it supports.  Applying an unsupported
operation raises :class:`~tanat.criterion.CriterionLevelError`.

----

EntityCriterion
===============

Filter entities or select sequences using any **Polars expression** evaluated
against the temporal data.

.. code-block:: python

   from tanat.criterion import EntityCriterion
   import polars as pl

   # Select sequences with at least one "error" row.
   ids = pool.which(EntityCriterion(query=pl.col("status") == "error"))

   # Keep only the "error" rows across all sequences.
   pool2 = pool.filter_entities(EntityCriterion(query=pl.col("status") == "error"))

   # Combine conditions with any Polars expression.
   pool3 = pool.filter_entities(
       EntityCriterion(query=(pl.col("status") == "error") & (pl.col("value") > 0.5))
   )

   # Single-sequence match.
   ok = seq.match(EntityCriterion(query=pl.col("status") == "error"))

The expression must return a Boolean column.  Rows where it evaluates to
``True`` are kept (``filter_entities``) or counted towards sequence selection
(``which``).

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``query``
     - ``pl.Expr``
     - A Polars expression evaluated per entity row against the temporal data.

----

StaticCriterion
===============

Select sequences or trajectories using a **Polars expression evaluated against
the static (per-ID) data**.  The pool must have static features attached.

.. code-block:: python

   from tanat.criterion import StaticCriterion
   import polars as pl

   # Select IDs whose age exceeds 50.
   ids = pool.which(StaticCriterion(query=pl.col("age") > 50))
   pool2 = pool.subset(ids)

   # Works identically on a TrajectoryPool.
   traj_ids = tpool.which(StaticCriterion(query=pl.col("group") == "A"))

   # Single match.
   ok = seq.match(StaticCriterion(query=pl.col("age") > 50))

``filter_entities()`` is **not supported**: static data has no entity rows.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``query``
     - ``pl.Expr``
     - A Polars expression evaluated per ID against the static data frame.

----

TimeCriterion
=============

Filter entities or select sequences based on **temporal bounds** on the start
and/or end time columns.  All bounds are inclusive.

.. code-block:: python

   import datetime as dt
   from tanat.criterion import TimeCriterion

   t0 = dt.datetime(2020, 1, 1)
   t1 = dt.datetime(2021, 1, 1)

   # Sequences with at least one entity starting on or after t0.
   ids = pool.which(TimeCriterion(start_ge=t0))

   # Sequences where ALL entities start on or after t0.
   ids = pool.which(TimeCriterion(start_ge=t0, all_entities=True))

   # Entity pruning: keep rows inside [t0, t1] (overlap mode, default).
   pool2 = pool.filter_entities(TimeCriterion(start_ge=t0, end_le=t1))

   # Containment mode: entity interval must be fully inside [t0, t1].
   pool3 = pool.filter_entities(
       TimeCriterion(start_ge=t0, end_le=t1, duration_within=True)
   )

   # Numeric bounds for timestep pools.
   ids = state_pool.which(TimeCriterion(start_ge=200.0, start_le=400.0))

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``start_ge``
     - :ref:`TimeBound <time-bound>` | ``None``
     - Minimum value for the start column (inclusive).
   * - ``start_le``
     - :ref:`TimeBound <time-bound>` | ``None``
     - Maximum value for the start column (inclusive).
   * - ``end_ge``
     - :ref:`TimeBound <time-bound>` | ``None``
     - Minimum value for the end column: interval/state pools only.
   * - ``end_le``
     - :ref:`TimeBound <time-bound>` | ``None``
     - Maximum value for the end column: interval/state pools only.
   * - ``duration_within``
     - ``bool``
     - ``False`` (default): overlap is sufficient.  ``True``: entity interval
       must be fully contained in the window.
   * - ``all_entities``
     - ``bool``
     - ``False`` (default): at least one row must match.  ``True``: every row
       must match.

.. _time-bound:

.. rubric:: TimeBound

``TimeBound = datetime.datetime | datetime.date | int | float``

All bounds within a single criterion call must share the same Python type.
``datetime`` and ``date`` may be mixed (``datetime`` takes precedence).
Use ``int`` or ``float`` for numeric timestep sequences.

Overlap vs containment (two-column pools)
-----------------------------------------

For interval and state sequences (duration-based sequences):

* **Overlap** (``duration_within=False``, default): entity ``[s, e]`` overlaps
  window ``[lo, hi]`` when ``s ≤ hi AND e ≥ lo``.
  Provide ``start_ge=lo, end_le=hi``.
* **Containment** (``duration_within=True``): entity is fully inside when
  ``s ≥ lo AND e ≤ hi``.
  Provide ``start_ge=lo, end_le=hi``.

Open-ended states (``end = null``) are treated as still-ongoing: their end is
considered ``+∞`` in overlap mode (they satisfy any ``end ≥ lo`` condition).

----

PatternCriterion
================

Select sequences or extract witness rows based on an **ordered pattern** of
string values in a feature column.  Elements are matched in temporal order.

.. code-block:: python

   from tanat.criterion import PatternCriterion, ANY, WILDCARD

   # A directly followed by B (adjacent).
   ids = pool.which(PatternCriterion(feature="status", pattern=["A", "B"]))

   # A before B with any number of rows in between.
   ids = pool.which(PatternCriterion(feature="status", pattern=["A", ANY, "B"]))

   # A, then exactly one element, then B.
   ids = pool.which(PatternCriterion(feature="status", pattern=["A", WILDCARD, "B"]))

   # Sequences that never contain A→B.
   ids = pool.which(
       PatternCriterion(feature="status", pattern=["A", "B"], present=False)
   )

   # Keep only the witness rows (greedy first match).
   pool2 = pool.filter_entities(
       PatternCriterion(feature="status", pattern=["A", "B"])
   )

Sentinels
---------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Constant
     - Value
     - Description
   * - ``ANY``
     - ``"..."``
     - Matches **zero or more** elements: free gap between adjacent
       sub-patterns.
   * - ``WILDCARD``
     - ``"*"``
     - Matches **exactly one** element of any value.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``feature``
     - ``str``
     - Name of the string feature column to match against.
   * - ``pattern``
     - ``str`` | ``list[str]``
     - Ordered pattern.  A bare string is a single-element pattern.
   * - ``present``
     - ``bool``
     - ``True`` (default): pattern must be present.  ``False``: pattern must
       be absent.
   * - ``regex``
     - ``bool``
     - ``True`` (default): elements are regular expressions.  ``False``:
       literal substring matching.
   * - ``case_sensitive``
     - ``bool``
     - ``True`` (default): case-sensitive.  ``False``: case-insensitive.

Entity-level behaviour
----------------------

* ``present=True``: keeps the **greedy first-match witness rows** only.
  Each ID contributes at most ``len(pattern)`` rows; IDs with no match
  contribute 0 rows.
* ``present=False``: keeps all rows that are **not** witnesses.  IDs with
  no match keep all their rows.

----

LengthCriterion
===============

Select sequences by their **number of entity rows** (sequence length).

.. code-block:: python

   from tanat.criterion import LengthCriterion

   # More than 6 entities.
   ids = pool.which(LengthCriterion(gt=6))

   # Between 3 and 10 entities (inclusive on both ends).
   ids = pool.which(LengthCriterion(ge=3, le=10))

   # Single match.
   ok = seq.match(LengthCriterion(ge=3, lt=20))

``filter_entities()`` is **not supported**.

.. list-table::
   :header-rows: 1
   :widths: 15 10 75

   * - Parameter
     - Type
     - Description
   * - ``gt``
     - ``int``
     - Strictly greater than.
   * - ``ge``
     - ``int``
     - Greater than or equal to.
   * - ``lt``
     - ``int``
     - Strictly less than.
   * - ``le``
     - ``int``
     - Less than or equal to.

At least one bound must be provided.  Contradictory bounds (e.g. ``gt=5,
lt=3``) raise ``ValueError`` at construction time.

----

RankCriterion
=============

Prune entity rows by their **0-based positional rank** within each sequence.

.. code-block:: python

   from tanat.criterion import RankCriterion

   # Keep the first 3 entities.
   pool2 = pool.filter_entities(RankCriterion(first=3))

   # Keep all except the last 2 entities.
   pool2 = pool.filter_entities(RankCriterion(first=-2))

   # Keep the last 2 entities.
   pool2 = pool.filter_entities(RankCriterion(last=2))

   # Python-slice: ranks 1, 2, 3 (0-based).
   pool2 = pool.filter_entities(RankCriterion(start=1, end=4))

   # Every other entity.
   pool2 = pool.filter_entities(RankCriterion(step=2))

   # First and last entity.
   pool2 = pool.filter_entities(RankCriterion(ranks=[0, -1]))

   # Relative to T0: entity at T0 and the one after it.
   pool.set_t0(position=0, anchor="start")
   pool2 = pool.filter_entities(RankCriterion(start=0, end=2, relative=True))

``which()`` and ``match()`` are **not supported**.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``first``
     - ``int``
     - Keep first N rows (``< 0`` → all except last ``|N|``). Cannot be 0.
   * - ``last``
     - ``int``
     - Keep last N rows (``< 0`` → all except first ``|N|``). Cannot be 0.
   * - ``start``
     - ``int``
     - Start rank inclusive (Python-style negative supported).
   * - ``end``
     - ``int``
     - End rank exclusive (Python-style negative supported).
   * - ``step``
     - ``int``
     - Sub-sample every N-th entity (≥ 1). Compatible with ``start``/``end``
       or standalone.
   * - ``ranks``
     - ``list[int]``
     - Explicit 0-based positions (negative = from end). A single ``int``
       is accepted.
   * - ``relative``
     - ``bool``
     - ``False`` (default): absolute ranks from start of sequence.
       ``True``: ranks relative to T0 (requires ``pool.set_t0()`` first).
       Not compatible with ``first``/``last``.

Exactly one parameter group must be active at a time.

----

Chaining criteria
=================

Criteria can be chained by passing the result of one operation as the target of
the next.  Each call returns a new pool view; the original is never modified.

.. code-block:: python

   # 1. Select IDs matching a static condition.
   ids = pool.which(StaticCriterion(query=pl.col("age") > 50))

   # 2. Restrict the pool to those IDs.
   pool2 = pool.subset(ids)

   # 3. Prune entity rows by time window.
   pool3 = pool2.filter_entities(
       TimeCriterion(start_ge=dt.datetime(2020, 1, 1), end_le=dt.datetime(2021, 1, 1))
   )

   # 4. Keep only the first 2 entities per sequence.
   pool4 = pool3.filter_entities(RankCriterion(first=2))

Alternatively, use ``which()`` results to drive multi-step pipelines:

.. code-block:: python

   ids_long = pool.which(LengthCriterion(gt=5))
   ids_error = pool.which(PatternCriterion(feature="status", pattern="error"))
   ids_target = ids_long & ids_error   # set intersection

   pool_target = pool.subset(ids_target)

----

See Also
--------

* :doc:`../user-guide/auto_examples/criterion/entity`   : EntityCriterion examples.
* :doc:`../user-guide/auto_examples/criterion/static`   : StaticCriterion examples.
* :doc:`../user-guide/auto_examples/criterion/time`     : TimeCriterion examples.
* :doc:`../user-guide/auto_examples/criterion/pattern`  : PatternCriterion examples.
* :doc:`../user-guide/auto_examples/criterion/length`   : LengthCriterion examples.
* :doc:`../user-guide/auto_examples/criterion/rank`     : RankCriterion examples.
* :doc:`manipulation` : Full operation reference (``which``, ``filter_entities``,
  ``subset``).
