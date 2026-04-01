.. _zeroing_reference:

===================
Zeroing & Alignment
===================

**Zeroing** aligns sequences to a common reference date (T0 / index date),
transforming absolute timestamps into relative ones. This is essential when
comparing sequences across individuals who were observed at different calendar
times, for example aligning patients to their first hospitalisation or users
to their registration date.

Once ``set_t0`` is called on a pool, every :py:class:`~tanat.sequence.base.sequence.Sequence`
object automatically exposes ``seq.t0`` and ``seq.t0_nearest_rank``.

----

Strategies
==========

Four strategies are available. Pass exactly **one** keyword to ``set_t0``.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Strategy
     - Keyword
     - Description
   * - ``position``
     - ``set_t0(position=N)``
     - T0 = temporal value at row index ``N`` (0-based; negative indices are
       supported: ``-1`` is the last row). For interval and state pools, the
       ``anchor`` parameter selects the reference point within the period
       (``"start"``, ``"end"``, or ``"middle"``).
   * - ``direct``
     - ``set_t0(direct=value)``
     - T0 = the same scalar timestamp for every sequence. Alternatively,
       pass a ``dict[id, timestamp]`` to assign a different T0 per individual;
       IDs absent from the dict receive ``_T0_ = null``.
   * - ``feature``
     - ``set_t0(feature="col")``
     - T0 = the value of a **static feature column**. The feature dtype must
       exactly match the pool's temporal dtype; use ``cast_features`` to align
       if needed. IDs with a ``null`` static value receive ``_T0_ = null``.
   * - ``query``
     - ``set_t0(query=expr)``
     - T0 = temporal value of the **first** (or last, with ``use_first=False``)
       entity row where the Polars expression evaluates to ``True``. The
       ``anchor`` parameter determines the reference point within the matched
       period. Sequences with no matching row receive ``_T0_ = null``.

The ``anchor`` parameter (``"start"`` | ``"end"`` | ``"middle"``) applies only to the
``position`` and ``query`` strategies, and only for
:py:class:`~tanat.sequence.type.interval.pool.IntervalSequencePool` and
:py:class:`~tanat.sequence.type.state.pool.StateSequencePool`; it is ignored
otherwise.

----

Usage
=====

.. code-block:: python

   import polars as pl
   from datetime import datetime

   # position: first row, start of interval
   pool.set_t0(position=0, anchor="start")

   # position: last row, end of interval
   pool.set_t0(position=-1, anchor="end")

   # direct: same T0 for all sequences
   pool.set_t0(direct=datetime(2000, 1, 1))

   # direct: per-id mapping
   pool.set_t0(direct={
       "pat_01": datetime(2020, 3, 15),
       "pat_02": datetime(2021, 6, 1),
   })

   # feature: read T0 from a static column
   pool.cast_features({"registration_date": pl.Datetime("us")}, is_static=True)
   pool.set_t0(feature="registration_date")

   # query: first row where status == "error"
   pool.set_t0(query=pl.col("status") == "error", anchor="start", use_first=True)

   # query: last row where value > 0.9
   pool.set_t0(query=pl.col("value") > 0.9, anchor="end", use_first=False)

----

Pool-Level Inspection
=====================

``pool.t0_data()`` returns the full T0 table as a DataFrame with columns
``[id, _T0_, _T0_NEAREST_RANK_]``.

The ``_T0_NEAREST_RANK_`` column holds the 0-based index of the entity whose
temporal start is the **floor** value at or just before T0. It is computed
from the ``start`` column regardless of the ``anchor`` used in ``set_t0``.

.. code-block:: python

   pool.set_t0(position=0, anchor="start")

   # pandas (default)
   pool.t0_data().head()

----

Sequence-Level Properties
==========================

Once ``set_t0`` has been called on the pool, every sequence object exposes
two read-only properties:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Property
     - Type
     - Description
   * - ``seq.t0``
     - scalar or ``None``
     - T0 value for this sequence. ``None`` when T0 could not be determined
       (sequence too short, no query match, ``null`` static feature value…).
   * - ``seq.t0_nearest_rank``
     - ``int`` or ``None``
     - 0-based index of the entity at or just before T0. ``None`` when
       ``seq.t0`` is ``None``.

.. code-block:: python

   seq = pool[pool.unique_ids[0]]
   print(seq.t0)               # e.g. datetime(2020, 3, 15, ...)
   print(seq.t0_nearest_rank)  # e.g. 2

----

Null Handling
=============

A sequence receives ``_T0_ = null`` in any of these situations:

* **position** - the index is out of range for that sequence.
* **direct (dict)** - the sequence ID is not a key in the dict.
* **feature** - the static feature value is ``null`` for that ID.
* **query** - no entity row matches the expression (or the sequence is empty).

Sequences with ``null`` T0 are **not dropped** from the pool. ``seq.t0``
returns ``None`` and ``seq.t0_nearest_rank`` returns ``None`` for those
individuals.

To inspect how many sequences are affected:

.. code-block:: python

   null_count = pool.t0_data()["_T0_"].isnull().sum()
   print(f"{null_count}/{len(pool)} sequences with T0 = null")

----

See Also
--------

* :doc:`manipulation` - Full operation reference including ``set_t0`` and ``t0_data``.
* :doc:`../user-guide/auto_examples/zeroing/sequence_t0` - Set T0 on sequence level.
* :doc:`../user-guide/auto_examples/zeroing/trajectory_t0` - Set T0 on trajectory level.
