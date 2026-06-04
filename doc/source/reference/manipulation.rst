.. _manipulation_reference:

.. role:: green
.. role:: red

=================
Data Manipulation
=================

Reference for main operations available on sequence pools, trajectory pools,
individual sequences, trajectories, and entities.

----

Navigation
==========

Look up a single item by ID or row index; the return type depends on the container.

.. code-block:: python

   # SequencePool: lookup a sequence by ID
   seq = pool["patient_001"]

   # TrajectoryPool: lookup a trajectory by ID
   traj = tpool["patient_001"]

   # Sequence: lookup an entity by position
   entity = seq[0]
   entity = seq[-1]  # last entity


----

Iteration
=========

All containers implement the standard Python iteration protocol.

.. list-table::
   :header-rows: 1
   :widths: 36 16 16 16 16

   * - Syntax → yields
     - SP
     - S
     - TP
     - T
   * - ``for x in obj`` → item
     - Sequence
     - Entity
     - Trajectory
     - alias (str)
   * - ``obj.items()`` → (key, item) pairs
     - :red:`✗`
     - :red:`✗`
     - (id, Trajectory)
     - (alias, Sequence)

*SP: SequencePool · TP: TrajectoryPool · S: Sequence · T: Trajectory*

.. code-block:: python

    for traj in tpool:                  # TP → Trajectory
        print(traj.id_value)

    for alias, seq in traj.items():     # T  → (alias, Sequence)
        print(alias, len(seq))

    for seq in pool:                    # SP → Sequence
        print(seq.id_value, len(seq))

    for entity in seq:                  # S  → Entity
        print(entity.temporal_extent, entity.data())

----

Subset
======

Restrict a pool to a subset of IDs **without copying data**.

.. code-block:: python

   view = pool.subset(ids=["id_001", "id_042", "id_099"])
   print(len(view))   # 3

The returned object is a **view**: it shares the same underlying store.
Changes to the view (casts, feature drops…) are visible through the view only.

----

Feature Engineering
===================

All methods below operate **lazily**: transformations are applied on the fly at
materialisation time and do not rewrite the store. Call ``pool.save`` to persist them.

Add and remove columns
----------------------

Attach new columns to the view or hide existing ones; the underlying store is never rewritten.

.. list-table::
   :header-rows: 1
   :widths: 40 10 50

   * - Method
     - Scope
     - Description
   * - ``pool.add_entity_features(df)``
     - SP
     - Append new entity-level columns. ``df`` must be positionally aligned
       with the **full** entity row set of the store. Blocked on filtered
       views; call ``pool.save()`` first.
   * - ``pool.add_static_features(df)``
     - SP, TP
     - Append new static columns joined by ID. Works on filtered views. Pass
       ``id_column`` if the join key column has a non-standard name.
   * - ``pool.drop_features(names, is_static)``
     - SP
     - Hide entity (default) or static features from the view.
       Pass ``permanently=True`` to also delete from disk.
   * - ``tpool.drop_static_features(names)``
     - TP
     - Hide static features from a TrajectoryPool view.
       Pass ``permanently=True`` to also delete from disk.

*SP: SequencePool · TP: TrajectoryPool*

Type casting
------------

All casts are lazy and scoped to the current view. Call ``pool.save`` to persist.

.. list-table::
   :header-rows: 1
   :widths: 49 9 44

   * - Method
     - Scope
     - Description
   * - ``pool.cast_features(schema, is_static)``
     - SP
     - Re-type entity (default) or static features.
       ``schema`` is a ``dict[str, pl.DataType]``.
   * - ``tpool.cast_static_features(schema)``
     - TP
     - Re-type static features. Entity features must be cast on each
       linked sequence pool directly.
   * - ``pool.cast_to_datetime(unit, time_zone)``
     - SP, TP
     - Convert the time index to ``pl.Datetime``.
       *unit*: ``"s"`` / ``"ms"`` / ``"us"`` (default) / ``"ns"``.
       On TP the cast propagates to all linked sequence pools.
   * - ``pool.cast_to_timestep(dtype)``
     - SP, TP
     - Convert the time index to an integer type (e.g. ``pl.Int64``).
       Cannot be applied if the time index is already in Datetime format.

*SP: SequencePool · TP: TrajectoryPool*

.. code-block:: python

   import polars as pl

   # SequencePool: cast entity feature
   pool.cast_features({"status": pl.Categorical})

   # SequencePool: cast static feature
   pool.cast_features({"age": pl.UInt8}, is_static=True)

   # TrajectoryPool: cast static feature (different method name!)
   tpool.cast_static_features({"group": pl.Categorical})

   # Both: convert time index
   pool.cast_to_datetime(unit="us", time_zone="UTC")
   pool.cast_to_timestep(pl.Int32)

   # Drop (SP only with is_static; TP: drop_static_features)
   pool.drop_features(["flag_valid"], is_static=False)
   tpool.drop_static_features(["debug_col"])

----

Transformation
==============

All methods in this section return a **new DataFrame** and do not modify the pool.

``apply``: evaluate an expression
----------------------------------

Evaluate a Polars expression against the pool's temporal **or static** data.
Available on SP and TP. Pass ``is_static=True`` to target static features.

At pool level, ``by_id=True`` groups the evaluation per ID, making it ideal
for deriving per-sequence aggregates. A natural follow-up is to pipe the result
directly into ``add_static_features`` or ``add_entity_features``:

.. code-block:: python

   # Per-sequence mean → attach as a static feature
   means = pool.apply(pl.col("value").mean().alias("value_mean"), by_id=True)
   pool.add_static_features(means)

   # Without by_id: expression runs over the full temporal data
   flags = pool.apply(pl.col("value") > 0)

   # On static data
   result = pool.apply(pl.col("age") > 65, is_static=True)

   # Works on TrajectoryPool too
   stats = tpool.apply(pl.col("score").max().alias("score_max"), by_id=True)

*SP: SequencePool · TP: TrajectoryPool*

``to_dummies``: one-hot encode
------------------------------

One-hot encode one or more ``Categorical`` features into binary indicator
columns. Pass ``is_static=True`` to target static features instead of
entity features.

.. code-block:: python

   # Entity features (default)
   dummies = pool.to_dummies(["status", "category"])

   # Static features
   dummies = pool.to_dummies(["site"], is_static=True)

``binned_data`` / ``to_tensor``: regular time bins
--------------------------------------------------

Project temporal features onto a regular time grid.

* :meth:`~tanat.sequence.base.pool.SequencePool.binned_data` returns a
  long-format DataFrame (pandas or polars). Useful for exploration, joins,
  and plotting.
* :meth:`~tanat.sequence.base.pool.SequencePool.to_tensor` returns a dense
  ``(N, M, K)`` ndarray together with IDs and K-axis feature labels. Useful
  for ML pipelines.

.. code-block:: python

   # Long-format dataframe
   df = pool.binned_data(features=["value", "score"], bin_size="1d")

   # ML-ready tensor with IDs and feature names
   arr, ids, feature_names = pool.to_tensor(features=["value", "score"], bin_size="1d")

----

Descriptive Statistics
======================

.. code-block:: python

   # One row per sequence (length, temporal span, …)
   pool.describe()

   # Cross-ID aggregated stats (equivalent to pandas .describe())
   pool.describe(by_id=False)

   # Attach stats as static features (side-effect)
   pool.describe(add_to_static=True)

   # Single sequence
   seq = pool[pool.unique_ids[0]]
   seq.describe()

   # TrajectoryPool: one row per trajectory, columns prefixed by alias
   tpool.describe()

----

Persistence
===========

Transformations are **lazy** by default. Save a snapshot to make them
permanent or to share a modified pool.

.. code-block:: python

   # Save under a new name (returns the new store path)
   saved_path = pool.save("my_pool_optimised", overwrite=True)

   # Copy the pool in-memory (deep copy of settings, same store)
   clone = pool.copy()

----

Composition
===========

``extend``
----------

Merge another pool into the current one. Two execution paths:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situation
     - Behaviour
   * - Both pools share the **same store**
     - Fast path: union of ID masks, no I/O
   * - Different stores
     - Cross-store: rebuilds a new store on disk; ``destination`` is required

.. code-block:: python

   # Same-store fast path (e.g. after train_test_split)
   train, test = pool.train_test_split(test_size=0.3)
   merged = train.extend(test)

   # Cross-store merge
   pool_a.extend(pool_b, destination="merged_pool",
                 on_duplicate="skip", overwrite=True)

``train_test_split``
--------------------

Split a pool by **unique IDs**. The interface mirrors
``sklearn.model_selection.train_test_split``.

.. code-block:: python

   train, test = pool.train_test_split(test_size=0.2, random_state=42)

   # Guarantee: zero ID overlap
   assert not set(train.unique_ids) & set(test.unique_ids)

----

Type Conversion
===============

Convert a pool between the three sequence types. The conversion is always
**view-level**: the original store is not modified.

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Method
     - Converts to
   * - ``pool.as_event()``
     - :py:class:`~tanat.sequence.type.event.pool.EventSequencePool`
   * - ``pool.as_interval()``
     - :py:class:`~tanat.sequence.type.interval.pool.IntervalSequencePool`
   * - ``pool.as_state()``
     - :py:class:`~tanat.sequence.type.state.pool.StateSequencePool`

.. code-block:: python

   event_view = interval_pool.as_event()   # treat interval start as event time

----

Temporal Alignment
==================

See :doc:`zeroing` for the full T0 reference.

.. code-block:: python

   # Set a reference date using the position strategy
   pool.set_t0(position=0, anchor="start")

   # Retrieve T0 values as a DataFrame
   pool.t0_data()

   # Sequence-level properties (available after set_t0)
   seq = pool[pool.unique_ids[0]]
   seq.t0               # T0 value for this sequence
   seq.t0_nearest_rank  # 0-based index of the entity at or just before T0

----

See Also
--------

* :doc:`builder` - How to build and load pools.
* :doc:`zeroing` - T0 strategies and temporal alignment.
* :doc:`metadata` - Inspect dtypes, feature info, and cast methods.
