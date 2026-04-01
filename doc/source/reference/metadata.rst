.. _metadata_reference:

.. role:: green
.. role:: red

========
Metadata
========

TanaT automatically infers rich metadata from your data at build time.
Metadata is attached to every pool via ``pool.metadata`` and describes
the time index, entity features, and static features.

----

Metadata Objects
================

``pool.metadata`` on a :py:class:`~tanat.sequence.base.pool.SequencePool`
returns a :py:class:`~tanat.metadata.sequence.SequenceMetadata` instance;
on a :py:class:`~tanat.trajectory.pool.TrajectoryPool` it returns a
:py:class:`~tanat.metadata.trajectory.TrajectoryMetadata`.

.. code-block:: python

   print(pool.metadata)              # human-readable summary
   pool.metadata.time_index          # TimeIndexInfo (dtype, range, tz…)
   pool.metadata.entity_features     # list[FeatureInfo], alphabetical
   pool.metadata.static_features     # list[FeatureInfo] | None

Both objects expose ``is_categorical_feature(name)``;
:py:class:`~tanat.metadata.sequence.SequenceMetadata` also has
``is_numeric_feature``, ``is_datetime_feature``, and ``is_duration_feature``.
All raise ``KeyError`` for unknown feature names.

.. tip::

   Printing any TanaT object renders a metadata summary: ``print(pool)``,
   ``print(sequence)``, ``print(trajectory)`` and even ``print(entity)`` all
   display type, features and temporal range in a human-readable form.

----

Feature Types
=============

TanaT maps each Polars dtype to a :py:class:`~tanat.metadata.feature.FeatureInfo`
subclass with type-specific extra attributes:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Class
     - Polars dtypes
     - Extra attributes
   * - :py:class:`~tanat.metadata.feature.NumericalInfo`
     - integers, floats
     - ``min``, ``max``
   * - :py:class:`~tanat.metadata.feature.CategoricalInfo`
     - ``Categorical``, ``Enum``
     - ``n_unique``, ``ordered``
   * - :py:class:`~tanat.metadata.feature.BooleanInfo`
     - ``Boolean``
     - ``true_count``, ``false_count``
   * - :py:class:`~tanat.metadata.feature.StringInfo`
     - ``String``
     - ``min_length``, ``max_length``
   * - :py:class:`~tanat.metadata.feature.TemporalInfo`
     - ``Date``, ``Datetime``, ``Duration``
     - ``min``, ``max``, ``is_duration``
   * - :py:class:`~tanat.metadata.feature.ArrayInfo`
     - ``Array``, ``List``
     - ``dimension``

.. code-block:: python

   info = pool.metadata.feature_info("status")
   print(info.summary)   # e.g. "Categorical (5 categories)"

----

Cast Methods
============

Casts are **lazy and view-local**: they are applied on the fly when data
is materialised, and do not touch the store files. Call ``pool.save()``
to persist them.

.. note::

   Cast methods are only available at **Pool level** (``SequencePool``,
   ``TrajectoryPool``). Casting directly on a ``Sequence``, ``Trajectory``,
   or ``Entity`` is intentionally not supported: those objects are views
   derived from a pool, and mutating their types independently would
   desynchronise them from their siblings in the pool.

SequencePool
------------

.. rubric:: cast_features()

Cast one or more entity or static feature columns.

.. code-block:: python

   # Entity features (default)
   pool.cast_features({"status": pl.Categorical})
   pool.cast_features({"response_time": pl.Duration("ms")})
   pool.cast_features({"severity": pl.Enum(["low", "medium", "high"])})

   # Static features
   pool.cast_features({"age": pl.UInt8, "group": pl.Categorical}, is_static=True)

----

.. rubric:: cast_id()

Cast the sequence ID column.

.. code-block:: python

   pool.cast_id(pl.Categorical)

----

.. rubric:: cast_to_datetime() / cast_to_timestep()

Change the type of the time index.

.. code-block:: python

   pool.cast_to_datetime()                            # us, no timezone
   pool.cast_to_datetime(unit="ms", time_zone="UTC")

   pool.cast_to_timestep(pl.UInt32)

.. note::
   ``cast_to_timestep()`` raises ``TypeError`` if the time index is already
   a ``Datetime`` type.

----

TrajectoryPool
--------------

Entity features live inside each linked sequence store; cast them directly
on ``tpool.sequence_pools["<alias>"]``.

.. rubric:: cast_static_features()

Cast trajectory-level static features.

.. code-block:: python

   tpool.cast_static_features({"group": pl.Categorical})

   # For entity features, go through the sequence pool:
   tpool.sequence_pools["pharmacy"].cast_features({"medication": pl.Categorical})

----

.. rubric:: cast_id() / cast_to_datetime() / cast_to_timestep()

Same signatures as on :py:class:`~tanat.sequence.base.pool.SequencePool`,
but the cast is **automatically propagated to all linked sequence pools**.

.. code-block:: python

   tpool.cast_id(pl.Categorical)                      # propagates to all sub-pools
   tpool.cast_to_datetime(unit="ms", time_zone="UTC") # idem
   tpool.cast_to_timestep(pl.UInt32)                  # idem

----

Compatibility Matrix
====================

.. list-table::
   :header-rows: 1
   :widths: 42 14 14

   * - Method
     - ``SequencePool``
     - ``TrajectoryPool``
   * - ``cast_features(schema, is_static=False)``
     - :green:`✓`
     - :red:`✗`
   * - ``cast_static_features(schema)``
     - :red:`✗`
     - :green:`✓`
   * - ``cast_id(dtype)``
     - :green:`✓`
     - :green:`✓`
   * - ``cast_to_datetime(unit, time_zone)``
     - :green:`✓`
     - :green:`✓`
   * - ``cast_to_timestep(dtype)``
     - :green:`✓`
     - :green:`✓`

----

See Also
--------

* :py:mod:`tanat.metadata`: full API for all metadata classes
* :py:meth:`tanat.sequence.base.pool.SequencePool.cast_features`
* :py:meth:`tanat.trajectory.pool.TrajectoryPool.cast_static_features`
