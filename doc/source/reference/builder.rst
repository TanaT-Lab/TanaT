.. _builder_reference:

=================
Builder & Storage
=================

Reference for building sequence and trajectory pools from various data sources.
The builder pattern lets you chain multiple sources of the same schema before
materialising a single store on disk.

----

Builder Lifecycle
=================

.. code-block:: text

   SequencePool.builder()  
                →  .add_*()  
                →  .add_*()  
                →  .build(name)

The result of ``.build()`` is a path to the store. Wrap it in the corresponding
pool class to start working with it:

.. code-block:: python

   from tanat.sequence import IntervalSequencePool

   store_path = (
       IntervalSequencePool.builder()
       .add_parquet(
           "data.parquet",
           id_column="id",
           start_column="start",
           end_column="end",
       )
       .build("my_pool")
   )
   pool = IntervalSequencePool(store=store_path)

----

Source Methods
==============

All source methods are available on every
:py:class:`~tanat.sequence.base.builder.SequenceStoreBuilder` regardless of
pool type. They share the same column-mapping parameters and can be chained
freely.

.. list-table::
   :header-rows: 1
   :widths: 27 35 38

   * - Method
     - Input
     - Notes
   * - ``add_dataframe(df)``
     - ``pandas`` or ``polars`` DataFrame
     - In-memory; no file path required
   * - ``add_parquet(path)``
     - ``.parquet`` file or glob
     - Glob patterns (``"data/*.parquet"``) are supported
   * - ``add_csv(path)``
     - ``.csv`` file
     - Set ``try_parse_dates=True`` to auto-parse temporal columns
   * - ``add_sql(con, query)``
     - SQL query + connection string
     - Requires ``connectorx``; ``con`` is a DB URI such as ``"sqlite:///path.db"``

Temporal column names differ by pool type:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pool type
     - Required temporal columns
   * - ``EventSequencePool``
     - ``time_column``
   * - ``IntervalSequencePool``
     - ``start_column``, ``end_column``
   * - ``StateSequencePool``
     - ``start_column`` (``end_column`` is optional; see `Builder Options`_)

----

Static Features
===============

Static features are time-invariant attributes of an individual (age, gender,
cohort…). They can be attached at build time or added to an existing pool.

**At build time** - pass ``is_static=True`` to any ``add_*()`` call:

.. code-block:: python

   store_path = (
       IntervalSequencePool.builder()
       .add_parquet(
           "sequences.parquet",
           id_column="id",
           start_column="start",
           end_column="end",
           features=["value", "label"],
       )
       .add_csv(
           "demographics.csv",
           id_column="id",
           is_static=True,
           features=["age", "gender"],
           try_parse_dates=True,
       )
       .build("my_pool")
   )

**Shortcut functions** (``build_events``, ``build_intervals``, ``build_states``) -
pass the static DataFrame directly via the ``static_data`` parameter:

.. code-block:: python

   from tanat import build_intervals

   pool = build_intervals(
       temporal_data=df,
       id_column="id", start_column="start", end_column="end",
       static_data=static_df,
   )

**Post-build** - attach static features to an already-built pool:

.. code-block:: python

   pool.add_static_features(df)          # id column auto-detected
   pool.add_static_features(df, id_column="pid")   # explicit join key

----

Multi-Source Chaining
=====================

Multiple ``.add_*()`` calls on the same builder merge all rows into one pool.
All sources must share the same schema (same ``id_column`` name, same temporal
column names, same feature names).

.. code-block:: python

   store_path = (
       IntervalSequencePool.builder()
       .add_sql(
           DB, admissions_query,
           id_column="subject_id",
           start_column="admittime",
           end_column="dischtime",
           features=["admission_type"],
       )
       .add_parquet(
           "extra_patients.parquet",
           id_column="subject_id",
           start_column="admittime",
           end_column="dischtime",
           features=["admission_type"],
       )
       .add_csv(
           "simulated.csv",
           id_column="subject_id",
           start_column="admittime",
           end_column="dischtime",
           features=["admission_type"],
       )
       .build("all_admissions")
   )

.. note::

   A temporal dtype mismatch between sources (e.g. one ``Datetime[us]``,
   another ``Datetime[ms]``) triggers a warning at registration time and
   causes an error at ``.build()``. Cast the source data to a consistent
   dtype before calling ``add_*``.

----

Builder Options
===============

.. list-table::
   :header-rows: 1
   :widths: 25 22 45

   * - Pool type
     - Option
     - Purpose
   * - ``IntervalSequencePool``
     - ``sort_anchor``
     - Controls row ordering within each sequence: ``"start"``, ``"end"``, or ``"middle"``
       (midpoint of the interval)
   * - ``StateSequencePool``
     - ``end_column``
     - When omitted, ``T_END`` is computed as ``next(T_START)`` within each sequence
   * - ``StateSequencePool``
     - ``end_value``
     - Sentinel value appended as the last state's ``T_END``; defaults to ``None`` (leaves it null)
   * - ``StateSequencePool``
     - ``validate_continuity``
     - When ``end_column`` is provided, raises ``ValueError`` if gaps exist between
       consecutive states

.. code-block:: python

   from tanat.sequence import IntervalSequencePool
   from tanat.sequence.type.state.pool import StateSequencePool
   from datetime import datetime

   # IntervalSequencePool: intervals sorted by their midpoint
   store_path = (
       IntervalSequencePool.builder(sort_anchor="middle")
       .add_dataframe(
           df,
           id_column="id",
           start_column="start",
           end_column="end",
           features=["score"],
       )
       .build("intervals_mid")
   )
   pool = IntervalSequencePool(store=store_path)

   # StateSequencePool: end derived from next start, sentinel closes the last state
   store_path = (
       StateSequencePool.builder(end_value=datetime(2025, 12, 31))
       .add_dataframe(
           df,
           id_column="id",
           start_column="start",
           features=["phase"],
       )
       .build("states_closed")
   )
   pool = StateSequencePool(store=store_path)

----

Trajectory Composition
======================

A :py:class:`~tanat.trajectory.pool.TrajectoryPool` wraps multiple sequence
pools under a shared ID space. Each pool is registered under an **alias** that
acts as the retrieval key.

.. code-block:: text

   TrajectoryPool.builder()  
                →  .add(alias, pool)  
                →  .add(alias, pool)  
                →  .build(name)

.. code-block:: python

   from tanat.trajectory.pool import TrajectoryPool

   store_path = (
       TrajectoryPool.builder()
       .add("admissions", admissions_pool)
       .add("pharmacy", pharmacy_pool)
       .add("procedures", procedures_pool)
       .build("patient_trajectories")
   )
   tpool = TrajectoryPool(store=store_path)

Static features can also be added at trajectory build time via the same
``add_static_*`` family of methods:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Description
   * - ``add_static_dataframe(df)``
     - In-memory static features for the trajectory
   * - ``add_static_csv(path)``
     - Static features from a CSV file
   * - ``add_static_parquet(path)``
     - Static features from a Parquet file
   * - ``add_static_sql(con, query)``
     - Static features from a SQL query

----

Workspace
=========

A **workspace** is a named registry that maps store names to their paths on
disk. Once a store is built under a workspace, you can reload it by name
without tracking the file path.

.. code-block:: python

   from tanat import set_workspace, get_workspace

   set_workspace("~/.tanat_workspace/my_project")
   ws = get_workspace()

   # Build and register
   pool = IntervalSequencePool(store=builder.build("my_pool"))

   # Reload from workspace (no path needed)
   pool = ws["my_pool"]

   # Save a modified pool back under a new name
   pool.cast_features({"status": pl.Categorical})
   pool.save("my_pool_v2")

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Operation
     - Code
   * - Set the active workspace
     - ``set_workspace(path)``
   * - Get the active workspace object
     - ``get_workspace()``
   * - Reload a store by name
     - ``ws["name"]``  or  ``IntervalSequencePool(store="name")``
   * - List all registered stores
     - ``ws``  (repr) or  ``ws.list()``
   * - Save pool with pending changes
     - ``pool.save("new_name")``

----

See Also
--------

* :doc:`manipulation` - All pool operations available after building.
* :doc:`zeroing` - Setting a reference date (T0) after building.
* :doc:`../user-guide/auto_tutorials/building_pools` - Building from multiple sources.
* :doc:`../user-guide/auto_examples/container/index` - Build and explore container types.
