.. _concepts_ref:

Core Concepts
=============

This page introduces the fundamental concepts of *TanaT*'s data model.
Understanding these concepts is essential for using the library effectively.

*TanaT* organises temporal data in three nested levels: *entities*, *sequences*, and *trajectories*.

For population-level analysis, TanaT groups sequences and trajectories into *pools*.

----

Entities, Sequences, and Trajectories
-------------------------------------

*TanaT* distinguishes three levels of temporal data structures:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Level
     - Description
     - Example
   * - **Entity**
     - A single observation with temporal extent
     - A medical visit, a hospitalization
   * - **Sequence**
     - Collection of entities for one individual
     - All visits of patient P001
   * - **Trajectory**
     - Multiple sequences for one individual
     - Visits + hospitalizations + lab results for P001

Entity
~~~~~~

An :term:`entity` is the atomic unit of temporal data in a :term:`sequence <Sequence>` object. It has:

- **Features**: One or more descriptive attributes (e.g., visit type, diagnosis code)
- **Temporal extent**: Either a single timestamp or a time interval

The temporal extent nature and feature structure are formalized through :term:`metadata`.

Sequence
~~~~~~~~

A :term:`sequence` is a temporal arrangement of entities described by the same :term:`metadata`.
All entities in a sequence share the same type (events, intervals, or states)
and the same feature structure.
See the :doc:`../user-guide/auto_examples/container/index` examples to build and explore each type.

The diagram below shows a sequence with 4 event entities.
Note that two events can share the same timestamp (Event `A` and Event `B` on *Nov 8*).

.. mermaid::

   gantt
      dateFormat  YYYY-MM-DD
      axisFormat %d
      title Event Sequence Example

      Event A (1st) : milestone, A1, 2023-11-01, 0d
      Event A (2nd) : milestone, A2, 2023-11-08, 0d
      Event B       : milestone, B1, 2023-11-08, 0d
      Event C       : milestone, C1, 2023-11-23, 0d

Trajectory
~~~~~~~~~~

A :term:`trajectory` combines multiple sequences of different types for the same individual.
For a complete walkthrough, see the :doc:`../user-guide/auto_examples/container/trajectory` example.

The diagram below shows a trajectory with three sequence types:

.. mermaid::

   gantt
      dateFormat  YYYY-MM-DD
      axisFormat %d
      title Trajectory Example

      section Visits
      Event A : milestone, A1, 2023-11-01, 0d
      Event A : milestone, A2, 2023-11-08, 0d
      Event B : milestone, B1, 2023-11-20, 0d

      section Hospitalizations
      Stay : I1, 2023-11-15, 2023-11-18

      section Lab Tests
      Test U : milestone, U1, 2023-11-09, 0d
      Test V : milestone, V1, 2023-11-13, 0d

----

Sequence Types
--------------

TanaT supports three types of temporal extent:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Type
     - Temporal Extent
     - Constraints
   * - **Event**
     - Single timestamp (punctual)
     - None
   * - **Interval**
     - Start and end dates
     - Can overlap, gaps allowed
   * - **State**
     - Start and end dates
     - Contiguous, no overlap, no gaps

.. mermaid::

   gantt
      dateFormat  YYYY-MM-DD
      axisFormat %d
      title       Comparison of Sequence Types

      section Event
      Event A : milestone, A1, 2023-11-01, 0d
      Event A : milestone, A2, 2023-11-08, 0d
      Event B : milestone, B1, 2023-11-20, 0d

      section Interval
      Interval K : K1, 2023-11-04, 2023-11-09
      Interval J : J1, 2023-11-12, 2023-11-17
      Interval I : I1, 2023-11-15, 2023-11-19

      section State
      State U : U1, 2023-11-01, 2023-11-08
      State V : V1, 2023-11-08, 2023-11-12
      State W : W1, 2023-11-12, 2023-11-18
      State X : X1, 2023-11-18, 2023-11-22

**When to use each type:**

- **Event**: Point-in-time occurrences (visits, purchases, clicks). Use :func:`~tanat.sequence.shortcuts.build_events`.
- **Interval**: Duration-based events that can overlap (treatments, projects). Use :func:`~tanat.sequence.shortcuts.build_intervals`.
- **State**: Continuous states without gaps (disease stages, employment status). Use :func:`~tanat.sequence.shortcuts.build_states`.

----

Pools
-----

A :term:`pool` is a collection of sequences or trajectories for multiple individuals.
All individual sequences of a pool share the same structure (same features, same temporal type).
Pools are the primary data structure for analysis operations like computing distance matrices or clustering.

Pools can be created with shortcut functions (:func:`~tanat.sequence.shortcuts.build_events`, :func:`~tanat.sequence.shortcuts.build_intervals`,
:func:`~tanat.sequence.shortcuts.build_states`) or via the lower-level builder pattern for multi-source ingestion.
See :doc:`first-steps` or :doc:`../reference/builder` for the full builder reference.

Static Data
-----

A sequence can be complemented by non-temporal features, so called :term:`static features <static feature>` (attributes like birth date or gender).
Similarly to temporal features, static features are also described through the :term:`metadata`. More specifically, each static feature has a type. 
Static features are the same for all sequences belonging to a pool.


----

See Also
--------

* :doc:`first-steps`: Minimal working example to get started quickly
* :doc:`../reference/glossary`: All terms defined in one place
* :doc:`../reference/builder`: Build pools from DataFrames, Parquet, CSV, or SQL
* :doc:`../reference/manipulation`: Iterate, navigate, transform, and split pools
* :doc:`../reference/zeroing`: Align sequences to a common reference date (T0)
* :doc:`../reference/metadata`: Inspect and cast feature types
* :doc:`../reference/api/index`: Complete API documentation
