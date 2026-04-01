Glossary
========

Key terms used throughout *TanaT* documentation and API.

For an overview of how these concepts relate, see :doc:`../getting-started/concepts`.

General terms
-------------

.. glossary::
   :sorted:

   Entity
      The atomic unit of temporal data. An entity represents a single observation
      for one individual at a given point or period in time, and carries one or more
      :term:`entity features <Entity feature>`.

   Entity feature
      A descriptive attribute of an :term:`entity`, such as a visit type, a
      diagnosis code, or a lab result value. Features can be categorical or numerical.

   Individual
      A unique subject of observation (patient, user, customer…). When an individual
      owns multiple :term:`sequences <Sequence>` of different types, they together
      form a :term:`trajectory <Trajectory>`.

   Sequence
      An ordered collection of :term:`entities <Entity>` of the same type belonging
      to one :term:`individual`, optionally enriched with :term:`static features
      <Static feature>`. All entities share the same feature structure and temporal
      type (event, interval, or state).

   Trajectory
      A collection of multiple :term:`sequences <Sequence>` of different types for
      the same :term:`individual`, optionally enriched with :term:`static features
      <Static feature>`. Trajectories give a multidimensional view of an
      individual's temporal evolution.

   Pool
      A collection of :term:`sequences <Sequence>` or :term:`trajectories
      <Trajectory>` across multiple :term:`individuals <Individual>`. Pools are
      the primary structure for batch analysis (distance matrices, clustering…).

   Static feature
      A time-invariant attribute of an :term:`individual` (e.g. birth date, gender,
      cohort). Can be attached to a :term:`sequence` as well as to a
      :term:`trajectory`.

   Metadata
      Descriptive information attached to a :term:`sequence` or :term:`pool` that
      characterises its temporal structure (granularity, timezone…) and the types
      of its :term:`entity features <Entity feature>`. TanaT infers metadata
      automatically and allows explicit overrides.

   Criterion
      A filtering rule applied to a data container (:term:`pool <Pool>`, :term:`sequence <Sequence>`, :term:`trajectory <Trajectory>`)
      based on temporal, pattern, or feature conditions.

   Zeroing
      The process of aligning :term:`sequences <Sequence>` or :term:`trajectories <Trajectory>` to a common reference
      date (T0 / index date), transforming absolute timestamps into relative ones
      to enable meaningful cross-individual comparison. See :py:mod:`tanat.zeroing`.

----

Sequence types
--------------

.. glossary::
   :sorted:

   Event sequence
      A :term:`sequence` of punctual events; each :term:`entity` occurs at a
      single timestamp with no duration.

   Interval sequence
      A :term:`sequence` of duration-based entities with explicit start and end
      dates. Intervals can overlap.

   State sequence
      A :term:`sequence` of contiguous, non-overlapping states. Together the
      entities cover the full observation period without gaps.