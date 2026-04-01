First Steps
===========

This guide walks you through the core TanaT workflow: loading data, choosing the right sequence type, and exploring your temporal data.

.. note::
   Make sure TanaT is installed: ``pip install tanat`` (see :doc:`installation`).


1. Prepare Your Data
--------------------

TanaT works with pandas DataFrames containing temporal data:

.. code-block:: python

   import pandas as pd

   # Sample data: patient visits
   data = pd.DataFrame({
       'patient_id': ['P001', 'P001', 'P001', 'P002', 'P002'],
       'visit_date': pd.to_datetime([
           '2023-01-15', '2023-02-20', '2023-03-10',
           '2023-01-20', '2023-03-15'
       ]),
       'visit_type': ['GP', 'SPECIALIST', 'GP', 'GP', 'EMERGENCY']
   })


2. Choose the Right Sequence Type
----------------------------------

Before creating a pool, identify which sequence type matches your data
(see :doc:`concepts` for a detailed comparison):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Type
     - Your data has...
     - Example
   * - **EventSequence**
     - Single timestamps (punctual events)
     - Medical visits, purchases, clicks
   * - **IntervalSequence**
     - Start + end dates (can overlap)
     - Treatments, hospital stays, projects
   * - **StateSequence**
     - Contiguous states (no gaps, no overlap)
     - Disease stages, employment status

For our example, visits are **punctual events** so we use ``EventSequencePool``.


3. Create a Sequence Pool
--------------------------

A :term:`pool` groups sequences from multiple individuals.
Use the :func:`~tanat.sequence.shortcuts.build_events` shortcut:
it infers every column that is not ``id`` or ``time`` as an :term:`entity feature`.

For more advanced data ingestion (Parquet, CSV, SQL, multi-source chaining),
see the :doc:`../reference/builder` reference.

.. code-block:: python

   from tanat import build_events

   pool = build_events(
       temporal_data=data,
       id_column="patient_id",
       time_column="visit_date",
   )


4. Verify Inferred Metadata
-----------------------------

Displaying the pool shows a summary of its content, structure and
automatically inferred :term:`metadata`. Verify the inference before proceeding
(see :doc:`../reference/metadata` for cast and override methods):

.. code-block:: python

   print(pool)

.. code-block:: text

   ┌──────────────────────────────────────────────┐
   │           EventSequencePool Summary          │
   └──────────────────────────────────────────────┘

   Overview
   ─────────────────────────
     Sequences          2
     Store              ~/.tanat/_quick_event_...
     id_column          patient_id

   Time Index
   ─────────────────────────
     Type               Datetime [2023-01-15 → 2023-03-15]
     Columns            ['visit_date']
     t0                 position=0, anchor=None

   Entity Features (1)
   ─────────────────────────
     • visit_type          String [len 2 → 10]


5. Access Individual Sequences
--------------------------------

.. code-block:: python

   # Get a specific patient's sequence
   patient = pool['P001']
   print(f"Patient P001: {len(patient)} visits")

   # View the temporal data (id + time + entity features)
   print(patient.temporal_data().head())

   # View the static data (id + static features or None if not provided)
   print(patient.static_data().head())


6. Access Individual Entities
--------------------------------

Within a sequence, individual entities are accessed by index.
Positive and negative indices are both supported:

.. code-block:: python

   # Get the first entity (visit) in the sequence
   first_visit = patient[0]

   # Access entity properties
   print(first_visit.temporal_extent)  # 2023-01-15 00:00:00
   print(first_visit.data())           # {'visit_type': 'GP'}

   # Iterate over all entities in the sequence
   for entity in patient:
       print(entity.temporal_extent, entity.data())

7. Iterate over pools and sequences
-------------------------------------

Pools and sequences follow the standard Python iteration protocol:

.. code-block:: python

   # Pool → Sequence : iterate over all sequences
   for seq in pool:
       print(seq.id_value, len(seq))

   # Sequence → Entity : iterate over all entities
   for entity in patient:
       print(entity.temporal_extent, entity.data())

Next Steps
----------

You now know how to build a pool, inspect metadata, and navigate sequences.
Here is the recommended reading order to deepen your understanding:

1. :doc:`concepts`: Understand the data model: entities, sequences, trajectories, and pools.
2. :doc:`../user-guide/auto_examples/index`: Self-contained examples for each container type, visualisation, and temporal alignment.
3. :doc:`../user-guide/auto_tutorials/index`: Step-by-step tutorials (multi-source ingestion, real-World applications, ...).
4. :doc:`../reference/index`: Full technical reference (builder, manipulation, zeroing, metadata, API).
