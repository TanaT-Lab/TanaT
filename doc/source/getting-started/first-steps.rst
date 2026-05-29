First Steps
===========

This guide walks you through the core TanaT workflow: loading data, choosing the right sequence type, and exploring your temporal data.

.. note::
   Make sure TanaT is installed: ``pip install tanat`` (see :doc:`installation`).


1. Prepare Your Data
--------------------

A typical data structure that fits the *TanaT*'s meeds is a pandas DataFrames containing the events of a cohort of individuals. 
Such table may be referred as a *table of events*. Each row describes one event and is indexed by both an identifier of the individual and a temporal extend.

This example illustrates of such a table inspired by the MIMIC database:

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


In this table, containing 5 events, the ``patient_id`` is the identifier of the individuals (there are two individuals). 
Each event is also timestamped by a ``visit_date``. The last column contains information about the event itself. 
In this case, it is a categorical attribute that gives a type of visit.
Note that events can be described by more than one attribute (see :doc:`concepts` for a detailed comparison).

This table of events contain the information about the temporal sequences you would like to manipulate.
Concretize them as *TanaT* objects, and more specifically a :term:`sequence` :term:`pool`, will ease your work.

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

For our example, visits are **punctual events** so we use :class:`~tanat.sequence.EventSequencePool`.


3. Create a Sequence Pool
--------------------------

A :term:`pool` is a *TanaT* object that groups sequences from multiple individuals.

An as we want sequence of punctual events, we use the :func:`~tanat.sequence.shortcuts.build_events` shortcut function to create the pool from the dataframe above (use :func:`~tanat.sequence.shortcuts.build_states` for state sequence, etc.).
This function requires to know which are the indexing columns for individuals and time, and it infers all other columns as :term:`entity feature`.


.. code-block:: python

   from tanat import build_events

   pool = build_events(
       temporal_data=data,
       id_column="patient_id",
       time_column="visit_date",
   )

The ``pool`` is now a *TanaT* object!

.. note::
   The content of the dataframe has been copied in the pool, meaning that you can delete it to free memory.


For more advanced data ingestion settings and format (Parquet, CSV, SQL, multi-source chaining),
see the :doc:`../reference/builder` reference.

4. Verify Inferred Metadata
-----------------------------

Displaying the pool shows a summary of its content, structure and
automatically inferred :term:`metadata`. 

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


Before further exploration of your data, this summary allows you to verify the type inference made by the building function.
For instance, we see that ``visit_type`` has been inferred as a string feature, while it could be considered a categorical feature.
In this case, we suggest simply casting it to suit your analysis needs (see :doc:`../reference/metadata` for cast and override methods).


5. Access Individual Sequences
--------------------------------

As a pool, this data structure contains a collection of sequences that can be access by their identifier. 

The code below illustrates how access one sequence, and its internal data.

.. code-block:: python

   # Get a specific patient's sequence
   patient = pool['P001']
   print(f"Patient P001: {len(patient)} visits")

   # View the temporal data (id + time + entity features)
   print(patient.temporal_data().head())

   # View the static data (id + static features or None if not provided)
   print(patient.static_data().head())

``patient.temporal_data()`` provides a pandas dataframe similar to the table of events introduced earlier.
``patient.static_data()`` will return only if sequence identifier in this case, as there is no static (non-temporal) data associated with individuals (see :doc:`concepts` for details).

Instead of accessing through an identifier, *TanaT* provides iterators to explore the sequences:

.. code-block:: python

   # Pool → Sequence : iterate over all sequences
   for seq in pool:
       print(seq.id_value, len(seq))

6. Access Individual Entities
--------------------------------

Within a sequence, entities are accessed by index (entities are ordered along time axis).
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


Entities of a sequences can also be iterated in a standard Python manner:

.. code-block:: python

   # Sequence → Entity : iterate over all entities
   for entity in patient:
       print(entity.temporal_extent, entity.data())


Next Steps
----------

You now know how to build a pool, inspect metadata, and navigate sequences.
You are on the right track to visualize, manipulate, and analyze your sequences.
Here is the recommended reading order to deepen your understanding:

1. :doc:`concepts`: Understand the data model: entities, sequences, trajectories, and pools.
2. :doc:`../user-guide/auto_examples/index`: Self-contained examples for each container type, visualisation, and temporal alignment.
3. :doc:`../user-guide/auto_tutorials/index`: Step-by-step tutorials (multi-source ingestion, real-world applications, ...).
4. :doc:`../reference/index`: Full technical reference (builder, manipulation, zeroing, metadata, API).
