Tutorials
=========

Comprehensive step-by-step guides that walk through complete workflows and real-world scenarios.
These tutorials help you learn TanaT systematically by combining multiple concepts into
practical examples.

.. toctree::
   :hidden:

   building_pools

Building & Ingesting Data
--------------------------

Master the builder API and load data from multiple heterogeneous sources:

* :doc:`Building pools from multiple sources <building_pools>`:
  combine Parquet and CSV sources into a single pool, configure builder options,
  compose trajectory pools, and manage the workspace


Working with Your Data
-----------------------

Master data manipulation and configuration:

* **Metadata management**: inspect, update, and control temporal and feature metadata
* **Type conversions**: convert between Event, State, and Interval sequence types
* **Data wrangling**: filter, transform, and prepare sequences and trajectories

Real-World Applications
------------------------

Apply TanaT to actual datasets:

* **Clinical data (MIMIC-IV)**: analyze electronic health records
* **MOOC student activity**: explore learner activity sequences from Massive Open Online Courses

----

See Also
--------

* :doc:`../../getting-started/first-steps`: Starting point before diving into tutorials.
* :doc:`../../getting-started/concepts`: TanaT data model explained.
* :doc:`../../reference/builder`: Builder API reference (source methods, options, workspace).
* :doc:`../../reference/manipulation`: All operations on pools and sequences.
* :doc:`../../reference/api/index`: Full API reference.
