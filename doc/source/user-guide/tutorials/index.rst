Tutorials
=========

Comprehensive step-by-step guides that walk through complete workflows and real-world scenarios.
These tutorials help you learn TanaT systematically by combining multiple concepts into
practical examples.

.. toctree::
   :hidden:

   building_pools
   mimic/index
   mooc/index

Building & Ingesting Data
--------------------------

Master the builder API and load data from multiple heterogeneous sources:

* :doc:`Building pools from multiple sources <building_pools>`:
  combine Parquet and CSV sources into a single pool, configure builder options,
  compose trajectory pools, and manage the workspace


Real-World Applications
------------------------

End-to-end workflows on real datasets:

.. rubric:: MIMIC-IV: Clinical Cohort Analysis

:doc:`mimic/index` · A series of tutorials on the MIMIC-IV demo dataset:

* :doc:`Exploring a patient cohort <mimic/explore_a_cohort>`: load, inspect, visualise, train/test split
* :doc:`Filtering and preparing a cohort <mimic/filter_and_prepare>`: criteria, T0 anchor, relative window
* :doc:`Analysing and clustering a cohort <mimic/analyse_and_cluster>`: distance matrix, hierarchical clustering, faceted timeline
* :doc:`Survival analysis by admission cluster <mimic/survival_analysis>`: mortality endpoint, survival target, per-cluster Kaplan-Meier curves


.. rubric:: Education: Learning Session Analysis

:doc:`mooc/index` · A series of tutorials on the MOOC demo dataset:

* :doc:`Exploring learner activity sequences <mooc/explore_sessions>`: load, inspect, visualise, train/test split
* :doc:`Clustering sessions by action patterns <mooc/cluster_sessions>`: criteria, T0 anchor, relative window

----

See Also
--------

* :doc:`../../getting-started/first-steps`: Starting point before diving into tutorials.
* :doc:`../../getting-started/concepts`: TanaT data model explained.
* :doc:`../../reference/builder`: Builder API reference (source methods, options, workspace).
* :doc:`../../reference/manipulation`: All operations on pools and sequences.
* :doc:`../../reference/api/index`: Full API reference.
