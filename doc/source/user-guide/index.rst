User Guide
==========

This section provides comprehensive guides and examples for using TanaT effectively.
Whether you're looking for quick examples or in-depth tutorials, you'll find the resources you need here.

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/index
   auto_tutorials/index


Examples Gallery
-----------------

:doc:`Browse the examples gallery → <auto_examples/index>`

**Quick, focused examples** showing how to use specific TanaT features: data containers, metrics,
clustering, visualization, and zeroing. Each example provides working code you can adapt for
your own projects.


In-Depth Tutorials
------------------

:doc:`Browse all tutorials → <auto_tutorials/index>`

**Comprehensive guides** that walk you through complete analysis workflows.
Learn how different TanaT components work together to solve real-world problems.

----

**Building & Ingesting Data**

   | :doc:`Building pools from multiple sources <auto_tutorials/building_pools>`
   | *Combine Parquet and CSV sources into a single pool, compose trajectory pools, and manage the workspace*

----

**MIMIC-IV: Clinical Cohort Analysis**

   | :doc:`Exploring a patient cohort <auto_tutorials/mimic/explore_a_cohort>`
   | :doc:`Filtering and preparing a cohort <auto_tutorials/mimic/filter_and_prepare>`
   | :doc:`Analysing and clustering a cohort <auto_tutorials/mimic/analyse_and_cluster>`
   | :doc:`Survival analysis by admission cluster <auto_tutorials/mimic/survival_analysis>`   
   | :doc:`Learning clinical phenotypes with SWoTTeD <auto_tutorials/mimic/phenotype_with_swotted>`   
   | *End-to-end pipeline on real EHR data: load → filter → T0 anchor → cluster → survival* 

----

**Education: Learning Session Analysis**

   | :doc:`Exploring learner activity sequences <auto_tutorials/mooc/explore_sessions>`
   | :doc:`Clustering sessions by action patterns <auto_tutorials/mooc/cluster_sessions>`
   | *Session detection, Optimal Matching distance, hierarchical clustering on MOOC data*

----

**Time Series to Sequences**

   | :doc:`Discretizing time series into sequences <auto_tutorials/aeon/discretize>`
   | *Apply Aeon segmentation/quantization to raw time series and ingest the result as a TanaT sequence pool*

----

Getting Help
------------

If you need additional help:

* Check the :doc:`../reference/glossary` for terminology
* Consult the :doc:`../reference/api/index` for detailed API documentation
* Visit our :doc:`../community/index` section for contributing guidelines and support
