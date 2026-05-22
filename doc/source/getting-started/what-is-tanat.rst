What is TanaT
==============

*TanaT* (*Temporal ANalysis of Trajectories*) is an extensible Python library for temporal sequence analysis. 

The name of the library refers to a variety of wine grape that originates from south of France, taking continuity with the `TraMineR library <http://traminer.unige.ch/>`_ which widely inspired this work (Traminer is also a variety of wine grape).

It has been thought to address analysis and modeling issues of patient care pathways. 
This application guides the choices of the first features to be implemented.
Nonetheless, our core data model fits the needs of multiple different contexts.

What Makes *TanaT* Different?
----------------------------

Unlike traditional time series libraries (more especially `aeon <https://www.aeon-toolkit.org>`_ or `sktime <https://www.sktime.net>`_ ), *TanaT* is 
designed for **irregularly sampled, symbolic event-based temporal data**.
In practice, this means:

* Sequences in a pool can have different lengths (different numbers of observations per individual).
* Observations carry complex features (e.g. symbolic labels ``"GP"``, ``"EMERGENCY"``) rather than purely numeric values.
* Classical data analysis methods do not apply directly. *TanaT* provides dedicated metrics and algorithms instead.

These characteristics are common in multiple contexts such as healthcare (patient pathways), web analytics
(user journeys), industry 4.0 (logs), tourism (human activities), etc.

Compared to the few alternative libraries for sequence analysis (`TraMineR <http://traminer.unige.ch/>`_ or `Sequenzo <https://sequenzo.yuqi-liang.tech/en/>`_ ), *TanaT* proposes an 
original and rich data model. The first fundamental difference lies in the way temporal information is represented: contrary to the above libraries, we do not discretize 
time. This makes our representation lighter for most cases. The second core difference is the richness of event description.


Core Framework Functionalities
-------------------------------

The *TanaT* framework provides a complete workflow for temporal sequence analysis, from data ingestion to advanced analytics and visualization.

:doc:`../user-guide/auto_examples/container/index`
   Flexible representations for events, intervals, and states, at both individual and population levels.

:doc:`../reference/metadata`
   Automatic inference and explicit control of temporal and feature metadata.

:doc:`../reference/builder`
   Persistent storage and retrieval of sequences and trajectories.

:doc:`../user-guide/auto_examples/visualization/index`
   Rich visualization tools for exploring and interpreting temporal sequences and analysis results.

:doc:`../reference/zeroing`
   Temporal alignment and reference-date management for comparative analysis.

Inspiration and Related Work
----------------------------

TanaT has been strongly inspired by:

* The `TraMineR <http://traminer.unige.ch/>`_ library for the analysis of state sequences in R
* Libraries dedicated to time series analysis such as `aeon <https://www.aeon-toolkit.org/>`_ and `tslearn <https://tslearn.readthedocs.io/>`_

**Core Team:**

- Arnaud Duvermy (design, core architecture, maintenance)
- Thomas Guyet (project leader, design, development of data analysis methods, documentation)

Links
-----

* `Homepage <https://tanat-lab.github.io/TanaT-Lab/TanaT>`_
* `Source Code <https://github.com/TanaT-Lab/TanaT>`_
* `Issues <https://github.com/TanaT-Lab/TanaT/issues>`_

Ready to try it?
----------------

Head to :doc:`installation` to set up TanaT, then follow :doc:`first-steps`
to build your first sequence pool in under five minutes.
