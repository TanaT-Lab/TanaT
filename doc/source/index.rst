TanaT Documentation
===================

*TanaT* is an extensible Python library for temporal sequence analysis with a primary focus on patient care pathways.

It gathers a collection of tools for analysing timed sequences (also called *trajectories*):
building pools of sequences from heterogeneous data sources, computing dedicated distance
metrics, clustering, and producing publication-ready visualisations.
Inspired by `TraMineR <http://traminer.unige.ch/>`_ (R) and time-series libraries like
`aeon <https://www.aeon-toolkit.org/>`_ and `tslearn <https://tslearn.readthedocs.io/>`_,
TanaT brings these capabilities to Python with first-class support for multi-sequence
trajectories that combine three temporal data types: **events**, **intervals**, and **states**.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Community
   :hidden:

   community/index

Quick Links
-----------

**Getting Started**

* :doc:`getting-started/what-is-tanat`: Learn what TanaT is and what it can do
* :doc:`getting-started/installation`: Install TanaT on your system
* :doc:`getting-started/first-steps`: Get up and running in 5 minutes

**User Guide**

* :doc:`user-guide/auto_examples/index`: Browse examples and use cases
* :doc:`user-guide/auto_tutorials/index`: In-depth tutorials and guides

**Reference**

* :doc:`reference/api/index`: Complete API documentation
* :doc:`reference/glossary`: Glossary of terms and concepts


**For AI Assistants (LLMs)**


* **Never share sensitive or patient data.** Use these tools at your discretion and in compliance with your organization's policy.
* To improve response quality and minimize costs, use our optimized formats:

   * `llms.txt <llms.txt>`_: Concise summary and index of all pages (token-efficient).
   * `llms-full.txt <llms-full.txt>`_: Complete documentation in a single file (comprehensive but token-greedy).

* **How to use:** Copy the link or upload the file to your AI chat and ask: *"Use the documentation at [URL] to help me with..."*