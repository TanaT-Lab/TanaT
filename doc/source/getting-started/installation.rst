Installation
============

Using PyPI
----------

Install the latest stable release from PyPI:

.. code-block:: bash

    python -m pip install tanat


Using the latest GitHub-hosted version
---------------------------------------

To get TanaT's latest development version directly from GitHub:

.. code-block:: bash

    python -m pip install git+https://github.com/TanaT-Lab/TanaT.git


Dependencies
------------

*TanaT* relies on several foundational libraries from the scientific Python ecosystem, including:

- ``pandas`` for convenient tabular data handling
- ``polars`` and ``pyarrow`` for high-performance columnar data processing
- ``numpy`` and ``scipy`` for numerical and scientific computing (transitive dependencies)
- ``matplotlib`` for basic visualization
- ``scikit-learn`` for machine learning utilities
- ``numba`` for performance optimization through JIT compilation

In addition, *TanaT* makes use of:

- ``tanat_utils`` for shared internal utilities
- ``tqdm`` for progress tracking in processing pipelines (transitive dependency)

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

SQL support (requires ``connectorx``):

.. code-block:: bash

    python -m pip install tanat[sql]
