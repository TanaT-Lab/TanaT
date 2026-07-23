Tips for developpers
============

The page is a guide for developers who would like to extend TanaT and propose pull requests.


Installation
---------------------------------------

Clone the TanaT's latest development version directly from GitHub:

.. code-block:: bash

    git clone git@github.com:TanaT-Lab/TanaT.git

Create and set up a virtual environment (illustrated with ``venv``, and to be adapted for ``conda``)

.. code-block:: bash

    python -m venv ./venv
    source ./venv/bin/activate
    pip install -U pip
    pip install -U -e .[test]

or run directly the installation script (with ``venv``)

.. code-block:: bash

    ./script/install_in_venv.sh

Tests
---------------------------------------

TanaT uses ``pytest`` tools to manage tests. 
``syrupy`` is used to check some complex tests, it requires first to run 
a the test once to create snapshots of results and, once the expected restults have been 
generated the next runs of test assert that the tests generate the same results.

Before starting to implement a feature, we recommand to first run the tests to prepare
the ``syrupy`` snapshots.


.. code-block:: bash

    ./script/test.sh --update



