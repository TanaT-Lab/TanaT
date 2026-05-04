.. _mimic_tutorials:

MIMIC-IV: Clinical Cohort Analysis
=====================================

A series of tutorials walking through a complete clinical analysis pipeline on the
`MIMIC-IV demo dataset <https://physionet.org/content/mimic-iv-demo/>`_
(100 de-identified patients from Beth Israel Deaconess Medical Center).

Each script is **self-contained** and downloads the database automatically on
first use via :func:`~tanat.dataset.access`.

The demo database covers:

- Patient demographics (gender, age)
- Hospital admissions with types and locations
- Medical procedures coded with ICD standards
- Pharmacy prescriptions and medication data

**Data access**

- Use ``access("mimic4")`` (automatic download, cached locally).
- Manual setup from `PhysioNet MIMIC-IV Demo
  <https://physionet.org/content/mimic-iv-demo/2.2/>`_ following the
  `MIT-LCP instructions <https://github.com/MIT-LCP/mimic-code>`_.

.. note::

   SQL ingestion requires ``connectorx``: ``pip install 'tanat[sql]'``