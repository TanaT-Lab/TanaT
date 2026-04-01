Reference
=========

Technical reference documentation for TanaT. This section is intended for looking up specific
information about classes, methods, parameters, and return types.

.. toctree::
   :maxdepth: 2
   :hidden:

   glossary
   builder
   manipulation
   zeroing
   metadata
   api/index

What's in this section?
------------------------

:doc:`glossary`
   Definitions of key terms and concepts used throughout TanaT documentation.

:doc:`builder`
   Builder lifecycle, all source methods (``add_dataframe``, ``add_parquet``,
   ``add_csv``, ``add_sql``), builder options, trajectory composition, and
   workspace management.

:doc:`manipulation`
   One-stop lookup for all operations on pools and sequences: navigation,
   feature engineering, persistence, composition, type conversion, and
   temporal alignment.

:doc:`zeroing`
   Setting a reference date (T0) with the four built-in strategies (position,
   direct, feature, query), pool-level inspection, and null handling.

:doc:`metadata`
   Metadata objects (``SequenceMetadata``, ``TrajectoryMetadata``, feature types),
   time index inspection, and all ``cast_*`` methods available on sequence and
   trajectory pools.

:doc:`api/index`
   Complete API reference automatically generated from source code.
   Browse by module to find all available classes, functions, and their signatures.
