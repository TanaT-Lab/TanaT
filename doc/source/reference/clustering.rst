Clustering
==========

Clustering partitions a pool of sequences or trajectories into groups based on pairwise distances.

Available Algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Algorithm
     - Characteristics
   * - :class:`~tanat.clustering.HierarchicalClusterer`
     - Produces nested partitions at all distance thresholds.
   * - :class:`~tanat.clustering.PAMClusterer`
     - Medoid-based; robust to outliers; slower on large datasets
   * - :class:`~tanat.clustering.CLARAClusterer`
     - Scalable variant of PAM; samples subsets repeatedly for speed

See Also
---------

* :doc:`../user-guide/auto_examples/clustering/index` - Worked examples for each algorithm
* :doc:`metrics` - Sequence and trajectory metrics used to compute distance matrices for clustering
