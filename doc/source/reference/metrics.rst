Metrics
=======

**Philosophy: Hierarchical Composition**

The metrics module implements a three-tier hierarchy of distance functions:

1. **EntityMetric**: Single point-in-time comparisons (categorical feature equality)
2. **SequenceMetric**: Built on EntityMetric, aligned/aggregated across timesteps
3. **TrajectoryMetric**: Aggregates SequenceMetrics across multiple sequence types

This composition allows flexible metric selection at each level while maintaining
consistent interfaces.

Entity Metrics
--------------

Entity metrics compare individual states or events on categorical features.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Purpose
   * - :class:`~tanat.metric.entity.HammingEntityMetric`
     - Categorical mismatch counter; building block for all sequence metrics

Sequence Metrics
----------------

Sequence metrics operate on entire sequences, leveraging an EntityMetric as a foundation.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Purpose
   * - :class:`~tanat.metric.sequence.LinearPairwiseSequenceMetric`
     - Position-wise alignment with configurable aggregation (mean, max, sum)
   * - :class:`~tanat.metric.sequence.EditSequenceMetric`
     - Needleman-Wunsch edit distance; insertions, deletions, substitutions
   * - :class:`~tanat.metric.sequence.LCSSequenceMetric`
     - Longest Common Subsequence distance
   * - :class:`~tanat.metric.sequence.DTWSequenceMetric`
     - Dynamic Time Warping; flexible time alignment
   * - :class:`~tanat.metric.sequence.SoftDTWSequenceMetric`
     - Differentiable DTW variant for optimization
   * - :class:`~tanat.metric.sequence.Chi2SequenceMetric`
     - Chi-squared distance between state-time distributions

Trajectory Metrics
------------------

Trajectory metrics compare trajectories (multi-sequence entities) by aggregating
sequence-level distances.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Purpose
   * - :class:`~tanat.metric.AggregationTrajectoryMetric`
     - Compute per-alias SequenceMetric, then aggregate via mean/min/max/sum

**Key feature:** Use different SequenceMetrics for different sequence types (aliases).
For example, use EditSequenceMetric for states and LCSSequenceMetric for events,
then aggregate the results.

Distance Matrix
---------------

:class:`~tanat.metric.DistanceMatrix` is a matrix wrapper that stores
pairwise distances and pool IDs, enabling efficient cluster operations.

SequenceMetric classes provide a :meth:`~tanat.metric.SequenceMetric.compute_matrix`
method to compute the full pairwise distance matrix:

.. code-block:: python

   metric = EditSequenceMetric(entity_metric=hamming, normalize=True)
   distance_matrix = metric.compute_matrix(pool)

Similarly, TrajectoryMetric classes have a :meth:`~tanat.metric.TrajectoryMetric.compute_matrix`
method that computes the full distance matrix across multiple trajectories.

----

See Also
--------

* :doc:`../user-guide/auto_examples/metric_entity/index` - Entity metric examples
* :doc:`../user-guide/auto_examples/metric_sequence/index` - Sequence metric examples
* :doc:`../user-guide/auto_examples/metric_trajectory/index` - Trajectory metric examples
* :doc:`clustering` - Clustering algorithms that operate on sequences or trajectories
