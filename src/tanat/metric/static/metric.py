#!/usr/bin/env python3
"""
StaticMetric Class: base class for defining a static metric.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ...trajectory.trajectory import Trajectory
from ...trajectory.pool import TrajectoryPool
from ...sequence.base.sequence import Sequence
from ...sequence.base.pool import SequencePool


class StaticMetric:
    """Class for comparing static attributes between sequences or trajectories.

    The comparison function between static information is *to be provided by the
    user* while creating the :class:`~tanat.metric.static.metric.StaticMetric`
    object.
    If not defined this class implements a binary similarity function that returns
    0 if all attributes values are similars and 1 otherwise. It is based on the
    shared static attributes between compared objects. If none, it raises an error.

    Computes a scalar distance between two
    :class:`~tanat.trajectory.trajectory.Trajectory` or
    :class:`~tanat.sequence.base.sequence.Sequence` objects.

    Warning::

        The comparison function must have two parameters that are dictionaries.
        Use brackets with attribute names to access the value of a static
        feature.

    Static metric are objects that made to compute metrics between pairs of
    trajectories or sequences. It can be defined as it.

    Example::

        # Comparison between two dictionaries representing
        # the static attributes of two trajectories/sequences.
        # Values of the attributes of static features are accessed
        # using named brackets only.

        # This function must have two parameters and return a
        # float.

        # In this exemple, it compares the ages of the individuals
        def static_metric(static1, static2) -> float:
            return abs(float(static1["age"]) - float(static2["age"]))


        # Definition of a metric based on static features
        csmet = StaticMetric(smet)

        # Evaluate the metrics between two trajectories
        val = csmet(traj1, traj2)


    Static metric are more useful while hidden in the
    :class:`~tanat.metric.trajectory.AggregationTrajectoryMetric`.
    A static object comparison function or a StaticMetric can be
    defined to take into account static features in the aggregated
    metric.

    Example::

        # Same function to compare the `age` static attribute
        def static_metric(static1, static2) -> float:
            return abs(float(static1["age"]) - float(static2["age"]))

        # When an aggregation trajectory metric is defined, the user can
        # add an additional component based on the static features by
        # providing the function used to compare static features and also
        # a weight in the weighted sum of metrics.
        metric = AggregationTrajectoryMetric(static_metric=static_metric, static_metric_weight=0.5)
        value = metric(traj1, traj2)


    Information ::

        StaticMetric can evaluate cross-distance matrix from a pool of trajectories
        or sequences. Nonetheless, their computation is not optimized for large
        datasets.

    Warning ::
        Contrary to the other functions that compute cross-distance matrix it does not
        provide a DistanceMatrix

    """

    def __init__(self, cmp_fnct: Callable | None = None) -> None:
        """
        Args:
            cmp_fnct: function to define the comparison between static
            data (must have two dictionaries as parameters).
        """

        if cmp_fnct is not None and not isinstance(cmp_fnct, Callable):
            raise TypeError("Expected a function for defining the static metric.")
        self._cmp_fnct = cmp_fnct

    # ------------------------------------------------------------------
    # Comparison function (with validation)
    # ------------------------------------------------------------------

    def __call__(
        self, obj_a: Trajectory | Sequence, obj_b: Trajectory | Sequence
    ) -> float:
        """Compute distance between the static information of two objects.

        Args:
            obj_a: First object (Trajectory or Sequence).
            obj_b: Second object (Trajectory or Sequence).

        Returns:
            Scalar distance.
        """
        self._validate_objects(obj_a, obj_b)
        return self._compute(obj_a, obj_b)

    def compute_cross_matrix(
        self,
        pool_rows: TrajectoryPool | SequencePool,
        pool_cols: TrajectoryPool | SequencePool,
    ) -> np.ndarray:
        """Compute cross distance matrix based on the static data
        between two pools.

        Args:
            pool_rows: Trajectory or Sequence pool for rows   (N trajectories).
            pool_cols: Trajectory or Sequence pool for columns (M trajectories).

        Returns:
            ``static_matrix``: (N x M) matrix containing the pairwise
            distances computed based on static data.
        """
        if (
            pool_rows.static_data(fmt="polars") is None
            or pool_cols.static_data(fmt="polars") is None
        ):
            raise ValueError(
                "Computing static metric values between pools: No static data",
                "in at least one of the pools.",
            )

        matrix = np.full((len(pool_rows), len(pool_cols)), np.nan, dtype=np.float32)

        for id_r, row_r in enumerate(
            pool_rows.static_data(fmt="polars").iter_rows(named=True)
        ):
            for id_c, row_c in enumerate(
                pool_cols.static_data(fmt="polars").iter_rows(named=True)
            ):
                if self._cmp_fnct is None:
                    matrix[id_r, id_c] = self._default_cmp(row_r, row_c)
                else:
                    matrix[id_r, id_c] = self._cmp_fnct(row_r, row_c)

        return matrix

    def compute_matrix(
        self,
        pool: TrajectoryPool | SequencePool,
    ) -> np.ndarray:
        return self.compute_cross_matrix(pool, pool)

    def _compute(self, obj_a: Trajectory, obj_b: Trajectory) -> float:
        """Core distance computation.

        Args:
            obj_a: First object (Trajectory or Sequence).
            obj_b: Second object (Trajectory or Sequence).

        Returns:
            Scalar distance.
        """

        if self._cmp_fnct is None:
            return self._default_cmp(
                obj_a.static_data(fmt="dict"), obj_b.static_data(fmt="dict")
            )
        return self._cmp_fnct(
            obj_a.static_data(fmt="dict"), obj_b.static_data(fmt="dict")
        )

    def _default_cmp(self, dict_a: dict, dict_b: dict) -> float:
        """Default distance computation. Returns 0.0 if the attributes
        are the same and 1.0 otherwise.

        Args:
            obj_a: First object (Trajectory or Sequence).
            obj_b: Second object (Trajectory or Sequence).

        Returns:
            Scalar distance.
        """

        shared_attributes = dict_a.keys() & dict_b.keys()
        return 1.0 - float(all(dict_a[att] == dict_b[att] for att in shared_attributes))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_objects(
        self, obj_a: Trajectory | Sequence, obj_b: Trajectory | Sequence
    ) -> None:
        """Type-check objects: must be both trajectories or both sequences,
        and must contain static data."""
        if not (
            (isinstance(obj_a, Trajectory) and isinstance(obj_b, Trajectory))
            or (isinstance(obj_a, Sequence) and isinstance(obj_b, Sequence))
        ):
            raise TypeError(
                "Static metric parameters must be a Trajectory or Sequence, "
                f"got {type(obj_a).__name__}/{type(obj_b).__name__}."
            )
        if obj_a.static_data() is None or obj_b.static_data() is None:
            raise TypeError("Static metric requires object with static data.")
