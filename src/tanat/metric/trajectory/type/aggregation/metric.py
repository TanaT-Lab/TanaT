#!/usr/bin/env python3
"""
AggregationTrajectoryMetric: trajectory distance by per-alias sequence distances
with weighted aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from tanat_utils import settings_dataclass as dataclass

from ...base import TrajectoryMetric
from ....sequence.base import SequenceMetric
from ....matrix import DistanceMatrix
from ...._storage import save_progress
from ....static import StaticMetric

if TYPE_CHECKING:
    from .....trajectory.trajectory import Trajectory
    from .....trajectory.pool import TrajectoryPool
    from ...._storage import StorageOptions


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _weighted_mean(distances, weights):
    return float(np.average(distances, weights=weights))


def _weighted_sum(distances, weights):
    return float(np.dot(distances, weights))


def _matrix_weighted_mean(stack, weights):
    w = weights[:, None, None]
    mask = ~np.isnan(stack)
    weighted = np.where(mask, stack * w, 0.0)
    w_sum = np.sum(mask * w, axis=0)
    return np.where(w_sum > 0, weighted.sum(axis=0) / w_sum, np.nan)


def _matrix_weighted_sum(stack, weights):
    w = weights[:, None, None]
    result = np.nansum(stack * w, axis=0)
    all_nan = np.all(np.isnan(stack), axis=0)
    return np.where(all_nan, np.nan, result)


_AGG_REGISTRY: dict[str, dict[str, Callable]] = {
    "mean": {"scalar": _weighted_mean, "matrix": _matrix_weighted_mean},
    "sum": {"scalar": _weighted_sum, "matrix": _matrix_weighted_sum},
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class AggregationSettings:
    """Settings for :class:`AggregationTrajectoryMetric`.

    ``agg_fun`` and ``weights`` are orthogonal: ``"mean"`` + weights
    computes a weighted mean, ``"sum"`` + weights a weighted sum.
    Aliases absent from ``weights`` default to ``1.0``.
    """

    sequence_metrics: dict[str, SequenceMetric] | None = None
    agg_fun: str = "mean"
    weights: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class AggregationTrajectoryMetric(TrajectoryMetric, register_name="aggregation"):
    """Trajectory distance by per-alias sequence distances, then weighted aggregation.

    For each store alias visible on both trajectories, computes the
    sequence-level distance using the configured
    :class:`~tanat.metric.sequence.base.SequenceMetric`. The resulting
    per-alias distances are aggregated (weighted mean/sum) into a scalar
    trajectory distance.

    Example::

        lp = LinearPairwiseSequenceMetric(entity_metric=hamming)

        agg = AggregationTrajectoryMetric(
            sequence_metric={"event": lp; "states": lp},
            agg_fun="mean",
            weights={"events": 1.0, "states": 0.5},
        )

        dist = agg(traj_a, traj_b)
        dm   = agg.compute_matrix(traj_pool)
    """

    SETTINGS_CLASS = AggregationSettings
    MEMMAP_SUPPORT = True

    def __init__(
        self,
        sequence_metrics: dict[str, SequenceMetric | str] | None = None,
        static_metric: StaticMetric | Callable | None = None,
        static_metric_weight: float = 1.0,
        agg_fun: str = "mean",
        weights: dict[str, float] | None = None,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
    ) -> None:
        if store_path is not None:
            storage_options: dict | None = {
                "store_path": store_path,
                "chunk_size": chunk_size,
                "resume": resume,
                "dtype": dtype,
            }
        else:
            storage_options = None

        super().__init__(
            settings=AggregationSettings(
                sequence_metrics=sequence_metrics,
                agg_fun=agg_fun,
                weights=weights,
            ),
            storage=storage_options,
        )

        if static_metric is None or isinstance(static_metric, StaticMetric):
            self._static_metric = static_metric
        else:
            self._static_metric = StaticMetric(static_metric)
        self._static_metric_weight = static_metric_weight

    def _validate_trajectories(self, traj_a: Trajectory, traj_b: Trajectory) -> None:
        """Type-check both trajectory arguments. Overload the parent function to
        handle static data."""
        super()._validate_trajectories(traj_a, traj_b)
        if self._static_metric is not None:
            if traj_a.static_data is None:
                raise ValueError(
                    "The first trajectory has no static data but the metric requires",
                    " some. Remove the `static_metric` from the metrics settings or",
                    "add compatible static values to the trajectory.",
                )
            if traj_b.static_data is None:
                raise ValueError(
                    "The second trajectory has no static data but the metric requires",
                    " some. Remove the `static_metric` from the metrics settings or",
                    "add compatible static values to the trajectory.",
                )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, traj_a: Trajectory, traj_b: Trajectory) -> float:
        """Compute distance on the intersection of common aliases.

        Strict version for direct calls: raises on undefined pairs.

        Args:
            traj_a: First trajectory.
            traj_b: Second trajectory.

        Returns:
            Scalar distance.

        Raises:
            ValueError: If the trajectories share no common sequence aliases.
        """
        if self.settings.sequence_metrics is None:
            raise ValueError("Undefined sequence metrics.")

        common = sorted(set(traj_a) & set(traj_b))
        if not common:
            raise ValueError(
                f"Trajectories {traj_a.id_value!r} and {traj_b.id_value!r} "
                f"share no common sequence aliases. "
                f"traj_a: {sorted(traj_a)}, traj_b: {sorted(traj_b)}"
            )

        if len(set(self.settings.sequence_metrics.keys()).difference(common)) > 0:
            raise ValueError(
                f"Trajectories {traj_a.id_value!r} or {traj_b.id_value!r} "
                "misses some sequence aliases required by the metrics. "
            )

        seq_metrics = self.settings.sequence_metrics or {}
        weights_map = self.settings.weights or {}
        agg_fn = self._get_agg_fn()

        distances = []
        weights = []
        for alias in seq_metrics.keys():
            metric = seq_metrics[alias]
            distances.append(metric(traj_a[alias], traj_b[alias]))
            weights.append(weights_map.get(alias, 1.0))

        # Add distance on static data if the metric has been defined.
        if self._static_metric is not None:
            distances.append(self._static_metric(traj_a, traj_b))
            weights.append(self._static_metric_weight)

        return float(agg_fn(distances, weights))

    # ------------------------------------------------------------------
    # Optimised matrix computation (two-step strategy)
    # ------------------------------------------------------------------

    def _compute_per_alias_matrices(
        self, pool: TrajectoryPool
    ) -> tuple[list[np.ndarray], list[float]]:
        """Compute per-alias (per type of sequence) distance matrices expanded to
        the full ID space.

        For each alias, delegates to the configured
        :class:`~tanat.metric.sequence.base.SequenceMetric`, then expands
        the result into a collection of ``(N, N)`` array (``nan`` for missing IDs) where `N`
        is the number of trajectories.

        Args:
            pool: Trajectory pool.

        Returns:
            ``(expanded_list, weight_list)`` aligned with ``pool.sequence_pools``. The size of
            the returned lists are the same and correspond to the number of aliases (sequence
            types)
        """
        all_ids = pool.unique_ids
        N = len(all_ids)
        id_to_idx = {tid: i for i, tid in enumerate(all_ids)}

        expanded_list: list[np.ndarray] = []
        weight_list: list[float] = []

        for alias, sub_pool in pool.sequence_pools.items():
            metric = self._metric_for(alias)
            if metric is None:
                # no metric defined for this alias
                continue

            with self._nested_display():
                dm = metric.compute_matrix(sub_pool)

            # Expand sub-pool matrix to full ID space; nan for missing IDs
            expanded = np.full((N, N), np.nan, dtype=np.float32)
            idx = np.array([id_to_idx[sid] for sid in dm.ids])
            expanded[np.ix_(idx, idx)] = dm.to_numpy()

            expanded_list.append(expanded)
            weight_list.append(self._get_weight_for(alias))

        return expanded_list, weight_list

    def _compute_per_alias_cross_matrices(
        self,
        pool_rows: TrajectoryPool,
        pool_cols: TrajectoryPool,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Compute per-alias cross distance matrices expanded to the full ID spaces.

        Mirror of :meth:`_compute_per_alias_matrices` for the asymmetric
        (n × k) case.  For each alias shared by both pools, delegates to
        :meth:`~tanat.metric.sequence.base.SequenceMetric.compute_cross_matrix`,
        then expands the result to an ``(N, K)`` array (``nan`` for IDs absent
        from the alias sub-pool).

        Args:
            pool_rows: Trajectory pool for rows   (n trajectories).
            pool_cols: Trajectory pool for columns (k trajectories).

        Returns:
            ``(expanded_list, weight_list)``: one entry per alias present in
            the ``sequence_metrics`` definition.
        """
        ids_r = pool_rows.unique_ids
        ids_c = pool_cols.unique_ids
        N, K = len(ids_r), len(ids_c)
        id_to_row = {tid: i for i, tid in enumerate(ids_r)}
        id_to_col = {tid: i for i, tid in enumerate(ids_c)}

        expanded_list: list[np.ndarray] = []
        weight_list: list[float] = []

        for alias in self.settings.sequence_metrics.keys():
            sub_r = pool_rows.sequence_pools.get(alias)
            sub_c = pool_cols.sequence_pools.get(alias)

            expanded = np.full((N, K), np.nan, dtype=np.float32)

            if sub_r is not None and sub_c is not None:
                metric = self._metric_for(alias)
                with self._nested_display():
                    cross = metric.compute_cross_matrix(sub_r, sub_c)  # (nr, kc)

                row_idx = np.array([id_to_row[sid] for sid in sub_r.unique_ids])
                col_idx = np.array([id_to_col[sid] for sid in sub_c.unique_ids])
                expanded[np.ix_(row_idx, col_idx)] = cross

            expanded_list.append(expanded)
            weight_list.append(self._get_weight_for(alias))

        return expanded_list, weight_list

    def _compute_cross_matrix_impl(
        self,
        pool_rows: TrajectoryPool,
        pool_cols: TrajectoryPool,
    ) -> np.ndarray:
        """Three-step optimised cross (N × M) matrix computation with the same
        K aliases (types of sequences).


        **Step 1**: per-alias cross sub-matrices: for each alias, delegates
        to the configured :class:`~tanat.metric.sequence.base.SequenceMetric`
        via :meth:`compute_cross_matrix` (which uses Numba when available),
        then expands to the full ``(N, M)`` trajectory ID space.

        **Step 2**: static metric sub-matrices: evaluate the pairwise
        static metric between trajectories.

        **Step 3**: weighted aggregation: stack the per-alias matrices and
        apply :meth:`_aggregate_matrices` (numpy, vectorised).

        Args:
            pool_rows: Trajectory pool for rows   (N trajectories, K sequences-types).
            pool_cols: Trajectory pool for columns (M trajectories, K sequence-types).

        Returns:
            float32 numpy array of shape ``(N, M)``.
        """

        # step 1 -- construct per-alias sub-matrices
        expanded_list, weight_list = self._compute_per_alias_cross_matrices(
            pool_rows, pool_cols
        )

        # step 2 -- compute distances with the static metric
        if self._static_metric is not None:
            expanded_list.append(
                self._static_metric.compute_matrix(pool_rows, pool_cols)
            )
            weight_list.append(self._static_metric_weight)

        # step 3 -- aggregate the distances
        stack = np.stack(expanded_list)  # (K_aliases, N, K)
        w = np.array(weight_list, dtype=np.float64)
        return self._aggregate_matrices(stack, w).astype(np.float32)

    def _aggregate_matrices(self, stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Aggregate a ``(K, N, N)`` stack of per-alias matrices into ``(N, N)``.

        A cell is ``nan`` only when *all* K aliases are ``nan``
        (no common alias for that pair).
        """
        return self._get_agg_fn(matrix=True)(stack, weights)

    def _compute_matrix_impl(
        self,
        pool: TrajectoryPool,
        *,
        storage: StorageOptions | None = None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Two-step optimised matrix computation with optional memmap support.

        **Step 1** - per-alias sub-matrices: for each alias, delegate to
        the configured :class:`~tanat.metric.sequence.base.SequenceMetric`
        (which may itself use Numba or memmap), then expand to the full
        trajectory ID space.

        **Step 2** - weighted aggregation: stack the per-alias matrices
        and apply :meth:`_aggregate_matrices` (numpy, vectorised).

        When *storage* is provided the final aggregated matrix is written
        to a memory-mapped file with chunk-level resume support.

        Args:
            pool:        Trajectory pool.
            storage:     Optional :class:`~tanat.metric._storage.StorageOptions`.
            result:      Pre-opened memmap injected by the base, or ``None``
                         for the in-memory path.
            is_resuming: Whether *result* already has partial chunks.
            completed:   Number of chunks already flushed.

        Returns:
            :class:`~tanat.metric.DistanceMatrix`.
        """
        all_ids = pool.unique_ids
        N = len(all_ids)

        # Step 1: per-alias matrices (sub-metrics handle their own optimisations)
        expanded_list, weight_list = self._compute_per_alias_matrices(pool)

        stack = np.stack(expanded_list)  # (K, N, N)
        w = np.array(weight_list, dtype=np.float64)  # (K,)

        # --- In-memory path ---
        if result is None:
            agg = self._aggregate_matrices(stack, w).astype(np.float32)
            return DistanceMatrix(agg, all_ids)

        # --- Memmap + chunks path ---
        chunk_size = storage.chunk_size
        chunks = list(range(0, N, chunk_size))

        with self._create_progress_bar(total=len(chunks), desc="Chunks") as pbar:
            for chunk_idx, chunk_start in enumerate(chunks):
                chunk_end = min(chunk_start + chunk_size, N)

                if is_resuming and chunk_idx < completed:
                    pbar.update(1)
                    continue

                # Aggregate rows [chunk_start:chunk_end] across all K aliases
                chunk_stack = stack[:, chunk_start:chunk_end, :]  # (K, chunk, N)
                result[chunk_start:chunk_end, :] = self._aggregate_matrices(
                    chunk_stack, w
                )

                result.flush()
                completed += 1
                save_progress(storage, completed, status="computing")
                pbar.update(1)

        result.flush()
        save_progress(storage, completed, status="complete")

        return DistanceMatrix(result, all_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _metric_for(self, alias: str) -> SequenceMetric | None:
        """Return the SequenceMetric for *alias* (per-alias override or default)
        or None if the metric is not defined for this `alias`."""
        mapping = self.settings.sequence_metrics or {}
        return mapping.get(alias, None)

    def _get_weight_for(self, alias: str) -> float:
        """Return weight for *alias* (from ``settings.weights``, default ``1.0``)."""
        if self.settings.weights is None:
            return 1.0
        return self.settings.weights.get(alias, 1.0)

    def _get_agg_fn(self, *, matrix: bool = False) -> Callable:
        """Return the aggregation callable for ``self.settings.agg_fun``.

        Args:
            matrix: If ``True``, return the matrix variant operating on a
                ``(K, N, N)`` stack. If ``False`` (default), return the
                scalar variant operating on a list of distances.

        Returns:
            Aggregation callable.

        Raises:
            ValueError: If ``agg_fun`` is not a supported key.
        """
        entry = _AGG_REGISTRY.get(self.settings.agg_fun)
        if entry is None:
            raise ValueError(
                f"Unknown agg_fun '{self.settings.agg_fun}'. "
                f"Supported: {list(_AGG_REGISTRY.keys())}"
            )
        return entry["matrix" if matrix else "scalar"]
