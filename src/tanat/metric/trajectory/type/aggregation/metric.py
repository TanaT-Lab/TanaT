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

    default_metric: SequenceMetric = "linearpairwise"
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

        hamming = HammingEntityMetric(entity_feature="status")
        lp = LinearPairwiseSequenceMetric(entity_metric=hamming)

        agg = AggregationTrajectoryMetric(
            default_metric=lp,
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
        default_metric: SequenceMetric | str = "linearpairwise",
        sequence_metrics: dict[str, SequenceMetric | str] | None = None,
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
                default_metric=default_metric,
                sequence_metrics=sequence_metrics,
                agg_fun=agg_fun,
                weights=weights,
            ),
            storage=storage_options,
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
        common = sorted(set(traj_a) & set(traj_b))
        if not common:
            raise ValueError(
                f"Trajectories {traj_a.id_value!r} and {traj_b.id_value!r} "
                f"share no common sequence aliases. "
                f"traj_a: {sorted(traj_a)}, traj_b: {sorted(traj_b)}"
            )

        distances = []
        for alias in common:
            metric = self._metric_for(alias)
            distances.append(metric(traj_a[alias], traj_b[alias]))

        return self._aggregate(distances, common)

    # ------------------------------------------------------------------
    # Optimised matrix computation (two-step strategy)
    # ------------------------------------------------------------------

    def _compute_per_alias_matrices(
        self, pool: TrajectoryPool
    ) -> tuple[list[np.ndarray], list[float]]:
        """Compute per-alias distance matrices expanded to the full ID space.

        For each alias, delegates to the configured
        :class:`~tanat.metric.sequence.base.SequenceMetric`, then expands
        the result to an ``(N, N)`` array (``nan`` for missing IDs).

        Args:
            pool: Trajectory pool.

        Returns:
            ``(expanded_list, weight_list)`` aligned with ``pool.sequence_pools``.
        """
        all_ids = pool.unique_ids
        N = len(all_ids)
        id_to_idx = {tid: i for i, tid in enumerate(all_ids)}

        expanded_list: list[np.ndarray] = []
        weight_list: list[float] = []

        for alias, sub_pool in pool.sequence_pools.items():
            metric = self._metric_for(alias)

            with self._nested_display():
                dm = metric.compute_matrix(sub_pool)

            # Expand sub-pool matrix to full ID space; nan for missing IDs
            expanded = np.full((N, N), np.nan, dtype=np.float32)
            idx = np.array([id_to_idx[sid] for sid in dm.ids])
            expanded[np.ix_(idx, idx)] = dm.to_numpy()

            expanded_list.append(expanded)
            weight_list.append(self._get_weight_for(alias))

        return expanded_list, weight_list

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

    def _metric_for(self, alias: str) -> SequenceMetric:
        """Return the SequenceMetric for *alias* (per-alias override or default)."""
        mapping = self.settings.sequence_metrics or {}
        return mapping.get(alias, self.settings.default_metric)

    def _get_weight_for(self, alias: str) -> float:
        """Return weight for *alias* (from ``settings.weights``, default ``1.0``)."""
        if self.settings.weights is None:
            return 1.0
        return self.settings.weights.get(alias, 1.0)

    def _get_agg_fn(self, *, matrix: bool = False) -> Callable:
        """Return the aggregation callable for the configured ``agg_fun``.

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

    def _aggregate(self, distances: list[float], aliases: list[str]) -> float:
        """Aggregate per-alias scalar distances using weights and ``agg_fun``."""
        weights = [self._get_weight_for(a) for a in aliases]
        return self._get_agg_fn()(distances, weights)
