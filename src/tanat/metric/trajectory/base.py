#!/usr/bin/env python3
"""
TrajectoryMetric ABC: base class for all trajectory-level distance metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tanat_utils import SettingsMixin, Registrable, DisplayMixin

from ..matrix import DistanceMatrix
from .._utils import (
    resolve_storage,
    default_pairwise_matrix,
    default_cross_matrix,
    validate_pair,
)
from .._storage import StorageOptions, open_or_create_matrix, compute_metric_config
from ...trajectory.pool import TrajectoryPool
from ...trajectory.trajectory import Trajectory


class TrajectoryMetric(SettingsMixin, Registrable, DisplayMixin, ABC):
    """Abstract base for trajectory-level distance metrics.

    Computes a scalar distance between two
    :class:`~tanat.trajectory.trajectory.Trajectory` objects and a full
    pairwise :class:`~tanat.metric.DistanceMatrix` over a
    :class:`~tanat.trajectory.pool.TrajectoryPool`.
    """

    _REGISTER: dict = {}
    _TYPE_SUBMODULE = "type"

    #: Set to ``True`` in subclasses that implement disk-backed (memmap) computation.
    MEMMAP_SUPPORT: bool = False

    def __init__(
        self, settings=None, storage: StorageOptions | dict | None = None
    ) -> None:
        super().__init__(settings)
        self._storage = resolve_storage(
            type(self).__name__, self.MEMMAP_SUPPORT, storage
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, traj_a: Trajectory, traj_b: Trajectory) -> float:
        """Compute distance between two trajectories.

        Args:
            traj_a: First trajectory.
            traj_b: Second trajectory.

        Returns:
            Scalar distance.
        """
        self._validate_trajectories(traj_a, traj_b)
        return self._compute(traj_a, traj_b)

    @abstractmethod
    def _compute(self, traj_a: Trajectory, traj_b: Trajectory) -> float:
        """Core distance computation.

        Args:
            traj_a: First trajectory.
            traj_b: Second trajectory.

        Returns:
            Scalar distance.
        """

    # ------------------------------------------------------------------
    # Public API — Matrix computation
    # ------------------------------------------------------------------

    def compute_matrix(
        self,
        pool: TrajectoryPool,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
    ) -> DistanceMatrix:
        """Compute full pairwise trajectory distance matrix.

        Storage kwargs are forwarded to
        :class:`~tanat.metric.StorageOptions`.

        Args:
            pool:       A :class:`~tanat.trajectory.pool.TrajectoryPool`.
            store_path: Storage directory (``None`` → in-memory).
            chunk_size: Rows per flush chunk (default 500).
            resume:     Skip already-computed chunks (default ``True``).
            dtype:      Numpy dtype for the matrix (default ``"float32"``).

        Returns:
            A :class:`~tanat.metric.DistanceMatrix`.
        """
        storage = resolve_storage(
            type(self).__name__,
            self.MEMMAP_SUPPORT,
            self._storage,
            store_path=store_path,
            chunk_size=chunk_size,
            resume=resume,
            dtype=dtype,
        )
        self._validate_pool(pool)

        result, is_resuming, completed = None, False, 0
        if storage is not None:
            ids = pool.unique_ids
            result, is_resuming, completed, is_complete = open_or_create_matrix(
                storage, len(ids), ids, compute_metric_config(self)
            )
            if is_complete:
                self._display_header()
                self._display_message("Cache hit: returning precomputed matrix")
                self._display_footer(f"{len(pool)} trajectories")
                return DistanceMatrix(result, ids)

        self._display_header()
        dm = self._compute_matrix_impl(
            pool,
            storage=storage,
            result=result,
            is_resuming=is_resuming,
            completed=completed,
        )
        self._display_footer(f"{len(pool)} trajectories")
        return dm

    def compute_cross_matrix(
        self,
        pool_rows: TrajectoryPool,
        pool_cols: TrajectoryPool,
    ) -> np.ndarray:
        """Compute an asymmetric (n × k) distance matrix between two pools.

        Row ``i`` ↔ trajectory ``i`` in *pool_rows*; column ``j`` ↔
        trajectory ``j`` in *pool_cols*.

        Validates both pools, then delegates to
        :meth:`_compute_cross_matrix_impl`.  Subclasses override
        :meth:`_compute_cross_matrix_impl` to use optimised kernels.

        Args:
            pool_rows: Pool whose trajectories form the rows   (n items).
            pool_cols: Pool whose trajectories form the columns (k items).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        self._validate_pool(pool_rows)
        self._validate_pool(pool_cols)
        return self._compute_cross_matrix_impl(pool_rows, pool_cols)

    # ------------------------------------------------------------------
    # Template Method chain (pairwise)
    # ------------------------------------------------------------------

    def _compute_matrix_impl(
        self,
        pool: TrajectoryPool,
        *,
        storage=None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Override point for optimised pairwise matrix computation.

        Subclasses override this method to replace the pure-Python loop with
        a faster kernel.  The default delegates to
        :meth:`_compute_matrix_python`, which handles both in-memory execution
        and optional disk-backed chunked computation.
        """
        return self._compute_matrix_python(
            pool,
            storage=storage,
            result=result,
            is_resuming=is_resuming,
            completed=completed,
        )

    def _compute_matrix_python(
        self,
        pool: TrajectoryPool,
        storage=None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Naive Python pairwise fallback.

        Delegates to :func:`~tanat.metric._utils.default_pairwise_matrix` and
        computes the full ``(n, n)`` square.

        Args:
            pool:        Trajectory pool.
            storage:     Optional storage options.
            result:      Pre-opened memmap or ``None`` for in-memory.
            is_resuming: Whether partial chunks are already on disk.
            completed:   Number of chunks already flushed.

        Returns:
            A :class:`~tanat.metric.DistanceMatrix` of shape ``(n, n)``.
        """
        ids = pool.unique_ids
        items = pool.get_trajectories()
        return default_pairwise_matrix(
            items,
            ids,
            self._compute,
            self._create_progress_bar,
            storage=storage,
            result=result,
            is_resuming=is_resuming,
            completed=completed,
        )

    # ------------------------------------------------------------------
    # Template Method chain (cross-matrix)
    # ------------------------------------------------------------------

    def _compute_cross_matrix_impl(
        self,
        pool_rows: TrajectoryPool,
        pool_cols: TrajectoryPool,
    ) -> np.ndarray:
        """Override point for optimised cross-pool distance computation.

        Subclasses override this method to replace the pure-Python loop with
        a faster kernel.  Pools are already validated when this method is
        called.  The default delegates to :meth:`_compute_cross_matrix_python`.

        Args:
            pool_rows: Pool whose trajectories form the rows   (n items).
            pool_cols: Pool whose trajectories form the columns (k items).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        return self._compute_cross_matrix_python(pool_rows, pool_cols)

    def _compute_cross_matrix_python(
        self,
        pool_rows: TrajectoryPool,
        pool_cols: TrajectoryPool,
    ) -> np.ndarray:
        """Naive Python cross-matrix fallback.

        Delegates to :func:`~tanat.metric._utils.default_cross_matrix`.

        Args:
            pool_rows: Pool whose trajectories form the rows.
            pool_cols: Pool whose trajectories form the columns.

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        ids_r = pool_rows.unique_ids
        ids_c = pool_cols.unique_ids
        trajs_r = pool_rows.get_trajectories()
        trajs_c = pool_cols.get_trajectories()
        return default_cross_matrix(trajs_r, ids_r, trajs_c, ids_c, self._compute)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_pool(self, pool: TrajectoryPool) -> None:
        """Type-check the pool argument."""
        if not isinstance(pool, TrajectoryPool):
            raise TypeError(f"pool must be a TrajectoryPool, got {type(pool).__name__}")

    def _validate_trajectories(self, traj_a: Trajectory, traj_b: Trajectory) -> None:
        """Type-check both trajectory arguments."""
        validate_pair(traj_a, traj_b, Trajectory, "traj_a", "traj_b")
