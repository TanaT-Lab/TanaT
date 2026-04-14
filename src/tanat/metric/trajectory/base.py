#!/usr/bin/env python3
"""
TrajectoryMetric ABC: base class for all trajectory-level distance metrics.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tanat_utils import SettingsMixin, Registrable, DisplayMixin

from ..matrix import DistanceMatrix
from .._storage import StorageOptions
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
        self._storage = self._resolve_storage(storage)

    # ------------------------------------------------------------------
    # Storage resolution
    # ------------------------------------------------------------------

    def _resolve_storage(
        self,
        storage: StorageOptions | dict | None = None,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
    ) -> StorageOptions | None:
        """Normalise *storage* and enforce :attr:`MEMMAP_SUPPORT`.

        An explicit ``store_path`` kwarg takes priority over *storage*.
        A plain ``dict`` is converted to :class:`~tanat.metric.StorageOptions`.
        Returns ``None`` when no storage is requested or when the subclass
        does not support memmap.

        Args:
            storage:    Existing :class:`~tanat.metric.StorageOptions`, plain
                        ``dict``, or ``None``.
            store_path: When provided, overrides *storage* entirely.
            chunk_size: Forwarded to :class:`~tanat.metric.StorageOptions`.
            resume:     Forwarded to :class:`~tanat.metric.StorageOptions`.
            dtype:      Forwarded to :class:`~tanat.metric.StorageOptions`.

        Returns:
            A resolved :class:`~tanat.metric.StorageOptions` instance, or
            ``None`` (in-memory fallback).
        """
        # 1. Explicit store_path wins over everything
        if store_path is not None:
            storage = StorageOptions(
                store_path=store_path,
                chunk_size=chunk_size,
                resume=resume,
                dtype=dtype,
            )
        # 2. Dict shorthand → StorageOptions
        elif isinstance(storage, dict):
            storage = StorageOptions(**storage)

        # 3. No storage requested → in-memory
        if storage is None:
            return None

        # 4. Guard: subclass must declare MEMMAP_SUPPORT
        if not self.MEMMAP_SUPPORT:
            warnings.warn(
                f"{type(self).__name__} does not support disk-backed computation "
                f"(MEMMAP_SUPPORT=False). Falling back to in-memory computation.",
                UserWarning,
                stacklevel=3,
            )
            return None

        return storage

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @SettingsMixin.shadow_dispatch
    def __call__(  # pylint: disable=unused-argument
        self, traj_a: Trajectory, traj_b: Trajectory, **kwargs
    ) -> float:
        """Compute distance between two trajectories.

        Settings-matching ``kwargs`` create a temporary shadow view.

        Args:
            traj_a: First trajectory.
            traj_b: Second trajectory.
            **kwargs: Settings overrides.

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
    # Matrix computation
    # ------------------------------------------------------------------

    @SettingsMixin.shadow_dispatch
    def compute_matrix(  # pylint: disable=unused-argument
        self,
        pool: TrajectoryPool,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
        **kwargs,
    ) -> DistanceMatrix:
        """Compute full pairwise trajectory distance matrix.

        Storage kwargs are forwarded to
        :class:`~tanat.metric.StorageOptions`. Other ``kwargs`` create a
        temporary settings override via shadow dispatch.

        Args:
            pool:       A :class:`~tanat.trajectory.pool.TrajectoryPool`.
            store_path: Storage directory (``None`` → in-memory).
            chunk_size: Rows per flush chunk (default 500).
            resume:     Skip already-computed chunks (default ``True``).
            dtype:      Numpy dtype for the matrix (default ``"float32"``).
            **kwargs:   Settings overrides.

        Returns:
            A :class:`~tanat.metric.DistanceMatrix`.
        """
        # call-site kwargs > instance default > None
        storage = self._resolve_storage(
            self._storage,
            store_path=store_path,
            chunk_size=chunk_size,
            resume=resume,
            dtype=dtype,
        )
        self._validate_pool(pool)
        self._display_header()
        dm = self._compute_matrix_impl(pool, storage)
        self._display_footer(f"{len(pool)} trajectories")
        return dm

    def _compute_matrix_impl(
        self,
        pool: TrajectoryPool,
        _storage=None,  # pylint: disable=unused-argument
    ) -> DistanceMatrix:
        """Default O(n²) double-loop implementation.

        Correct for all metrics. Computes all n*(n-1) ordered pairs
        without assuming symmetry. Subclasses may override to use
        batch-optimised kernels or exploit symmetry.

        Args:
            pool:     Trajectory pool.
            _storage: Not used.  Present only to satisfy the interface expected
                      by subclasses that override this method with memmap support.
                      The base class always computes in-memory.

        Returns:
            :class:`DistanceMatrix`.
        """
        ids = pool.unique_ids
        n = len(ids)
        result = np.zeros((n, n), dtype=np.float32)
        trajs = {tid: pool[tid] for tid in ids}

        with self._create_progress_bar(total=n * (n - 1), desc="Pairs") as pbar:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    d = self._compute(trajs[ids[i]], trajs[ids[j]])
                    result[i, j] = float(d)
                    pbar.update(1)

        return DistanceMatrix(result, ids)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_pool(self, pool: TrajectoryPool) -> None:
        """Type-check: *pool* must be a :class:`~tanat.trajectory.pool.TrajectoryPool`.

        Raises:
            TypeError: If *pool* is not a ``TrajectoryPool``.
        """
        if not isinstance(pool, TrajectoryPool):
            raise TypeError(f"pool must be a TrajectoryPool, got {type(pool).__name__}")

    def _validate_trajectories(self, traj_a: Trajectory, traj_b: Trajectory) -> None:
        """Type-check both trajectory arguments.

        Raises:
            TypeError: If either argument is not a
                :class:`~tanat.trajectory.trajectory.Trajectory`.
        """
        if not isinstance(traj_a, Trajectory):
            raise TypeError(f"traj_a must be a Trajectory, got {type(traj_a).__name__}")
        if not isinstance(traj_b, Trajectory):
            raise TypeError(f"traj_b must be a Trajectory, got {type(traj_b).__name__}")
