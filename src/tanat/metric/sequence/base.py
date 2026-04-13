#!/usr/bin/env python3
"""
SequenceMetric ABC: base class for all sequence-level distance metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import warnings

import numpy as np
from tanat_utils import SettingsMixin, Registrable, DisplayMixin

from ..matrix import DistanceMatrix
from ..entity.base import EntityMetric
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from .._storage import StorageOptions


class SequenceMetric(SettingsMixin, Registrable, DisplayMixin, ABC):
    """Abstract base for sequence-level distance metrics.

    Computes a scalar distance between two :class:`~tanat.sequence.base.sequence.Sequence`
    objects and a full pairwise :class:`~tanat.metric.DistanceMatrix` over a pool.
    """

    _REGISTER: dict = {}
    _TYPE_SUBMODULE = "type"

    #: Set to ``True`` in subclasses that implement disk-backed (memmap) computation.
    #: When ``False``, passing ``store_path`` or an instance-level ``StorageOptions``
    #: raises :class:`NotImplementedError` early with a clear message.
    MEMMAP_SUPPORT: bool = False

    def __init__(
        self, settings=None, storage: StorageOptions | dict | None = None
    ) -> None:
        super().__init__(settings)
        self._storage = self._resolve_storage(storage)

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

    @SettingsMixin.shadow_dispatch
    def __call__(  # pylint: disable=unused-argument
        self, seq_a: Sequence, seq_b: Sequence, **kwargs
    ) -> float:
        """Compute distance between two sequences.

        Settings-matching ``kwargs`` create a temporary shadow view.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.
            **kwargs: Settings overrides.

        Returns:
            Scalar distance.
        """
        self._validate_sequences(seq_a, seq_b)
        self.validate_composition(seq_a, seq_b)
        return self._compute(seq_a, seq_b)

    @abstractmethod
    def _compute(self, seq_a: Sequence, seq_b: Sequence) -> float:
        """Core distance computation.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            Scalar distance.
        """

    # ------------------------------------------------------------------
    # Matrix computation
    # ------------------------------------------------------------------

    @SettingsMixin.shadow_dispatch
    def compute_matrix(  # pylint: disable=unused-argument
        self,
        pool: SequencePool,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
        **kwargs,
    ) -> DistanceMatrix:
        """Compute the full pairwise distance matrix for *pool*.

        Storage kwargs are forwarded to
        :class:`~tanat.metric.StorageOptions`.  Other ``kwargs``
        (e.g. ``agg_fun``) create a temporary settings override.

        Args:
            pool:       A :class:`~tanat.sequence.base.pool.SequencePool`.
            store_path: Storage directory (``None`` → in-memory).
            chunk_size: Rows per flush chunk (default 500).
            resume:     Skip already-computed chunks (default ``True``).
            dtype:      Numpy dtype for the matrix (default ``"float32"``).
            **kwargs:   Settings overrides (e.g. ``agg_fun``, ``padding_penalty``).

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
        for sid in pool.unique_ids:
            seq = pool[sid]
            if seq:
                self.validate_composition(seq)
                break
        self._display_header()
        dm = self._compute_matrix_impl(pool, storage)
        self._display_footer(f"{len(pool)} sequences")
        return dm

    def _compute_matrix_impl(
        self,
        pool: SequencePool,
        _storage=None,  # pylint: disable=unused-argument
    ) -> DistanceMatrix:
        """Default O(n²) double-loop implementation.

        Correct for all metrics. Computes all n*(n-1) ordered pairs
        without assuming symmetry. Subclasses may override to use
        batch-optimised kernels (e.g. Numba) or exploit symmetry.

        Args:
            pool:     Sequence pool.
            _storage: Not used.  Present only to satisfy the interface expected
                      by subclasses that override this method with memmap support.
                      The base class always computes in-memory.

        Returns:
            :class:`DistanceMatrix`.
        """
        ids = pool.unique_ids
        n = len(ids)
        result = np.zeros((n, n), dtype=np.float32)
        seqs = {sid: pool[sid] for sid in ids}

        with self._create_progress_bar(total=n * (n - 1), desc="Pairs") as pbar:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    d = self._compute(seqs[ids[i]], seqs[ids[j]])
                    result[i, j] = float(d)
                    pbar.update(1)

        return DistanceMatrix(result, ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def entity_metric(self) -> EntityMetric:
        """Resolve ``settings.entity_metric`` (string → instance or pass-through).

        Returns:
            The resolved :class:`~tanat.metric.entity.base.EntityMetric`.

        Raises:
            AttributeError: If the concrete settings class has no
                ``entity_metric`` field.
        """
        metric = self.settings.entity_metric
        if isinstance(metric, str):
            return EntityMetric.get_registered(metric)()
        return metric

    def _validate_pool(self, pool: SequencePool) -> None:
        """Type-check the pool argument.

        Raises:
            TypeError: If *pool* is not a
                :class:`~tanat.sequence.base.pool.SequencePool`.
        """
        if not isinstance(pool, SequencePool):
            raise TypeError(f"pool must be a SequencePool, got {type(pool).__name__}")

    @abstractmethod
    def validate_composition(
        self, seq_a: Sequence, seq_b: Sequence | None = None
    ) -> None:
        """Composition compatibility check between this metric and the given sequence(s).

        Called from :meth:`__call__` with both sequences, and from
        :meth:`compute_matrix` with a single sample sequence (``seq_b=None``).
        Subclasses that compose with an :class:`~tanat.metric.entity.base.EntityMetric`
        probe a sample entity to surface schema errors early.

        Args:
            seq_a: Primary sequence to probe.
            seq_b: Optional second sequence.

        Raises:
            TypeError: If the entity feature has an incompatible dtype.
            KeyError:  If a required feature is absent.
        """

    def _validate_sequences(self, seq_a: Sequence, seq_b: Sequence) -> None:
        """Type-check both sequence arguments.

        Raises:
            TypeError: If either argument is not a
                :class:`~tanat.sequence.base.sequence.Sequence`.
        """
        if not isinstance(seq_a, Sequence):
            raise TypeError(f"seq_a must be a Sequence, got {type(seq_a).__name__}")
        if not isinstance(seq_b, Sequence):
            raise TypeError(f"seq_b must be a Sequence, got {type(seq_b).__name__}")
