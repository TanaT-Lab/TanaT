#!/usr/bin/env python3
"""
SequenceMetric ABC: base class for all sequence-level distance metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tanat_utils import SettingsMixin, Registrable, DisplayMixin

from ..matrix import DistanceMatrix
from ..entity.base import EntityMetric
from .._utils import (
    resolve_storage,
    default_pairwise_matrix,
    default_cross_matrix,
    validate_pair,
)
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from .._storage import (
    StorageOptions,
    open_or_create_matrix,
    compute_metric_config,
    save_progress,
)


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
        self._storage = resolve_storage(
            type(self).__name__, self.MEMMAP_SUPPORT, storage
        )

    def __call__(self, seq_a: Sequence, seq_b: Sequence) -> float:
        """Compute distance between two sequences.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

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

    def compute_matrix(
        self,
        pool: SequencePool,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
    ) -> DistanceMatrix:
        """Compute the full pairwise distance matrix for *pool*.


        Args:
            pool:       A :class:`~tanat.sequence.base.pool.SequencePool`.
            store_path: Storage directory (``None`` → in-memory).
            chunk_size: Rows per flush chunk (default 500).
            resume:     Skip already-computed chunks (default ``True``).
            dtype:      Numpy dtype for the matrix (default ``"float32"``).

        Returns:
            A :class:`~tanat.metric.DistanceMatrix` of shape ``(n, n)``.
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
        self._probe_composition(pool)

        result, is_resuming, completed = None, False, 0
        if storage is not None:
            ids = pool.unique_ids
            result, is_resuming, completed, is_complete = open_or_create_matrix(
                storage, len(ids), ids, compute_metric_config(self)
            )
            if is_complete:
                self._display_header()
                self._display_message("Cache hit: returning precomputed matrix")
                self._display_footer(f"{len(pool)} sequences")
                return DistanceMatrix(result, ids)

        self._display_header()
        dm = self._compute_matrix_impl(
            pool,
            storage=storage,
            result=result,
            is_resuming=is_resuming,
            completed=completed,
        )
        self._display_footer(f"{len(pool)} sequences")
        return dm

    def compute_cross_matrix(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Compute an asymmetric (n × k) distance matrix between two pools.

        Row ``i`` ↔ sequence ``i`` in *pool_rows*; column ``j`` ↔ sequence
        ``j`` in *pool_cols*.  The result is **not** symmetric.

        Validates both pools (type-check + composition probe), then delegates
        to :meth:`_compute_cross_matrix_impl`.  Subclasses override
        :meth:`_compute_cross_matrix_impl` to use Numba kernels when available.

        Args:
            pool_rows: Pool whose sequences form the rows   (n items).
            pool_cols: Pool whose sequences form the columns (k items).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        self._validate_pool(pool_rows)
        self._validate_pool(pool_cols)
        self._probe_composition(pool_rows)
        self._probe_composition(pool_cols)
        return self._compute_cross_matrix_impl(pool_rows, pool_cols)

    def _compute_cross_matrix_impl(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Override point for optimised cross-pool distance computation.

        Subclasses (e.g. Numba-enabled metrics) override this method to
        replace the pure-Python loop with a faster kernel.  Pools are
        already validated when this method is called.  The default
        delegates to :meth:`_compute_cross_matrix_python`.

        Args:
            pool_rows: Pool whose sequences form the rows   (n items).
            pool_cols: Pool whose sequences form the columns (k items).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        return self._compute_cross_matrix_python(pool_rows, pool_cols)

    def _compute_matrix_impl(
        self,
        pool: SequencePool,
        *,
        storage=None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Override point for optimised pairwise matrix computation.

        Subclasses (e.g. Numba-enabled metrics) override this method to
        replace the pure-Python loop with a faster kernel.  The default
        delegates to :meth:`_compute_matrix_python`, which handles both
        in-memory execution and optional disk-backed chunked computation.
        """
        return self._compute_matrix_python(
            pool,
            storage=storage,
            result=result,
            is_resuming=is_resuming,
            completed=completed,
        )

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
        """Type-check the pool argument."""
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

    def _probe_composition(self, pool: SequencePool) -> None:
        """Validate composition on the first non-empty sequence in *pool*."""
        for sid in pool.unique_ids:
            seq = pool[sid]
            if seq:
                self.validate_composition(seq)
                break

    def _validate_sequences(self, seq_a: Sequence, seq_b: Sequence) -> None:
        """Type-check both sequence arguments."""
        validate_pair(seq_a, seq_b, Sequence, "seq_a", "seq_b")

    # ------------------------------------------------------------------
    # Shared Numba / Python boilerplate (used by entity-metric subtypes)
    # ------------------------------------------------------------------

    def _run_numba_matrix(
        self,
        pool: SequencePool,
        kernel,
        kernel_args: tuple,
        *,
        storage=None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
        symmetric: bool,
    ) -> DistanceMatrix:
        """Execute a Numba kernel over chunks with progress bar and optional storage.

        Handles the entire chunk loop, resume logic, progress bar, and
        storage flushing.  Subclasses call this from ``_compute_matrix_numba``
        and only provide the metric-specific *kernel* and *kernel_args*.

        Args:
            pool:        Sequence pool.
            kernel:      Numba matrix kernel callable.
            kernel_args: Tuple of metric-specific positional args appended
                         after ``context`` (and before ``symmetric``).
            storage:     Optional storage options.
            result:      Pre-opened memmap or ``None`` for in-memory.
            is_resuming: Whether partial chunks are already on disk.
            completed:   Number of chunks already flushed.
            symmetric:   Whether the kernel should exploit symmetry.

        Returns:
            A :class:`~tanat.metric.DistanceMatrix` of shape ``(n, n)``.
        """
        em = self.entity_metric
        arrays, lengths, context = em.prepare_batch_data(pool)
        n = len(arrays)

        if result is None:
            result = np.full((n, n), np.nan, dtype=np.float32)

        chunk_size = storage.chunk_size if storage is not None else n
        chunks = list(range(0, n, chunk_size))

        with self._create_progress_bar(total=len(chunks), desc="Chunks") as pbar:
            for chunk_idx, chunk_start in enumerate(chunks):
                chunk_end = min(chunk_start + chunk_size, n)
                if is_resuming and chunk_idx < completed:
                    pbar.update(1)
                    continue
                kernel(
                    result,
                    chunk_start,
                    chunk_end,
                    arrays,
                    lengths,
                    arrays,
                    lengths,
                    em.distance_kernel,
                    context,
                    *kernel_args,
                    symmetric,
                )
                if storage is not None:
                    result.flush()
                    completed += 1
                    save_progress(storage, completed, status="computing")
                pbar.update(1)

        if storage is not None:
            result.flush()
            save_progress(storage, completed, status="complete")

        return DistanceMatrix(result, pool.unique_ids)

    def _run_numba_cross_matrix(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
        kernel,
        kernel_args: tuple,
    ) -> np.ndarray:
        """Execute a Numba kernel for cross (n × k) distance matrices.

        Args:
            pool_rows:   Pool whose sequences form the rows.
            pool_cols:   Pool whose sequences form the columns.
            kernel:      Numba matrix kernel callable.
            kernel_args: Tuple of metric-specific positional args appended
                         after ``context`` (and before ``symmetric=False``).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        em = self.entity_metric
        arrays_r, lengths_r, arrays_c, lengths_c, context = em.prepare_cross_batch_data(
            pool_rows, pool_cols
        )
        n, k = len(lengths_r), len(lengths_c)
        result = np.empty((n, k), dtype=np.float32)
        if n > 0 and k > 0:
            kernel(
                result,
                0,
                n,
                arrays_r,
                lengths_r,
                arrays_c,
                lengths_c,
                em.distance_kernel,
                context,
                *kernel_args,
                False,
            )
        return result

    def _compute_matrix_python(
        self,
        pool: SequencePool,
        storage=None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Naive Python pairwise fallback.

        Delegates to :func:`~tanat.metric._utils.default_pairwise_matrix` and
        computes the full ``(n, n)`` square.

        Args:
            pool:        Sequence pool.
            storage:     Optional storage options.
            result:      Pre-opened memmap or ``None`` for in-memory.
            is_resuming: Whether partial chunks are already on disk.
            completed:   Number of chunks already flushed.

        Returns:
            A :class:`~tanat.metric.DistanceMatrix` of shape ``(n, n)``.
        """
        ids = pool.unique_ids
        items = pool.get_sequences()
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

    def _compute_cross_matrix_python(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Naive Python cross-matrix fallback.

        Delegates to :func:`~tanat.metric._utils.default_cross_matrix`.

        Args:
            pool_rows: Pool whose sequences form the rows.
            pool_cols: Pool whose sequences form the columns.

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        ids_r = pool_rows.unique_ids
        ids_c = pool_cols.unique_ids
        seqs_r = pool_rows.get_sequences()
        seqs_c = pool_cols.get_sequences()
        return default_cross_matrix(seqs_r, ids_r, seqs_c, ids_c, self._compute)
