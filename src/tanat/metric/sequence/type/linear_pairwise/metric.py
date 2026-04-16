#!/usr/bin/env python3
"""
LinearPairwiseSequenceMetric: align sequences position-by-position and aggregate entity distances.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from tanat_utils import settings_dataclass as dataclass

from ...base import SequenceMetric
from ....entity.base import EntityMetric
from ....matrix import DistanceMatrix
from ...._storage import save_progress
from .kernels import (
    compute_matrix_kernel,
    compute_matrix_chunk,
    _AGG_NUMBA_KERNELS,
)

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence
    from .....sequence.base.pool import SequencePool
    from ...._storage import StorageOptions


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_AGG_FUNCTIONS: dict[str, Callable] = {
    "mean": np.mean,
    "sum": np.sum,
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class LinearPairwiseSettings:
    """Settings for :class:`LinearPairwiseSequenceMetric`.

    Args:
        entity_metric: Entity-level metric.  Accepts a registration name (string) or an instance.
            Pydantic auto-resolves strings via ``Registrable.__get_pydantic_core_schema__``.
            Default: ``"hamming"``.
        agg_fun: Aggregation function applied to the vector of entity
            distances.  One of ``"mean"`` (default) or ``"sum"``.
        padding_penalty: Distance value used for unmatched positions when
            sequences have different lengths.  ``None`` → unmatched
            positions are ignored (only the overlap is aggregated).
            When the overlap is empty (one sequence has length 0),
            ``None`` makes the distance undefined (``nan`` in a matrix,
            :class:`ValueError` on a direct call).
    """

    entity_metric: EntityMetric = "hamming"
    agg_fun: str = "mean"
    padding_penalty: float | None = None


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class LinearPairwiseSequenceMetric(SequenceMetric, register_name="linearpairwise"):
    """Sequence metric by linear (position-wise) alignment of entities.

    Aligns ``seq_a`` and ``seq_b`` rank-by-rank and applies the
    configured entity metric to each aligned pair.  The resulting vector
    of entity distances is aggregated (mean, sum, ...) to produce a single
    scalar sequence distance.

    When sequences differ in length, ``padding_penalty`` is applied for
    each unmatched position of the longer sequence.  If
    ``padding_penalty`` is ``None``, only the overlapping prefix is used.

    Empty-sequence behaviour:

    * **Both empty** → ``nan`` (distance is undefined).
    * **One empty, padding_penalty is set** → all positions are padded.
    * **One empty, padding_penalty is None** → direct call raises
      :class:`ValueError`; matrix computation inserts ``nan``.

    Example::

        hamming = HammingEntityMetric(entity_feature="status")
        lp = LinearPairwiseSequenceMetric(entity_metric=hamming)

        dist = lp(seq_a, seq_b)
        dm   = lp.compute_matrix(pool)
    """

    SETTINGS_CLASS = LinearPairwiseSettings
    MEMMAP_SUPPORT = True

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        agg_fun: str = "mean",
        padding_penalty: float | None = None,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
    ) -> None:
        if store_path is not None:
            storage_options = {
                "store_path": store_path,
                "chunk_size": chunk_size,
                "resume": resume,
                "dtype": dtype,
            }
        else:
            storage_options = None

        super().__init__(
            settings=LinearPairwiseSettings(
                entity_metric=entity_metric,
                agg_fun=agg_fun,
                padding_penalty=padding_penalty,
            ),
            storage=storage_options,
        )

    # ------------------------------------------------------------------
    # Composition validation
    # ------------------------------------------------------------------

    def validate_composition(
        self, seq_a: Sequence, seq_b: Sequence | None = None
    ) -> None:
        """Probe the first entity of each sequence through the entity metric."""
        em = self.entity_metric
        if seq_a:
            em.validate_entity(seq_a[0])
        if seq_b:
            em.validate_entity(seq_b[0])

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, seq_a: Sequence, seq_b: Sequence) -> float:
        """Compute distance for a single pair of sequences (already validated).

        Delegates to :meth:`_compute_pair` after resolving the entity metric
        and aggregation function.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            Aggregated scalar distance.

        Raises:
            ValueError: If one sequence is empty and the other is not,
                with ``padding_penalty`` set to ``None``.
        """
        n_a, n_b = len(seq_a), len(seq_b)
        if (
            min(n_a, n_b) == 0
            and max(n_a, n_b) > 0
            and self.settings.padding_penalty is None
        ):
            raise ValueError(
                f"Cannot compute distance: one sequence is empty "
                f"(lengths {n_a} vs {n_b}) and padding_penalty is None. "
                f"Set padding_penalty to a numeric value to handle "
                f"length-mismatched sequences."
            )
        return self._compute_pair(seq_a, seq_b, self.entity_metric, self._get_agg_fn())

    def _compute_cross_matrix_impl(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Dispatch to the Numba fast path or the Python fallback.

        Pools are already validated by :meth:`compute_cross_matrix`.
        Uses the Numba path when the entity metric declares
        ``NUMBA_OPTIM = True``, otherwise falls back to
        :meth:`_compute_cross_matrix_python`.

        Args:
            pool_rows: Pool whose sequences form the rows   (n items).
            pool_cols: Pool whose sequences form the columns (k items).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        em = self.entity_metric
        if em.NUMBA_OPTIM:
            return self._compute_cross_matrix_numba(pool_rows, pool_cols)
        return self._compute_cross_matrix_python(pool_rows, pool_cols)

    def _compute_cross_matrix_python(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Python double-loop fallback for the cross (n × k) distance matrix.

        Uses :meth:`_compute_pair` rather than :meth:`_compute` so that
        undefined distances (empty sequence with no padding) produce ``nan``
        instead of raising :class:`ValueError`: consistent with
        :meth:`_compute_matrix_python`.

        Args:
            pool_rows: Pool whose sequences form the rows   (n items).
            pool_cols: Pool whose sequences form the columns (k items).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        em = self.entity_metric
        agg_fn = self._get_agg_fn()
        ids_r = pool_rows.unique_ids
        ids_c = pool_cols.unique_ids
        seqs_r = {sid: pool_rows[sid] for sid in ids_r}
        seqs_c = {sid: pool_cols[sid] for sid in ids_c}
        n, k = len(ids_r), len(ids_c)
        result = np.empty((n, k), dtype=np.float32)
        for i, id_r in enumerate(ids_r):
            for j, id_c in enumerate(ids_c):
                result[i, j] = self._compute_pair(
                    seqs_r[id_r], seqs_c[id_c], em, agg_fn
                )
        return result

    def _compute_cross_matrix_numba(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Numba fast path for the cross (n × k) distance matrix.

        Delegates shared-vocabulary encoding to the entity metric's
        :meth:`prepare_cross_batch_data`, then calls the parallel Numba
        kernel.  The first call triggers JIT compilation; subsequent calls
        reuse the compiled binary.

        Args:
            pool_rows: Pool whose sequences form the rows   (n items).
            pool_cols: Pool whose sequences form the columns (k items).

        Returns:
            float32 numpy array of shape ``(n, k)``.
        """
        em = self.entity_metric
        arrays_r, lengths_r, arrays_c, lengths_c, context = em.prepare_cross_batch_data(
            pool_rows, pool_cols
        )
        agg_kernel = self._get_agg_fn(numba=True)
        padding = (
            np.float32(self.settings.padding_penalty)
            if self.settings.padding_penalty is not None
            else np.float32(np.nan)
        )
        n, k = len(lengths_r), len(lengths_c)
        result = np.empty((n, k), dtype=np.float32)
        if n > 0 and k > 0:
            compute_matrix_kernel(
                result,
                arrays_r,
                lengths_r,
                arrays_c,
                lengths_c,
                em.distance_kernel,
                context,
                agg_kernel,
                padding,
                False,  # rectangular (n×k): triangle optimisation requires a square same-pool matrix
            )
        return result

    def _compute_matrix_impl(
        self,
        pool: SequencePool,
        *,
        storage: StorageOptions | None = None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Dispatch to the Numba fast path or the Python fallback.

        Uses the Numba batch path when the entity metric declares
        ``NUMBA_OPTIM = True`` (e.g. :class:`~.entity.HammingEntityMetric`).
        Falls back to the Python double-loop otherwise.

        Args:
            pool:        Sequence pool.
            storage:     Optional :class:`~tanat.metric.StorageOptions`.
            result:      Pre-opened memmap injected by the base, or ``None``
                         for the in-memory path.
            is_resuming: Whether *result* already has partial data.
            completed:   Number of chunks already flushed.

        Returns:
            Symmetric :class:`DistanceMatrix` (may contain ``nan``).
        """
        em = self.entity_metric
        if em.NUMBA_OPTIM:
            return self._compute_matrix_numba(
                pool, storage, result, is_resuming, completed
            )
        return self._compute_matrix_python(
            pool, storage, result, is_resuming, completed
        )

    def _compute_matrix_python(
        self,
        pool: SequencePool,
        storage: StorageOptions | None = None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Python double-loop fallback.

        Iterates all O(n^2) pairs.  Undefined distances produce ``nan``.
        Uses the memmap + chunks path when ``storage`` is set.

        Args:
            pool:        Sequence pool.
            storage:     Optional :class:`~tanat.metric.StorageOptions`.
            result:      Pre-opened memmap (disk path) or ``None`` (in-memory).
            is_resuming: Whether partial chunks are already on disk.
            completed:   Number of chunks already flushed.

        Returns:
            Symmetric :class:`DistanceMatrix` (may contain ``nan``).
        """
        em = self.entity_metric
        agg_fn = self._get_agg_fn()
        ids = pool.unique_ids
        n = len(ids)
        seqs = {sid: pool[sid] for sid in ids}

        if result is None:
            result = np.zeros((n, n), dtype=np.float32)

        chunk_size = storage.chunk_size if storage is not None else n
        chunks = list(range(0, n, chunk_size))

        with self._create_progress_bar(total=n * (n - 1) // 2, desc="Pairs") as pbar:
            for chunk_idx, chunk_start in enumerate(chunks):
                chunk_end = min(chunk_start + chunk_size, n)

                if is_resuming and chunk_idx < completed:
                    for i in range(chunk_start, chunk_end):
                        pbar.update(n - i - 1)
                    continue

                for i in range(chunk_start, chunk_end):
                    for j in range(i + 1, n):
                        d = self._compute_pair(seqs[ids[i]], seqs[ids[j]], em, agg_fn)
                        result[i, j] = result[j, i] = float(d)
                        pbar.update(1)

                if storage is not None:
                    result.flush()
                    completed += 1
                    save_progress(storage, completed, status="computing")

        if storage is not None:
            np.fill_diagonal(result, 0.0)
            result.flush()
            save_progress(storage, completed, status="complete")

        return DistanceMatrix(result, ids)

    def _compute_matrix_numba(
        self,
        pool: SequencePool,
        storage: StorageOptions | None = None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Numba fast path: batch-extract data then run the parallel kernel.

        Delegates feature extraction to the entity metric's
        :meth:`prepare_batch_data`, then calls the Numba kernel.
        The first call triggers JIT compilation; subsequent calls
        reuse the compiled binary.

        Undefined distances produce ``nan``.

        Args:
            pool:        Sequence pool.
            storage:     Optional :class:`~tanat.metric.StorageOptions`.
            result:      Pre-opened memmap (disk path) or ``None`` (in-memory).
            is_resuming: Whether partial chunks are already on disk.
            completed:   Number of chunks already flushed.

        Returns:
            Symmetric :class:`DistanceMatrix` (may contain ``nan``).
        """
        em = self.entity_metric
        arrays, lengths, context = em.prepare_batch_data(pool)

        agg_kernel = self._get_agg_fn(numba=True)

        padding = (
            np.float32(self.settings.padding_penalty)
            if self.settings.padding_penalty is not None
            else np.float32(np.nan)
        )

        n = len(arrays)

        if result is None:
            # --- In-memory path ---
            result = np.zeros((n, n), dtype=np.float32)
            if n > 1:
                compute_matrix_kernel(
                    result,
                    arrays,
                    lengths,
                    arrays,
                    lengths,
                    em.distance_kernel,
                    context,
                    agg_kernel,
                    padding,
                    em.IS_SYMMETRIC,
                )
            return DistanceMatrix(result, pool.unique_ids)

        # --- Memmap + chunks path ---
        chunk_size = storage.chunk_size
        chunks = list(range(0, n, chunk_size))
        total_chunks = len(chunks)

        with self._create_progress_bar(total=total_chunks, desc="Chunks") as pbar:
            for chunk_idx, chunk_start in enumerate(chunks):
                chunk_end = min(chunk_start + chunk_size, n)

                if is_resuming and chunk_idx < completed:
                    pbar.update(1)
                    continue

                if em.IS_SYMMETRIC:
                    compute_matrix_chunk(
                        result,
                        chunk_start,
                        chunk_end,
                        arrays,
                        lengths,
                        arrays,
                        lengths,
                        em.distance_kernel,
                        context,
                        agg_kernel,
                        padding,
                        True,
                    )
                else:
                    compute_matrix_chunk(
                        result,
                        chunk_start,
                        chunk_end,
                        arrays,
                        lengths,
                        arrays,
                        lengths,
                        em.distance_kernel,
                        context,
                        agg_kernel,
                        padding,
                        False,
                    )

                result.flush()
                completed += 1
                save_progress(storage, completed, status="computing")
                pbar.update(1)

        np.fill_diagonal(result, 0.0)
        result.flush()
        save_progress(storage, completed, status="complete")
        return DistanceMatrix(result, pool.unique_ids)

    def _compute_pair(
        self,
        seq_a: Sequence,
        seq_b: Sequence,
        em: EntityMetric,
        agg_fn: Callable,
    ) -> float:
        """Compute distance for a single pair of sequences.

        Aligns entities by rank and aggregates distances.  Returns ``nan``
        when the distance is undefined (both empty, or one empty with no
        padding).

        Args:
            seq_a:  First sequence.
            seq_b:  Second sequence.
            em:     Resolved entity metric instance.
            agg_fn: Resolved aggregation callable.

        Returns:
            Aggregated scalar distance, or ``nan`` for undefined pairs.
        """
        n_a = len(seq_a)
        n_b = len(seq_b)

        if n_a == 0 and n_b == 0:
            return float("nan")

        if min(n_a, n_b) == 0 and self.settings.padding_penalty is None:
            return float("nan")

        min_len = min(n_a, n_b)
        max_len = max(n_a, n_b)
        distances: list[float] = []

        for i in range(min_len):
            distances.append(em(seq_a[i], seq_b[i]))

        if max_len > min_len and self.settings.padding_penalty is not None:
            for _ in range(max_len - min_len):
                distances.append(float(self.settings.padding_penalty))

        return float(agg_fn(distances))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_agg_fn(self, *, numba: bool = False) -> Callable:
        """Return the aggregation callable for the configured ``agg_fun``.

        Args:
            numba: When ``True``, return the Numba JIT kernel from
                   :data:`_AGG_NUMBA_KERNELS`; otherwise return the
                   plain Python callable from :data:`_AGG_FUNCTIONS`.

        Raises:
            ValueError: If ``agg_fun`` is not supported by the requested registry.
        """
        registry = _AGG_NUMBA_KERNELS if numba else _AGG_FUNCTIONS
        fn = registry.get(self.settings.agg_fun)
        if fn is None:
            raise ValueError(
                f"Unknown agg_fun '{self.settings.agg_fun}'. "
                f"Supported: {list(registry.keys())}"
            )
        return fn
