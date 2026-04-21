#!/usr/bin/env python3
"""
EditSequenceMetric: Needleman-Wunsch edit distance between sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from tanat_utils import settings_dataclass as dataclass

from ...base import SequenceMetric
from ....entity.base import EntityMetric
from ....matrix import DistanceMatrix
from .kernels import compute_edit_matrix

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence
    from .....sequence.base.pool import SequencePool
    from ...._storage import StorageOptions


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class EditSettings:
    """Settings for :class:`EditSequenceMetric`.

    Args:
        entity_metric: Entity-level substitution cost metric.  Default: ``"hamming"``.
        indel_cost: Cost per insertion or deletion.  Must be > 0.  Default: 1.0.
        normalize: When ``True``, divide the raw edit distance by
            ``max(len_a, len_b)`` to obtain a value in ``[0, 1]``.  Default: ``False``.
    """

    entity_metric: EntityMetric = "hamming"
    indel_cost: float = Field(default=1.0, gt=0)
    normalize: bool = False


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class EditSequenceMetric(SequenceMetric, register_name="edit"):
    """Needleman-Wunsch edit distance between two sequences.

    Computes the minimum-cost alignment between two sequences using a full
    O(n × m) DP matrix (Needleman-Wunsch).  Substitution cost comes from the
    entity metric; insertions and deletions cost ``indel_cost`` each.

    When ``normalize=True``, the raw distance is divided by
    ``max(len_a, len_b)`` so the result lies in ``[0, 1]``.

    Empty-sequence behaviour:

    * **Both empty** → ``0.0`` (no edits needed).
    * **One empty** → ``n × indel_cost`` (all insertions/deletions).

    Example::

        edit = EditSequenceMetric(indel_cost=0.5, normalize=True)
        d    = edit(seq_a, seq_b)
        dm   = edit.compute_matrix(pool)
    """

    SETTINGS_CLASS = EditSettings
    MEMMAP_SUPPORT = True

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        indel_cost: float = 1.0,
        normalize: bool = False,
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
            settings=EditSettings(
                entity_metric=entity_metric,
                indel_cost=indel_cost,
                normalize=normalize,
            ),
            storage=storage_options,
        )

    # ------------------------------------------------------------------
    # Composition
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
    # Core
    # ------------------------------------------------------------------

    def _compute(self, seq_a: Sequence, seq_b: Sequence) -> float:
        """Compute Needleman-Wunsch edit distance.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            Raw or normalised edit distance.
        """
        em = self.entity_metric
        indel = self.settings.indel_cost
        n, m = len(seq_a), len(seq_b)

        # Degenerate cases
        if n == 0:
            return indel * m
        if m == 0:
            return indel * n

        # Needleman-Wunsch full DP matrix
        matrix = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            matrix[i][0] = i * indel
        for j in range(m + 1):
            matrix[0][j] = j * indel

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sub = em(seq_a[i - 1], seq_b[j - 1])
                matrix[i][j] = min(
                    matrix[i - 1][j - 1] + sub,
                    matrix[i - 1][j] + indel,
                    matrix[i][j - 1] + indel,
                )

        d = matrix[n][m]
        if self.settings.normalize:
            max_len = max(n, m)
            d = d / max_len if max_len > 0 else 0.0
        return d

    # ------------------------------------------------------------------
    # Matrix computation: Numba optimisation
    # ------------------------------------------------------------------

    def _compute_matrix_impl(
        self,
        pool: SequencePool,
        *,
        storage: StorageOptions | None = None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Dispatch to Numba or Python path based on entity metric capability."""
        em = self.entity_metric
        if em.NUMBA_OPTIM:
            return self._compute_matrix_numba(
                pool, storage, result, is_resuming, completed
            )
        return self._compute_matrix_python(
            pool, storage, result, is_resuming, completed
        )

    def _compute_cross_matrix_impl(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Dispatch to Numba or Python path for cross (n × k) matrices."""
        em = self.entity_metric
        if em.NUMBA_OPTIM:
            return self._compute_cross_matrix_numba(pool_rows, pool_cols)
        return self._compute_cross_matrix_python(pool_rows, pool_cols)

    def _compute_matrix_numba(
        self,
        pool: SequencePool,
        storage: StorageOptions | None = None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Numba fast path for the pairwise distance matrix."""
        em = self.entity_metric
        indel_cost = np.float32(self.settings.indel_cost)
        normalize = self.settings.normalize
        return self._run_numba_matrix(
            pool,
            compute_edit_matrix,
            (indel_cost, normalize),
            storage=storage,
            result=result,
            is_resuming=is_resuming,
            completed=completed,
            symmetric=em.IS_SYMMETRIC,
        )

    def _compute_cross_matrix_numba(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Numba fast path for cross (n × k) distance matrix."""
        indel_cost = np.float32(self.settings.indel_cost)
        normalize = self.settings.normalize
        return self._run_numba_cross_matrix(
            pool_rows, pool_cols, compute_edit_matrix, (indel_cost, normalize)
        )
