#!/usr/bin/env python3
"""
LCSSequenceMetric: Longest Common Subsequence distance between sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from tanat_utils import settings_dataclass as dataclass

from ...base import SequenceMetric
from ....entity.base import EntityMetric
from ....matrix import DistanceMatrix
from .kernels import compute_lcs_matrix

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence
    from .....sequence.base.pool import SequencePool
    from ...._storage import StorageOptions

# Integer encoding for mode parameter passed to Numba kernels.
_MODE_MAP = {"length": 0, "distance": 1, "normalized": 2}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class LCSSettings:
    """Settings for :class:`LCSSequenceMetric`.

    Args:
        entity_metric: Entity-level metric.  Default: ``"hamming"``.
        equality_threshold: Two entities are considered equal when their
            entity distance is ≤ this threshold. Default: 0.0.
        mode: Output mode.

            * ``"length"``     → raw LCS length (not a proper distance).
            * ``"distance"``   → additive distance: ``len_a + len_b − 2·LCS``.
            * ``"normalized"`` → Jaccard-like: ``1 − 2·LCS / (len_a + len_b)``,
              in [0, 1].
    """

    entity_metric: EntityMetric = "hamming"
    equality_threshold: float = 0.0
    mode: Literal["length", "distance", "normalized"] = "distance"


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class LCSSequenceMetric(SequenceMetric, register_name="lcs"):
    """Longest Common Subsequence distance between two sequences.

    Computes the LCS length using a space-optimised DP (2-row rolling array).
    Two entities are considered equal when their entity distance ≤
    ``equality_threshold``.

    Three output ``mode``\\ s are available:

    * ``"length"``     → raw LCS length (not a proper distance).
    * ``"distance"``   → ``len_a + len_b − 2·LCS`` (always ≥ 0).
    * ``"normalized"`` → ``1 − 2·LCS / (len_a + len_b)`` ∈ [0, 1].

    Empty-sequence behaviour:

    * **Both empty** → ``0.0`` (for all modes).
    * **One empty** (length *n* vs 0) → *length*: ``0.0``,
      *distance*: ``n``, *normalized*: ``1.0``.

    Example::

        lcs = LCSSequenceMetric(mode="normalized", equality_threshold=0.1)
        d   = lcs(seq_a, seq_b)
        dm  = lcs.compute_matrix(pool)
    """

    SETTINGS_CLASS = LCSSettings
    MEMMAP_SUPPORT = True

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        equality_threshold: float = 0.0,
        mode: Literal["length", "distance", "normalized"] = "distance",
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
            settings=LCSSettings(
                entity_metric=entity_metric,
                equality_threshold=equality_threshold,
                mode=mode,
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
        """Compute LCS distance between two sequences.

        Uses a space-optimised O(n × m) DP with 2-row rolling arrays.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            Scalar value depending on ``mode``.
        """
        em = self.entity_metric
        threshold = self.settings.equality_threshold
        n, m = len(seq_a), len(seq_b)

        if n == 0 or m == 0:
            lcs_len = 0.0
        else:
            prev = [0.0] * (m + 1)
            curr = [0.0] * (m + 1)
            for i in range(n):
                for j in range(m):
                    if em(seq_a[i], seq_b[j]) <= threshold:
                        curr[j + 1] = prev[j] + 1.0
                    else:
                        curr[j + 1] = max(curr[j], prev[j + 1])
                prev, curr = curr, [0.0] * (m + 1)
            lcs_len = prev[m]

        mode = self.settings.mode
        if mode == "length":
            return lcs_len
        if mode == "distance":
            return float(n + m) - 2.0 * lcs_len
        # normalized
        total = n + m
        return 1.0 - 2.0 * lcs_len / total if total > 0 else 0.0

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
        threshold = np.float32(self.settings.equality_threshold)
        mode_int = _MODE_MAP[self.settings.mode]
        return self._run_numba_matrix(
            pool,
            compute_lcs_matrix,
            (threshold, mode_int),
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
        threshold = np.float32(self.settings.equality_threshold)
        mode_int = _MODE_MAP[self.settings.mode]
        return self._run_numba_cross_matrix(
            pool_rows, pool_cols, compute_lcs_matrix, (threshold, mode_int)
        )
