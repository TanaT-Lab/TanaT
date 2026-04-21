#!/usr/bin/env python3
"""
DTWSequenceMetric: Dynamic Time Warping between sequences.
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
from .kernels import compute_dtw_matrix

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence
    from .....sequence.base.pool import SequencePool
    from ...._storage import StorageOptions


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class DTWSettings:
    """Settings for :class:`DTWSequenceMetric`.

    Args:
        entity_metric: Entity-level distance metric.  Default: ``"hamming"``.
        window: Sakoe-Chiba band width (number of cells off the diagonal).
            ``None`` means no constraint (full DTW).  Must be > 0 when set.
        normalize: When ``True``, divide the DTW cost by ``len_a + len_b``
            (approximation that avoids O(n×m) backtracking).  Default: ``False``.
    """

    entity_metric: EntityMetric = "hamming"
    window: int | None = Field(default=None, gt=0)
    normalize: bool = False


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class DTWSequenceMetric(SequenceMetric, register_name="dtw"):
    """Dynamic Time Warping distance between two sequences.

    Uses a space-optimised 2-row DP.  The Sakoe-Chiba band is applied when
    ``window`` is set, limiting the warping path to stay within ``window``
    cells of the diagonal.

    Empty-sequence behaviour:

    * **Both empty** → ``nan`` (no alignment possible).
    * **One empty** → ``nan`` (no alignment possible).

    When ``normalize=True``, divides the raw DTW cost by ``len_a + len_b``
    (an approximation that does not require path backtracking).

    Example::

        dtw = DTWSequenceMetric(window=3, normalize=True)
        d   = dtw(seq_a, seq_b)
        dm  = dtw.compute_matrix(pool)
    """

    SETTINGS_CLASS = DTWSettings
    MEMMAP_SUPPORT = True

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        window: int | None = None,
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
            settings=DTWSettings(
                entity_metric=entity_metric,
                window=window,
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
        """Compute DTW distance using a 2-row Sakoe-Chiba DP.

        The DP initialises with a sentinel ``prev[0] = 0`` so that
        ``dp[0][0] = cost(a[0], b[0])`` and boundary cells remain ``inf``.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            DTW distance (``nan`` when either sequence is empty).
        """
        em = self.entity_metric
        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return float("nan")

        window = self.settings.window
        INF = float("inf")  # DP sentinel for unreachable cells

        # prev[j+1] = dp value at column j of the previous row.
        # prev[0] = 0 is the sentinel (conceptually dp[-1][-1] = 0).
        prev = [INF] * (m + 1)
        prev[0] = 0.0

        for i in range(n):
            curr = [INF] * (m + 1)
            j_lo = max(0, i - window) if window is not None else 0
            j_hi = min(m - 1, i + window) if window is not None else m - 1
            for j in range(j_lo, j_hi + 1):
                cost = em(seq_a[i], seq_b[j])
                # dp[i][j] = cost + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
                # shifted indices: prev[j] = dp[i-1][j-1], prev[j+1] = dp[i-1][j],
                # curr[j] = dp[i][j-1]
                curr[j + 1] = cost + min(prev[j], prev[j + 1], curr[j])
            prev = curr

        d = prev[m]
        if self.settings.normalize:
            d = d / (n + m)
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
        window_int = self.settings.window if self.settings.window is not None else -1
        normalize = self.settings.normalize
        return self._run_numba_matrix(
            pool,
            compute_dtw_matrix,
            (window_int, normalize),
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
        window_int = self.settings.window if self.settings.window is not None else -1
        normalize = self.settings.normalize
        return self._run_numba_cross_matrix(
            pool_rows, pool_cols, compute_dtw_matrix, (window_int, normalize)
        )
