#!/usr/bin/env python3
"""
SoftDTWSequenceMetric: Soft Dynamic Time Warping between sequences.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from tanat_utils import settings_dataclass as dataclass

from ...base import SequenceMetric
from ....entity.base import EntityMetric
from ....matrix import DistanceMatrix
from .kernels import compute_softdtw_matrix

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence
    from .....sequence.base.pool import SequencePool
    from ...._storage import StorageOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    """Numerically stable soft-minimum of three values.

    soft-min(a, b, c; γ) = −γ · log( exp(−a/γ) + exp(−b/γ) + exp(−c/γ) )

    Uses the log-sum-exp trick for numerical stability.
    """
    if a == float("inf") and b == float("inf") and c == float("inf"):
        return float("inf")
    neg_gamma = -gamma
    # scaled values: -v/gamma
    sa = -a / gamma if a != float("inf") else -float("inf")
    sb = -b / gamma if b != float("inf") else -float("inf")
    sc = -c / gamma if c != float("inf") else -float("inf")
    max_s = max(sa, sb, sc)
    log_sum = max_s + math.log(
        math.exp(sa - max_s) + math.exp(sb - max_s) + math.exp(sc - max_s)
    )
    return neg_gamma * log_sum


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class SoftDTWSettings:
    """Settings for :class:`SoftDTWSequenceMetric`.

    Args:
        entity_metric: Entity-level distance metric.  Default: ``"hamming"``.
        gamma: Regularisation parameter for the soft-min operator.  Must be > 0.
            Large values produce a smoother (mean-like) approximation;
            small values approach standard DTW.  Default: 1.0.
    """

    entity_metric: EntityMetric = "hamming"
    gamma: float = Field(default=1.0, gt=0)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class SoftDTWSequenceMetric(SequenceMetric, register_name="softdtw"):
    """Soft Dynamic Time Warping distance between two sequences.

    Replaces the ``min`` operator in the DTW recurrence with a differentiable
    soft-minimum parameterised by ``gamma``:

    .. math::

        \\text{soft-min}(a, b, c; \\gamma) = -\\gamma \\log\\bigl(
            e^{-a/\\gamma} + e^{-b/\\gamma} + e^{-c/\\gamma}\\bigr)

    As ``gamma → 0``, SoftDTW converges to standard DTW.
    As ``gamma → ∞``, it approaches the mean of all alignment costs.

    Empty-sequence behaviour:

    * **Both empty** → ``nan`` (no alignment possible).
    * **One empty** → ``nan`` (no alignment possible).

    References
    ----------
    Cuturi & Blondel (2017) — *Soft-DTW: a Differentiable Loss Function for
    Time-Series*, ICML.

    Example::

        sdtw = SoftDTWSequenceMetric(gamma=0.1)
        d    = sdtw(seq_a, seq_b)
        dm   = sdtw.compute_matrix(pool)
    """

    SETTINGS_CLASS = SoftDTWSettings
    MEMMAP_SUPPORT = True

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        gamma: float = 1.0,
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
            settings=SoftDTWSettings(
                entity_metric=entity_metric,
                gamma=gamma,
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
        """Compute SoftDTW distance using full O(n × m) DP matrix.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            SoftDTW distance (``nan`` when either sequence is empty).
        """
        em = self.entity_metric
        gamma = self.settings.gamma
        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return float("nan")

        INF = float("inf")
        # R[i+1][j+1] = SoftDTW DP value at (i, j).
        # R[0][0] = 0 is the starting sentinel; all other boundary cells = inf.
        R = [[INF] * (m + 2) for _ in range(n + 2)]
        R[0][0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = em(seq_a[i - 1], seq_b[j - 1])
                R[i][j] = cost + _softmin3(
                    R[i - 1][j], R[i - 1][j - 1], R[i][j - 1], gamma
                )

        return R[n][m]

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
        gamma = np.float32(self.settings.gamma)
        return self._run_numba_matrix(
            pool,
            compute_softdtw_matrix,
            (gamma,),
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
        gamma = np.float32(self.settings.gamma)
        return self._run_numba_cross_matrix(
            pool_rows, pool_cols, compute_softdtw_matrix, (gamma,)
        )
