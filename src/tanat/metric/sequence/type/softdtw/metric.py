#!/usr/bin/env python3
"""
SoftDTWSequenceMetric: Soft Dynamic Time Warping between sequences.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import Field
from tanat_utils import settings_dataclass as dataclass

from ...base import SequenceMetric
from ....entity.base import EntityMetric

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence


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
    MEMMAP_SUPPORT = False

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        gamma: float = 1.0,
    ) -> None:
        super().__init__(
            settings=SoftDTWSettings(
                entity_metric=entity_metric,
                gamma=gamma,
            )
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
