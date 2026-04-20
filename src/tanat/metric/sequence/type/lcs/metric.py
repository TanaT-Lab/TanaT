#!/usr/bin/env python3
"""
LCSSequenceMetric: Longest Common Subsequence distance between sequences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tanat_utils import settings_dataclass as dataclass

from ...base import SequenceMetric
from ....entity.base import EntityMetric

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence


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
    MEMMAP_SUPPORT = False

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        equality_threshold: float = 0.0,
        mode: Literal["length", "distance", "normalized"] = "distance",
    ) -> None:
        super().__init__(
            settings=LCSSettings(
                entity_metric=entity_metric,
                equality_threshold=equality_threshold,
                mode=mode,
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
