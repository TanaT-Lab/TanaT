#!/usr/bin/env python3
"""
LCPSequenceMetric: Longest Common Prefix distance between sequences.
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
class LCPSettings:
    """Settings for :class:`LCPSequenceMetric`.

    Args:
        entity_metric: Entity-level metric.  Default: ``"hamming"``.
        equality_threshold: Two entities are considered equal when their
            entity distance is ≤ this threshold.  Must be ≥ 0.  Default: 0.0.
        mode: Output mode.

            * ``"length"``    →  raw LCP length (not a distance, can be > 1).
            * ``"distance"``  →  additive distance: ``len_a + len_b − 2·LCP``.
            * ``"normalized"``→  Jaccard-like distance: ``1 − 2·LCP / (len_a + len_b)``,
              in ``[0, 1]`` (default).
    """

    entity_metric: EntityMetric = "hamming"
    equality_threshold: float = 0.0
    mode: Literal["length", "distance", "normalized"] = "distance"


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class LCPSequenceMetric(SequenceMetric, register_name="lcp"):
    """Longest Common Prefix distance between two sequences.

    Scans the sequences from the start and counts consecutive positions where
    the two entities are *equal* (i.e. their entity distance ≤
    ``equality_threshold``).  The scan stops at the first mismatch.

    Three output ``mode``\\ s are available:

    * ``"length"``     → raw prefix length (not a proper distance).
    * ``"distance"``   → ``len_a + len_b − 2·LCP`` (always ≥ 0).
    * ``"normalized"`` → ``1 − 2·LCP / (len_a + len_b)`` ∈ [0, 1].

    Empty-sequence behaviour:

    * **Both empty** → ``0.0`` (for all modes).
    * **One empty** (length *n* vs 0) → *length*: ``0.0``,
      *distance*: ``n``, *normalized*: ``1.0``.

    Example::

        lcp = LCPSequenceMetric(mode="normalized")
        d   = lcp(seq_a, seq_b)
        dm  = lcp.compute_matrix(pool)
    """

    SETTINGS_CLASS = LCPSettings
    MEMMAP_SUPPORT = False

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        equality_threshold: float = 0.0,
        mode: Literal["length", "distance", "normalized"] = "distance",
    ) -> None:
        super().__init__(
            settings=LCPSettings(
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
        """Compute LCP distance between two sequences.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            Scalar distance (or length, depending on ``mode``).
        """
        em = self.entity_metric
        threshold = self.settings.equality_threshold
        n, m = len(seq_a), len(seq_b)

        lcp_len = 0.0
        for i in range(min(n, m)):
            if em(seq_a[i], seq_b[i]) > threshold:
                break
            lcp_len += 1.0

        mode = self.settings.mode
        if mode == "length":
            return lcp_len
        if mode == "distance":
            return float(n + m) - 2.0 * lcp_len
        # normalized
        total = n + m
        return 1.0 - 2.0 * lcp_len / total if total > 0 else 0.0
