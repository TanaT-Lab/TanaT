#!/usr/bin/env python3
"""
EditSequenceMetric: Needleman-Wunsch edit distance between sequences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from tanat_utils import settings_dataclass as dataclass

from ...base import SequenceMetric
from ....entity.base import EntityMetric

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence


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
    MEMMAP_SUPPORT = False

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        indel_cost: float = 1.0,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            settings=EditSettings(
                entity_metric=entity_metric,
                indel_cost=indel_cost,
                normalize=normalize,
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
