#!/usr/bin/env python3
"""
DTWSequenceMetric: Dynamic Time Warping between sequences.
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

    .. note::
        Full path normalisation (by actual path length) requires O(n×m)
        memory for backtracking and is deferred to Phase 3.

    Example::

        dtw = DTWSequenceMetric(window=3, normalize=True)
        d   = dtw(seq_a, seq_b)
        dm  = dtw.compute_matrix(pool)
    """

    SETTINGS_CLASS = DTWSettings
    MEMMAP_SUPPORT = False

    def __init__(
        self,
        entity_metric: EntityMetric | str = "hamming",
        window: int | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            settings=DTWSettings(
                entity_metric=entity_metric,
                window=window,
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
