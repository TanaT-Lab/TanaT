#!/usr/bin/env python3
"""
Chi2SequenceMetric: Chi-squared distance between state-time distributions.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from tanat_utils import settings_dataclass as dataclass

from .....metadata.feature import CategoricalInfo
from ...base import SequenceMetric

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_histogram(seq: Sequence, feature: str) -> dict:
    """Build a category → total-weight histogram from a sequence.

    Weight per entity:

    * Interval / state sequences: ``temporal_extent`` returns ``[start, end]``;
      ``weight = end − start``.
    * Event sequences: ``temporal_extent`` returns a scalar; ``weight = 1.0``.

    Args:
        seq:     Sequence to extract the histogram from.
        feature: Feature name to use as category key.

    Returns:
        Dict mapping category value → cumulative weight.
    """
    hist: dict = {}
    for entity in seq:
        extent = entity.temporal_extent
        if isinstance(extent, (list, tuple)) and len(extent) == 2:
            start, end = extent
            if start is None or end is None:
                warnings.warn(
                    f"Chi2SequenceMetric: an entity in sequence '{seq.id_value}' has an "
                    f"undefined temporal bound (extent={extent!r}). "
                    "It will be ignored in the histogram.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            else:
                diff = end - start
                # timedelta (datetime sequences) → total seconds
                if hasattr(diff, "total_seconds"):
                    weight = diff.total_seconds()
                else:
                    weight = float(diff)
        else:
            weight = 1.0

        key = entity.data(features=[feature])[feature]
        hist[key] = hist.get(key, 0.0) + weight

    return hist


def _chi2_distance(hist_a: dict, hist_b: dict) -> float:
    """Chi-squared distance between two histograms.

    .. math::

        d(a, b) = \\sqrt{\\sum_j \\frac{(p_{aj} - p_{bj})^2}{p_{aj} + p_{bj}}}

    where ``p`` values are proportions (sum-normalised weights).  Categories
    present in one histogram but not the other contribute normally (the
    missing proportion is 0).

    Returns 0.0 when both histograms are empty.

    Args:
        hist_a: Category → weight histogram for sequence *a*.
        hist_b: Category → weight histogram for sequence *b*.

    Returns:
        Chi-squared distance (float ≥ 0).
    """
    total_a = sum(hist_a.values())
    total_b = sum(hist_b.values())

    if total_a == 0.0 and total_b == 0.0:
        return 0.0
    if total_a == 0.0 or total_b == 0.0:
        return 1.0

    categories = set(hist_a) | set(hist_b)
    result = 0.0
    for cat in categories:
        pa = hist_a.get(cat, 0.0) / total_a
        pb = hist_b.get(cat, 0.0) / total_b
        denom = pa + pb
        if denom > 0.0:
            result += (pa - pb) ** 2 / denom

    return math.sqrt(result)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Chi2Settings:
    """Settings for :class:`Chi2SequenceMetric`.

    Args:
        entity_feature: Categorical feature name used as the histogram key
            (same semantics as :class:`~tanat.metric.entity.HammingEntityMetric`).
            ``None`` \u2192 resolved from the first entity feature of the sequence
            at :meth:`validate_composition` time.
    """

    entity_feature: str | None = None


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class Chi2SequenceMetric(SequenceMetric, register_name="chi2"):
    """Chi-squared distance between the state-time distributions of two sequences.

    Rather than comparing sequences element-by-element, Chi² compares the
    *proportion of time* (or event count) spent in each categorical state.

    * **Event** sequences: each event contributes a weight of 1.
    * **Interval / State** sequences: each entity contributes
      ``end − start`` as its weight.

    The per-category proportions are computed independently for each sequence,
    then compared via the Chi-squared distance formula:

    .. math::

        d(a, b) = \\sqrt{\\sum_j \\frac{(p_{aj} - p_{bj})^2}{p_{aj} + p_{bj}}}

    .. note::
        Chi² does **not** use an entity metric; ``entity_metric`` is absent
        from its settings.  The ``validate_composition`` method checks only
        that the requested feature is present.

    Empty-sequence behaviour:

    * **Both empty** → ``0.0`` (identical empty distributions).
    * **One empty** → ``1.0`` (maximally different distributions).

    Example::

        chi2 = Chi2SequenceMetric(entity_feature="status")
        d    = chi2(seq_a, seq_b)
        dm   = chi2.compute_matrix(pool)
    """

    SETTINGS_CLASS = Chi2Settings
    MEMMAP_SUPPORT = False

    def __init__(
        self,
        entity_feature: str | None = None,
    ) -> None:
        super().__init__(settings=Chi2Settings(entity_feature=entity_feature))

    def _resolve_feature(self, seq: Sequence) -> str:
        """Return the target feature name, resolving from seq if needed."""
        if self.settings.entity_feature is not None:
            return self.settings.entity_feature
        return seq.settings.entity_features[0]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def validate_composition(
        self, seq_a: Sequence, seq_b: Sequence | None = None
    ) -> None:
        """Resolve and validate the target feature.

        If ``entity_feature`` was not specified, resolves to the first entity
        feature of ``seq_a`` and stores it in :attr:`target_feature`.
        Then checks that the feature is present and categorical in every
        provided sequence.

        Args:
            seq_a: Primary sequence.
            seq_b: Optional second sequence.

        Raises:
            KeyError:  If the feature is absent from a sequence.
            TypeError: If the feature is not categorical.
        """
        feature = self._resolve_feature(seq_a)

        for seq in (s for s in (seq_a, seq_b) if s is not None):
            info = seq.metadata.feature_info(feature)
            if info is None:
                raise KeyError(
                    f"Feature '{feature}' not found in sequence entity features. "
                    f"Available: {sorted(seq.settings.entity_features)}"
                )
            if not isinstance(info, CategoricalInfo):
                raise TypeError(
                    f"Chi2SequenceMetric requires a Categorical or Enum feature, "
                    f"got '{feature}' ({type(info).__name__}). "
                    f"Cast it first to pl.Categorical or pl.Enum."
                )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _compute(self, seq_a: Sequence, seq_b: Sequence) -> float:
        """Compute Chi-squared distance between two sequence distributions.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            Chi-squared distance >= 0.
        """
        feature = self._resolve_feature(seq_a)
        hist_a = _build_histogram(seq_a, feature)
        hist_b = _build_histogram(seq_b, feature)
        return _chi2_distance(hist_a, hist_b)
