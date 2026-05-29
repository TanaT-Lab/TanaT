#!/usr/bin/env python3
"""Feature-based T0 strategy: T0 = value from a static feature column."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from tanat_utils import settings_dataclass as dataclass

from ..base import T0Setter, _T0

if TYPE_CHECKING:
    from ...sequence.base.pool import SequencePool
    from ...sequence.base.sequence import Sequence
    from ...trajectory.pool import TrajectoryPool


@dataclass
class FeatureT0Settings:
    """Settings for the feature-based T0 strategy."""

    feature: str


class FeatureT0Setter(T0Setter, register_name="feature"):
    """T0 = value of a static feature column, one value per sequence.

    The named feature must exist in the pool's static features and its dtype
    must exactly match the pool's time index dtype.
    """

    def __init__(self, *, feature: str):
        super().__init__(FeatureT0Settings(feature=feature))

    @property
    def _strategy_label(self) -> str:
        """e.g. ``"feature='admission_date'"``."""
        return f"feature='{self.settings.feature}'"

    def _compute_t0(self, target: SequencePool | Sequence) -> pl.LazyFrame:
        id_col = target.settings.id_column
        feature = self.settings.feature

        # Validate the feature exists in static features.
        target.settings.validate_features(feature, is_static=True)

        # Validate dtype compatibility: feature dtype must match temporal dtype.
        feat_info = target.metadata.feature_info(feature, is_static=True)
        temporal_dtype = target.metadata.time_index.dtype
        if feat_info is not None and feat_info.dtype != temporal_dtype:
            raise TypeError(
                f"Static feature '{feature}' has dtype {feat_info.dtype!r} but the pool's "
                f"time index has dtype {temporal_dtype!r}. "
                f"Use pool.cast_features({{'{feature}': <target_dtype>}}, is_static=True) "
                "to align the feature dtype before calling set_t0()."
            )

        # pylint: disable=protected-access
        return target._frames.static(feature).select(id_col, pl.col(feature).alias(_T0))

    def compute_from_trajectory(
        self,
        target: TrajectoryPool,
        on: str | None = None,
    ) -> pl.DataFrame:
        """Read T0 from a trajectory-level static feature column.

        Uses ``target.static_data`` (public API) so no coupling to
        trajectory store internals is needed.

        Args:
            target: The trajectory pool.
            on:     Ignored for the feature strategy (T0 comes from the
                    trajectory-level static feature, not a sub-pool).

        Returns:
            Complete ``[id_col, _T0_]`` DataFrame stored in ``self._df``.

        Raises:
            KeyError:  If the feature does not exist in trajectory static features.
            TypeError: If the feature dtype does not match the trajectory's
                       time index dtype.
        """
        feature = self.settings.feature
        id_col = target.settings.id_column

        # Validate feature exists and dtype matches temporal dtype.
        meta = target.metadata
        temporal_dtype = meta.time_index.dtype
        feat_map = {f.name: f for f in (meta.static_features or [])}
        if feature not in feat_map:
            raise KeyError(
                f"Static feature '{feature}' not found in trajectory "
                f"static features. Available: {sorted(feat_map)}"
            )
        if feat_map[feature].dtype != temporal_dtype:
            raise TypeError(
                f"Static feature '{feature}' has dtype "
                f"{feat_map[feature].dtype!r} but the trajectory's time "
                f"index has dtype {temporal_dtype!r}. Use "
                f"cast_static_features({{'{feature}': <target_dtype>}}) "
                "to align the feature dtype before calling set_t0()."
            )

        static_df = target.static_data(features=[feature], fmt="polars")
        partial_lf = static_df.lazy().select(id_col, pl.col(feature).alias(_T0))
        return self._finalize(partial_lf, target._id_lf, id_col)
