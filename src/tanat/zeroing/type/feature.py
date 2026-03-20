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


@dataclass
class FeatureT0Settings:
    """Settings for the feature-based T0 strategy."""

    feature: str


class FeatureT0Setter(T0Setter, register_name="feature"):
    """T0 = value of a static feature column, one value per sequence.

    The named feature must exist in the pool's static features and its dtype
    must exactly match the pool's temporal index dtype.
    """

    def __init__(self, *, feature: str):
        super().__init__(FeatureT0Settings(feature=feature))

    def compute(self, target: SequencePool | Sequence) -> pl.DataFrame:
        feature = self.settings.feature
        id_col = target.settings.id_column

        # 1. Validate the feature exists in static features.
        target.settings.validate_features(feature, is_static=True)

        # 2. Validate dtype compatibility: feature dtype must match temporal dtype.
        feat_info = target.metadata.feature_info(feature, is_static=True)
        temporal_dtype = target.metadata.temporal.dtype
        if feat_info is not None and feat_info.dtype != temporal_dtype:
            raise TypeError(
                f"Static feature '{feature}' has dtype {feat_info.dtype!r} but the pool's "
                f"temporal index has dtype {temporal_dtype!r}. "
                f"Use pool.cast_features({{'{feature}': <target_dtype>}}, is_static=True) "
                "to align the feature dtype before calling set_t0()."
            )

        # 3. Build [id_col, _T0_] via a left-join so that sequences without a static value receive null.
        ids: list = (
            target.unique_ids if hasattr(target, "unique_ids") else [target.id_value]
        )
        t0_lf = target._static_data_lf(feature).select(
            id_col, pl.col(feature).alias(_T0)
        )
        t0_df = pl.LazyFrame({id_col: ids}).join(t0_lf, on=id_col, how="left").collect()

        self._warn_nulls(t0_df.filter(pl.col(_T0).is_null())[id_col].to_list())
        self._df = t0_df
        return self._df
