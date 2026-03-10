#!/usr/bin/env python3
"""Tests: SequencePool.add_entity_features and add_static_features."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolAddFeatures:
    """add_entity_features and add_static_features write into the virtual context."""

    # ------------------------------------------------------------------
    # add_entity_features
    # ------------------------------------------------------------------

    def test_add_entity_feature_visible(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """New column appears in entity_features settings after add_entity_features."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"noise": [0.5] * n_rows}))
        assert snapshot == sorted(pool.settings.entity_features)

    def test_add_entity_marks_dirty(self, pools_dict: dict, pool_type: str) -> None:
        """add_entity_features sets is_dirty to True."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"extra_e": [1.0] * n_rows}))
        assert pool.is_dirty

    def test_add_entity_via_apply_by_id(self, pools_dict: dict, pool_type: str) -> None:
        """Per-sequence normalization (apply by_id=True) can be passed to add_entity_features."""
        pool = pools_dict[pool_type].copy()
        normed = pool.apply(
            (pl.col("value") - pl.col("value").mean()).alias("value_normed"),
            by_id=True,
            output_format="polars",
        )
        pool.add_entity_features(normed)
        assert "value_normed" in pool.settings.entity_features

    def test_add_entity_collision_raises(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """ValueError raised on column collision without overwrite=True."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        feat = pl.DataFrame({"dup_feat": [0.5] * n_rows})
        pool.add_entity_features(feat)
        with pytest.raises(ValueError, match="collision"):
            pool.add_entity_features(feat)

    def test_add_entity_overwrite_replaces(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """add_entity_features(overwrite=True) replaces an existing column without error."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        feat = pl.DataFrame({"overwritable": [0.5] * n_rows})
        pool.add_entity_features(feat)
        pool.add_entity_features(feat, overwrite=True)  # must not raise
        assert "overwritable" in pool.settings.entity_features

    def test_add_entity_blocked_on_view(self, pools_dict: dict, pool_type: str) -> None:
        """add_entity_features raises RuntimeError on a filtered view."""
        pool = pools_dict[pool_type]
        view = pool.subset(pool.unique_ids[:2])
        n_view = view.sequence_data(output_format="polars").height
        with pytest.raises(RuntimeError):
            view.add_entity_features(pl.DataFrame({"blocked": [0.0] * n_view}))

    # ------------------------------------------------------------------
    # add_static_features
    # ------------------------------------------------------------------

    def test_add_static_feature_visible(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """New column appears in static_features settings after add_static_features."""
        pool = pools_dict[pool_type].copy()
        summary = pool.apply(
            pl.col("value").mean().alias("value_mean"),
            by_id=True,
            output_format="polars",
        )
        pool.add_static_features(summary)
        assert snapshot == sorted(pool.settings.static_features)

    def test_add_static_partial_gives_nulls(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Partial df: IDs absent from the input receive null in static_data()."""
        pool = pools_dict[pool_type].copy()
        partial = pool.apply(
            pl.col("value").std().alias("value_std"),
            by_id=True,
            output_format="polars",
        ).head(3)
        pool.add_static_features(partial)
        sd = pool.static_data(output_format="polars")
        assert sd["value_std"].null_count() > 0

    def test_add_static_custom_id_column(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """add_static_features accepts a custom id_column name for the join key."""
        pool = pools_dict[pool_type].copy()
        custom_df = pl.DataFrame(
            {
                "pid": pool.unique_ids,
                "priority": list(range(len(pool))),
            }
        )
        pool.add_static_features(custom_df, id_column="pid")
        assert "priority" in pool.settings.static_features

    def test_add_static_collision_raises(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """ValueError raised on static column collision without overwrite=True."""
        pool = pools_dict[pool_type].copy()
        df = pl.DataFrame({"id": pool.unique_ids, "dup_static": [0.0] * len(pool)})
        pool.add_static_features(df)
        with pytest.raises(ValueError, match="collision"):
            pool.add_static_features(df)
