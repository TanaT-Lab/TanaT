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

    def test_add_entity_feature_in_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After add_entity_features, new name appears in pool.metadata.entity_features."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"meta_e": [1.0] * n_rows}))
        names = {f.name for f in pool.metadata.entity_features}
        assert "meta_e" in names

    def test_add_static_feature_in_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After add_static_features, new name appears in pool.metadata.static_features."""
        pool = pools_dict[pool_type].copy()
        df = pl.DataFrame({"id": pool.unique_ids, "meta_s": [2.0] * len(pool)})
        pool.add_static_features(df)
        assert pool.metadata.static_features is not None
        names = {f.name for f in pool.metadata.static_features}
        assert "meta_s" in names


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolAddFeaturesPropagation:
    """Features added at pool level propagate all the way down to Entity children.

    pool.settings → Sequence.parent_metadata → Entity._parent_metadata
    so virtual entity features are immediately visible in entity.metadata,
    entity.feature_names and entity.data() without any extra step.
    """

    def test_entity_feature_visible_on_entity(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Entity feature added to pool appears in entity.metadata on a child Entity."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"propagated_e": [7.0] * n_rows}))
        entity = pool[pool.unique_ids[0]][0]
        assert "propagated_e" in entity.metadata

    def test_entity_feature_in_feature_names(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Added entity feature is visible in entity.feature_names on a child Entity."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"propagated_e": [7.0] * n_rows}))
        entity = pool[pool.unique_ids[0]][0]
        assert entity.feature_names is not None
        assert "propagated_e" in entity.feature_names

    def test_entity_feature_in_data(self, pools_dict: dict, pool_type: str) -> None:
        """Added entity feature key is present in entity.data() on a child Entity."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"propagated_e": [7.0] * n_rows}))
        entity = pool[pool.unique_ids[0]][0]
        assert "propagated_e" in entity.data()

    def test_static_feature_visible_on_sequence(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Static feature added to pool is present in static_data() on a child Sequence."""
        pool = pools_dict[pool_type].copy()
        df = pl.DataFrame({"id": pool.unique_ids, "propagated_s": [3.0] * len(pool)})
        pool.add_static_features(df)
        seq = pool[pool.unique_ids[0]]
        sd = seq.static_data(output_format="polars")
        assert sd is not None
        assert "propagated_s" in sd.columns

    def test_entity_feature_in_sequence_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After add_entity_features, new name is in seq.metadata.entity_features."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"meta_e": [1.0] * n_rows}))
        seq = pool[pool.unique_ids[0]]
        names = {f.name for f in seq.metadata.entity_features}
        assert "meta_e" in names

    def test_static_feature_in_sequence_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After add_static_features, new name is in seq.metadata.static_features."""
        pool = pools_dict[pool_type].copy()
        df = pl.DataFrame({"id": pool.unique_ids, "meta_s": [2.0] * len(pool)})
        pool.add_static_features(df)
        seq = pool[pool.unique_ids[0]]
        assert seq.metadata.static_features is not None
        names = {f.name for f in seq.metadata.static_features}
        assert "meta_s" in names
