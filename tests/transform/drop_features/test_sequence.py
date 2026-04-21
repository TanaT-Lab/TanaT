#!/usr/bin/env python3
"""Tests: SequencePool.drop_features (soft drop, view isolation, guard-rails)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolDropFeatures:
    """drop_features hides columns from the pool view without touching disk by default."""

    def test_drop_entity_absent_from_settings(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Dropped entity feature no longer appears in entity_features settings."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        assert "flag_valid" not in pool.settings.entity_features

    def test_drop_entity_absent_from_data(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Dropped entity feature is absent from temporal_data() output columns."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        assert "flag_valid" not in pool.temporal_data(fmt="polars").columns

    def test_drop_static_absent_from_settings(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Dropped static feature no longer appears in static_features settings."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["age"], is_static=True)
        assert "age" not in pool.settings.static_features

    def test_drop_marks_dirty(self, pools_dict: dict, pool_type: str) -> None:
        """drop_features marks the pool as dirty."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        assert pool.is_dirty

    def test_soft_drop_does_not_affect_original(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Soft drop on a copy never modifies the session-scoped pool."""
        original = pools_dict[pool_type]
        copy = original.copy()
        copy.drop_features(["flag_valid"], is_static=False)
        assert "flag_valid" in original.settings.entity_features

    def test_drop_unknown_feature_raises(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Requesting a non-existent feature raises an error."""
        pool = pools_dict[pool_type].copy()
        with pytest.raises((ValueError, KeyError)):
            pool.drop_features(["nonexistent_xyz"])

    def test_entity_features_snapshot_after_drop(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Entity feature list after dropping flag_valid matches snapshot."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        assert snapshot == sorted(pool.settings.entity_features)

    def test_drop_entity_feature_absent_from_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After drop_features, dropped name is absent from pool.metadata.entity_features."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        names = {f.name for f in pool.metadata.entity_features}
        assert "flag_valid" not in names

    def test_drop_static_feature_absent_from_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After drop_features(is_static=True), dropped name absent from pool.metadata.static_features."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["age"], is_static=True)
        if pool.metadata.static_features is not None:
            names = {f.name for f in pool.metadata.static_features}
            assert "age" not in names


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolDropFeaturesPropagation:
    """Dropped features at pool level are absent from Entity children.

    A soft drop removes the column from settings, which flows down as
    Sequence.parent_metadata → Entity._parent_metadata, so the column
    is invisible in entity.metadata, entity.feature_names and entity.data()
    immediately.
    """

    def test_entity_drop_absent_on_entity(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Dropped entity feature is absent from entity.metadata on a child Entity."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        entity = pool[pool.unique_ids[0]][0]
        assert "flag_valid" not in entity.metadata

    def test_entity_drop_absent_from_feature_names(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Dropped entity feature is absent from entity.feature_names on a child Entity."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        entity = pool[pool.unique_ids[0]][0]
        assert entity.feature_names is not None
        assert "flag_valid" not in entity.feature_names

    def test_entity_drop_absent_from_data(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Dropped entity feature is absent from entity.data() keys on a child Entity."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        entity = pool[pool.unique_ids[0]][0]
        assert "flag_valid" not in entity.data()

    def test_static_drop_absent_on_sequence(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Dropped static feature is absent from static_data() on a child Sequence."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["age"], is_static=True)
        seq = pool[pool.unique_ids[0]]
        sd = seq.static_data(fmt="polars")
        if sd is not None:
            assert "age" not in sd.columns

    def test_drop_entity_feature_absent_from_sequence_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After drop_features, dropped name is absent from seq.metadata.entity_features."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        seq = pool[pool.unique_ids[0]]
        names = {f.name for f in seq.metadata.entity_features}
        assert "flag_valid" not in names

    def test_drop_static_feature_absent_from_sequence_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After drop_features(is_static=True), dropped name absent from seq.metadata.static_features."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["age"], is_static=True)
        seq = pool[pool.unique_ids[0]]
        if seq.metadata.static_features is not None:
            names = {f.name for f in seq.metadata.static_features}
            assert "age" not in names
