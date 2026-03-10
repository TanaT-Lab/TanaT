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
        """Dropped entity feature is absent from sequence_data() output columns."""
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        assert "flag_valid" not in pool.sequence_data(output_format="polars").columns

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
