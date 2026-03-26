#!/usr/bin/env python3
"""Tests: SequencePool.copy (independent virtual context, mutation isolation)."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolCopy:
    """copy() returns an independent pool sharing the same physical store."""

    def test_copy_preserves_len(self, pools_dict: dict, pool_type: str) -> None:
        """copy() produces a pool with the same number of sequences."""
        pool = pools_dict[pool_type]
        assert len(pool.copy()) == len(pool)

    def test_copy_preserves_entity_features(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """copy() preserves the entity feature list."""
        pool = pools_dict[pool_type]
        assert pool.copy().settings.entity_features == pool.settings.entity_features

    def test_copy_preserves_static_features(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """copy() preserves the static feature list."""
        pool = pools_dict[pool_type]
        assert pool.copy().settings.static_features == pool.settings.static_features

    def test_add_entity_on_copy_isolated(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """add_entity_features on copy does not expose the column on the original."""
        pool = pools_dict[pool_type]
        copy = pool.copy()
        n = copy.temporal_data(output_format="polars").height
        copy.add_entity_features(pl.DataFrame({"copy_only": [1.0] * n}))
        assert "copy_only" not in pool.settings.entity_features

    def test_add_static_on_copy_isolated(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """add_static_features on copy does not expose the column on the original."""
        pool = pools_dict[pool_type]
        copy = pool.copy()
        df = pl.DataFrame({"id": copy.unique_ids, "copy_stat": [0.0] * len(copy)})
        copy.add_static_features(df)
        assert "copy_stat" not in pool.settings.static_features

    def test_cast_on_copy_isolated(self, pools_dict: dict, pool_type: str) -> None:
        """cast_features on copy does not alter the original pool's schema."""
        pool = pools_dict[pool_type]
        copy = pool.copy()
        copy.cast_features({"status": pl.Categorical})
        original_schema = pool.temporal_data(output_format="polars").schema
        assert original_schema["status"] != pl.Categorical

    def test_drop_on_copy_isolated(self, pools_dict: dict, pool_type: str) -> None:
        """drop_features on copy does not modify the original feature list."""
        pool = pools_dict[pool_type]
        copy = pool.copy()
        copy.drop_features(["flag_valid"], is_static=False)
        assert "flag_valid" in pool.settings.entity_features


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolCopyAfterVirtualFeatures:
    """
    Regression: copy() must not raise when the pool already carries virtual features.
    """

    def test_copy_with_virtual_static_does_not_raise(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """copy() must not raise KeyError when the pool has virtual static features."""
        pool = pools_dict[pool_type].copy()
        df = pl.DataFrame({"id": pool.unique_ids, "virt_copy_stat": [1.0] * len(pool)})
        pool.add_static_features(df)
        copy = pool.copy()  # must not raise
        assert "virt_copy_stat" in copy.settings.static_features

    def test_copy_with_virtual_static_data_readable(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Virtual static feature is accessible via static_data() on the copy."""
        pool = pools_dict[pool_type].copy()
        df = pl.DataFrame({"id": pool.unique_ids, "virt_readable": [7.0] * len(pool)})
        pool.add_static_features(df)
        copy = pool.copy()
        sd = copy.static_data(output_format="polars")
        assert sd is not None
        assert "virt_readable" in sd.columns

    def test_copy_with_virtual_entity_does_not_raise(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """copy() must not raise when the pool has virtual entity features."""
        pool = pools_dict[pool_type].copy()
        n_rows = pool.temporal_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"virt_copy_entity": [0.5] * n_rows}))
        copy = pool.copy()  # must not raise
        assert "virt_copy_entity" in copy.settings.entity_features
