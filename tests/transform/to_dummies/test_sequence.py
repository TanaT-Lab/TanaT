#!/usr/bin/env python3
"""Tests: SequencePool.to_dummies (one-hot encoding of Categorical features)."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolToDummies:
    """to_dummies() one-hot encodes Categorical features into binary columns."""

    def test_column_names_snapshot(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Column names of the dummies output match snapshot."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        dummies = pool.to_dummies(["status"], output_format="polars")
        assert snapshot == dummies.columns

    def test_original_column_absent(self, pools_dict: dict, pool_type: str) -> None:
        """The source column ('status') is absent from the dummies result."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        dummies = pool.to_dummies(["status"], output_format="polars")
        assert "status" not in dummies.columns

    def test_dummy_columns_are_uint8(self, pools_dict: dict, pool_type: str) -> None:
        """All dummy columns contain UInt8 values (0 or 1)."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        dummies = pool.to_dummies(["status"], output_format="polars")
        dummy_cols = [c for c in dummies.columns if c.startswith("status_")]
        assert dummy_cols, "Expected at least one dummy column for 'status'"
        for col in dummy_cols:
            assert dummies[col].dtype == pl.UInt8

    def test_drop_first_removes_one_column(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """drop_first=True produces one fewer dummy column than drop_first=False."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        full = pool.to_dummies(["status"], drop_first=False, output_format="polars")
        reduced = pool.to_dummies(["status"], drop_first=True, output_format="polars")
        assert len(reduced.columns) == len(full.columns) - 1

    def test_non_categorical_raises(self, pools_dict: dict, pool_type: str) -> None:
        """to_dummies on a non-Categorical column raises TypeError."""
        pool = pools_dict[pool_type].copy()
        with pytest.raises(TypeError):
            pool.to_dummies(["value"], output_format="polars")

    def test_does_not_mutate_pool(self, pools_dict: dict, pool_type: str) -> None:
        """to_dummies is a consumption method: entity_features settings unchanged."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        features_before = list(pool.settings.entity_features)
        pool.to_dummies(["status"], output_format="polars")
        assert pool.settings.entity_features == features_before
