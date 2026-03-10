#!/usr/bin/env python3
"""Tests: SequencePool.apply (read-only expression evaluation)."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolApply:
    """apply() evaluates Polars expressions without mutating the pool."""

    def test_global_expr_column_present(self, pools_dict: dict, pool_type: str) -> None:
        """apply() result contains the aliased output column."""
        result = pools_dict[pool_type].apply(
            (pl.col("value") * 2).alias("value_doubled"),
            output_format="polars",
        )
        assert "value_doubled" in result.columns

    def test_global_expr_dtype(self, pools_dict: dict, pool_type: str) -> None:
        """Result dtype of an arithmetic expression matches the source column."""
        result = pools_dict[pool_type].apply(
            (pl.col("value") * 2).alias("value_doubled"),
            output_format="polars",
        )
        assert result["value_doubled"].dtype == pl.Float64

    def test_by_id_has_id_column(self, pools_dict: dict, pool_type: str) -> None:
        """by_id=True always includes the ID column in the result."""
        pool = pools_dict[pool_type]
        result = pool.apply(
            pl.col("value").mean().alias("value_mean"),
            by_id=True,
            output_format="polars",
        )
        assert pool.settings.id_column in result.columns

    def test_by_id_row_count(self, pools_dict: dict, pool_type: str) -> None:
        """by_id=True returns exactly one row per sequence that has entity data."""
        pool = pools_dict[pool_type]
        result = pool.apply(
            pl.col("value").mean().alias("value_mean"),
            by_id=True,
            output_format="polars",
        )
        id_col = pool.settings.id_column
        # One row per ID (sparse pools: not all IDs have entity data)
        assert result.height == result[id_col].n_unique()

    def test_by_id_result_schema(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """by_id=True output schema (column → dtype) matches snapshot."""
        pool = pools_dict[pool_type]
        result = pool.apply(
            pl.col("value").mean().alias("value_mean"),
            by_id=True,
            output_format="polars",
        )
        assert snapshot == dict(result.schema)

    def test_by_id_static_raises(self, pools_dict: dict, pool_type: str) -> None:
        """by_id=True with is_static=True raises ValueError."""
        pool = pools_dict[pool_type]
        with pytest.raises(ValueError):
            pool.apply(
                pl.col("age").mean().alias("age_mean"),
                is_static=True,
                by_id=True,
                output_format="polars",
            )

    def test_apply_is_read_only(self, pools_dict: dict, pool_type: str) -> None:
        """apply() does not alter the pool's entity feature list."""
        pool = pools_dict[pool_type]
        features_before = list(pool.settings.entity_features)
        pool.apply(
            (pl.col("value") * 2).alias("value_doubled"),
            output_format="polars",
        )
        assert pool.settings.entity_features == features_before

    def test_static_expr(self, pools_dict: dict, pool_type: str) -> None:
        """apply(is_static=True) evaluates expressions against static features."""
        result = pools_dict[pool_type].apply(
            pl.col("age").cast(pl.Float64).alias("age_f"),
            is_static=True,
            output_format="polars",
        )
        assert "age_f" in result.columns
