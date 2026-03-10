#!/usr/bin/env python3
"""Tests: TrajectoryPool.apply (read-only computation on trajectory-level static features)."""

from __future__ import annotations

import polars as pl

from tanat.trajectory.pool import TrajectoryPool


class TestTrajectoryPoolApply:
    """TrajectoryPool.apply() evaluates expressions against trajectory-level static features."""

    def test_result_column_present(self, traj_pool: TrajectoryPool) -> None:
        """apply() returns a DataFrame containing the aliased output column."""
        result = traj_pool.apply(pl.col("age").cast(pl.Float64).alias("age_f"))
        assert "age_f" in result.columns

    def test_result_schema(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """apply() output schema matches snapshot."""
        result = traj_pool.apply(pl.col("age").cast(pl.Float64).alias("age_f"))
        assert snapshot == dict(result.schema)

    def test_result_row_count(self, traj_pool: TrajectoryPool) -> None:
        """apply() returns one row per unique trajectory ID."""
        result = traj_pool.apply(pl.col("age").cast(pl.Float64).alias("age_f"))
        assert result.height == len(traj_pool)

    def test_does_not_mutate_feature_list(self, traj_pool: TrajectoryPool) -> None:
        """apply() is read-only: static feature list is unchanged after the call."""
        features_before = list(traj_pool.settings.static_features)
        traj_pool.apply(pl.col("age").cast(pl.Float64).alias("age_f"))
        assert traj_pool.settings.static_features == features_before

    def test_multiple_exprs(self, traj_pool: TrajectoryPool) -> None:
        """apply() accepts a list of expressions and returns all output columns."""
        result = traj_pool.apply(
            [
                pl.col("age").cast(pl.Float64).alias("age_f"),
                pl.col("is_active").cast(pl.Int8).alias("active_int"),
            ]
        )
        assert "age_f" in result.columns
        assert "active_int" in result.columns
