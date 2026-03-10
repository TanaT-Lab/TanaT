#!/usr/bin/env python3
"""Tests: TrajectoryPool.add_static_features."""

from __future__ import annotations

import polars as pl
import pytest

from tanat.trajectory.pool import TrajectoryPool


class TestTrajectoryPoolAddFeatures:
    """add_static_features adds trajectory-level features via an ID-keyed join."""

    def test_add_static_visible(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """New column appears in static_features settings after add_static_features."""
        pool = traj_pool.copy()
        scores = pl.DataFrame(
            {"id": pool.unique_ids, "score": [float(i) for i in range(len(pool))]}
        )
        pool.add_static_features(scores)
        assert snapshot == pool.settings.static_features

    def test_add_static_marks_dirty(self, traj_pool: TrajectoryPool) -> None:
        """add_static_features marks the pool as dirty."""
        pool = traj_pool.copy()
        scores = pl.DataFrame(
            {"id": pool.unique_ids, "score2": [float(i) for i in range(len(pool))]}
        )
        pool.add_static_features(scores)
        assert pool.is_dirty

    def test_add_static_partial_gives_nulls(self, traj_pool: TrajectoryPool) -> None:
        """Partial df: trajectory IDs absent from input receive null in static_data()."""
        pool = traj_pool.copy()
        partial = pl.DataFrame(
            {
                "id": pool.unique_ids[:3],
                "partial_score": [1.0, 2.0, 3.0],
            }
        )
        pool.add_static_features(partial)
        sd = pool.static_data(output_format="polars")
        assert sd["partial_score"].null_count() > 0

    def test_add_static_collision_raises(self, traj_pool: TrajectoryPool) -> None:
        """ValueError raised on column collision without overwrite=True."""
        pool = traj_pool.copy()
        df = pl.DataFrame({"id": pool.unique_ids, "dup_col": [0.0] * len(pool)})
        pool.add_static_features(df)
        with pytest.raises(ValueError, match="collision"):
            pool.add_static_features(df)

    def test_add_static_missing_id_column_raises(
        self, traj_pool: TrajectoryPool
    ) -> None:
        """KeyError raised when the id column is absent from the input DataFrame."""
        pool = traj_pool.copy()
        df = pl.DataFrame({"wrong_key": pool.unique_ids, "val": [0.0] * len(pool)})
        with pytest.raises(KeyError):
            pool.add_static_features(df)
