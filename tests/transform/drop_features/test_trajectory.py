#!/usr/bin/env python3
"""Tests: TrajectoryPool.drop_static_features (soft drop, view isolation)."""

from __future__ import annotations

from tanat.trajectory.pool import TrajectoryPool


class TestTrajectoryPoolDropFeatures:
    """drop_static_features hides trajectory-level static columns from the view."""

    def test_drop_absent_from_settings(self, traj_pool: TrajectoryPool) -> None:
        """Dropped feature no longer appears in static_features settings."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        assert "age" not in pool.settings.static_features

    def test_drop_absent_from_data(self, traj_pool: TrajectoryPool) -> None:
        """Dropped feature is absent from static_data() output columns."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        sd = pool.static_data(output_format="polars")
        if sd is not None:
            assert "age" not in sd.columns

    def test_drop_marks_dirty(self, traj_pool: TrajectoryPool) -> None:
        """drop_static_features marks the pool as dirty."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        assert pool.is_dirty

    def test_soft_drop_does_not_affect_original(
        self, traj_pool: TrajectoryPool
    ) -> None:
        """Soft drop on a copy never modifies the session-scoped pool."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        assert "age" in traj_pool.settings.static_features

    def test_static_features_snapshot_after_drop(
        self, traj_pool: TrajectoryPool, snapshot
    ) -> None:
        """Static feature list after dropping 'age' matches snapshot."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        assert snapshot == pool.settings.static_features
