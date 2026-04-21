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
        sd = pool.static_data(fmt="polars")
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

    def test_drop_static_feature_absent_from_metadata(
        self, traj_pool: TrajectoryPool
    ) -> None:
        """After drop_static_features, dropped name absent from traj_pool.metadata.static_features."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        names = {f.name for f in pool.metadata.static_features}
        assert "age" not in names


class TestTrajectoryPoolDropFeaturesPropagation:
    """Dropped static features at trajectory level are absent from Trajectory children.

    A soft drop removes the column from ``settings``, which is passed as
    ``parent_metadata`` to each ``traj_pool[id]`` call — so the column is
    invisible on individual Trajectory objects immediately.
    """

    def test_static_drop_absent_on_trajectory(self, traj_pool: TrajectoryPool) -> None:
        """Dropped static feature is absent from static_data() on a child Trajectory."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        traj = pool[pool.unique_ids[0]]
        sd = traj.static_data(fmt="polars")
        assert "age" not in sd.columns

    def test_drop_static_feature_absent_from_trajectory_metadata(
        self, traj_pool: TrajectoryPool
    ) -> None:
        """After drop_static_features, dropped name absent from traj.metadata.static_features."""
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        traj = pool[pool.unique_ids[0]]
        names = {f.name for f in traj.metadata.static_features}
        assert "age" not in names
