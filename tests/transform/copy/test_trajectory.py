#!/usr/bin/env python3
"""Tests: TrajectoryPool.copy (independent virtual context, mutation isolation)."""

from __future__ import annotations

import polars as pl

from tanat.trajectory.pool import TrajectoryPool


class TestTrajectoryPoolCopy:
    """copy() returns an independent trajectory pool sharing the same physical store."""

    def test_copy_preserves_len(self, traj_pool: TrajectoryPool) -> None:
        """copy() produces a pool with the same number of trajectories."""
        assert len(traj_pool.copy()) == len(traj_pool)

    def test_copy_preserves_static_features(self, traj_pool: TrajectoryPool) -> None:
        """copy() preserves the static feature list."""
        assert (
            traj_pool.copy().settings.static_features
            == traj_pool.settings.static_features
        )

    def test_add_static_on_copy_isolated(self, traj_pool: TrajectoryPool) -> None:
        """add_static_features on copy does not expose the column on the original."""
        copy = traj_pool.copy()
        scores = pl.DataFrame(
            {"id": copy.unique_ids, "copy_score": [float(i) for i in range(len(copy))]}
        )
        copy.add_static_features(scores)
        assert "copy_score" not in traj_pool.settings.static_features

    def test_drop_on_copy_isolated(self, traj_pool: TrajectoryPool) -> None:
        """drop_static_features on copy does not modify the original feature list."""
        copy = traj_pool.copy()
        copy.drop_static_features(["age"])
        assert "age" in traj_pool.settings.static_features

    def test_subset_of_copy_restricts_child_pools(
        self, traj_pool: TrajectoryPool
    ) -> None:
        """subset() applied to a copy propagates the ID mask to all child sequence pools."""
        copy = traj_pool.copy()
        ids = set(copy.unique_ids[:3])
        view = copy.subset(list(ids))
        for sp in view.sequence_pools.values():
            assert set(sp.unique_ids).issubset(ids)

    def test_subset_of_copy_does_not_affect_original(
        self, traj_pool: TrajectoryPool
    ) -> None:
        """Subsetting a copy does not reduce the size of the original pool."""
        original_len = len(traj_pool)
        copy = traj_pool.copy()
        copy.subset(copy.unique_ids[:2])
        assert len(traj_pool) == original_len
