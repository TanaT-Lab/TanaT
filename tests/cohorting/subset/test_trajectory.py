#!/usr/bin/env python3
"""Tests: TrajectoryPool.subset (view scoping and guard-rails)."""

from __future__ import annotations

import pytest


class TestTrajectoryPoolSubset:
    """subset() returns a trajectory pool view restricted to the requested IDs."""

    def test_subset_len(self, traj_pool) -> None:
        """Subset contains exactly the requested number of IDs."""
        ids = traj_pool.unique_ids[:3]
        view = traj_pool.subset(ids)
        assert len(view) == 3

    def test_subset_ids_match(self, traj_pool) -> None:
        """Subset unique_ids equals the requested IDs."""
        ids = traj_pool.unique_ids[:5]
        view = traj_pool.subset(ids)
        assert set(view.unique_ids) == set(ids)

    def test_subset_does_not_mutate_original(self, traj_pool) -> None:
        """subset() returns a new view; the original pool is unchanged."""
        original_len = len(traj_pool)
        traj_pool.subset(traj_pool.unique_ids[:3])
        assert len(traj_pool) == original_len

    def test_subset_propagates_to_sequence_pools(self, traj_pool) -> None:
        """Each child sequence_pool in the view is scoped to the same IDs."""
        ids = set(traj_pool.unique_ids[:4])
        view = traj_pool.subset(list(ids))
        for seq_pool in view.sequence_pools.values():
            assert set(seq_pool.unique_ids).issubset(ids)

    def test_subset_unknown_id_raises(self, traj_pool) -> None:
        """Requesting an ID not in the pool raises an error."""
        with pytest.raises((ValueError, KeyError)):
            traj_pool.subset([-9999])
