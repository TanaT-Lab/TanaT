#!/usr/bin/env python3
"""Tests: TrajectoryPool.extend (fast path, on_duplicate guard)."""

from __future__ import annotations

import pytest


class TestTrajectoryPoolExtend:
    """extend() merges two trajectory pool views or stores into a single pool."""

    def test_fast_path_same_store(self, traj_pool) -> None:
        """Fast path: train + test from the same store reunite with zero I/O."""
        train, test = traj_pool.train_test_split(test_size=0.3, random_state=0)
        reunited = train.extend(test)
        assert set(reunited.unique_ids) == set(train.unique_ids) | set(test.unique_ids)

    def test_fast_path_len(self, traj_pool) -> None:
        """Fast path: reunited pool has exactly train + test IDs."""
        train, test = traj_pool.train_test_split(test_size=0.3, random_state=0)
        reunited = train.extend(test)
        assert len(reunited) == len(train) + len(test)

    def test_on_duplicate_raise(self, traj_pool) -> None:
        """on_duplicate='raise' (default) raises ValueError on overlapping IDs."""
        with pytest.raises(ValueError, match="(?i)duplicate"):
            traj_pool.extend(traj_pool)

    def test_on_duplicate_skip(self, traj_pool) -> None:
        """on_duplicate='skip' silently ignores already-present IDs."""
        train, _ = traj_pool.train_test_split(test_size=0.4, random_state=1)
        result = train.extend(traj_pool, on_duplicate="skip")
        assert set(traj_pool.unique_ids).issubset(set(result.unique_ids))
