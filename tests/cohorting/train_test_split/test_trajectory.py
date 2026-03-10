#!/usr/bin/env python3
"""Tests: TrajectoryPool.train_test_split (sizes, disjointness, reproducibility)."""

from __future__ import annotations


class TestTrajectoryPoolTrainTestSplit:
    """train_test_split splits by trajectory ID with the same guarantees as SequencePool."""

    def test_default_split_sizes(self, traj_pool) -> None:
        """Default split (test_size=0.25): train + test == pool size."""
        train, test = traj_pool.train_test_split(random_state=0)
        assert len(train) + len(test) == len(traj_pool)

    def test_ids_disjoint(self, traj_pool) -> None:
        """Train and test ID sets are strictly disjoint."""
        train, test = traj_pool.train_test_split(test_size=0.3, random_state=42)
        assert set(train.unique_ids).isdisjoint(set(test.unique_ids))

    def test_reproducible_with_random_state(self, traj_pool) -> None:
        """Same random_state produces identical splits on repeated calls."""
        train1, test1 = traj_pool.train_test_split(test_size=0.25, random_state=7)
        train2, test2 = traj_pool.train_test_split(test_size=0.25, random_state=7)
        assert train1.unique_ids == train2.unique_ids
        assert test1.unique_ids == test2.unique_ids

    def test_sequence_pools_scoped_to_split(self, traj_pool) -> None:
        """Child sequence_pools in each split are scoped to their respective IDs."""
        train, test = traj_pool.train_test_split(test_size=0.3, random_state=0)
        train_ids = set(train.unique_ids)
        test_ids = set(test.unique_ids)
        for seq_pool in train.sequence_pools.values():
            assert set(seq_pool.unique_ids).issubset(train_ids)
        for seq_pool in test.sequence_pools.values():
            assert set(seq_pool.unique_ids).issubset(test_ids)
