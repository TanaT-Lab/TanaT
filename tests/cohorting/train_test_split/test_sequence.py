#!/usr/bin/env python3
"""Tests: SequencePool.train_test_split (sizes, disjointness, reproducibility)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolTrainTestSplit:
    """train_test_split splits by unique ID into two non-overlapping pool views."""

    def test_default_split_sizes(self, pools_dict: dict, pool_type: str) -> None:
        """Default split (test_size=0.25): train + test == pool size."""
        pool = pools_dict[pool_type]
        train, test = pool.train_test_split(random_state=0)
        assert len(train) + len(test) == len(pool)

    def test_float_test_size(self, pools_dict: dict, pool_type: str) -> None:
        """test_size=0.2 → test is approximately 20% of the pool."""
        pool = pools_dict[pool_type]
        train, test = pool.train_test_split(test_size=0.2, random_state=0)
        assert len(train) + len(test) == len(pool)
        # allow ±1 due to rounding
        assert abs(len(test) - round(0.2 * len(pool))) <= 1

    def test_ids_disjoint(self, pools_dict: dict, pool_type: str) -> None:
        """Train and test ID sets are strictly disjoint."""
        pool = pools_dict[pool_type]
        train, test = pool.train_test_split(test_size=0.3, random_state=42)
        assert set(train.unique_ids).isdisjoint(set(test.unique_ids))

    def test_reproducible_with_random_state(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Same random_state produces identical splits on repeated calls."""
        pool = pools_dict[pool_type]
        train1, test1 = pool.train_test_split(test_size=0.25, random_state=7)
        train2, test2 = pool.train_test_split(test_size=0.25, random_state=7)
        assert train1.unique_ids == train2.unique_ids
        assert test1.unique_ids == test2.unique_ids

    def test_shuffle_false_preserves_order(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """shuffle=False: train IDs come first in store order, test IDs come after."""
        pool = pools_dict[pool_type]
        n_test = max(1, len(pool) // 5)
        train, test = pool.train_test_split(test_size=n_test, shuffle=False)
        all_ids = pool.unique_ids
        assert train.unique_ids == all_ids[: len(train)]
        assert test.unique_ids == all_ids[len(train) : len(train) + len(test)]
