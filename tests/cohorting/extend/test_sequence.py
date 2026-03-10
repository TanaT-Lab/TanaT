#!/usr/bin/env python3
"""Tests: SequencePool.extend (fast path, cross-store, on_duplicate guard)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolExtend:
    """extend() merges two pool views or stores into a single pool."""

    def test_fast_path_same_store(self, pools_dict: dict, pool_type: str) -> None:
        """Fast path: train + test from the same store reunite with zero I/O.

        The result shares the same store root; no destination argument is needed.
        """
        pool = pools_dict[pool_type]
        train, test = pool.train_test_split(test_size=0.3, random_state=0)
        reunited = train.extend(test)
        assert set(reunited.unique_ids) == set(train.unique_ids) | set(test.unique_ids)

    def test_fast_path_len(self, pools_dict: dict, pool_type: str) -> None:
        """Fast path: reunited pool has exactly train + test IDs."""
        pool = pools_dict[pool_type]
        train, test = pool.train_test_split(test_size=0.3, random_state=0)
        reunited = train.extend(test)
        assert len(reunited) == len(train) + len(test)

    def test_on_duplicate_raise(self, pools_dict: dict, pool_type: str) -> None:
        """on_duplicate='raise' (default) raises ValueError on overlapping IDs."""
        pool = pools_dict[pool_type]
        with pytest.raises(ValueError, match="(?i)duplicate"):
            pool.extend(pool)

    def test_on_duplicate_skip(self, pools_dict: dict, pool_type: str) -> None:
        """on_duplicate='skip' silently ignores already-present IDs."""
        pool = pools_dict[pool_type]
        # Split so train ∩ test = ∅, then make a partial-overlap view
        train, _ = pool.train_test_split(test_size=0.4, random_state=1)
        # Extend train with the full pool (contains all train IDs → all duplicates)
        result = train.extend(pool, on_duplicate="skip")
        # All pool IDs should now be present (skipped dupes, added new ones)
        assert set(pool.unique_ids).issubset(set(result.unique_ids))
