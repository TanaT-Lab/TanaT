#!/usr/bin/env python3
"""Tests: SequencePool.subset (view scoping and guard-rails)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolSubset:
    """subset() returns a pool view restricted to the requested IDs."""

    def test_subset_len(self, pools_dict: dict, pool_type: str) -> None:
        """Subset contains exactly the requested number of IDs."""
        pool = pools_dict[pool_type]
        ids = pool.unique_ids[:3]
        view = pool.subset(ids)
        assert len(view) == 3

    def test_subset_ids_match(self, pools_dict: dict, pool_type: str) -> None:
        """Subset unique_ids equals the requested IDs (order may differ)."""
        pool = pools_dict[pool_type]
        ids = pool.unique_ids[:5]
        view = pool.subset(ids)
        assert set(view.unique_ids) == set(ids)

    def test_subset_does_not_mutate_original(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """subset() returns a new view; the original pool is unchanged."""
        pool = pools_dict[pool_type]
        original_len = len(pool)
        pool.subset(pool.unique_ids[:3])
        assert len(pool) == original_len

    def test_subset_unknown_id_raises(self, pools_dict: dict, pool_type: str) -> None:
        """Requesting an ID not in the pool raises ValueError."""
        pool = pools_dict[pool_type]
        with pytest.raises((ValueError, KeyError)):
            pool.subset([-9999])
