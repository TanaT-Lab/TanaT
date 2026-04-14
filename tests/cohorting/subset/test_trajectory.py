#!/usr/bin/env python3
"""Tests: TrajectoryPool.subset (view scoping and guard-rails)."""

from __future__ import annotations

import polars as pl
import pytest

from tanat.metadata.sequence import CategoricalInfo


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


# ---------------------------------------------------------------------------
# Cast preservation (regression for TrajectoryPool._propagate_id_mask_to_pools)
# ---------------------------------------------------------------------------


class TestSubsetCastPreservation:
    """Entity casts applied to sequence_pools must survive TrajectoryPool.subset()."""

    def test_entity_cast_preserved_after_subset(self, traj_pool) -> None:
        """Cast applied before subset() is still active after subset()."""
        # Work on a copy so the session-scoped traj_pool is not mutated.
        pool = traj_pool.copy()
        for alias in pool._store_aliases:  # pylint: disable=protected-access
            pool.sequence_pools[alias].cast_features({"status": pl.Categorical})

        ids = pool.unique_ids[:4]
        sub = pool.subset(ids)

        for alias in sub._store_aliases:  # pylint: disable=protected-access
            meta = sub.sequence_pools[alias].metadata
            status_info = meta.feature_info("status", is_static=False)
            assert isinstance(status_info, CategoricalInfo), (
                f"alias={alias!r}: expected CategoricalInfo after subset(), "
                f"got {type(status_info).__name__}"
            )

    def test_entity_cast_preserved_after_subset_inplace(self, traj_pool) -> None:
        """Cast applied before subset(inplace=True) is still active afterwards."""
        # Work on a copy so the session-scoped traj_pool is not mutated.
        pool = traj_pool.copy()
        for alias in pool._store_aliases:  # pylint: disable=protected-access
            pool.sequence_pools[alias].cast_features({"status": pl.Categorical})

        ids = pool.unique_ids[:4]
        pool.subset(ids, inplace=True)

        for alias in pool._store_aliases:  # pylint: disable=protected-access
            meta = pool.sequence_pools[alias].metadata
            status_info = meta.feature_info("status", is_static=False)
            assert isinstance(status_info, CategoricalInfo), (
                f"alias={alias!r}: expected CategoricalInfo after subset(inplace=True), "
                f"got {type(status_info).__name__}"
            )
