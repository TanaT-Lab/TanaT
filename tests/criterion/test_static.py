#!/usr/bin/env python3
"""Tests: StaticCriterion.

StaticCriterion filters on static (per-ID) data via a Polars expression.
Supported levels: SEQUENCE, TRAJECTORY.
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.criterion.base import CriterionLevelError
from tanat.criterion.type.static import StaticCriterion

# ---------------------------------------------------------------------------
# which(): sequence pool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestStaticCriterionWhichSequence:
    """which() on a SequencePool filters by static attributes."""

    def test_which_age_gt50(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """IDs with age > 50 match the snapshot."""
        pool = pools_dict[pool_type]
        result = pool.which(StaticCriterion(query=pl.col("age") > 50))
        assert snapshot == result

    def test_which_group_a(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """IDs in group 'A' match the snapshot."""
        pool = pools_dict[pool_type]
        result = pool.which(StaticCriterion(query=pl.col("group") == "A"))
        assert snapshot == result

    def test_which_all_match(self, pools_dict: dict, pool_type: str) -> None:
        """A literal True expression returns every ID in the pool."""
        pool = pools_dict[pool_type]
        result = pool.which(StaticCriterion(query=pl.lit(True)))
        assert result == set(pool.unique_ids)

    def test_which_none_match(self, pools_dict: dict, pool_type: str) -> None:
        """A literal False expression returns an empty set."""
        pool = pools_dict[pool_type]
        result = pool.which(StaticCriterion(query=pl.lit(False)))
        assert result == set()

    def test_which_present_false_is_complement(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """age>50 and age<=50 partition the IDs that have a non-null age."""
        pool = pools_dict[pool_type]
        # IDs where age is not null: only those can appear in either partition.
        ids_with_age = pool.which(StaticCriterion(query=pl.col("age").is_not_null()))
        ids_old = pool.which(StaticCriterion(query=pl.col("age") > 50))
        ids_young = pool.which(StaticCriterion(query=pl.col("age") <= 50))
        assert ids_old | ids_young == ids_with_age
        assert ids_old & ids_young == set()


# ---------------------------------------------------------------------------
# which(): trajectory pool
# ---------------------------------------------------------------------------


class TestStaticCriterionWhichTrajectory:
    """which() on a TrajectoryPool filters by trajectory-level static attributes."""

    def test_which_age_gt50_traj(self, traj_pool) -> None:
        """Trajectory IDs with age > 50 are a non-empty subset of all IDs."""
        result = traj_pool.which(StaticCriterion(query=pl.col("age") > 50))
        assert result  # non-empty
        assert result.issubset(set(traj_pool.unique_ids))

    def test_which_group_a_traj(self, traj_pool) -> None:
        """Trajectory IDs in group 'A' are a non-empty subset of all IDs."""
        result = traj_pool.which(StaticCriterion(query=pl.col("group") == "A"))
        assert result
        assert result.issubset(set(traj_pool.unique_ids))

    def test_which_sequence_and_trajectory_agree(
        self, interval_pool, traj_pool
    ) -> None:
        """Same static expression returns the same IDs on sequence and trajectory pools."""
        expr = pl.col("age") > 50
        seq_ids = interval_pool.which(StaticCriterion(query=expr))
        traj_ids = traj_pool.which(StaticCriterion(query=expr))
        # trajectory pool shares the same static data → must be equal
        assert seq_ids == traj_ids


# ---------------------------------------------------------------------------
# match(): single sequence and trajectory
# ---------------------------------------------------------------------------


class TestStaticCriterionMatch:
    """match() returns True iff the static row of the target satisfies the expression."""

    def test_match_sequence_true(self, interval_pool) -> None:
        """Sequence whose age > 50 matches the criterion."""
        seq = interval_pool[11]
        assert seq.match(StaticCriterion(query=pl.col("age") > 50)) is True

    def test_match_sequence_false(self, interval_pool) -> None:
        """Sequence whose age <= 50 does not match."""
        seq = interval_pool[2]
        assert seq.match(StaticCriterion(query=pl.col("age") > 50)) is False

    def test_match_trajectory_true(self, traj_pool) -> None:
        """Trajectory with age > 50 matches."""
        traj = traj_pool[11]
        assert traj.match(StaticCriterion(query=pl.col("age") > 50)) is True

    def test_match_trajectory_false(self, traj_pool) -> None:
        """Trajectory with age <= 50 does not match."""
        traj = traj_pool[2]
        assert traj.match(StaticCriterion(query=pl.col("age") > 50)) is False


# ---------------------------------------------------------------------------
# Level compatibility guard-rails
# ---------------------------------------------------------------------------


class TestStaticCriterionLevelGuards:
    """StaticCriterion does not support ENTITY level."""

    def test_filter_entities_raises(self, interval_pool) -> None:
        """filter_entities() raises CriterionLevelError for StaticCriterion."""
        with pytest.raises(CriterionLevelError):
            interval_pool.filter_entities(StaticCriterion(query=pl.col("age") > 50))
