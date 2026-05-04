#!/usr/bin/env python3
"""Tests: LengthCriterion.

LengthCriterion selects sequences by their number of entity rows.
Supported levels: SEQUENCE.
"""

from __future__ import annotations

import pytest

from tanat.criterion.base import CriterionLevelError
from tanat.criterion.type.length import LengthCriterion

from .conftest import _temporal_ids

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestLengthCriterionValidation:
    """Settings are validated at construction time."""

    def test_no_bound_raises(self) -> None:
        """Providing no bound raises ValueError."""
        with pytest.raises(ValueError, match="At least one bound"):
            LengthCriterion()

    def test_contradictory_bounds_raise(self) -> None:
        """gt=5 and lt=3 is an empty range → ValueError."""
        with pytest.raises(ValueError, match="[Cc]ontradictory|empty"):
            LengthCriterion(gt=5, lt=3)

    def test_no_bounds_raises(self) -> None:
        """Providing no bounds raises ValueError."""
        with pytest.raises(ValueError):
            LengthCriterion()

    def test_valid_single_bound(self) -> None:
        """A single bound is sufficient."""
        c = LengthCriterion(gt=3)
        assert c is not None


# ---------------------------------------------------------------------------
# which(): sequence-level selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestLengthCriterionWhich:
    """which() returns IDs whose sequence length satisfies the criterion."""

    def test_which_gt6(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """IDs with more than 6 rows match the snapshot."""
        pool = pools_dict[pool_type]
        result = pool.which(LengthCriterion(gt=6))
        assert snapshot == result

    def test_which_le3(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """IDs with at most 3 rows match the snapshot."""
        pool = pools_dict[pool_type]
        result = pool.which(LengthCriterion(le=3))
        assert snapshot == result

    def test_which_range_is_intersection(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """gt=3, le=6 result is the intersection of the two individual results."""
        pool = pools_dict[pool_type]
        ids_gt3 = pool.which(LengthCriterion(gt=3))
        ids_le6 = pool.which(LengthCriterion(le=6))
        ids_range = pool.which(LengthCriterion(gt=3, le=6))
        assert ids_range == ids_gt3 & ids_le6

    def test_which_gt_and_ge_consistent(self, pools_dict: dict, pool_type: str) -> None:
        """gt=N is equivalent to ge=N+1."""
        pool = pools_dict[pool_type]
        assert pool.which(LengthCriterion(gt=4)) == pool.which(LengthCriterion(ge=5))

    def test_which_lt_and_le_consistent(self, pools_dict: dict, pool_type: str) -> None:
        """lt=N is equivalent to le=N-1."""
        pool = pools_dict[pool_type]
        assert pool.which(LengthCriterion(lt=5)) == pool.which(LengthCriterion(le=4))

    def test_which_all_match(self, pools_dict: dict, pool_type: str) -> None:
        """A very wide range returns every ID."""
        pool = pools_dict[pool_type]
        result = pool.which(LengthCriterion(ge=1))
        assert result == _temporal_ids(pool)

    def test_which_none_match(self, pools_dict: dict, pool_type: str) -> None:
        """An impossible bound returns an empty set."""
        pool = pools_dict[pool_type]
        result = pool.which(LengthCriterion(gt=9999))
        assert result == set()


# ---------------------------------------------------------------------------
# match(): single-sequence evaluation
# ---------------------------------------------------------------------------


class TestLengthCriterionMatch:
    """match() returns True iff the sequence length satisfies the criterion."""

    def test_match_true(self, seq_long) -> None:
        """Long sequence (n>6) matches LengthCriterion(gt=6)."""
        assert seq_long.match(LengthCriterion(gt=6)) is True

    def test_match_false(self, seq_short) -> None:
        """Short sequence (n==3) does not match LengthCriterion(gt=6)."""
        assert seq_short.match(LengthCriterion(gt=6)) is False

    def test_match_exact_boundary(self, seq_short) -> None:
        """Boundary: seq of length 3 matches ge=3 but not gt=3."""
        assert seq_short.match(LengthCriterion(ge=3)) is True
        assert seq_short.match(LengthCriterion(gt=3)) is False


# ---------------------------------------------------------------------------
# Level compatibility guard-rails
# ---------------------------------------------------------------------------


class TestLengthCriterionLevelGuards:
    """LengthCriterion does not support ENTITY or TRAJECTORY levels."""

    def test_filter_entities_raises(self, interval_pool) -> None:
        """filter_entities() raises CriterionLevelError for LengthCriterion."""
        with pytest.raises(CriterionLevelError):
            interval_pool.filter_entities(LengthCriterion(gt=3))

    def test_which_on_trajectory_raises(self, traj_pool) -> None:
        """which() on a TrajectoryPool raises CriterionLevelError."""
        with pytest.raises(CriterionLevelError):
            traj_pool.which(LengthCriterion(gt=3))
