#!/usr/bin/env python3
"""Tests: TimeCriterion.

TimeCriterion filters on temporal bounds (start/end columns).
Supported levels: ENTITY, SEQUENCE.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tanat.criterion.base import CriterionLevelError
from tanat.criterion.type.time import TimeCriterion

from .conftest import _temporal_ids

# ---------------------------------------------------------------------------
# Known facts from frozen test data
# ---------------------------------------------------------------------------

# Date used for all_entities=True tests (datetime variant).
_ALL_ENTITIES_START_GE_DATE = dt.datetime(2000, 9, 30)

# Datetime wide window constants (used in match() tests which always use dt seq fixtures).
_DT_START_MIN = dt.datetime(2000, 1, 1)
_DT_START_MAX = dt.datetime(2001, 12, 31)


# ---------------------------------------------------------------------------
# Helpers: adapt bounds to the pool's time-index type
# ---------------------------------------------------------------------------


def _wide_criterion(pool) -> TimeCriterion:
    """Return a TimeCriterion whose window covers the entire pool."""
    if pool.metadata.time_index.is_datetime:
        return TimeCriterion(
            start_ge=dt.datetime(2000, 1, 1), start_le=dt.datetime(2001, 12, 31)
        )
    return TimeCriterion(start_ge=0.0, start_le=9999.0)


def _impossible_criterion(pool) -> TimeCriterion:
    """Return a TimeCriterion that matches nothing in the pool.

    For state pools (null end = still open), ``start_ge`` alone is not enough
    because an open-ended entity satisfies ``end >= future_date`` via fill_null.
    We add ``start_le`` set before all data so both bounds must hold, impossible.
    """
    if pool.metadata.time_index.is_datetime:
        return TimeCriterion(
            start_ge=dt.datetime(2099, 1, 1),
            start_le=dt.datetime(1900, 1, 1),
        )
    return TimeCriterion(start_ge=999_999.0, start_le=-999_999.0)


def _partial_criterion(pool) -> TimeCriterion:
    """Return a TimeCriterion that matches a meaningful subset of the pool."""
    if pool.metadata.time_index.is_datetime:
        return TimeCriterion(start_ge=dt.datetime(2000, 7, 1))
    # ts start range: 26–442; 200 gives a non-trivial split
    return TimeCriterion(start_ge=200.0)


def _filter_criterion(pool) -> TimeCriterion:
    """Return a filter-level criterion cutting roughly half the pool."""
    if pool.metadata.time_index.is_datetime:
        return TimeCriterion(start_ge=dt.datetime(2000, 9, 30))
    return TimeCriterion(start_ge=300.0)


# ---------------------------------------------------------------------------
# which(): sequence-level selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestTimeCriterionWhich:
    """which() selects sequences that have at least one (or all) rows in the window.

    pools_dict is parametrized over datetime and timestep variants; bounds are
    adapted automatically via the helper functions above.
    """

    def test_which_wide_window_returns_all(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """A window covering the entire data range returns every ID."""
        pool = pools_dict[pool_type]
        result = pool.which(_wide_criterion(pool))
        assert result == _temporal_ids(pool)

    def test_which_impossible_window_returns_empty(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """A window beyond the data range returns an empty set."""
        pool = pools_dict[pool_type]
        result = pool.which(_impossible_criterion(pool))
        assert result == set()

    def test_which_partial_window_snapshot(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Partial window result matches snapshot."""
        pool = pools_dict[pool_type]
        result = pool.which(_partial_criterion(pool))
        assert snapshot == result


# ---------------------------------------------------------------------------
# which(): all_entities semantics (only interval_pool, not parametrized)
# ---------------------------------------------------------------------------


class TestTimeCriterionWhichAllEntities:
    """Tests specific to all_entities=True semantics (datetime interval pool)."""

    def test_any_entity_vs_all_entities(self, interval_pool) -> None:
        """all_entities=True ⊆ all_entities=False (stricter condition)."""
        bound = dt.datetime(2000, 9, 30)
        ids_any = interval_pool.which(TimeCriterion(start_ge=bound, all_entities=False))
        ids_all = interval_pool.which(TimeCriterion(start_ge=bound, all_entities=True))
        assert ids_all.issubset(ids_any)

    def test_all_entities_matches_known_ids(self, interval_pool, snapshot) -> None:
        """all_entities=True with mid-point date matches the snapshot."""
        result = interval_pool.which(
            TimeCriterion(start_ge=_ALL_ENTITIES_START_GE_DATE, all_entities=True)
        )
        assert snapshot == result


# ---------------------------------------------------------------------------
# filter_entities(): entity-level row pruning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event"])
class TestTimeCriterionFilterEntities:
    """filter_entities() keeps only entity rows within the time window."""

    def test_filter_snapshot(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """Filtered temporal data after applying a time bound matches snapshot."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(_filter_criterion(pool))
        assert snapshot == filtered.temporal_data(fmt="polars")

    def test_filter_wide_window_keeps_all_rows(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Window covering all data keeps every entity row."""
        pool = pools_dict[pool_type]
        original = pool.temporal_data(fmt="polars").height
        filtered = pool.filter_entities(_wide_criterion(pool))
        assert filtered.temporal_data(fmt="polars").height == original

    def test_filter_impossible_window_empties_rows(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Impossible window removes all entity rows."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(_impossible_criterion(pool))
        assert filtered.temporal_data(fmt="polars").height == 0

    def test_filter_does_not_mutate_original(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """filter_entities() returns a new view; original pool is unchanged."""
        pool = pools_dict[pool_type]
        original_rows = pool.describe(fmt="polars")["length"].sum()
        pool.filter_entities(_filter_criterion(pool))
        assert pool.describe(fmt="polars")["length"].sum() == original_rows


class TestTimeCriterionFilterEntitiesDurationWithin:
    """Tests specific to duration_within semantics (datetime interval pool)."""

    def test_filter_duration_within_is_stricter(self, interval_pool) -> None:
        """duration_within=True keeps fewer rows than duration_within=False."""
        criterion_any = TimeCriterion(
            start_ge=dt.datetime(2000, 6, 1),
            end_le=dt.datetime(2001, 1, 1),
            duration_within=False,
        )
        criterion_within = TimeCriterion(
            start_ge=dt.datetime(2000, 6, 1),
            end_le=dt.datetime(2001, 1, 1),
            duration_within=True,
        )
        rows_any = (
            interval_pool.filter_entities(criterion_any)
            .describe(fmt="polars")["length"]
            .sum()
        )
        rows_within = (
            interval_pool.filter_entities(criterion_within)
            .describe(fmt="polars")["length"]
            .sum()
        )
        assert rows_within <= rows_any


# ---------------------------------------------------------------------------
# match(): single-sequence evaluation (datetime fixtures, always dt)
# ---------------------------------------------------------------------------


class TestTimeCriterionMatch:
    """match() returns True iff the sequence has ≥1 (or all) rows in the window."""

    def test_match_true_wide_window(self, seq_with_error) -> None:
        """Any sequence matches a window covering the entire data range."""
        criterion = TimeCriterion(start_ge=_DT_START_MIN, start_le=_DT_START_MAX)
        assert seq_with_error.match(criterion) is True

    def test_match_false_impossible_window(self, seq_with_error) -> None:
        """No sequence matches an impossible future window."""
        criterion = TimeCriterion(start_ge=dt.datetime(2099, 1, 1))
        assert seq_with_error.match(criterion) is False


# ---------------------------------------------------------------------------
# Level compatibility guard-rails
# ---------------------------------------------------------------------------


class TestTimeCriterionLevelGuards:
    """TimeCriterion does not support TRAJECTORY level."""

    def test_which_on_trajectory_raises(self, traj_pool) -> None:
        """which() on a TrajectoryPool raises CriterionLevelError."""
        with pytest.raises(CriterionLevelError):
            traj_pool.which(TimeCriterion(start_ge=_DT_START_MIN))

    def test_no_bound_raises(self) -> None:
        """Providing no bound raises ValueError."""
        with pytest.raises(ValueError, match="At least one bound"):
            TimeCriterion()


# ---------------------------------------------------------------------------
# filter_entities() guard on StateSequencePool
# ---------------------------------------------------------------------------


class TestTimeCriterionStateGuard:
    """filter_entities() is not supported on StateSequencePool nor StateSequence."""

    def test_filter_entities_raises_on_state_pool(self, pools_dict: dict) -> None:
        """StateSequencePool.filter_entities() must raise TypeError."""
        pool = pools_dict["state"]
        with pytest.raises(TypeError, match="filter_entities"):
            pool.filter_entities(TimeCriterion(start_ge=dt.datetime(2020, 1, 1)))

    def test_filter_entities_raises_on_state_sequence(self, state_seq) -> None:
        """StateSequence.filter_entities() must raise TypeError."""
        with pytest.raises(TypeError, match="filter_entities"):
            state_seq.filter_entities(TimeCriterion(start_ge=dt.datetime(2020, 1, 1)))
