#!/usr/bin/env python3
"""Tests: RankCriterion.

RankCriterion selects entity rows by their 0-based positional rank within
each sequence.  Supported levels: ENTITY.

Parameter groups (mutually exclusive):
    first=N     : keep first N rows (N<0: all except last |N|)
    last=N      : keep last N rows  (N<0: all except first |N|)
    start/end/step : slice with optional step
    ranks=[…]   : explicit 0-based positions (negative = from end)
    relative=True : ranks relative to T0 (nearest entity to time 0)
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.criterion.base import CriterionLevelError
from tanat.criterion.type.rank import RankCriterion

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestRankCriterionValidation:
    """Settings validated at construction time."""

    def test_no_group_raises(self) -> None:
        """Providing no parameter group raises ValueError."""
        with pytest.raises(ValueError):
            RankCriterion()

    def test_first_zero_raises(self) -> None:
        """first=0 is invalid."""
        with pytest.raises(ValueError):
            RankCriterion(first=0)

    def test_last_zero_raises(self) -> None:
        """last=0 is invalid."""
        with pytest.raises(ValueError):
            RankCriterion(last=0)

    def test_step_zero_raises(self) -> None:
        """step=0 is invalid (must be >= 1)."""
        with pytest.raises(ValueError):
            RankCriterion(step=0)

    def test_valid_first(self) -> None:
        """first=3 is valid."""
        assert RankCriterion(first=3) is not None

    def test_valid_ranks(self) -> None:
        """ranks=[0, 2, 4] is valid."""
        assert RankCriterion(ranks=[0, 2, 4]) is not None


# ---------------------------------------------------------------------------
# filter_entities(): the primary API for RankCriterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestRankCriterionFilterEntities:
    """filter_entities() keeps only entity rows at the specified ranks."""

    def test_first_2_keeps_at_most_2_rows_per_id(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """first=2: each ID has at most 2 rows in the result."""
        pool = pools_dict[pool_type]
        id_col = pool.settings.id_column
        filtered = pool.filter_entities(RankCriterion(first=2))
        counts = (
            filtered.temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n"))
        )
        assert (counts["n"] <= 2).all()

    def test_last_2_keeps_at_most_2_rows_per_id(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """last=2: each ID has at most 2 rows in the result."""
        pool = pools_dict[pool_type]
        id_col = pool.settings.id_column
        filtered = pool.filter_entities(RankCriterion(last=2))
        counts = (
            filtered.temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n"))
        )
        assert (counts["n"] <= 2).all()

    def test_first_negative_drops_last_rows(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """first=-1: each ID keeps all rows except the last one."""
        pool = pools_dict[pool_type]
        id_col = pool.settings.id_column
        orig = (
            pool.temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n_orig"))
        )
        result = (
            pool.filter_entities(RankCriterion(first=-1))
            .temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n_result"))
        )
        joined = orig.join(result, on=id_col, how="left").fill_null(0)
        # Each ID: n_result == max(0, n_orig - 1)
        assert (joined["n_result"] == (joined["n_orig"] - 1).clip(lower_bound=0)).all()

    def test_last_negative_drops_first_rows(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """last=-1: each ID keeps all rows except the first one."""
        pool = pools_dict[pool_type]
        id_col = pool.settings.id_column
        orig = (
            pool.temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n_orig"))
        )
        result = (
            pool.filter_entities(RankCriterion(last=-1))
            .temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n_result"))
        )
        joined = orig.join(result, on=id_col, how="left").fill_null(0)
        assert (joined["n_result"] == (joined["n_orig"] - 1).clip(lower_bound=0)).all()

    def test_ranks_explicit(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """Explicit ranks=[0, 2]: temporal data matches snapshot."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(RankCriterion(ranks=[0, 2]))
        assert snapshot == filtered.temporal_data(fmt="polars")

    def test_start_end_step(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """start=0, end=4, step=2: temporal data matches snapshot."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(RankCriterion(start=0, end=4, step=2))
        assert snapshot == filtered.temporal_data(fmt="polars")

    def test_first_and_last_give_same_rows_for_single_row_seqs(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """For single-row sequences, first=1 and last=1 are identical."""
        pool = pools_dict[pool_type]
        id_col = pool.settings.id_column
        single_ids = set(
            pool.temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") == 1)[id_col]
            .to_list()
        )
        if not single_ids:
            pytest.skip("No single-row sequences in this pool")
        f1 = (
            pool.filter_entities(RankCriterion(first=1))
            .temporal_data(fmt="polars")
            .filter(pl.col(id_col).is_in(list(single_ids)))
            .group_by(id_col)
            .agg(pl.len().alias("n"))
            .sort(id_col)
        )
        l1 = (
            pool.filter_entities(RankCriterion(last=1))
            .temporal_data(fmt="polars")
            .filter(pl.col(id_col).is_in(list(single_ids)))
            .group_by(id_col)
            .agg(pl.len().alias("n"))
            .sort(id_col)
        )
        assert f1.equals(l1)

    def test_filter_does_not_mutate_original(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """filter_entities() returns a new view; original pool is unchanged."""
        pool = pools_dict[pool_type]
        original_rows = pool.temporal_data(fmt="polars").height
        pool.filter_entities(RankCriterion(first=2))
        assert pool.temporal_data(fmt="polars").height == original_rows

    def test_relative_mode_snapshot(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """relative=True, ranks=[0]: one entity per ID, describe() matches snapshot."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(RankCriterion(ranks=[0], relative=True))
        assert snapshot == filtered.temporal_data(fmt="polars")


# ---------------------------------------------------------------------------
# Level compatibility guard-rails
# ---------------------------------------------------------------------------


class TestRankCriterionLevelGuards:
    """RankCriterion supports ENTITY only: which() and match() must raise."""

    def test_which_raises(self, interval_pool) -> None:
        """which() raises CriterionLevelError for RankCriterion."""
        with pytest.raises(CriterionLevelError):
            interval_pool.which(RankCriterion(first=2))

    def test_match_raises(self, seq_with_error) -> None:
        """match() raises CriterionLevelError for RankCriterion."""
        with pytest.raises(CriterionLevelError):
            seq_with_error.match(RankCriterion(first=2))

    def test_which_on_trajectory_raises(self, traj_pool) -> None:
        """which() on TrajectoryPool also raises CriterionLevelError."""
        with pytest.raises(CriterionLevelError):
            traj_pool.which(RankCriterion(first=2))
