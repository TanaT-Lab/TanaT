#!/usr/bin/env python3
"""Tests: EntityCriterion.

EntityCriterion filters on entity (temporal) data via a Polars expression.
Supported levels: ENTITY, SEQUENCE.
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.criterion.type.entity import EntityCriterion

from .conftest import _temporal_ids

# ---------------------------------------------------------------------------
# which(): sequence-level selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestEntityCriterionWhich:
    """which() returns the set of IDs that have at least one matching row."""

    def test_which_partial_match(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """IDs with at least one 'error' status row match the snapshot."""
        pool = pools_dict[pool_type]
        result = pool.which(EntityCriterion(query=pl.col("status") == "error"))
        assert snapshot == result

    def test_which_all_match(self, pools_dict: dict, pool_type: str) -> None:
        """A literal True expression returns every ID in the pool."""
        pool = pools_dict[pool_type]
        result = pool.which(EntityCriterion(query=pl.lit(True)))
        assert result == _temporal_ids(pool)

    def test_which_none_match(self, pools_dict: dict, pool_type: str) -> None:
        """A literal False expression returns an empty set."""
        pool = pools_dict[pool_type]
        result = pool.which(EntityCriterion(query=pl.lit(False)))
        assert result == set()

    def test_which_present_false_is_complement(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """IDs with ≥1 error and IDs with zero errors partition the full temporal ID set."""
        pool = pools_dict[pool_type]
        # "at least one error" vs "no error at all", these are true complements.
        ids_any_error = pool.which(EntityCriterion(query=pl.col("status") == "error"))
        ids_no_error = _temporal_ids(pool) - ids_any_error
        # Verify: sequences with zero errors do NOT appear in ids_any_error.
        assert ids_any_error & ids_no_error == set()
        assert ids_any_error | ids_no_error == _temporal_ids(pool)


# ---------------------------------------------------------------------------
# match(): single-sequence evaluation
# ---------------------------------------------------------------------------


class TestEntityCriterionMatch:
    """match() returns True iff the sequence has ≥1 matching row."""

    def test_match_true(self, seq_with_error) -> None:
        """Sequence with an 'error' row matches the criterion."""
        criterion = EntityCriterion(query=pl.col("status") == "error")
        assert seq_with_error.match(criterion) is True

    def test_match_false(self, seq_without_error) -> None:
        """Sequence with no 'error' row does not match the criterion."""
        criterion = EntityCriterion(query=pl.col("status") == "error")
        assert seq_without_error.match(criterion) is False

    def test_match_inverted(self, seq_with_error, seq_without_error) -> None:
        """Negated expression inverts the match result."""
        criterion = EntityCriterion(query=pl.col("status") != "error")
        assert seq_with_error.match(criterion) is True  # has other statuses too
        assert seq_without_error.match(criterion) is True


# ---------------------------------------------------------------------------
# filter_entities(): entity-level row pruning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event"])
class TestEntityCriterionFilterEntities:
    """filter_entities() keeps only rows satisfying the expression."""

    def test_filter_keeps_only_matching_rows(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Filtered pool entity data matches snapshot (schema + content)."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(
            EntityCriterion(query=pl.col("status") == "error")
        )
        assert snapshot == filtered.temporal_data(fmt="polars")

    def test_filter_does_not_mutate_original(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """filter_entities() returns a new view; original pool is unchanged."""
        pool = pools_dict[pool_type]
        original_rows = pool.describe(fmt="polars")["length"].sum()
        pool.filter_entities(EntityCriterion(query=pl.col("status") == "error"))
        assert pool.describe(fmt="polars")["length"].sum() == original_rows

    def test_filter_literal_false_empties_rows(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Filtering with a False expression removes all entity rows."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(EntityCriterion(query=pl.lit(False)))
        assert filtered.temporal_data(fmt="polars").height == 0

    def test_filter_literal_true_keeps_all_rows(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Filtering with a True expression keeps all entity rows."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(EntityCriterion(query=pl.lit(True)))
        assert (
            filtered.temporal_data(fmt="polars").height
            == pool.temporal_data(fmt="polars").height
        )


# ---------------------------------------------------------------------------
# filter_entities() guard on StateSequencePool
# ---------------------------------------------------------------------------


class TestEntityCriterionStateGuard:
    """filter_entities() is not supported on StateSequencePool nor StateSequence."""

    def test_filter_entities_raises_on_state_pool(self, pools_dict: dict) -> None:
        """StateSequencePool.filter_entities() must raise TypeError."""
        pool = pools_dict["state"]
        with pytest.raises(TypeError, match="filter_entities"):
            pool.filter_entities(EntityCriterion(query=pl.lit(True)))

    def test_filter_entities_raises_on_state_sequence(self, state_seq) -> None:
        """StateSequence.filter_entities() must raise TypeError."""
        with pytest.raises(TypeError, match="filter_entities"):
            state_seq.filter_entities(EntityCriterion(query=pl.lit(True)))
