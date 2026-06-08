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

    def test_which_after_casted_feature(self, pools_dict: dict, pool_type: str) -> None:
        """which() works when a string-backed numeric feature is cast before querying."""
        pool = pools_dict[pool_type].copy()
        rows = pool.temporal_data(fmt="polars").height
        status_num = ["1" if i % 2 == 0 else "2" for i in range(rows)]
        pool.add_entity_features(pl.DataFrame({"status_num": status_num}))
        pool.cast_features({"status_num": pl.Int64})

        result = pool.which(EntityCriterion(query=pl.col("status_num") == 1))

        id_col = pool.settings.id_column
        ids_with_1 = (
            pool.temporal_data(fmt="polars")
            .filter(pl.col("status_num") == 1)
            .select(id_col)
            .unique()[id_col]
            .to_list()
        )
        assert result == set(ids_with_1)


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

    def test_match_after_casted_feature(self, pools_dict: dict) -> None:
        """match() works when a string-backed numeric feature is cast before evaluation."""
        pool = pools_dict["interval"].copy()
        rows = pool.temporal_data(fmt="polars").height
        status_num = ["1" if i % 2 == 0 else "2" for i in range(rows)]
        pool.add_entity_features(pl.DataFrame({"status_num": status_num}))
        pool.cast_features({"status_num": pl.Int64})

        seq = pool[pool.unique_ids[0]]
        expected = seq.temporal_data(fmt="polars")["status_num"].to_list().count(1) > 0
        result = seq.match(EntityCriterion(query=pl.col("status_num") == 1))

        assert result is expected


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

    def test_filter_after_casted_text_feature(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """filter_entities() works when a string-backed numeric feature is cast before filtering."""
        pool = pools_dict[pool_type].copy()
        rows = pool.temporal_data(fmt="polars").height
        status_num = ["1" if i % 2 == 0 else "2" for i in range(rows)]
        pool.add_entity_features(pl.DataFrame({"status_num": status_num}))
        pool.cast_features({"status_num": pl.Int64})

        filtered = pool.filter_entities(
            EntityCriterion(query=pl.col("status_num") == 1)
        )
        values = filtered.temporal_data(fmt="polars")["status_num"].to_list()

        assert values
        assert all(value == 1 for value in values)

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


# ---------------------------------------------------------------------------
# _store_index_df path: entity filter + feature cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event"])
class TestEntityCriterionStoreIndexPath:
    """len(seq) and rank-based access go through _store_index_df, a separate
    pipeline from temporal(). This class ensures that pipeline also applies
    feature casts before evaluating the entity filter expression.
    """

    def test_len_sequence_from_filtered_pool_with_casted_feature(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """len(seq) does not crash when the entity filter depends on a cast feature."""
        pool = pools_dict[pool_type].copy()
        rows = pool.temporal_data(fmt="polars").height
        status_num = ["1" if i % 2 == 0 else "2" for i in range(rows)]
        pool.add_entity_features(pl.DataFrame({"status_num": status_num}))
        pool.cast_features({"status_num": pl.Int64})

        filtered = pool.filter_entities(
            EntityCriterion(query=pl.col("status_num") == 1)
        )
        seq_id = filtered.unique_ids[0]
        seq = filtered[seq_id]

        # len(seq) triggers _store_index_df on filtered (has_entity_filter_expr=True)
        n = len(seq)
        assert n > 0

    def test_all_rows_in_filtered_sequence_match_cast_value(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Rank resolution is correct: every returned row satisfies the cast-based filter."""
        pool = pools_dict[pool_type].copy()
        rows = pool.temporal_data(fmt="polars").height
        status_num = ["1" if i % 2 == 0 else "2" for i in range(rows)]
        pool.add_entity_features(pl.DataFrame({"status_num": status_num}))
        pool.cast_features({"status_num": pl.Int64})

        filtered = pool.filter_entities(
            EntityCriterion(query=pl.col("status_num") == 1)
        )
        seq_id = filtered.unique_ids[0]
        seq = filtered[seq_id]

        values = seq.temporal_data(fmt="polars")["status_num"].to_list()
        assert values
        assert all(v == 1 for v in values)
