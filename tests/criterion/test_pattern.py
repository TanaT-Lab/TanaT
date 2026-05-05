#!/usr/bin/env python3
"""Tests: PatternCriterion.

PatternCriterion filters on ordered patterns of string values in temporal data.
Supported levels: ENTITY, SEQUENCE.

Pattern semantics recap:
    ["error", "ok"]           : 'error' directly followed by 'ok' (adjacent)
    ["error", ANY, "ok"]      : 'error' anywhere before 'ok' (free gap)
    ["error", WILDCARD, "ok"] : 'error', exactly one element, then 'ok'
    present=False             : sequences that do NOT contain the pattern
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.criterion.type.pattern import ANY, WILDCARD, PatternCriterion

from .conftest import _temporal_ids

# ---------------------------------------------------------------------------
# which(): sequence-level selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestPatternCriterionWhich:
    """which() selects sequences containing (or not) the ordered pattern."""

    def test_which_single_element(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Single-element pattern matches any sequence with ≥1 'error' row."""
        pool = pools_dict[pool_type]
        result = pool.which(PatternCriterion(feature="status", pattern="error"))
        assert snapshot == result

    def test_which_adjacent(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """Adjacent pattern ['error','ok'] matches the snapshot."""
        pool = pools_dict[pool_type]
        result = pool.which(PatternCriterion(feature="status", pattern=["error", "ok"]))
        assert snapshot == result

    def test_which_any_gap(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """['error', ANY, 'ok'] matches any ID with error before ok (free gap)."""
        pool = pools_dict[pool_type]
        result = pool.which(
            PatternCriterion(feature="status", pattern=["error", ANY, "ok"])
        )
        assert snapshot == result

    def test_which_any_gap_is_superset_of_adjacent(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Free-gap result ⊇ adjacent result (more permissive)."""
        pool = pools_dict[pool_type]
        adj = pool.which(PatternCriterion(feature="status", pattern=["error", "ok"]))
        gap = pool.which(
            PatternCriterion(feature="status", pattern=["error", ANY, "ok"])
        )
        assert adj.issubset(gap)

    def test_which_wildcard(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """['error', WILDCARD, 'ok'] matches sequences where error, one element, ok."""
        pool = pools_dict[pool_type]
        result = pool.which(
            PatternCriterion(feature="status", pattern=["error", WILDCARD, "ok"])
        )
        assert snapshot == result

    def test_which_present_false_is_complement(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """present=True and present=False partition the full ID set."""
        pool = pools_dict[pool_type]
        ids_yes = pool.which(
            PatternCriterion(feature="status", pattern=["error", "ok"])
        )
        ids_no = pool.which(
            PatternCriterion(feature="status", pattern=["error", "ok"], present=False)
        )
        assert ids_yes | ids_no == _temporal_ids(pool)
        assert ids_yes & ids_no == set()

    def test_which_regex(self, pools_dict: dict, pool_type: str) -> None:
        """Regex pattern 'err.*' is equivalent to the literal 'error' match."""
        pool = pools_dict[pool_type]
        literal = pool.which(PatternCriterion(feature="status", pattern="error"))
        regex = pool.which(
            PatternCriterion(feature="status", pattern="err.*", regex=True)
        )
        assert regex == literal

    def test_which_case_insensitive(self, pools_dict: dict, pool_type: str) -> None:
        """case_sensitive=False matches regardless of case."""
        pool = pools_dict[pool_type]
        sensitive = pool.which(PatternCriterion(feature="status", pattern="error"))
        insensitive = pool.which(
            PatternCriterion(
                feature="status", pattern="ERROR", case_sensitive=False, regex=False
            )
        )
        assert insensitive == sensitive


# ---------------------------------------------------------------------------
# match(): single-sequence evaluation
# ---------------------------------------------------------------------------


class TestPatternCriterionMatch:
    """match() returns True iff the sequence contains the ordered pattern."""

    def test_match_adjacent_true(self, seq_with_adjacent_error_ok) -> None:
        """Sequence with adjacent error→ok matches the adjacent pattern."""
        criterion = PatternCriterion(feature="status", pattern=["error", "ok"])
        assert seq_with_adjacent_error_ok.match(criterion) is True

    def test_match_adjacent_false(self, seq_without_error) -> None:
        """Sequence without 'error' does not match the adjacent pattern."""
        criterion = PatternCriterion(feature="status", pattern=["error", "ok"])
        assert seq_without_error.match(criterion) is False

    def test_match_present_false_inverts(self, seq_without_error) -> None:
        """present=False: sequence without the pattern returns True."""
        criterion = PatternCriterion(
            feature="status", pattern=["error", "ok"], present=False
        )
        assert seq_without_error.match(criterion) is True

    def test_match_single_element(self, seq_with_error, seq_without_error) -> None:
        """Single-element pattern matches iff the value appears at least once."""
        c = PatternCriterion(feature="status", pattern="error")
        assert seq_with_error.match(c) is True
        assert seq_without_error.match(c) is False


# ---------------------------------------------------------------------------
# filter_entities(): entity-level row pruning (witness rows only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event"])
class TestPatternCriterionFilterEntities:
    """filter_entities() keeps only the greedy-first-match witness rows."""

    def test_filter_witnesses_snapshot(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Witness rows after adjacent pattern filter match snapshot."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(
            PatternCriterion(feature="status", pattern=["error", "ok"])
        )
        assert snapshot == filtered.temporal_data(fmt="polars")

    def test_filter_witness_count_le_pattern_length(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Each ID in the filtered pool has at most len(pattern) witness rows."""
        pool = pools_dict[pool_type]
        pattern = ["error", "ok"]
        filtered = pool.filter_entities(
            PatternCriterion(feature="status", pattern=pattern)
        )
        id_col = pool.settings.id_column
        counts = (
            filtered.temporal_data(fmt="polars")
            .group_by(id_col)
            .agg(pl.len().alias("n"))
        )
        assert (counts["n"] <= len(pattern)).all()

    def test_filter_present_false_excludes_witnesses(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """present=False: all non-witness rows are kept (snapshot)."""
        pool = pools_dict[pool_type]
        filtered = pool.filter_entities(
            PatternCriterion(feature="status", pattern=["error", "ok"], present=False)
        )
        assert snapshot == filtered.temporal_data(fmt="polars")

    def test_filter_does_not_mutate_original(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """filter_entities() returns a new view; original pool is unchanged."""
        pool = pools_dict[pool_type]
        original_rows = pool.describe(fmt="polars")["length"].sum()
        pool.filter_entities(
            PatternCriterion(feature="status", pattern=["error", "ok"])
        )
        assert pool.describe(fmt="polars")["length"].sum() == original_rows

    def test_filter_id_without_match_has_no_rows(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """An ID with no pattern match contributes 0 rows with present=True."""
        pool = pools_dict[pool_type]
        # Use an impossible pattern
        filtered = pool.filter_entities(
            PatternCriterion(
                feature="status", pattern=["error", "error", "error", "error", "error"]
            )
        )
        assert filtered.temporal_data(fmt="polars").height == 0


# ---------------------------------------------------------------------------
# filter_entities() guard on StateSequencePool
# ---------------------------------------------------------------------------


class TestPatternCriterionStateGuard:
    """filter_entities() is not supported on StateSequencePool nor StateSequence."""

    def test_filter_entities_raises_on_state_pool(self, pools_dict: dict) -> None:
        """StateSequencePool.filter_entities() must raise TypeError."""
        pool = pools_dict["state"]
        with pytest.raises(TypeError, match="filter_entities"):
            pool.filter_entities(PatternCriterion(feature="status", pattern="error"))

    def test_filter_entities_raises_on_state_sequence(self, state_seq) -> None:
        """StateSequence.filter_entities() must raise TypeError."""
        with pytest.raises(TypeError, match="filter_entities"):
            state_seq.filter_entities(
                PatternCriterion(feature="status", pattern="error")
            )
