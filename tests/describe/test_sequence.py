#!/usr/bin/env python3
"""Tests for SequencePool.describe() and Sequence.describe().

Test organisation
-----------------
- ``TestSequencePoolDescribe``: parametrized over [interval, event, state]
  and the dt/ts temporal variants.  Every test uses ``pool_copy``. Safe
  for both read-only and mutating assertions.
- ``TestEvent/State/IntervalStats``: assert type-specific describe columns,
  each parametrized to a single pool type via ``pool_copy``.
- ``TestSequenceDescribe``: individual :class:`Sequence` object, via the
  local ``sequence`` fixture.
"""

from __future__ import annotations

import warnings

import polars as pl
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Column-set constants
# ---------------------------------------------------------------------------

SHARED_COLS = {"length", "n_unique_entities", "temporal_span"}
EVENT_COLS = {"median_gap", "gap_std"}
INTERVAL_COLS = {"mean_duration", "median_duration", "duration_std"}
STATE_COLS = {"mean_duration", "median_duration", "duration_std", "n_transitions"}


# ---------------------------------------------------------------------------
# Pool-level tests (parametrized over type × temporal variant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolDescribe:
    """SequencePool.describe(): all sequence types, both temporal variants."""

    def test_returns_pandas_by_default(self, pool_copy):
        """Default output_format is pandas."""
        result = pool_copy.describe()
        assert isinstance(result, pd.DataFrame)

    def test_returns_polars_when_requested(self, pool_copy):
        """output_format='polars' returns a pl.DataFrame."""
        result = pool_copy.describe(output_format="polars")
        assert isinstance(result, pl.DataFrame)

    def test_by_id_one_row_per_id(self, pool_copy):
        """by_id=True (default) produces exactly one row per sequence ID."""
        result = pool_copy.describe(output_format="polars")
        assert result.height == len(pool_copy.unique_ids)

    def test_by_id_contains_id_column(self, pool_copy):
        """Result always includes the sequence ID column."""
        result = pool_copy.describe(output_format="polars")
        assert pool_copy.settings.id_column in result.columns

    def test_shared_columns_present(self, pool_copy):
        """All types expose the three shared describe columns."""
        result = pool_copy.describe(output_format="polars")
        assert SHARED_COLS.issubset(set(result.columns))

    def test_aggregated_returns_pandas_describe(self, pool_copy):
        """by_id=False returns cross-ID aggregated statistics (count, mean, std, min, 25%, ...)."""
        result = pool_copy.describe(by_id=False)
        assert isinstance(result, pd.DataFrame)
        assert "count" in result.index

    def test_aggregated_has_no_id_column(self, pool_copy):
        """The ID column is dropped before aggregation."""
        result = pool_copy.describe(by_id=False)
        assert pool_copy.settings.id_column not in result.columns

    def test_add_to_static(self, pool_copy):
        """add_to_static=True persists describe columns into the static store."""
        pool_copy.describe(add_to_static=True)
        static = pool_copy.static_data(output_format="polars")
        assert static is not None
        assert "length" in static.columns

    def test_add_to_static_ignored_with_by_id_false(self, pool_copy):
        """add_to_static=True is silently ignored (with a warning) when by_id=False."""
        initial_static = pool_copy.static_data(output_format="polars")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pool_copy.describe(by_id=False, add_to_static=True)
        assert len(w) == 1
        assert "add_to_static=True is ignored" in str(w[0].message)
        # Static features must be unchanged
        new_static = pool_copy.static_data(output_format="polars")
        if initial_static is None:
            assert new_static is None
        else:
            assert new_static.columns == initial_static.columns

    def test_caching_returns_same_object(self, pool_copy):
        """Identical calls return the exact same cached object (no recomputation)."""
        r1 = pool_copy.describe(output_format="polars")
        r2 = pool_copy.describe(output_format="polars")
        assert r1 is r2

    def test_cache_invalidated_after_mutation(self, pool_copy):
        """Cache is cleared after a structural mutation (subset)."""
        r1 = pool_copy.describe(output_format="polars")
        pool_sub = pool_copy.subset(pool_copy.unique_ids[:5])
        r2 = pool_sub.describe(output_format="polars")
        assert r1 is not r2
        assert r2.height == 5

    def test_invalid_output_format_raises(self, pool_copy):
        """An unsupported output_format raises ValueError."""
        with pytest.raises(ValueError, match="output_format"):
            pool_copy.describe(output_format="csv")  # type: ignore[arg-type]

    def test_describe_output_snapshot(self, pool_copy, snapshot):
        """Full describe() output is stable across runs."""
        result = pool_copy.describe()
        assert result.to_csv() == snapshot


# ---------------------------------------------------------------------------
# Type-specific describe columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["event"])
class TestEventStats:
    """EventSequence-specific describe columns: gap stats only."""

    def test_has_gap_columns(self, pool_copy):
        """Event pools expose median_gap and gap_std."""
        result = pool_copy.describe(output_format="polars")
        assert EVENT_COLS.issubset(set(result.columns))

    def test_no_duration_columns(self, pool_copy):
        """Event pools have no duration columns (events are point-in-time)."""
        result = pool_copy.describe(output_format="polars")
        assert "mean_duration" not in result.columns

    def test_no_transition_column(self, pool_copy):
        """Event pools have no n_transitions column."""
        result = pool_copy.describe(output_format="polars")
        assert "n_transitions" not in result.columns


@pytest.mark.parametrize("pool_type", ["state"])
class TestStateStats:
    """StateSequence-specific describe columns: durations + transitions, no gaps."""

    def test_has_duration_and_transition_columns(self, pool_copy):
        """State pools expose all duration stats and n_transitions."""
        result = pool_copy.describe(output_format="polars")
        assert STATE_COLS.issubset(set(result.columns))

    def test_no_gap_columns(self, pool_copy):
        """State pools have no gap columns (states are contiguous, no gaps)."""
        result = pool_copy.describe(output_format="polars")
        assert "median_gap" not in result.columns
        assert "gap_std" not in result.columns


@pytest.mark.parametrize("pool_type", ["interval"])
class TestIntervalStats:
    """IntervalSequence-specific describe columns: durations only."""

    def test_has_duration_columns(self, pool_copy):
        """Interval pools expose all three duration stats."""
        result = pool_copy.describe(output_format="polars")
        assert INTERVAL_COLS.issubset(set(result.columns))

    def test_no_transition_or_gap_columns(self, pool_copy):
        """Interval pools have neither n_transitions nor gap columns."""
        result = pool_copy.describe(output_format="polars")
        assert "n_transitions" not in result.columns
        assert "median_gap" not in result.columns


# ---------------------------------------------------------------------------
# Individual Sequence.describe()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequenceDescribe:
    """Sequence.describe(): individual sequence objects, via the ``sequence`` fixture."""

    def test_returns_one_row(self, sequence):
        """describe() on a single sequence always returns exactly 1 row."""
        result = sequence.describe(output_format="polars")
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_returns_pandas_by_default(self, sequence):
        """Default output_format is pandas with 1 row."""
        result = sequence.describe()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_shared_columns_present(self, sequence):
        """The three shared describe columns are always present (no ID column)."""
        result = sequence.describe(output_format="polars")
        assert SHARED_COLS.issubset(set(result.columns))

    def test_caching_returns_same_object(self, sequence):
        """Repeated calls on the same Sequence return the cached object."""
        r1 = sequence.describe(output_format="polars")
        r2 = sequence.describe(output_format="polars")
        assert r1 is r2

    def test_describe_output_snapshot(self, sequence, snapshot):
        """describe() output for a single sequence is stable across runs."""
        result = sequence.describe()
        assert result.to_csv() == snapshot
