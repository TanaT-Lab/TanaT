#!/usr/bin/env python3
"""Tests: IntervalSequencePool type conversions.

Conversion matrix for IntervalSequencePool (source):
    as_event(anchor)   →  EventSequencePool    (anchor = "start" / "end" / "middle")
    as_state()         →  NotImplementedError  (gaps / overlaps make it ambiguous)
    as_interval()      →  no-op + UserWarning
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tanat import get_workspace
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.zeroing import _T0

# ---------------------------------------------------------------------------
# as_event  (parametrized over anchor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["start", "end", "middle"])
class TestIntervalPoolAsEvent:
    """as_event(anchor) collapses (start, end) to a single event timestamp.

    The ``anchor`` selects which endpoint (or their midpoint) becomes the
    event time.  All three anchors produce an EventSequencePool with the
    same column structure (only the timestamp values differ).
    """

    def test_returns_event_type(self, pools_dict: dict, anchor: str) -> None:
        """Result is an EventSequencePool."""
        pool = pools_dict["interval"]
        assert isinstance(pool.as_event(anchor), EventSequencePool)

    def test_row_count_preserved(self, pools_dict: dict, anchor: str) -> None:
        """Conversion does not add or remove rows."""
        pool = pools_dict["interval"]
        n = pool.temporal_data(fmt="polars").height
        assert pool.as_event(anchor).temporal_data(fmt="polars").height == n

    def test_custom_time_column_name(self, pools_dict: dict, anchor: str) -> None:
        """time_column controls the name of the event timestamp in the output."""
        pool = pools_dict["interval"]
        converted = pool.as_event(anchor, time_column="ts")
        assert "ts" in converted.temporal_data(fmt="polars").columns

    def test_end_column_absent_from_output(self, pools_dict: dict, anchor: str) -> None:
        """The interval end column is not present in the event pool output."""
        pool = pools_dict["interval"]
        converted = pool.as_event(anchor, time_column="time")
        assert (
            pool.settings.end_column
            not in converted.temporal_data(fmt="polars").columns
        )

    def test_entity_features_preserved(self, pools_dict: dict, anchor: str) -> None:
        """Entity feature list is carried over unchanged."""
        pool = pools_dict["interval"]
        converted = pool.as_event(anchor)
        assert sorted(converted.settings.entity_features) == sorted(
            pool.settings.entity_features
        )

    def test_static_features_preserved(self, pools_dict: dict, anchor: str) -> None:
        """Static feature list is carried over unchanged."""
        pool = pools_dict["interval"]
        converted = pool.as_event(anchor)
        assert sorted(converted.settings.static_features) == sorted(
            pool.settings.static_features
        )


class TestIntervalPoolAsEventColumns:
    """Column structure of the converted EventSequencePool (anchor-independent)."""

    def test_columns_snapshot(self, pools_dict: dict, snapshot) -> None:
        """Column set of the converted EventSequencePool matches snapshot."""
        pool = pools_dict["interval"]
        converted = pool.as_event("start", time_column="time")
        assert snapshot == sorted(converted.temporal_data(fmt="polars").columns)


# ---------------------------------------------------------------------------
# as_state  (blocked)
# ---------------------------------------------------------------------------


class TestIntervalPoolAsState:
    """as_state() raises NotImplementedError for IntervalSequencePool.

    Intervals may overlap or contain gaps, which cannot be automatically
    resolved into contiguous non-overlapping states.
    """

    def test_raises_not_implemented(self, pools_dict: dict) -> None:
        """as_state() always raises NotImplementedError."""
        pool = pools_dict["interval"]
        with pytest.raises(NotImplementedError):
            pool.as_state()


# ---------------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------------


class TestIntervalPoolNoop:
    """as_interval() on an IntervalSequencePool is a no-op. Returns self with a UserWarning."""

    def test_noop_returns_self(self, pools_dict: dict) -> None:
        """as_interval() returns the exact same pool object."""
        pool = pools_dict["interval"]
        with pytest.warns(UserWarning):
            result = pool.as_interval()
        assert result is pool

    def test_noop_warns(self, pools_dict: dict) -> None:
        """as_interval() emits a UserWarning mentioning the no-op."""
        pool = pools_dict["interval"]
        with pytest.warns(UserWarning, match="no-op"):
            pool.as_interval()


# ---------------------------------------------------------------------------
# Temporal cast forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["start", "end", "middle"])
class TestIntervalPoolTemporalCast:
    """temporal_cast is forwarded to _fork_period_to_event during as_event().

    When cast_to_timestep(Int64) is set on a Float64 timestep pool, the
    forked virtual event temporal must be written as Int64.  The resulting
    pool must carry no pending temporal cast.
    """

    def test_temporal_dtype(
        self, interval_pool_ts: IntervalSequencePool, anchor: str
    ) -> None:
        """Event time comes out as Int64 when cast_to_timestep(Int64) is pending."""
        pool = interval_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        df = pool.as_event(anchor).temporal_data(fmt="polars")
        assert df["time"].dtype == pl.Int64

    def test_no_pending_cast(
        self, interval_pool_ts: IntervalSequencePool, anchor: str
    ) -> None:
        """After conversion the temporal cast recipe is cleared on the result."""
        pool = interval_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        # pylint: disable=protected-access
        assert pool.as_event(anchor)._casts.time_index == []

    def test_row_count_preserved(
        self, interval_pool_ts: IntervalSequencePool, anchor: str
    ) -> None:
        """Cast + conversion does not add or remove rows."""
        pool = interval_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        n = pool.temporal_data(fmt="polars").height
        assert pool.as_event(anchor).temporal_data(fmt="polars").height == n


# ---------------------------------------------------------------------------
# Persist (datetime variant, function-scoped)
# ---------------------------------------------------------------------------


class TestIntervalPoolPersist:
    """as_event with destination= writes a named store to the workspace."""

    def test_persist_event_reloadable(
        self, interval_pool: IntervalSequencePool, tmp_path: Path
    ) -> None:
        """as_event('start', destination=name) → store reloadable as EventSequencePool."""
        store_name = tmp_path.name
        interval_pool.as_event(
            "start",
            time_column="time",
            destination=store_name,
            overwrite=True,
        )
        reloaded = get_workspace()[store_name]
        assert isinstance(reloaded, EventSequencePool)
        assert len(reloaded) == len(interval_pool)


# ---------------------------------------------------------------------------
# T0 propagation (non-regression for _persist_as)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["start", "end", "middle"])
class TestT0PropagationPersist:
    """
    T0 strategy is preserved through a persistent as_event() conversion.
    """

    def test_t0_propagated_as_event(
        self, interval_pool: IntervalSequencePool, anchor: str, tmp_path: Path
    ) -> None:
        """set_t0(position=2, anchor=anchor) on IntervalPool → persisted EventPool keeps the same T0."""
        pool = interval_pool.copy()
        pool.set_t0(position=2, anchor=anchor)
        source_t0 = pool.t0_data(fmt="polars")

        converted = pool.as_event(
            anchor,
            destination=str(tmp_path / f"t0_event_{anchor}"),
            overwrite=True,
        )
        converted_t0 = converted.t0_data(fmt="polars")

        assert source_t0[_T0].equals(converted_t0[_T0], null_equal=True)
