#!/usr/bin/env python3
"""Tests: StateSequencePool type conversions.

Conversion matrix for StateSequencePool (source):
    as_interval()    →  IntervalSequencePool  (zero I/O, pure type relabelling)
    as_event(anchor) →  EventSequencePool     (anchor = "start" / "end" / "middle")
    as_state()       →  no-op + UserWarning
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tanat import get_workspace
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.state.pool import StateSequencePool
from tanat.zeroing import _T0

# ---------------------------------------------------------------------------
# as_interval  (zero I/O reinterpretation)
# ---------------------------------------------------------------------------


class TestStatePoolAsInterval:
    """as_interval() reinterprets a StateSequencePool as an IntervalSequencePool.

    States and intervals share the same ``(start, end)`` physical layout.
    The conversion is a pure type relabelling with no temporal recomputation
    and no extra I/O.
    """

    def test_returns_interval_type(self, pools_dict: dict) -> None:
        """Result is an IntervalSequencePool."""
        pool = pools_dict["state"]
        assert isinstance(pool.as_interval(), IntervalSequencePool)

    def test_row_count_preserved(self, pools_dict: dict) -> None:
        """Zero-I/O conversion preserves row count exactly."""
        pool = pools_dict["state"]
        n = pool.sequence_data(output_format="polars").height
        assert pool.as_interval().sequence_data(output_format="polars").height == n

    def test_unique_ids_preserved(self, pools_dict: dict) -> None:
        """The set of sequence IDs is identical between state and interval views."""
        pool = pools_dict["state"]
        assert set(pool.as_interval().unique_ids) == set(pool.unique_ids)

    def test_temporal_values_identical(self, pools_dict: dict) -> None:
        """Start and end values are byte-for-byte identical after pure type relabelling."""
        pool = pools_dict["state"]
        converted = pool.as_interval()
        start_col = pool.settings.start_column
        end_col = pool.settings.end_column
        state_data = pool.sequence_data(output_format="polars")
        interval_data = converted.sequence_data(output_format="polars")
        assert state_data[start_col].equals(interval_data[start_col])
        assert state_data[end_col].equals(interval_data[end_col])

    def test_entity_features_preserved(self, pools_dict: dict) -> None:
        """Entity feature list is carried over unchanged."""
        pool = pools_dict["state"]
        converted = pool.as_interval()
        assert sorted(converted.settings.entity_features) == sorted(
            pool.settings.entity_features
        )

    def test_static_features_preserved(self, pools_dict: dict) -> None:
        """Static feature list is carried over unchanged."""
        pool = pools_dict["state"]
        converted = pool.as_interval()
        assert sorted(converted.settings.static_features) == sorted(
            pool.settings.static_features
        )

    def test_custom_start_end_column_names(self, pools_dict: dict) -> None:
        """start_column / end_column override the temporal column names in the output."""
        pool = pools_dict["state"]
        converted = pool.as_interval(start_column="t_start", end_column="t_end")
        cols = converted.sequence_data(output_format="polars").columns
        assert "t_start" in cols
        assert "t_end" in cols

    def test_columns_snapshot(self, pools_dict: dict, snapshot) -> None:
        """Column set of the converted IntervalSequencePool matches snapshot."""
        pool = pools_dict["state"]
        assert snapshot == sorted(
            pool.as_interval().sequence_data(output_format="polars").columns
        )


# ---------------------------------------------------------------------------
# as_event  (parametrized over anchor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["start", "end", "middle"])
class TestStatePoolAsEvent:
    """as_event(anchor) collapses (start, end) to a single event timestamp.

    Identical semantics to IntervalSequencePool.as_event() since states and
    intervals share the same physical layout.
    """

    def test_returns_event_type(self, pools_dict: dict, anchor: str) -> None:
        """Result is an EventSequencePool."""
        pool = pools_dict["state"]
        assert isinstance(pool.as_event(anchor), EventSequencePool)

    def test_row_count_preserved(self, pools_dict: dict, anchor: str) -> None:
        """Conversion does not add or remove rows."""
        pool = pools_dict["state"]
        n = pool.sequence_data(output_format="polars").height
        assert pool.as_event(anchor).sequence_data(output_format="polars").height == n

    def test_custom_time_column_name(self, pools_dict: dict, anchor: str) -> None:
        """time_column controls the name of the event timestamp in the output."""
        pool = pools_dict["state"]
        converted = pool.as_event(anchor, time_column="ts")
        assert "ts" in converted.sequence_data(output_format="polars").columns

    def test_end_column_absent_from_output(self, pools_dict: dict, anchor: str) -> None:
        """The state end column is not present in the event pool output."""
        pool = pools_dict["state"]
        converted = pool.as_event(anchor, time_column="time")
        assert (
            pool.settings.end_column
            not in converted.sequence_data(output_format="polars").columns
        )

    def test_entity_features_preserved(self, pools_dict: dict, anchor: str) -> None:
        """Entity feature list is carried over unchanged."""
        pool = pools_dict["state"]
        converted = pool.as_event(anchor)
        assert sorted(converted.settings.entity_features) == sorted(
            pool.settings.entity_features
        )

    def test_static_features_preserved(self, pools_dict: dict, anchor: str) -> None:
        """Static feature list is carried over unchanged."""
        pool = pools_dict["state"]
        converted = pool.as_event(anchor)
        assert sorted(converted.settings.static_features) == sorted(
            pool.settings.static_features
        )


# ---------------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------------


class TestStatePoolNoop:
    """as_state() on a StateSequencePool is a no-op. Returns self with a UserWarning."""

    def test_noop_returns_self(self, pools_dict: dict) -> None:
        """as_state() returns the exact same pool object."""
        pool = pools_dict["state"]
        with pytest.warns(UserWarning):
            result = pool.as_state()
        assert result is pool

    def test_noop_warns(self, pools_dict: dict) -> None:
        """as_state() emits a UserWarning mentioning the no-op."""
        pool = pools_dict["state"]
        with pytest.warns(UserWarning, match="no-op"):
            pool.as_state()


# ---------------------------------------------------------------------------
# Temporal cast forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["start", "end", "middle"])
class TestStatePoolTemporalCast:
    """temporal_cast is forwarded to _fork_period_to_event during as_event().

    When cast_to_timestep(Int64) is set on a Float64 timestep pool, the
    forked virtual event temporal must be written as Int64.  The resulting
    pool must carry no pending temporal cast.
    """

    def test_temporal_dtype(
        self, state_pool_ts: StateSequencePool, anchor: str
    ) -> None:
        """Event time comes out as Int64 when cast_to_timestep(Int64) is pending."""
        pool = state_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        df = pool.as_event(anchor).sequence_data(output_format="polars")
        assert df["time"].dtype == pl.Int64

    def test_no_pending_cast(
        self, state_pool_ts: StateSequencePool, anchor: str
    ) -> None:
        """After conversion the temporal cast recipe is cleared on the result."""
        pool = state_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        # pylint: disable=protected-access
        assert pool.as_event(anchor)._casts.temporal is None

    def test_row_count_preserved(
        self, state_pool_ts: StateSequencePool, anchor: str
    ) -> None:
        """Cast + conversion does not add or remove rows."""
        pool = state_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        n = pool.sequence_data(output_format="polars").height
        assert pool.as_event(anchor).sequence_data(output_format="polars").height == n


# ---------------------------------------------------------------------------
# Persist (datetime variant, function-scoped)
# ---------------------------------------------------------------------------


class TestStatePoolPersist:
    """as_interval / as_event with destination= writes a named store to the workspace."""

    def test_persist_interval_reloadable(
        self, state_pool: StateSequencePool, tmp_path: Path
    ) -> None:
        """as_interval(destination=name) → store reloadable as IntervalSequencePool."""
        store_name = tmp_path.name
        state_pool.as_interval(destination=store_name, overwrite=True)
        reloaded = get_workspace()[store_name]
        assert isinstance(reloaded, IntervalSequencePool)
        assert len(reloaded) == len(state_pool)

    def test_persist_event_reloadable(
        self, state_pool: StateSequencePool, tmp_path: Path
    ) -> None:
        """as_event('start', destination=name) → store reloadable as EventSequencePool."""
        store_name = tmp_path.name
        state_pool.as_event(
            "start", time_column="time", destination=store_name, overwrite=True
        )
        reloaded = get_workspace()[store_name]
        assert isinstance(reloaded, EventSequencePool)
        assert len(reloaded) == len(state_pool)


# ---------------------------------------------------------------------------
# T0 propagation (non-regression for _persist_as)
# ---------------------------------------------------------------------------


class TestT0PropagationPersist:
    """
    T0 strategy is preserved through a persistent type conversion.
    """

    def test_t0_propagated_as_event(
        self, state_pool: StateSequencePool, tmp_path: Path
    ) -> None:
        """set_t0(position=2, anchor='start') on StatePool → persisted EventPool keeps the same T0."""
        pool = state_pool.copy()
        pool.set_t0(position=2, anchor="start")
        source_t0 = pool.t0_data(output_format="polars")

        converted = pool.as_event(
            "start",
            destination=str(tmp_path / "t0_event"),
            overwrite=True,
        )
        converted_t0 = converted.t0_data(output_format="polars")

        # T0 values must be identical after a persist conversion.
        assert source_t0[_T0].equals(converted_t0[_T0], null_equal=True)
