#!/usr/bin/env python3
"""Tests: EventSequencePool type conversions.

Conversion matrix for EventSequencePool (source):
    as_interval(duration)  →  IntervalSequencePool  (timedelta / numeric / feature col)
    as_state(end_value)    →  StateSequencePool     (next-event shift)
    as_event()             →  no-op + UserWarning
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from tanat import get_workspace
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.state.pool import StateSequencePool
from tanat.zeroing import _T0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _duration(pool: EventSequencePool):
    """Return the right duration type for a given pool's temporal variant."""
    return timedelta(days=7) if pool.metadata.time_index.is_datetime else 7.0


def _sentinel(pool: EventSequencePool):
    """Return a closed-end sentinel compatible with the pool's temporal variant."""
    return datetime(2200, 12, 31) if pool.metadata.time_index.is_datetime else 99999.0


# ---------------------------------------------------------------------------
# as_interval
# ---------------------------------------------------------------------------


class TestEventPoolAsInterval:
    """as_interval(duration) converts an EventSequencePool to an IntervalSequencePool.

    Each event timestamp becomes ``start``; ``end`` is computed as
    ``start + duration``.  Duration can be a timedelta (datetime pools),
    a numeric scalar (timestep pools), or the name of a per-row feature column.
    """

    def test_returns_interval_type(self, pools_dict: dict) -> None:
        """Result is an IntervalSequencePool."""
        pool = pools_dict["event"]
        assert isinstance(
            pool.as_interval(duration=_duration(pool)), IntervalSequencePool
        )

    def test_row_count_preserved(self, pools_dict: dict) -> None:
        """Conversion does not add or remove rows."""
        pool = pools_dict["event"]
        n = pool.temporal_data(output_format="polars").height
        converted = pool.as_interval(duration=_duration(pool))
        assert converted.temporal_data(output_format="polars").height == n

    def test_unique_ids_preserved(self, pools_dict: dict) -> None:
        """All sequence IDs from the source appear in the converted pool."""
        pool = pools_dict["event"]
        converted = pool.as_interval(duration=_duration(pool))
        assert set(converted.unique_ids) == set(pool.unique_ids)

    def test_custom_start_end_column_names(self, pools_dict: dict) -> None:
        """start_column / end_column control the output time column names."""
        pool = pools_dict["event"]
        converted = pool.as_interval(
            duration=_duration(pool), start_column="t_start", end_column="t_end"
        )
        cols = converted.temporal_data(output_format="polars").columns
        assert "t_start" in cols
        assert "t_end" in cols

    def test_entity_features_preserved(self, pools_dict: dict) -> None:
        """Entity feature list is carried over unchanged."""
        pool = pools_dict["event"]
        converted = pool.as_interval(duration=_duration(pool))
        assert sorted(converted.settings.entity_features) == sorted(
            pool.settings.entity_features
        )

    def test_static_features_preserved(self, pools_dict: dict) -> None:
        """Static feature list is carried over unchanged."""
        pool = pools_dict["event"]
        converted = pool.as_interval(duration=_duration(pool))
        assert sorted(converted.settings.static_features) == sorted(
            pool.settings.static_features
        )

    def test_feature_col_duration(self, pools_dict: dict, snapshot) -> None:
        """as_interval(duration='duration') uses the per-row entity feature column.

        On datetime pools the 'duration' column is stored as Int64 (CSV source), so
        it must be cast to pl.Duration first via cast_features().  The store reads
        entity features with the cast applied at query time, so no save() is needed.
        On timestep pools the column is already numeric.
        """
        pool = pools_dict["event"].copy()
        if pool.metadata.time_index.is_datetime:
            pool.cast_features({"duration": pl.Duration("ms")})
        converted = pool.as_interval(duration="duration")
        assert isinstance(converted, IntervalSequencePool)
        df = converted.temporal_data(output_format="polars")
        assert snapshot == df.select(sorted(df.columns))
        assert converted.temporal_data(output_format="polars").height == (
            pool.temporal_data(output_format="polars").height
        )

    def test_feature_col_duration_uncast_raises(self, pools_dict: dict) -> None:
        """Passing an Int64 column name as duration raises TypeError on datetime pools.

        The datetime test pool stores 'duration' as Int64.  The API must reject it
        because a plain integer column is neither a timedelta nor a Duration column.
        """
        pool = pools_dict["event"]
        if not pool.metadata.time_index.is_datetime:
            pytest.skip("Int64 guard-rail only applies to datetime pools")
        with pytest.raises(TypeError):
            pool.as_interval(duration="duration")

    def test_timedelta_invalid_on_timestep(
        self, event_pool_ts: EventSequencePool
    ) -> None:
        """timedelta is not valid for timestep (non-datetime) pools. Should raises ValueError."""
        with pytest.raises(ValueError, match="(?i)timedelta"):
            event_pool_ts.as_interval(duration=timedelta(days=7))

    def test_numeric_scalar_invalid_on_datetime(
        self, event_pool: EventSequencePool
    ) -> None:
        """Numeric scalar is not valid for datetime pools. Should raises ValueError."""
        with pytest.raises(ValueError):
            event_pool.as_interval(duration=7.0)

    def test_columns_snapshot(self, pools_dict: dict, snapshot) -> None:
        """Column set of the converted IntervalSequencePool matches snapshot."""
        pool = pools_dict["event"]
        converted = pool.as_interval(
            duration=_duration(pool), start_column="start", end_column="end"
        )
        assert snapshot == sorted(
            converted.temporal_data(output_format="polars").columns
        )


# ---------------------------------------------------------------------------
# as_state
# ---------------------------------------------------------------------------


class TestEventPoolAsState:
    """as_state() converts an EventSequencePool to a StateSequencePool.

    Each event timestamp becomes ``start``; ``end`` is taken from the **next**
    event in the same sequence (shift-based).  The last row's end is set to
    ``end_value`` (``None`` → null; explicit value → closed end).
    """

    def test_returns_state_type(self, pools_dict: dict) -> None:
        """Result is a StateSequencePool."""
        pool = pools_dict["event"]
        assert isinstance(pool.as_state(), StateSequencePool)

    def test_row_count_preserved(self, pools_dict: dict) -> None:
        """Conversion does not add or remove rows."""
        pool = pools_dict["event"]
        n = pool.temporal_data(output_format="polars").height
        assert pool.as_state().temporal_data(output_format="polars").height == n

    def test_last_end_null_when_none(self, pools_dict: dict) -> None:
        """end_value=None → the last event per sequence gets a null end."""
        pool = pools_dict["event"]
        converted = pool.as_state(end_value=None, end_column="end")
        data = converted.temporal_data(output_format="polars")
        n_ids = data[pool.settings.id_column].n_unique()
        null_count = data["end"].is_null().sum()
        assert null_count == n_ids

    def test_last_end_closed_with_sentinel(self, pools_dict: dict) -> None:
        """end_value provided → no null ends; every row has a closed end."""
        pool = pools_dict["event"]
        converted = pool.as_state(end_value=_sentinel(pool), end_column="end")
        assert (
            converted.temporal_data(output_format="polars")["end"].is_null().sum() == 0
        )

    def test_entity_features_preserved(self, pools_dict: dict) -> None:
        """Entity feature list is carried over unchanged."""
        pool = pools_dict["event"]
        converted = pool.as_state()
        assert sorted(converted.settings.entity_features) == sorted(
            pool.settings.entity_features
        )

    def test_static_features_preserved(self, pools_dict: dict) -> None:
        """Static feature list is carried over unchanged."""
        pool = pools_dict["event"]
        converted = pool.as_state()
        assert sorted(converted.settings.static_features) == sorted(
            pool.settings.static_features
        )


# ---------------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------------


class TestEventPoolNoop:
    """as_event() on an EventSequencePool is a no-op. Returns self with a UserWarning."""

    def test_noop_returns_self(self, pools_dict: dict) -> None:
        """as_event() returns the exact same pool object."""
        pool = pools_dict["event"]
        with pytest.warns(UserWarning):
            result = pool.as_event()
        assert result is pool

    def test_noop_warns(self, pools_dict: dict) -> None:
        """as_event() emits a UserWarning mentioning the no-op."""
        pool = pools_dict["event"]
        with pytest.warns(UserWarning, match="no-op"):
            pool.as_event()


# ---------------------------------------------------------------------------
# Persist (datetime variant, function-scoped)
# ---------------------------------------------------------------------------


class TestEventPoolPersist:
    """as_interval / as_state with destination= writes a named store to the workspace."""

    def test_persist_interval_reloadable(
        self, event_pool: EventSequencePool, tmp_path: Path
    ) -> None:
        """as_interval(destination=name) → store reloadable as IntervalSequencePool."""
        store_name = tmp_path.name
        event_pool.as_interval(
            duration=timedelta(days=7),
            start_column="start",
            end_column="end",
            destination=store_name,
            overwrite=True,
        )
        reloaded = get_workspace()[store_name]
        assert isinstance(reloaded, IntervalSequencePool)
        assert len(reloaded) == len(event_pool)

    def test_persist_state_reloadable(
        self, event_pool: EventSequencePool, tmp_path: Path
    ) -> None:
        """as_state(destination=name) → store reloadable as StateSequencePool."""
        store_name = tmp_path.name
        event_pool.as_state(
            end_value=None,
            start_column="start",
            end_column="end",
            destination=store_name,
            overwrite=True,
        )
        reloaded = get_workspace()[store_name]
        assert isinstance(reloaded, StateSequencePool)
        assert len(reloaded) == len(event_pool)


# ---------------------------------------------------------------------------
# Temporal cast forwarding
# ---------------------------------------------------------------------------


class TestEventPoolTemporalCast:
    """temporal_cast is applied inside the fork helpers during type conversions.

    When cast_to_timestep(Int64) is set on a Float64 timestep pool, the forked
    virtual temporal index must be written as Int64 (not as the raw Float64
    stored on disk).  The resulting pool must carry no pending temporal cast
    (recipe.time_index is None), since the cast was already materialised in the fork.
    """

    # -- as_interval ----------------------------------------------------------

    def test_as_interval_temporal_dtype(self, event_pool_ts: EventSequencePool) -> None:
        """start/end come out as Int64 when cast_to_timestep(Int64) is pending."""
        pool = event_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        df = pool.as_interval(duration=7).temporal_data(output_format="polars")
        assert df["start"].dtype == pl.Int64
        assert df["end"].dtype == pl.Int64

    def test_as_interval_end_equals_start_plus_duration(
        self, event_pool_ts: EventSequencePool
    ) -> None:
        """end = start + duration for every row (scalar integer duration)."""
        pool = event_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        df = pool.as_interval(duration=7).temporal_data(output_format="polars")
        assert ((df["end"] - df["start"]) == 7).all()

    def test_as_interval_no_pending_cast(
        self, event_pool_ts: EventSequencePool
    ) -> None:
        """After conversion the temporal cast recipe is cleared on the result."""
        pool = event_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        # pylint: disable=protected-access
        assert pool.as_interval(duration=7)._casts.time_index == []

    # -- as_state -------------------------------------------------------------

    def test_as_state_temporal_dtype(self, event_pool_ts: EventSequencePool) -> None:
        """start/end come out as Int64 when cast_to_timestep(Int64) is pending."""
        pool = event_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        df = pool.as_state().temporal_data(output_format="polars")
        assert df["start"].dtype == pl.Int64
        assert df["end"].dtype == pl.Int64

    def test_as_state_no_pending_cast(self, event_pool_ts: EventSequencePool) -> None:
        """After conversion the temporal cast recipe is cleared on the result."""
        pool = event_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        # pylint: disable=protected-access
        assert pool.as_state()._casts.time_index == []

    # -- common ---------------------------------------------------------------

    def test_row_count_preserved(self, event_pool_ts: EventSequencePool) -> None:
        """Cast + conversion does not add or remove rows."""
        pool = event_pool_ts.copy()
        pool.cast_to_timestep(pl.Int64)
        n = pool.temporal_data(output_format="polars").height
        assert (
            pool.as_interval(duration=7).temporal_data(output_format="polars").height
            == n
        )
        assert pool.as_state().temporal_data(output_format="polars").height == n


# ---------------------------------------------------------------------------
# T0 propagation (non-regression for _persist_as)
# ---------------------------------------------------------------------------


class TestT0PropagationPersist:
    """
    T0 strategy is preserved through a persistent type conversion.
    """

    def test_t0_propagated_as_interval(
        self, event_pool: EventSequencePool, tmp_path: Path
    ) -> None:
        """set_t0(position=2) on EventPool → persisted IntervalPool keeps the same T0."""
        pool = event_pool.copy()
        pool.set_t0(position=2)
        source_t0 = pool.t0_data(output_format="polars")

        converted = pool.as_interval(
            duration=_duration(pool),
            destination=str(tmp_path / "t0_interval"),
            overwrite=True,
        )
        converted_t0 = converted.t0_data(output_format="polars")

        # T0 values must be identical after a persist conversion.
        assert source_t0[_T0].equals(converted_t0[_T0], null_equal=True)

    def test_t0_propagated_as_state(
        self, event_pool: EventSequencePool, tmp_path: Path
    ) -> None:
        """set_t0(position=2) on EventPool → persisted StatePool keeps the same T0."""
        pool = event_pool.copy()
        pool.set_t0(position=2)
        source_t0 = pool.t0_data(output_format="polars")

        converted = pool.as_state(
            destination=str(tmp_path / "t0_state"),
            overwrite=True,
        )
        converted_t0 = converted.t0_data(output_format="polars")

        assert source_t0[_T0].equals(converted_t0[_T0], null_equal=True)
