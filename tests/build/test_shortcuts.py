#!/usr/bin/env python3
"""
Tests: quick-build helper functions.
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat import (
    build_events,
    build_intervals,
    build_states,
    build_trajectories,
)
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.state.pool import StateSequencePool
from tanat.trajectory.pool import TrajectoryPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Column names present in the test data files (see build/conftest.py)
# sequence_main.parquet : id, start, end, value, status, flag_valid, duration
# static.csv           : id, age, group, is_active, membership_duration

_SEQ_ID = "id"
_SEQ_START = "start"
_SEQ_END = "end"
_SEQ_TIME = "start"  # reuse "start" as the event timestamp


# ---------------------------------------------------------------------------
# 1) build_events
# ---------------------------------------------------------------------------


class TestBuildEvents:
    """Tests for build_events(): single timestamp per row."""

    def test_build_events(self, temporal_df_fixture, snapshot) -> None:
        """Returns EventSequencePool; length and temporal schema match snapshot."""
        pool = build_events(
            temporal_data=temporal_df_fixture,
            id_column=_SEQ_ID,
            time_column=_SEQ_TIME,
        )
        assert isinstance(pool, EventSequencePool)
        assert len(pool) == snapshot
        assert dict(pool.temporal_data(output_format="polars").schema) == snapshot

    def test_build_events_with_static(
        self, temporal_df_fixture, static_df_fixture, snapshot
    ) -> None:
        """Static schema includes all inferred features."""
        pool = build_events(
            temporal_data=temporal_df_fixture,
            id_column=_SEQ_ID,
            time_column=_SEQ_TIME,
            static_data=static_df_fixture,
        )
        assert isinstance(pool, EventSequencePool)
        sd = pool.static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# 2) build_intervals
# ---------------------------------------------------------------------------


class TestBuildIntervals:
    """Tests for build_intervals(): start/end pair per row."""

    def test_build_intervals(self, temporal_df_fixture, snapshot) -> None:
        """Returns IntervalSequencePool; length and temporal schema match snapshot."""
        pool = build_intervals(
            temporal_data=temporal_df_fixture,
            id_column=_SEQ_ID,
            start_column=_SEQ_START,
            end_column=_SEQ_END,
        )
        assert isinstance(pool, IntervalSequencePool)
        assert len(pool) == snapshot
        assert dict(pool.temporal_data(output_format="polars").schema) == snapshot

    def test_build_intervals_with_static(
        self, temporal_df_fixture, static_df_fixture, snapshot
    ) -> None:
        """Static schema matches snapshot."""
        pool = build_intervals(
            temporal_data=temporal_df_fixture,
            id_column=_SEQ_ID,
            start_column=_SEQ_START,
            end_column=_SEQ_END,
            static_data=static_df_fixture,
        )
        assert isinstance(pool, IntervalSequencePool)
        sd = pool.static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# 3) build_states
# ---------------------------------------------------------------------------


class TestBuildStates:
    """Tests for build_states(): contiguous state sequences."""

    def test_build_states_no_end(self, temporal_df_fixture, snapshot) -> None:
        """Returns StateSequencePool; end auto-derived; schema matches snapshot."""
        # Drop the 'end' column so it is not treated as a feature: the
        # StatePool already adds a derived 'end' column from the time index,
        # and having both would cause a DuplicateError.
        if isinstance(temporal_df_fixture, pl.LazyFrame):
            data = temporal_df_fixture.drop("end")
        elif isinstance(temporal_df_fixture, pl.DataFrame):
            data = temporal_df_fixture.drop("end")
        else:
            data = temporal_df_fixture.drop(columns=["end"])
        pool = build_states(
            temporal_data=data,
            id_column=_SEQ_ID,
            start_column=_SEQ_START,
        )
        assert isinstance(pool, StateSequencePool)
        assert len(pool) == snapshot
        assert dict(pool.temporal_data(output_format="polars").schema) == snapshot

    def test_build_states_with_end(self, snapshot) -> None:
        """Explicit end column respected; schema matches snapshot."""
        # Use a small inline DataFrame with genuinely contiguous states
        # (end[i] == start[i+1]), which is the requirement for StateSequencePool.
        contiguous_df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "start": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "end": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "value": [10, 20, 30, 40, 50, 60],
                "status": ["a", "b", "c", "d", "e", "f"],
            }
        )
        pool = build_states(
            temporal_data=contiguous_df,
            id_column="id",
            start_column="start",
            end_column="end",
        )
        assert isinstance(pool, StateSequencePool)
        assert len(pool) == snapshot
        assert dict(pool.temporal_data(output_format="polars").schema) == snapshot


# ---------------------------------------------------------------------------
# 4 – build_trajectories  (non-parametrized: fixed polars format)
# ---------------------------------------------------------------------------


class TestBuildTrajectories:
    """
    Tests for build_trajectories(): composition of pre-built sequence pools.
    """

    @pytest.fixture()
    def interval_pool(self, temporal_data_pl):
        """An IntervalSequencePool built from the session temporal data."""
        return build_intervals(
            temporal_data=temporal_data_pl,
            id_column=_SEQ_ID,
            start_column=_SEQ_START,
            end_column=_SEQ_END,
        )

    @pytest.fixture()
    def event_pool(self, temporal_data_pl):
        """An EventSequencePool built from the session temporal data."""
        return build_events(
            temporal_data=temporal_data_pl,
            id_column=_SEQ_ID,
            time_column=_SEQ_TIME,
        )

    def test_build_trajectories(self, interval_pool, event_pool, snapshot) -> None:
        """Returns TrajectoryPool with correct aliases and length snapshot."""
        tpool = build_trajectories(
            pools={"admissions": interval_pool, "events": event_pool},
        )
        assert isinstance(tpool, TrajectoryPool)
        assert set(tpool.sequence_pools.keys()) == {"admissions", "events"}
        assert len(tpool) == snapshot

    def test_build_trajectories_with_static(
        self, interval_pool, event_pool, static_data_pl, snapshot
    ) -> None:
        """Static data is attached to the trajectory pool."""
        tpool = build_trajectories(
            pools={"admissions": interval_pool, "events": event_pool},
            static_data=static_data_pl,
            id_column=_SEQ_ID,
        )
        assert isinstance(tpool, TrajectoryPool)
        sd = tpool.static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# 5 – Validation error cases
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Tests that invalid inputs raise ValueError with informative messages."""

    def test_missing_id_column_raises(self, temporal_data_pl) -> None:
        """ValueError when id_column is absent from temporal_data."""
        with pytest.raises(ValueError, match="missing"):
            build_intervals(
                temporal_data=temporal_data_pl,
                id_column="nonexistent_id",
                start_column=_SEQ_START,
                end_column=_SEQ_END,
            )

    def test_missing_time_column_raises(self, temporal_data_pl) -> None:
        """ValueError when time_column is absent from temporal_data."""
        with pytest.raises(ValueError, match="missing"):
            build_events(
                temporal_data=temporal_data_pl,
                id_column=_SEQ_ID,
                time_column="nonexistent_time",
            )

    def test_no_features_raises(self) -> None:
        """DataFrame with only structural columns raises ValueError."""
        two_col_df = pl.DataFrame({"id": [1, 2], "start": [0.0, 1.0]})
        with pytest.raises(ValueError, match="[Nn]o feature"):
            build_events(
                temporal_data=two_col_df,
                id_column="id",
                time_column="start",
            )

    def test_custom_store_name(self, temporal_data_pl) -> None:
        """Store is created with the user-provided name."""
        custom_name = "my_custom_event_store"
        pool = build_events(
            temporal_data=temporal_data_pl,
            id_column=_SEQ_ID,
            time_column=_SEQ_TIME,
            store_name=custom_name,
        )
        assert isinstance(pool, EventSequencePool)
        # The store path must contain the custom name as a component
        # pylint: disable=protected-access
        assert custom_name in str(pool._store.root_path)

    def test_static_data_without_id_column_raises(self, temporal_data_pl) -> None:
        """ValueError on build_trajectories when static_data given without id_column."""
        pool = build_intervals(
            temporal_data=temporal_data_pl,
            id_column=_SEQ_ID,
            start_column=_SEQ_START,
            end_column=_SEQ_END,
        )
        static_without_id = pl.DataFrame({"age": [30, 40], "group": ["A", "B"]})
        with pytest.raises(ValueError, match="id_column"):
            build_trajectories(
                pools={"main": pool},
                static_data=static_without_id,
                # id_column intentionally omitted
            )

    def test_build_states_forbidden_end_column_raises(self) -> None:
        """ValueError when temporal_data has an 'end' column and end_column is not passed."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "start": [0.0, 1.0, 0.0, 1.0],
                "end": [1.0, 2.0, 1.0, 2.0],   # present but NOT declared
                "value": ["a", "b", "x", "y"],
            }
        )
        with pytest.raises(ValueError, match="end_column"):
            build_states(
                temporal_data=df,
                id_column="id",
                start_column="start",
                # end_column intentionally omitted, 'end' should be flagged
            )

    def test_build_states_custom_end_column_name_forbidden_raises(self) -> None:
        """ValueError when data has 'end' column and end_column is not passed, regardless of start name."""
        df = pl.DataFrame(
            {
                "id": [1, 1],
                "t_start": [0.0, 1.0],
                "end": [1.0, 2.0],   # shadows the default output name
                "value": ["a", "b"],
            }
        )
        with pytest.raises(ValueError, match="end_column"):
            build_states(
                temporal_data=df,
                id_column="id",
                start_column="t_start",
                # end_column not passed → output end col defaults to "end"
            )
