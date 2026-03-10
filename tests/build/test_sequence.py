#!/usr/bin/env python3
"""
Tests: all sequence pools are correctly built.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.state.pool import StateSequencePool


# ---------------------------------------------------------------------------
# Fixture-driven: canonical pools from the session conftest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolFromFixtures:
    """Fixture-driven: verifies every pool built by the session conftest."""

    def test_len(self, pools_dict, pool_type, snapshot) -> None:
        """Pool size matches snapshot."""
        assert len(pools_dict[pool_type]) == snapshot

    def test_sequence_data_schema(self, pools_dict, pool_type, snapshot) -> None:
        """sequence_data() column schema matches snapshot."""
        df = pools_dict[pool_type].sequence_data(output_format="polars")
        assert dict(df.schema) == snapshot

    def test_static_data_schema(self, pools_dict, pool_type, snapshot) -> None:
        """static_data() column schema matches snapshot."""
        sd = pools_dict[pool_type].static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# Custom build: from DataFrames (polars df, polars lf, pandas df)
# ---------------------------------------------------------------------------


class TestBuildFromDataFrame:
    """Build each pool type from in-memory DataFrames (polars eager, lazy, pandas)."""

    def test_interval_pool(self, sequence_df_fixture, tmp_path: Path, snapshot) -> None:
        """IntervalSequencePool built from a DataFrame has the expected length and schema."""
        store = (
            IntervalSequencePool.builder()
            .add_dataframe(
                sequence_df_fixture,
                id_column="id",
                start_column="start",
                end_column="end",
                features=["value", "status", "flag_valid", "duration"],
            )
            .build(tmp_path / "interval_pool")
        )
        pool = IntervalSequencePool(store=store)
        assert len(pool) == snapshot
        assert dict(pool.sequence_data(output_format="polars").schema) == snapshot

    def test_event_pool(self, sequence_df_fixture, tmp_path: Path, snapshot) -> None:
        """EventSequencePool built from a DataFrame has the expected length and schema."""
        store = (
            EventSequencePool.builder()
            .add_dataframe(
                sequence_df_fixture,
                id_column="id",
                time_column="start",
                features=["value", "status", "flag_valid", "duration"],
            )
            .build(tmp_path / "event_pool")
        )
        pool = EventSequencePool(store=store)
        assert len(pool) == snapshot
        assert dict(pool.sequence_data(output_format="polars").schema) == snapshot

    def test_state_pool(self, sequence_df_fixture, tmp_path: Path, snapshot) -> None:
        """StateSequencePool built from a DataFrame has the expected length and schema."""
        store = (
            StateSequencePool.builder()
            .add_dataframe(
                sequence_df_fixture,
                id_column="id",
                start_column="start",
                features=["value", "status", "flag_valid", "duration"],
            )
            .build(tmp_path / "state_pool")
        )
        pool = StateSequencePool(store=store)
        assert len(pool) == snapshot
        assert dict(pool.sequence_data(output_format="polars").schema) == snapshot

    def test_with_static(
        self, static_df_fixture, sequence_data_pl, tmp_path: Path, snapshot
    ) -> None:
        """Static schema is exposed after registering static via add_dataframe()."""
        store = (
            IntervalSequencePool.builder()
            .add_dataframe(
                sequence_data_pl,
                id_column="id",
                start_column="start",
                end_column="end",
                features=["value"],
            )
            .add_dataframe(
                static_df_fixture,
                id_column="id",
                is_static=True,
                features=["age", "group"],
            )
            .build(tmp_path / "interval_with_static")
        )
        pool = IntervalSequencePool(store=store)
        sd = pool.static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# Custom build: from SQL (skipped when connectorx is absent)
# ---------------------------------------------------------------------------


class TestBuildFromSQL:
    """Build each pool type from a SQLite database via add_sql()."""

    @pytest.fixture(autouse=True)
    def _require_connectorx(self) -> None:
        pytest.importorskip("connectorx")

    def test_interval_pool(self, sqlite_db: str, tmp_path: Path, snapshot) -> None:
        """IntervalSequencePool built from SQL has the expected length and schema."""
        store = (
            IntervalSequencePool.builder()
            .add_sql(
                sqlite_db,
                "SELECT id, start, end, value, status FROM sequence_data",
                id_column="id",
                start_column="start",
                end_column="end",
                features=["value", "status"],
            )
            .build(tmp_path / "sql_interval")
        )
        pool = IntervalSequencePool(store=store)
        assert len(pool) == snapshot
        assert dict(pool.sequence_data(output_format="polars").schema) == snapshot

    def test_event_pool(self, sqlite_db: str, tmp_path: Path, snapshot) -> None:
        """EventSequencePool built from SQL has the expected length and schema."""
        store = (
            EventSequencePool.builder()
            .add_sql(
                sqlite_db,
                "SELECT id, start, value FROM sequence_data",
                id_column="id",
                time_column="start",
                features=["value"],
            )
            .build(tmp_path / "sql_event")
        )
        pool = EventSequencePool(store=store)
        assert len(pool) == snapshot
        assert dict(pool.sequence_data(output_format="polars").schema) == snapshot

    def test_state_pool(self, sqlite_db: str, tmp_path: Path, snapshot) -> None:
        """StateSequencePool built from SQL has the expected length and schema."""
        store = (
            StateSequencePool.builder()
            .add_sql(
                sqlite_db,
                "SELECT id, start, value FROM sequence_data",
                id_column="id",
                start_column="start",
                features=["value"],
            )
            .build(tmp_path / "sql_state")
        )
        pool = StateSequencePool(store=store)
        assert len(pool) == snapshot
        assert dict(pool.sequence_data(output_format="polars").schema) == snapshot

    def test_with_static(self, sqlite_db: str, tmp_path: Path, snapshot) -> None:
        """Static schema is exposed after registering static via add_sql()."""
        store = (
            IntervalSequencePool.builder()
            .add_sql(
                sqlite_db,
                "SELECT id, start, end, value FROM sequence_data",
                id_column="id",
                start_column="start",
                end_column="end",
                features=["value"],
            )
            .add_sql(
                sqlite_db,
                "SELECT id, age FROM static_data",
                id_column="id",
                is_static=True,
                features=["age"],
            )
            .build(tmp_path / "sql_with_static")
        )
        pool = IntervalSequencePool(store=store)
        sd = pool.static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# Builder options: sort_anchor (interval) and end_value / validate_continuity (state)
# ---------------------------------------------------------------------------


class TestBuilderOptions:
    """Exercise type-specific builder options for IntervalSequencePool and StateSequencePool."""

    # --- IntervalSequencePool: sort_anchor ---

    @pytest.mark.parametrize("anchor", ["start", "end", "middle"])
    def test_interval_sort_anchor(
        self, sequence_data_pl, tmp_path: Path, anchor: str, snapshot
    ) -> None:
        """sequence_data() row order reflects the chosen sort_anchor, locked by snapshot."""
        store = (
            IntervalSequencePool.builder(sort_anchor=anchor)
            .add_dataframe(
                sequence_data_pl,
                id_column="id",
                start_column="start",
                end_column="end",
                features=["value"],
            )
            .build(tmp_path / f"interval_{anchor}")
        )
        df = IntervalSequencePool(store=store).sequence_data(output_format="polars")
        assert snapshot == df

    def test_interval_sort_anchor_invalid(self) -> None:
        """An unrecognised sort_anchor raises ValueError at builder instantiation."""
        with pytest.raises(ValueError, match="sort_anchor"):
            IntervalSequencePool.builder(sort_anchor="invalid")

    # --- StateSequencePool: end_value ---

    def test_state_without_end_column_last_null(self, tmp_path: Path, snapshot) -> None:
        """Without end_column and no end_value, sequence_data() shows null end for last state."""
        df = pl.DataFrame(
            {"id": [1, 1, 2], "start": [0.0, 1.0, 0.0], "value": ["a", "b", "x"]}
        )
        store = (
            StateSequencePool.builder()
            .add_dataframe(
                df,
                id_column="id",
                start_column="start",
                features=["value"],
            )
            .build(tmp_path / "state_no_end")
        )
        data = StateSequencePool(store=store).sequence_data(output_format="polars")
        assert snapshot == data

    def test_state_end_value_fills_last_end(self, tmp_path: Path, snapshot) -> None:
        """end_value sentinel appears in sequence_data() for the last state of every entity."""
        df = pl.DataFrame(
            {"id": [1, 1, 2], "start": [0.0, 1.0, 0.0], "value": ["a", "b", "x"]}
        )
        store = (
            StateSequencePool.builder(end_value=99.0)
            .add_dataframe(
                df,
                id_column="id",
                start_column="start",
                features=["value"],
            )
            .build(tmp_path / "state_end_value")
        )
        data = StateSequencePool(store=store).sequence_data(output_format="polars")
        assert snapshot == data

    # --- StateSequencePool: validate_continuity ---

    def test_state_continuity_violation_raises(self, tmp_path: Path) -> None:
        """validate_continuity=True raises ValueError when states have gaps."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1],
                "start": [
                    0.0,
                    1.0,
                    3.0,
                ],  # gap: end of state 1 (1.5) ≠ start of state 2 (3.0)
                "end": [1.5, 2.5, 4.0],
                "value": ["a", "b", "c"],
            }
        )
        with pytest.raises(ValueError, match="continuity"):
            (
                StateSequencePool.builder(validate_continuity=True)
                .add_dataframe(
                    df,
                    id_column="id",
                    start_column="start",
                    end_column="end",
                    features=["value"],
                )
                .build(tmp_path / "state_gaps")
            )

    def test_state_continuity_skip_builds(self, tmp_path: Path, snapshot) -> None:
        """validate_continuity=False accepts non-contiguous data; sequence_data() locked by snapshot."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1],
                "start": [
                    0.0,
                    1.0,
                    3.0,
                ],  # gap: end of state 1 (1.5) ≠ start of state 2 (3.0)
                "end": [1.5, 2.5, 4.0],
                "value": ["a", "b", "c"],
            }
        )
        store = (
            StateSequencePool.builder(validate_continuity=False)
            .add_dataframe(
                df,
                id_column="id",
                start_column="start",
                end_column="end",
                features=["value"],
            )
            .build(tmp_path / "state_skip_continuity")
        )
        data = StateSequencePool(store=store).sequence_data(output_format="polars")
        assert snapshot == data
