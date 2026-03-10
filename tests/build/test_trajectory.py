#!/usr/bin/env python3
"""
Tests: TrajectoryPool is correctly built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tanat.trajectory.pool import TrajectoryPool


# ---------------------------------------------------------------------------
# Fixture-driven: canonical pools from the session conftest
# ---------------------------------------------------------------------------


class TestTrajectoryPoolFromFixtures:
    """Fixture-driven: verifies every TrajectoryPool built by the session conftest."""

    def test_len(self, traj_pool, snapshot) -> None:
        """Pool size matches snapshot."""
        assert len(traj_pool) == snapshot

    def test_sequence_pools_aliases(self, traj_pool) -> None:
        """The three expected aliases are present."""
        assert set(traj_pool.sequence_pools.keys()) == {"intervals", "events", "states"}

    def test_static_data_is_none(self, traj_pool) -> None:
        """No trajectory-level static was registered in the fixture."""
        assert traj_pool.static_data(output_format="polars") is None


# ---------------------------------------------------------------------------
# Custom build: trajectory-level static from DataFrames
# Reuses the pre-built sequence pools from the session conftest.
# ---------------------------------------------------------------------------


class TestTrajectoryStaticFromDataFrame:
    """Add trajectory-level static to a pool whose sequences come from the session fixtures."""

    def test_with_static(
        self,
        static_df_fixture,
        interval_pool_ts,
        event_pool_ts,
        state_pool_ts,
        tmp_path: Path,
        snapshot,
    ) -> None:
        """Static schema matches snapshot for all three DF flavours."""
        store = (
            TrajectoryPool.builder()
            .add("intervals", interval_pool_ts)
            .add("events", event_pool_ts)
            .add("states", state_pool_ts)
            .add_dataframe(
                static_df_fixture,
                id_column="id",
                features=["age", "group"],
            )
            .build(tmp_path / "traj_with_static")
        )
        pool = TrajectoryPool(store=store)
        sd = pool.static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# Custom build: trajectory-level static from SQL
# Reuses the pre-built sequence pools from the session conftest.
# ---------------------------------------------------------------------------


class TestTrajectoryStaticFromSQL:
    """Add trajectory-level static via add_sql(); skipped when connectorx is absent."""

    @pytest.fixture(autouse=True)
    def _require_connectorx(self) -> None:
        pytest.importorskip("connectorx")

    def test_with_static(
        self,
        sqlite_db: str,
        interval_pool_ts,
        event_pool_ts,
        state_pool_ts,
        tmp_path: Path,
        snapshot,
    ) -> None:
        """Static schema matches snapshot."""
        store = (
            TrajectoryPool.builder()
            .add("intervals", interval_pool_ts)
            .add("events", event_pool_ts)
            .add("states", state_pool_ts)
            .add_sql(
                sqlite_db,
                "SELECT id, age FROM static_data",
                id_column="id",
                features=["age"],
            )
            .build(tmp_path / "traj_sql_static")
        )
        pool = TrajectoryPool(store=store)
        sd = pool.static_data(output_format="polars")
        assert sd is not None
        assert dict(sd.schema) == snapshot


# ---------------------------------------------------------------------------
# Incompatibility: mixing datetime and timestep pools in the same trajectory
# ---------------------------------------------------------------------------


class TestTrajectoryIncompatiblePools:
    """TrajectoryStoreBuilder.add() raises TypeError when pools have incompatible temporal schemas."""

    def test_dt_then_ts_raises(self, interval_pool, interval_pool_ts) -> None:
        """Adding a timestep pool after a datetime pool raises TypeError."""
        with pytest.raises(TypeError, match="temporal schema"):
            (
                TrajectoryPool.builder()
                .add("dt_pool", interval_pool)
                .add("ts_pool", interval_pool_ts)
            )

    def test_ts_then_dt_raises(self, interval_pool_ts, interval_pool) -> None:
        """Adding a datetime pool after a timestep pool raises TypeError."""
        with pytest.raises(TypeError, match="temporal schema"):
            (
                TrajectoryPool.builder()
                .add("ts_pool", interval_pool_ts)
                .add("dt_pool", interval_pool)
            )

    def test_dt_event_then_ts_state_raises(self, event_pool, state_pool_ts) -> None:
        """Mixing different sequence types with different temporal schemas still raises TypeError."""
        with pytest.raises(TypeError, match="temporal schema"):
            (
                TrajectoryPool.builder()
                .add("events_dt", event_pool)
                .add("states_ts", state_pool_ts)
            )
