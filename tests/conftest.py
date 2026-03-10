#!/usr/bin/env python3
"""
Shared session-scoped fixtures for the TanaT test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tanat import set_workspace, get_workspace
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.state.pool import StateSequencePool
from tanat.trajectory.pool import TrajectoryPool

if TYPE_CHECKING:
    from tanat.core.workspace import Workspace
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Absolute path to tests/data/ (exposes static/, datetime/, timestep/ subdirs)."""
    return Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """Isolated workspace in a temp directory, cleared and torn down after the session.

    autouse=True ensures this runs before any other session-scoped fixture
    (including the pool builders), so all stores land in the temp workspace,
    never in ~/.tanat_workspace.
    """
    ws_path = tmp_path_factory.mktemp("tanat_workspace")
    set_workspace(str(ws_path))
    ws = get_workspace()
    ws.clear()
    return ws


# ---------------------------------------------------------------------------
# Sequence pools: datetime variant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def interval_pool(data_dir: Path) -> IntervalSequencePool:
    """IntervalSequencePool built from datetime/ (two sequence sources + shared static)."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    store_path = (
        IntervalSequencePool.builder()
        # Entity sources (non-overlapping IDs: main=1-35, extra=36-50)
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            start_column="start",
            end_column="end",
            features=[
                "value",
                "status",
                "flag_valid",
                "token_emb",
                "observed_at",
                "duration",
            ],
        )
        .add_csv(
            seq / "sequence_extra.csv",
            id_column="id",
            start_column="start",
            end_column="end",
            features=["value", "status", "response_time", "duration"],
        )
        # Static sources (shared folder)
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .add_parquet(
            sta / "static_embeddings.parquet",
            id_column="id",
            is_static=True,
            features=["summary_embedding"],
        )
        .build("interval_pool")
    )
    return IntervalSequencePool(store=store_path)


@pytest.fixture(scope="session")
def event_pool(data_dir: Path) -> EventSequencePool:
    """EventSequencePool built from datetime/ (uses start as event time)."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    store_path = (
        EventSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            time_column="start",
            features=[
                "value",
                "status",
                "flag_valid",
                "token_emb",
                "observed_at",
                "duration",
            ],
        )
        .add_csv(
            seq / "sequence_extra.csv",
            id_column="id",
            time_column="start",
            features=["value", "status", "response_time", "duration"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .add_parquet(
            sta / "static_embeddings.parquet",
            id_column="id",
            is_static=True,
            features=["summary_embedding"],
        )
        .build("event_pool")
    )
    return EventSequencePool(store=store_path)


@pytest.fixture(scope="session")
def state_pool(data_dir: Path) -> StateSequencePool:
    """StateSequencePool built from datetime/ (contiguous non-overlapping states)."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    store_path = (
        StateSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            start_column="start",
            features=[
                "value",
                "status",
                "flag_valid",
                "token_emb",
                "observed_at",
                "duration",
            ],
        )
        .add_csv(
            seq / "sequence_extra.csv",
            id_column="id",
            start_column="start",
            features=["value", "status", "response_time", "duration"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .add_parquet(
            sta / "static_embeddings.parquet",
            id_column="id",
            is_static=True,
            features=["summary_embedding"],
        )
        .build("state_pool")
    )
    return StateSequencePool(store=store_path)


# ---------------------------------------------------------------------------
# Trajectory pool
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trajectory_pool_dt(
    interval_pool: IntervalSequencePool,
    event_pool: EventSequencePool,
    state_pool: StateSequencePool,
) -> TrajectoryPool:
    """TrajectoryPool built from the datetime sequence pools."""
    store_path = (
        TrajectoryPool.builder()
        .add("intervals", interval_pool)
        .add("events", event_pool)
        .add("states", state_pool)
        .build("trajectory_pool_dt")
    )
    return TrajectoryPool(store=store_path)


@pytest.fixture(scope="session")
def trajectory_pool_ts(
    interval_pool_ts: IntervalSequencePool,
    event_pool_ts: EventSequencePool,
    state_pool_ts: StateSequencePool,
) -> TrajectoryPool:
    """TrajectoryPool built from the timestep sequence pools."""
    store_path = (
        TrajectoryPool.builder()
        .add("intervals", interval_pool_ts)
        .add("events", event_pool_ts)
        .add("states", state_pool_ts)
        .build("trajectory_pool_ts")
    )
    return TrajectoryPool(store=store_path)


# ---------------------------------------------------------------------------
# Sequence pools: timestep variant (float day-offset T_START / T_END)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def interval_pool_ts(data_dir: Path) -> IntervalSequencePool:
    """IntervalSequencePool built from timestep/ (float timestamps, no observed_at)."""
    seq = data_dir / "timestep"
    sta = data_dir / "static"
    store_path = (
        IntervalSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            start_column="start",
            end_column="end",
            features=["value", "status", "flag_valid", "token_emb", "duration"],
        )
        .add_csv(
            seq / "sequence_extra.csv",
            id_column="id",
            start_column="start",
            end_column="end",
            features=["value", "status", "response_time", "duration"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .add_parquet(
            sta / "static_embeddings.parquet",
            id_column="id",
            is_static=True,
            features=["summary_embedding"],
        )
        .build("interval_pool_ts")
    )
    return IntervalSequencePool(store=store_path)


@pytest.fixture(scope="session")
def event_pool_ts(data_dir: Path) -> EventSequencePool:
    """EventSequencePool built from timestep/."""
    seq = data_dir / "timestep"
    sta = data_dir / "static"
    store_path = (
        EventSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            time_column="start",
            features=["value", "status", "flag_valid", "token_emb", "duration"],
        )
        .add_csv(
            seq / "sequence_extra.csv",
            id_column="id",
            time_column="start",
            features=["value", "status", "response_time", "duration"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .add_parquet(
            sta / "static_embeddings.parquet",
            id_column="id",
            is_static=True,
            features=["summary_embedding"],
        )
        .build("event_pool_ts")
    )
    return EventSequencePool(store=store_path)


@pytest.fixture(scope="session")
def state_pool_ts(data_dir: Path) -> StateSequencePool:
    """StateSequencePool built from timestep/."""
    seq = data_dir / "timestep"
    sta = data_dir / "static"
    store_path = (
        StateSequencePool.builder()
        .add_parquet(
            seq / "sequence_main.parquet",
            id_column="id",
            start_column="start",
            features=["value", "status", "flag_valid", "token_emb", "duration"],
        )
        .add_csv(
            seq / "sequence_extra.csv",
            id_column="id",
            start_column="start",
            features=["value", "status", "response_time", "duration"],
        )
        .add_csv(
            sta / "static.csv",
            id_column="id",
            is_static=True,
            features=["age", "group", "is_active", "membership_duration"],
        )
        .add_parquet(
            sta / "static_embeddings.parquet",
            id_column="id",
            is_static=True,
            features=["summary_embedding"],
        )
        .build("state_pool_ts")
    )
    return StateSequencePool(store=store_path)


# ---------------------------------------------------------------------------
# Combined dicts: sequence pools grouped by temporal variant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sequence_pools_dt(
    interval_pool: IntervalSequencePool,
    event_pool: EventSequencePool,
    state_pool: StateSequencePool,
) -> dict:
    """Dict of the three datetime-variant sequence pools."""
    return {
        "interval": interval_pool,
        "event": event_pool,
        "state": state_pool,
    }


@pytest.fixture(scope="session")
def sequence_pools_ts(
    interval_pool_ts: IntervalSequencePool,
    event_pool_ts: EventSequencePool,
    state_pool_ts: StateSequencePool,
) -> dict:
    """Dict of the three timestep-variant sequence pools."""
    return {
        "interval": interval_pool_ts,
        "event": event_pool_ts,
        "state": state_pool_ts,
    }


# ---------------------------------------------------------------------------
# Parametrized convenience fixtures (usable anywhere in tests/)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["dt", "ts"], ids=["datetime", "timestep"])
def pools_dict(request: pytest.FixtureRequest) -> dict:
    """Parametrized over datetime/timestep; yields the matching sequence_pools_* dict."""
    return request.getfixturevalue(f"sequence_pools_{request.param}")


@pytest.fixture(params=["dt", "ts"], ids=["datetime", "timestep"])
def traj_pool(request: pytest.FixtureRequest) -> TrajectoryPool:
    """Parametrized over datetime/timestep; yields the matching trajectory_pool_*."""
    return request.getfixturevalue(f"trajectory_pool_{request.param}")
