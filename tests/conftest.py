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
def interval_store(data_dir: Path) -> Path:
    """Store path for the datetime IntervalSequencePool (two entity sources + shared static)."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    return (
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


@pytest.fixture(scope="session")
def interval_pool(interval_store: Path) -> IntervalSequencePool:
    """IntervalSequencePool built from datetime/ (two sequence sources + shared static)."""
    return IntervalSequencePool(store=interval_store)


@pytest.fixture(scope="session")
def event_store(data_dir: Path) -> Path:
    """Store path for the datetime EventSequencePool (uses start as event time)."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    return (
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


@pytest.fixture(scope="session")
def event_pool(event_store: Path) -> EventSequencePool:
    """EventSequencePool built from datetime/ (uses start as event time)."""
    return EventSequencePool(store=event_store)


@pytest.fixture(scope="session")
def state_store(data_dir: Path) -> Path:
    """Store path for the datetime StateSequencePool (contiguous non-overlapping states)."""
    seq = data_dir / "datetime"
    sta = data_dir / "static"
    return (
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


@pytest.fixture(scope="session")
def state_pool(state_store: Path) -> StateSequencePool:
    """StateSequencePool built from datetime/ (contiguous non-overlapping states)."""
    return StateSequencePool(store=state_store)


# ---------------------------------------------------------------------------
# Sequence pools: timestep variant (float day-offset T_START / T_END)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def interval_store_ts(data_dir: Path) -> Path:
    """Store path for the timestep IntervalSequencePool (float timestamps, no observed_at)."""
    seq = data_dir / "timestep"
    sta = data_dir / "static"
    return (
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


@pytest.fixture(scope="session")
def interval_pool_ts(interval_store_ts: Path) -> IntervalSequencePool:
    """IntervalSequencePool built from timestep/ (float timestamps, no observed_at)."""
    return IntervalSequencePool(store=interval_store_ts)


@pytest.fixture(scope="session")
def event_store_ts(data_dir: Path) -> Path:
    """Store path for the timestep EventSequencePool."""
    seq = data_dir / "timestep"
    sta = data_dir / "static"
    return (
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


@pytest.fixture(scope="session")
def event_pool_ts(event_store_ts: Path) -> EventSequencePool:
    """EventSequencePool built from timestep/."""
    return EventSequencePool(store=event_store_ts)


@pytest.fixture(scope="session")
def state_store_ts(data_dir: Path) -> Path:
    """Store path for the timestep StateSequencePool."""
    seq = data_dir / "timestep"
    sta = data_dir / "static"
    return (
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


@pytest.fixture(scope="session")
def state_pool_ts(state_store_ts: Path) -> StateSequencePool:
    """StateSequencePool built from timestep/."""
    return StateSequencePool(store=state_store_ts)


# ---------------------------------------------------------------------------
# Trajectory pools (dt + ts) — declared after all sequence pool dependencies
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trajectory_store_dt(
    interval_pool: IntervalSequencePool,
    event_pool: EventSequencePool,
    state_pool: StateSequencePool,
    data_dir: Path,
) -> Path:
    """Store path for the datetime TrajectoryPool (three datetime sequence pools + static)."""
    sta = data_dir / "static"
    return (
        TrajectoryPool.builder()
        .add("intervals", interval_pool)
        .add("events", event_pool)
        .add("states", state_pool)
        .add_csv(
            sta / "static.csv",
            id_column="id",
            features=["age", "group", "is_active"],
        )
        .build("trajectory_pool_dt")
    )


@pytest.fixture(scope="session")
def trajectory_pool_dt(trajectory_store_dt: Path) -> TrajectoryPool:
    """TrajectoryPool built from the datetime sequence pools, with trajectory-level static features."""
    return TrajectoryPool(store=trajectory_store_dt)


@pytest.fixture(scope="session")
def trajectory_store_ts(
    interval_pool_ts: IntervalSequencePool,
    event_pool_ts: EventSequencePool,
    state_pool_ts: StateSequencePool,
    data_dir: Path,
) -> Path:
    """Store path for the timestep TrajectoryPool (three timestep sequence pools + static)."""
    sta = data_dir / "static"
    return (
        TrajectoryPool.builder()
        .add("intervals", interval_pool_ts)
        .add("events", event_pool_ts)
        .add("states", state_pool_ts)
        .add_csv(
            sta / "static.csv",
            id_column="id",
            features=["age", "group", "is_active"],
        )
        .build("trajectory_pool_ts")
    )


@pytest.fixture(scope="session")
def trajectory_pool_ts(trajectory_store_ts: Path) -> TrajectoryPool:
    """TrajectoryPool built from the timestep sequence pools, with trajectory-level static features."""
    return TrajectoryPool(store=trajectory_store_ts)


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


@pytest.fixture(scope="session")
def sequence_stores_dt(
    interval_store: Path,
    event_store: Path,
    state_store: Path,
) -> dict:
    """Dict of the three datetime-variant sequence store paths."""
    return {
        "interval": interval_store,
        "event": event_store,
        "state": state_store,
    }


@pytest.fixture(scope="session")
def sequence_stores_ts(
    interval_store_ts: Path,
    event_store_ts: Path,
    state_store_ts: Path,
) -> dict:
    """Dict of the three timestep-variant sequence store paths."""
    return {
        "interval": interval_store_ts,
        "event": event_store_ts,
        "state": state_store_ts,
    }


# ---------------------------------------------------------------------------
# Parametrized convenience fixtures (usable anywhere in tests/)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["dt", "ts"], ids=["datetime", "timestep"])
def pools_dict(request: pytest.FixtureRequest) -> dict:
    """Parametrized over datetime/timestep; yields the matching sequence_pools_* dict."""
    return request.getfixturevalue(f"sequence_pools_{request.param}")


@pytest.fixture(params=["dt", "ts"], ids=["datetime", "timestep"])
def stores_dict(request: pytest.FixtureRequest) -> dict:
    """Parametrized over datetime/timestep; yields the matching sequence_stores_* dict."""
    return request.getfixturevalue(f"sequence_stores_{request.param}")


@pytest.fixture(params=["dt", "ts"], ids=["datetime", "timestep"])
def traj_pool(request: pytest.FixtureRequest) -> TrajectoryPool:
    """Parametrized over datetime/timestep; yields the matching trajectory_pool_*."""
    return request.getfixturevalue(f"trajectory_pool_{request.param}")


@pytest.fixture(params=["dt", "ts"], ids=["datetime", "timestep"])
def traj_store(request: pytest.FixtureRequest) -> Path:
    """Parametrized over datetime/timestep; yields the matching trajectory_store_*."""
    return request.getfixturevalue(f"trajectory_store_{request.param}")
