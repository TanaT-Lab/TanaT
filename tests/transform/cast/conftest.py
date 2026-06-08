#!/usr/bin/env python3
"""Fixtures for cast recipe tests.

Provides:

* ``cast_pools_dict``: dict mapping ``pool_type`` (``"interval"``,
  ``"event"``, ``"state"``) to a session-scoped :class:`SequencePool` with
  intentionally mixed-castability entity data.

* ``strict_traj``: :class:`TrajectoryPool` with mixed-castability static
  data, built on top of the interval pool.

All fixtures are **session-scoped**.  Any test that mutates a pool must
call ``.copy()`` first to avoid cross-test interference.

Data layout
-----------
Entity (all pool types)::

    id=1 : ["10", "20"]  (both castable to Int64)
    id=2 : ["bad", "40"] ("bad" is non-castable)
    id=3 : ["50"]        (castable)

Trajectory static (``strict_traj`` only)::

    id=1 → "1.5"  (valid Float32)
    id=2 → "bad"  (non-castable)
    id=3 → "3.5"  (valid Float32)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.state.pool import StateSequencePool
from tanat.trajectory.pool import TrajectoryPool

# ---------------------------------------------------------------------------
# Shared entity parquet
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _cast_entity_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the shared entity parquet once for the entire test session."""
    path = tmp_path_factory.mktemp("strict_cast_entity") / "entity.parquet"
    pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "start": [0, 1, 0, 1, 0],
            "end": [1, 2, 1, 2, 1],
            "age_str": ["10", "20", "bad", "40", "50"],
        }
    ).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# One pool per sequence type (private, consumed by cast_pools_dict)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _strict_interval_pool(_cast_entity_parquet: Path) -> IntervalSequencePool:
    store = (
        IntervalSequencePool.builder()
        .add_parquet(
            _cast_entity_parquet,
            id_column="id",
            start_column="start",
            end_column="end",
            features=["age_str"],
        )
        .build("strict_cast_interval_pool")
    )
    return IntervalSequencePool(store=store)


@pytest.fixture(scope="session")
def _strict_event_pool(_cast_entity_parquet: Path) -> EventSequencePool:
    store = (
        EventSequencePool.builder()
        .add_parquet(
            _cast_entity_parquet,
            id_column="id",
            time_column="start",
            features=["age_str"],
        )
        .build("strict_cast_event_pool")
    )
    return EventSequencePool(store=store)


@pytest.fixture(scope="session")
def _strict_state_pool(_cast_entity_parquet: Path) -> StateSequencePool:
    store = (
        StateSequencePool.builder()
        .add_parquet(
            _cast_entity_parquet,
            id_column="id",
            start_column="start",
            features=["age_str"],
        )
        .build("strict_cast_state_pool")
    )
    return StateSequencePool(store=store)


# ---------------------------------------------------------------------------
# Public: dict of all sequence pool types
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cast_pools_dict(
    _strict_interval_pool: IntervalSequencePool,
    _strict_event_pool: EventSequencePool,
    _strict_state_pool: StateSequencePool,
) -> dict:
    """Session-scoped dict mapping pool_type → mixed-castability SequencePool."""
    return {
        "interval": _strict_interval_pool,
        "event": _strict_event_pool,
        "state": _strict_state_pool,
    }


# ---------------------------------------------------------------------------
# Trajectory pool (uses interval as the linked sequence pool)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def strict_traj(
    _strict_interval_pool: IntervalSequencePool,
    tmp_path_factory: pytest.TempPathFactory,
) -> TrajectoryPool:
    """Session-scoped TrajectoryPool with mixed-castability static data."""
    tmp = tmp_path_factory.mktemp("strict_cast_traj")
    static_path = tmp / "traj_static.parquet"
    pl.DataFrame(
        {
            "id": [1, 2, 3],
            "score_str": ["1.5", "bad", "3.5"],
        }
    ).write_parquet(static_path)
    store = (
        TrajectoryPool.builder()
        .add("sequences", _strict_interval_pool)
        .add_parquet(static_path, id_column="id", features=["score_str"])
        .build("strict_cast_traj_pool")
    )
    return TrajectoryPool(store=store)
