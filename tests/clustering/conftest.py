#!/usr/bin/env python3
"""
Shared fixtures for tests/clustering/.
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.metric.sequence.base import SequenceMetric
from tanat.metric.trajectory.base import TrajectoryMetric
from tanat import build_trajectories
from tanat.trajectory.pool import TrajectoryPool

# ---------------------------------------------------------------------------
# Registry names
# ---------------------------------------------------------------------------

#: All registered sequence metric names.
SEQUENCE_METRIC_NAMES = SequenceMetric.list_registered()

#: All registered trajectory metric names.
TRAJECTORY_METRIC_NAMES = TrajectoryMetric.list_registered()


@pytest.fixture(params=[pytest.param(name, id=name) for name in SEQUENCE_METRIC_NAMES])
def seq_metric_name(request) -> str:
    """One registered sequence metric name per parametrised run."""
    return request.param


@pytest.fixture(
    params=[pytest.param(name, id=name) for name in TRAJECTORY_METRIC_NAMES]
)
def traj_metric_name(request) -> str:
    """One registered trajectory metric name per parametrised run."""
    return request.param


# ---------------------------------------------------------------------------
# cat_pool: all 6 variants × 5 IDs, 'status' cast to Categorical
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(("dt", "interval"), id="datetime-interval"),
        pytest.param(("dt", "event"), id="datetime-event"),
        pytest.param(("dt", "state"), id="datetime-state"),
        pytest.param(("ts", "interval"), id="timestep-interval"),
        pytest.param(("ts", "event"), id="timestep-event"),
        pytest.param(("ts", "state"), id="timestep-state"),
    ],
)
def cat_pool(
    request,
    interval_pool,
    event_pool,
    state_pool,
    interval_pool_ts,
    event_pool_ts,
    state_pool_ts,
):
    """5-sequence pool with 'status' cast to Categorical.

    Parametrised over 6 variants (datetime/timestep × interval/event/state).
    Uses only IDs 1–5 so O(n²) PAM stays instantaneous.
    """
    variant, pool_type = request.param
    mapping = {
        ("dt", "interval"): interval_pool,
        ("dt", "event"): event_pool,
        ("dt", "state"): state_pool,
        ("ts", "interval"): interval_pool_ts,
        ("ts", "event"): event_pool_ts,
        ("ts", "state"): state_pool_ts,
    }
    pool = mapping[(variant, pool_type)].copy()
    pool = pool.subset(list(range(1, 6)))
    pool.cast_features({"status": pl.Categorical})
    pool.update_settings(entity_features=["status"])
    return pool


# ---------------------------------------------------------------------------
# small_traj_pool: TrajectoryPool, 5 IDs, 'status' cast to Categorical
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=["dt", "ts"],
    ids=["datetime", "timestep"],
)
def small_traj_pool(
    request,
    interval_pool,
    event_pool,
    state_pool,
    interval_pool_ts,
    event_pool_ts,
    state_pool_ts,
) -> TrajectoryPool:
    """5-trajectory pool with 'status' cast to Categorical.

    Parametrised over datetime and timestep variants.
    """
    raw = {
        "dt": {"intervals": interval_pool, "events": event_pool, "states": state_pool},
        "ts": {
            "intervals": interval_pool_ts,
            "events": event_pool_ts,
            "states": state_pool_ts,
        },
    }[request.param]

    traj = build_trajectories(pools=raw)
    sub = traj.subset(list(range(1, 6)))
    for alias in sub._store_aliases:  # pylint: disable=protected-access
        sub.sequence_pools[alias].cast_features({"status": pl.Categorical})
        sub.sequence_pools[alias].update_settings(entity_features=["status"])
    return sub
