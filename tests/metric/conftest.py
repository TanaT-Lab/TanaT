#!/usr/bin/env python3
"""
Shared fixtures for tests/metric/.

Registry lists (ENTITY_METRIC_REGISTER_NAMES, SEQUENCE_METRIC_REGISTER_NAMES,
TRAJECTORY_METRIC_REGISTER_NAMES) expose all registered metric names so future
tests can parametrize over them with a single import.
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat import build_trajectories
from tanat.metric.entity.base import EntityMetric
from tanat.metric.sequence.base import SequenceMetric
from tanat.metric.trajectory.base import TrajectoryMetric
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.state.pool import StateSequencePool
from tanat.trajectory.pool import TrajectoryPool

# ---------------------------------------------------------------------------
# Metric registries
# ---------------------------------------------------------------------------

#: All registered entity metric names.
ENTITY_METRIC_REGISTER_NAMES = EntityMetric.list_registered()

#: All registered sequence metric names.
SEQUENCE_METRIC_REGISTER_NAMES = SequenceMetric.list_registered()

#: All registered trajectory metric names.
TRAJECTORY_METRIC_REGISTER_NAMES = TrajectoryMetric.list_registered()

# ---------------------------------------------------------------------------
# cat_pool: all pool types × both temporal variants, 'status' cast to Categorical
# Parametrized: 6 combinations (datetime/timestep × interval/event/state)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="session",
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
    """Any pool type / temporal variant with 'status' cast to pl.Categorical.

    Runs 6× per test (datetime × timestep) × (interval × event × state).
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
    # Keep a small subset of non-empty sequences (IDs 1-10) so that the
    # O(n²) Python pairwise path stays fast.  IDs 51-60 are static-only
    # (length-0) and must be excluded for base metric tests.
    pool = pool.subset(list(range(1, 11)))
    pool.cast_features({"status": pl.Categorical})
    return pool


# ---------------------------------------------------------------------------
# cat_pool_status_only: same as cat_pool but with only 'status' as entity feature
# ---------------------------------------------------------------------------


@pytest.fixture
def cat_pool_status_only(cat_pool):
    """cat_pool restricted to 'status' as the sole entity feature.

    First entity feature is 'status', so entity_feature=None resolves to it.
    """
    pool = cat_pool.copy()
    pool.update_settings(entity_features=["status"])
    return pool


# ---------------------------------------------------------------------------
# empty_cat_pool: pool where every sequence has length 0 (static-only IDs)
# Parametrized: 3 pool types (event / interval / state)
# ---------------------------------------------------------------------------


def _build_empty_pool(pool_cls, name, *, entity_df, time_kwargs):
    """Build a *pool_cls* where visible IDs have length-0 sequences.

    The builder requires at least one entity source, so *entity_df* carries a
    single dummy row.  Only the static IDs survive the final ``subset()``.
    """
    static_df = pl.DataFrame(
        {
            "id": ["empty_1", "empty_2", "empty_3"],
            "status": ["A", "B", "C"],
        }
    )
    store = (
        pool_cls.builder()
        .add_dataframe(
            entity_df,
            id_column="id",
            features=["status"],
            **time_kwargs,
        )
        .add_dataframe(
            static_df,
            id_column="id",
            is_static=True,
            features=["status"],
        )
        .build(name, exist_ok=True)
    )
    pool = pool_cls(store=store)
    pool = pool.subset(["empty_1", "empty_2", "empty_3"])
    pool.cast_features({"status": pl.Categorical})
    return pool


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("event", id="event"),
        pytest.param("interval", id="interval"),
        pytest.param("state", id="state"),
    ],
)
def empty_cat_pool(request):
    """SequencePool where all visible sequences are empty (length 0).

    Built by injecting a single dummy entity row for a throwaway ID,
    plus a static source with the real IDs.  The dummy ID is excluded
    via ``subset()``, leaving only 0-length sequences with ``'status'``
    cast to ``pl.Categorical``.

    Parametrized over the 3 pool types (event / interval / state).
    """
    pool_type = request.param
    if pool_type == "event":
        return _build_empty_pool(
            EventSequencePool,
            "empty_event",
            entity_df=pl.DataFrame({"id": ["__dummy__"], "time": [0], "status": ["X"]}),
            time_kwargs={"time_column": "time"},
        )
    if pool_type == "interval":
        return _build_empty_pool(
            IntervalSequencePool,
            "empty_interval",
            entity_df=pl.DataFrame(
                {"id": ["__dummy__"], "start": [0], "end": [1], "status": ["X"]}
            ),
            time_kwargs={"start_column": "start", "end_column": "end"},
        )
    # state
    return _build_empty_pool(
        StateSequencePool,
        "empty_state",
        entity_df=pl.DataFrame({"id": ["__dummy__"], "start": [0], "status": ["X"]}),
        time_kwargs={"start_column": "start"},
    )


# ---------------------------------------------------------------------------
# Entity metric fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[pytest.param(name, id=name) for name in ENTITY_METRIC_REGISTER_NAMES],
)
def entity_metric(request):
    """One EntityMetric instance per registered type, configured on 'status'.

    Parametrized over all registered entity metrics.
    Adding a new metric automatically includes it here.
    """
    return EntityMetric.get_registered(request.param)(entity_feature="status")


# ---------------------------------------------------------------------------
# Trajectory metric fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=["dt", "ts"],
    ids=["datetime", "timestep"],
)
def small_traj_pool(
    request: pytest.FixtureRequest,
    interval_pool,
    event_pool,
    state_pool,
    interval_pool_ts,
    event_pool_ts,
    state_pool_ts,
) -> TrajectoryPool:
    """Small trajectory pool (IDs 1-10) with 'status' cast to Categorical.

    Built from individual session-scoped sequence pools via
    :func:`~tanat.build_trajectories` — same pattern as ``cat_pool``.
    Parametrized over datetime and timestep variants.
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
    sub = traj.subset(list(range(1, 11)))
    for alias in sub._store_aliases:  # pylint: disable=protected-access
        sub.sequence_pools[alias].cast_features({"status": pl.Categorical})
        sub.sequence_pools[alias].update_settings(entity_features=["status"])
    return sub


@pytest.fixture
def traj_pair(small_traj_pool: TrajectoryPool):
    """A pair (traj_a, traj_b) from small_traj_pool."""
    ids = small_traj_pool.unique_ids
    return small_traj_pool[ids[0]], small_traj_pool[ids[1]]


@pytest.fixture(scope="session")
def disjoint_alias_traj_pool() -> TrajectoryPool:
    """TrajectoryPool whose aliases cover intentionally non-overlapping ID subsets.

    Alias → IDs:

    - ``"events"``    → 1, 6
    - ``"states"``    → 3, 8
    - ``"intervals"`` → 6, 11

    Trajectories for IDs **1** (events-only) and **11** (intervals-only) share
    no common alias, so any pairwise metric call between them must raise
    :exc:`ValueError`.
    """
    event_df = pl.DataFrame(
        {"id": [1, 1, 6, 6], "time": [0, 1, 0, 2], "status": ["A", "B", "A", "C"]}
    )
    state_df = pl.DataFrame({"id": [3, 8], "start": [0, 0], "status": ["A", "B"]})
    interval_df = pl.DataFrame(
        {
            "id": [6, 6, 11, 11],
            "start": [0, 2, 0, 3],
            "end": [2, 4, 3, 6],
            "status": ["A", "B", "C", "A"],
        }
    )

    events_store = (
        EventSequencePool.builder()
        .add_dataframe(
            event_df, id_column="id", features=["status"], time_column="time"
        )
        .build("disjoint_events", exist_ok=True)
    )
    states_store = (
        StateSequencePool.builder()
        .add_dataframe(
            state_df, id_column="id", features=["status"], start_column="start"
        )
        .build("disjoint_states", exist_ok=True)
    )
    intervals_store = (
        IntervalSequencePool.builder()
        .add_dataframe(
            interval_df,
            id_column="id",
            features=["status"],
            start_column="start",
            end_column="end",
        )
        .build("disjoint_intervals", exist_ok=True)
    )

    events = EventSequencePool(store=events_store)
    states = StateSequencePool(store=states_store)
    intervals = IntervalSequencePool(store=intervals_store)
    for pool in (events, states, intervals):
        pool.cast_features({"status": pl.Categorical})
        pool.update_settings(entity_features=["status"])

    return build_trajectories(
        pools={"events": events, "states": states, "intervals": intervals}
    )
