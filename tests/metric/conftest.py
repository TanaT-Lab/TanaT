#!/usr/bin/env python3
"""
Shared fixtures for tests/metric/.

Registry lists (ENTITY_METRIC_REGISTER_NAMES, SEQUENCE_METRIC_REGISTER_NAMES) expose all registered
metric names so future tests can parametrize over them with a single import.
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.metric.entity.base import EntityMetric
from tanat.metric.sequence.base import SequenceMetric
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.sequence.type.state.pool import StateSequencePool

# ---------------------------------------------------------------------------
# Metric registries
# ---------------------------------------------------------------------------

#: All registered entity metric names.
ENTITY_METRIC_REGISTER_NAMES = EntityMetric.list_registered()

#: All registered sequence metric names.
SEQUENCE_METRIC_REGISTER_NAMES = SequenceMetric.list_registered()

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
