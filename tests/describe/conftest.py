#!/usr/bin/env python3
"""Local fixtures for tests/describe/.

Tests that call describe(add_to_static=True) mutate the pool's virtual
store and would corrupt other tests if they shared the session-scoped
instance.  ``pool_copy`` and ``traj_pool_copy`` provide fresh copies.
"""

from __future__ import annotations

import pytest

from tanat.sequence.type.interval.sequence import IntervalSequence
from tanat.sequence.type.event.sequence import EventSequence
from tanat.sequence.type.state.sequence import StateSequence

_SEQ_CLS: dict = {
    "interval": IntervalSequence,
    "event": EventSequence,
    "state": StateSequence,
}

# ID guaranteed to have temporal rows
_ID_WITH_DATA = 1


@pytest.fixture
def pool_copy(pools_dict: dict, pool_type: str):
    """Fresh per-test copy of the session pool; safe to call add_to_static."""
    return pools_dict[pool_type].copy()


@pytest.fixture
def traj_pool_copy(traj_pool):
    """Fresh per-test copy of the trajectory pool; safe to call add_to_static."""
    return traj_pool.copy()


@pytest.fixture
def sequence(stores_dict: dict, pool_type: str):
    """Standalone Sequence for ID=1; guaranteed to have temporal rows."""
    cls = _SEQ_CLS[pool_type]
    return cls(id_value=_ID_WITH_DATA, store=stores_dict[pool_type])


@pytest.fixture
def trajectory(traj_pool):
    """Single Trajectory for the first ID; for individual Trajectory.describe() tests."""
    return traj_pool[traj_pool.unique_ids[0]]
