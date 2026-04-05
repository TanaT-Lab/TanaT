#!/usr/bin/env python3
"""
Local fixtures for tests/visualization/ — plain pool copies, no setup.
Cast and T0 are applied inline in each test.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend: no display required


@pytest.fixture
def state_pool_copy(state_pool):
    """Fresh copy of the session StateSequencePool (datetime)."""
    return state_pool.copy()


@pytest.fixture
def interval_pool_copy(interval_pool):
    """Fresh copy of the session IntervalSequencePool (datetime)."""
    return interval_pool.copy()


@pytest.fixture
def state_pool_ts_copy(state_pool_ts):
    """Fresh copy of the session StateSequencePool (timestep)."""
    return state_pool_ts.copy()


@pytest.fixture
def interval_pool_ts_copy(interval_pool_ts):
    """Fresh copy of the session IntervalSequencePool (timestep)."""
    return interval_pool_ts.copy()


@pytest.fixture
def trajectory_pool_dt_copy(trajectory_pool_dt):
    """Fresh copy of the session TrajectoryPool (datetime)."""
    return trajectory_pool_dt.copy()
