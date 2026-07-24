#!/usr/bin/env python3
"""Tests for TrajectoryPool.describe() and Trajectory.describe().

Test organisation
-----------------
- ``TestTrajectoryPoolDescribe``: covers both dt/ts temporal variants via
  the local ``traj_pool_copy`` fixture.  Each test receives a fresh copy --
  safe for both read-only and mutating assertions.
- ``TestTrajectoryDescribe``: individual :class:`Trajectory` object, via
  the local ``trajectory`` fixture.
"""

from __future__ import annotations

import warnings

import polars as pl
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# TrajectoryPool.describe()
# ---------------------------------------------------------------------------


class TestTrajectoryPoolDescribe:
    """TrajectoryPool.describe(): both temporal variants."""

    def test_returns_pandas_by_default(self, traj_pool_copy):
        """Default fmt is pandas."""
        result = traj_pool_copy.describe()
        assert isinstance(result, pd.DataFrame)

    def test_returns_polars_when_requested(self, traj_pool_copy):
        """fmt='polars' returns a pl.DataFrame."""
        result = traj_pool_copy.describe(fmt="polars")
        assert isinstance(result, pl.DataFrame)

    def test_one_row_per_trajectory(self, traj_pool_copy):
        """by_id=True (default) produces exactly one row per trajectory ID."""
        result = traj_pool_copy.describe(fmt="polars")
        assert result.height == len(traj_pool_copy.unique_ids)

    def test_id_column_present(self, traj_pool_copy):
        """Result always includes the trajectory ID column."""
        result = traj_pool_copy.describe(fmt="polars")
        assert traj_pool_copy.settings.id_column in result.columns

    def test_n_sequences_column_present(self, traj_pool_copy):
        """A n_sequences column is always present."""
        result = traj_pool_copy.describe(fmt="polars")
        assert "n_sequences" in result.columns

    def test_n_sequences_value(self, traj_pool_copy):
        """n_sequences equals the number of visible sequence pools (constant per row)."""
        result = traj_pool_copy.describe(fmt="polars")
        n_aliases = len(list(traj_pool_copy.sequence_pools.keys()))
        assert result["n_sequences"].n_unique() == 1
        assert result["n_sequences"][0] == n_aliases

    def test_prefixed_columns_present(self, traj_pool_copy):
        """Each alias contributes a prefixed 'length' describe column."""
        result = traj_pool_copy.describe(fmt="polars")
        cols = set(result.columns)
        for alias in traj_pool_copy.sequence_pools:
            assert f"{alias}_length" in cols

    def test_custom_separator(self, traj_pool_copy):
        """separator='.' changes the delimiter between alias and column name."""
        result = traj_pool_copy.describe(separator=".", fmt="polars")
        cols = set(result.columns)
        for alias in traj_pool_copy.sequence_pools:
            assert f"{alias}.length" in cols
            assert f"{alias}_length" not in cols

    def test_aggregated_returns_pandas_describe(self, traj_pool_copy):
        """by_id=False returns cross-trajectory aggregated statistics (count, mean, std, min, 25%, ...)."""
        result = traj_pool_copy.describe(by_id=False)
        assert isinstance(result, pd.DataFrame)
        assert "count" in result.index

    def test_add_to_static(self, traj_pool_copy):
        """add_to_static=True persists prefixed describe columns into trajectory static."""
        traj_pool_copy.describe(add_to_static=True)
        static = traj_pool_copy.static_data(fmt="polars")
        assert static is not None
        assert any("length" in c for c in static.columns)

    def test_add_to_static_ignored_with_by_id_false(self, traj_pool_copy):
        """add_to_static=True is silently ignored (with a warning) when by_id=False."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            traj_pool_copy.describe(by_id=False, add_to_static=True)
        assert len(ws) >= 1
        assert any("add_to_static=True is ignored" in str(w.message) for w in ws)

    def test_caching_returns_same_object(self, traj_pool_copy):
        """Identical calls return the exact same cached object (no recomputation)."""
        r1 = traj_pool_copy.describe(fmt="polars")
        r2 = traj_pool_copy.describe(fmt="polars")
        assert r1 is r2

    def test_invalid_fmt_raises(self, traj_pool_copy):
        """An unsupported fmt raises ValueError."""
        with pytest.raises(ValueError, match="fmt"):
            traj_pool_copy.describe(fmt="csv")  # type: ignore[arg-type]

    def test_describe_output_snapshot(self, traj_pool_copy, snapshot):
        """Full describe() output is stable across runs."""
        result = traj_pool_copy.describe()
        assert result.to_csv() == snapshot


# ---------------------------------------------------------------------------
# Trajectory.describe() (individual)
# ---------------------------------------------------------------------------


class TestTrajectoryDescribe:
    """Trajectory.describe(): individual trajectory objects, via the ``trajectory`` fixture."""

    def test_returns_one_row(self, trajectory):
        """describe() on a single trajectory always returns exactly 1 row."""
        result = trajectory.describe(fmt="polars")
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_returns_pandas_by_default(self, trajectory):
        """Default fmt is pandas with 1 row."""
        result = trajectory.describe()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_has_n_sequences(self, trajectory):
        """n_sequences column is present on the single-trajectory result."""
        result = trajectory.describe(fmt="polars")
        assert "n_sequences" in result.columns

    def test_prefixed_columns_present(self, trajectory):
        """Each visible alias contributes a prefixed describe column."""
        result = trajectory.describe(fmt="polars")
        cols = set(result.columns)
        for alias in trajectory:
            assert f"{alias}_length" in cols

    def test_custom_separator(self, trajectory):
        """separator='.' changes the delimiter on the single-trajectory result."""
        result = trajectory.describe(separator=".", fmt="polars")
        cols = set(result.columns)
        for alias in trajectory:
            assert f"{alias}.length" in cols

    def test_caching_returns_same_object(self, trajectory):
        """Repeated calls on the same Trajectory return the cached object."""
        r1 = trajectory.describe(fmt="polars")
        r2 = trajectory.describe(fmt="polars")
        assert r1 is r2

    def test_describe_output_snapshot(self, trajectory, snapshot):
        """describe() output for a single trajectory is stable across runs."""
        result = trajectory.describe()
        assert result.to_csv() == snapshot
