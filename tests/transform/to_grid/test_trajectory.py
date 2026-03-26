#!/usr/bin/env python3
"""Tests: TrajectoryPool.to_grid (shared temporal axis across sub-pools)."""

from __future__ import annotations

import numpy as np
import pytest

from tanat.trajectory.pool import TrajectoryPool

_MAX_BINS = 5


def _bin_size(pool: TrajectoryPool) -> str | int:
    """Return an appropriate bin_size for the pool's temporal encoding."""
    return "1D" if pool.metadata.time_index.is_datetime else 1


class TestTrajectoryPoolToGrid:
    """to_grid() projects multi-modal trajectories onto a single shared temporal axis."""

    def test_output_columns_snapshot(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """Column names in the polars output match snapshot."""
        result = traj_pool.to_grid(
            {"intervals": "value", "events": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            output_format="polars",
        )
        assert snapshot == result.columns

    def test_numpy_shape(self, traj_pool: TrajectoryPool) -> None:
        """numpy output has shape (N_trajectories, max_bins, N_feature_columns)."""
        arr = traj_pool.to_grid(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            output_format="numpy",
        )
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[0] == len(traj_pool)
        assert arr.shape[1] == _MAX_BINS

    def test_unknown_alias_raises(self, traj_pool: TrajectoryPool) -> None:
        """to_grid with an unrecognised alias raises KeyError."""
        with pytest.raises(KeyError):
            traj_pool.to_grid(
                {"unknown_alias": "value"},
                bin_size=_bin_size(traj_pool),
                max_bins=_MAX_BINS,
            )

    def test_polars_row_count(self, traj_pool: TrajectoryPool) -> None:
        """Wide-format polars output has exactly one row per trajectory."""
        result = traj_pool.to_grid(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            output_format="polars",
        )
        assert result.height == len(traj_pool)
