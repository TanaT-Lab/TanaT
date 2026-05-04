#!/usr/bin/env python3
"""Tests: TrajectoryPool.binned_data and TrajectoryPool.to_tensor (shared temporal axis)."""

from __future__ import annotations

import numpy as np
import pytest

from tanat.trajectory.pool import TrajectoryPool

_MAX_BINS = 5


def _bin_size(pool: TrajectoryPool) -> str | int:
    """Return an appropriate bin_size for the pool's temporal encoding."""
    return "1D" if pool.metadata.time_index.is_datetime else 1


class TestTrajectoryPoolBinnedData:
    """binned_data() projects multi-modal trajectories onto a single shared temporal axis."""

    def test_output_columns_snapshot(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """Column names in the polars output match snapshot."""
        result = traj_pool.binned_data(
            {"intervals": "value", "events": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            fmt="polars",
        )
        assert snapshot == result.columns

    def test_unknown_alias_raises(self, traj_pool: TrajectoryPool) -> None:
        """binned_data with an unrecognised alias raises KeyError."""
        with pytest.raises(KeyError):
            traj_pool.binned_data(
                {"unknown_alias": "value"},
                bin_size=_bin_size(traj_pool),
                max_bins=_MAX_BINS,
            )

    def test_polars_row_count(self, traj_pool: TrajectoryPool) -> None:
        """Long-format polars output has N × max_bins rows."""
        result = traj_pool.binned_data(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            fmt="polars",
        )
        assert result.height == len(traj_pool) * _MAX_BINS

    def test_has_id_and_bin_columns(self, traj_pool: TrajectoryPool) -> None:
        """Long-format output contains both the ID column and the bin-index column."""
        result = traj_pool.binned_data(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            fmt="polars",
        )
        assert traj_pool.settings.id_column in result.columns
        assert "__bin__" in result.columns

    def test_fill_value_removes_nulls(self, traj_pool: TrajectoryPool) -> None:
        """fill_value=0.0 leaves no null in the feature column."""
        result = traj_pool.binned_data(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            fill_value=0.0,
            fmt="polars",
        )
        assert result["intervals_value"].null_count() == 0

    def test_custom_bin_col_name(self, traj_pool: TrajectoryPool) -> None:
        """bin_col parameter renames the bin-index column in the output."""
        result = traj_pool.binned_data(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
            bin_col="t",
            fmt="polars",
        )
        assert "t" in result.columns
        assert "__bin__" not in result.columns

    def test_fmt_invalid_raises(self, traj_pool: TrajectoryPool) -> None:
        """fmt='invalid' on binned_data raises ValueError pointing to to_tensor."""
        with pytest.raises(ValueError, match="Invalid fmt"):
            traj_pool.binned_data(
                {"intervals": "value"},
                bin_size=_bin_size(traj_pool),
                max_bins=_MAX_BINS,
                fmt="invalid",
            )


class TestTrajectoryPoolToTensor:
    """to_tensor() returns a (N, M, K) ndarray with feature_names."""

    def test_shape(self, traj_pool: TrajectoryPool) -> None:
        """ndarray has shape (N_trajectories, max_bins, N_feature_columns)."""
        arr, ids, feature_names = traj_pool.to_tensor(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
        )
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[0] == len(traj_pool)
        assert arr.shape[1] == _MAX_BINS

    def test_feature_names_contract(self, traj_pool: TrajectoryPool) -> None:
        """feature_names are prefixed and K-axis matches."""
        arr, ids, feature_names = traj_pool.to_tensor(
            {"intervals": "value"},
            bin_size=_bin_size(traj_pool),
            max_bins=_MAX_BINS,
        )
        assert feature_names == ["intervals_value"]
        assert arr.shape[2] == len(feature_names)
