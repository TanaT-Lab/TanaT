#!/usr/bin/env python3
"""Tests: SequencePool.to_grid (temporal grid projection)."""

from __future__ import annotations

import numpy as np
import pytest

_MAX_BINS = 5


def _bin_size(pool) -> str | int:
    """Return an appropriate bin_size for the pool's temporal encoding."""
    return "1D" if pool.metadata.temporal.is_datetime else 1


@pytest.mark.parametrize("pool_type", ["interval", "event"])
class TestSequencePoolToGrid:
    """to_grid() projects sequences onto a regular temporal grid."""

    def test_output_columns_snapshot(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Column names in the polars output match snapshot."""
        pool = pools_dict[pool_type]
        result = pool.to_grid(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            output_format="polars",
        )
        assert snapshot == result.columns

    def test_has_id_and_bin_columns(self, pools_dict: dict, pool_type: str) -> None:
        """Output contains both the ID column and the bin-index column."""
        pool = pools_dict[pool_type]
        result = pool.to_grid(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            output_format="polars",
        )
        assert pool.settings.id_column in result.columns
        assert "__bin__" in result.columns

    def test_fill_value_removes_nulls(self, pools_dict: dict, pool_type: str) -> None:
        """fill_value=0.0 leaves no null in the feature column."""
        pool = pools_dict[pool_type]
        result = pool.to_grid(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            fill_value=0.0,
            output_format="polars",
        )
        assert result["value"].null_count() == 0

    def test_numpy_shape(self, pools_dict: dict, pool_type: str) -> None:
        """numpy output has shape (N_sequences, max_bins, N_features)."""
        pool = pools_dict[pool_type]
        arr = pool.to_grid(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            output_format="numpy",
        )
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[0] == len(pool)
        assert arr.shape[1] == _MAX_BINS
        assert arr.shape[2] == 1  # one feature

    def test_custom_bin_col_name(self, pools_dict: dict, pool_type: str) -> None:
        """bin_col parameter renames the bin-index column in the output."""
        pool = pools_dict[pool_type]
        result = pool.to_grid(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            bin_col="t",
            output_format="polars",
        )
        assert "t" in result.columns
        assert "__bin__" not in result.columns


class TestStatePoolToGridNullEnd:
    """The ``state`` fixture has null end times, so ``to_grid`` raises ValueError."""

    def test_raises_on_null_end_column(self, pools_dict: dict) -> None:
        """Fixture contains nulls → to_grid raises ValueError."""
        pool = pools_dict["state"]
        with pytest.raises(ValueError, match="null"):
            pool.to_grid("value", bin_size=_bin_size(pool), max_bins=_MAX_BINS)
