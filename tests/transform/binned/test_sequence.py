#!/usr/bin/env python3
"""Tests: SequencePool.binned_data and SequencePool.to_tensor (temporal bin projection)."""

from __future__ import annotations

import numpy as np
import pytest

_MAX_BINS = 5


def _bin_size(pool) -> str | int:
    """Return an appropriate bin_size for the pool's temporal encoding."""
    return "1D" if pool.metadata.time_index.is_datetime else 1


@pytest.mark.parametrize("pool_type", ["interval", "event"])
class TestSequencePoolBinnedData:
    """binned_data() projects sequences onto a regular temporal grid."""

    def test_output_columns_snapshot(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """Column names in the polars output match snapshot."""
        pool = pools_dict[pool_type]
        result = pool.binned_data(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            fmt="polars",
        )
        assert snapshot == result.columns

    def test_has_id_and_bin_columns(self, pools_dict: dict, pool_type: str) -> None:
        """Output contains both the ID column and the bin-index column."""
        pool = pools_dict[pool_type]
        result = pool.binned_data(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            fmt="polars",
        )
        assert pool.settings.id_column in result.columns
        assert "__bin__" in result.columns

    def test_fill_value_removes_nulls(self, pools_dict: dict, pool_type: str) -> None:
        """fill_value=0.0 leaves no null in the feature column."""
        pool = pools_dict[pool_type]
        result = pool.binned_data(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            fill_value=0.0,
            fmt="polars",
        )
        assert result["value"].null_count() == 0

    def test_custom_bin_col_name(self, pools_dict: dict, pool_type: str) -> None:
        """bin_col parameter renames the bin-index column in the output."""
        pool = pools_dict[pool_type]
        result = pool.binned_data(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
            bin_col="t",
            fmt="polars",
        )
        assert "t" in result.columns
        assert "__bin__" not in result.columns

    def test_fmt_invalid_raises(self, pools_dict: dict, pool_type: str) -> None:
        """fmt='invalid' on binned_data raises ValueError pointing to to_tensor."""
        pool = pools_dict[pool_type]
        with pytest.raises(ValueError, match="Invalid fmt"):
            pool.binned_data(
                "value", bin_size=_bin_size(pool), max_bins=_MAX_BINS, fmt="invalid"
            )


@pytest.mark.parametrize("pool_type", ["interval", "event"])
class TestSequencePoolToTensor:
    """to_tensor() returns a (N, M, K) ndarray with feature_names."""

    def test_shape(self, pools_dict: dict, pool_type: str) -> None:
        """ndarray has shape (N_sequences, max_bins, N_features)."""
        pool = pools_dict[pool_type]
        arr, ids, feature_names = pool.to_tensor(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
        )
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[0] == len(pool)
        assert arr.shape[1] == _MAX_BINS
        assert arr.shape[2] == 1  # one feature

    def test_feature_names_contract(self, pools_dict: dict, pool_type: str) -> None:
        """feature_names matches the K-axis of the returned array."""
        pool = pools_dict[pool_type]
        arr, ids, feature_names = pool.to_tensor(
            "value",
            bin_size=_bin_size(pool),
            max_bins=_MAX_BINS,
        )
        assert feature_names == ["value"]
        assert arr.shape[2] == len(feature_names)


class TestStatePoolBinnedDataNullEnd:
    """The ``state`` fixture has null end times, so ``binned_data`` raises ValueError."""

    def test_raises_on_null_end_column(self, pools_dict: dict) -> None:
        """Fixture contains nulls → binned_data raises ValueError."""
        pool = pools_dict["state"]
        with pytest.raises(ValueError, match="null"):
            pool.binned_data("value", bin_size=_bin_size(pool), max_bins=_MAX_BINS)
