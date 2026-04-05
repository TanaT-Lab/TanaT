#!/usr/bin/env python3
"""
Tests for DistributionVizBuilder.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import polars as pl
import pytest

from tanat.visualization.sequence.base.exceptions import UnsupportedSequenceTypeError
from tanat.visualization.sequence.core import SequenceVisualizer

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


class TestDistributionSmoke:
    """Basic draw completes without error."""

    def test_relative_mode_draws(self, state_pool_copy) -> None:
        """Relative mode (datetime state pool, default unit) renders without error."""
        state_pool_copy.set_t0(position=0, anchor="start")
        # fmt: off
        result = (
            SequenceVisualizer.distribution(time_mode="relative", bin_size="1d", allow_large=True)
            .draw(state_pool_copy, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    def test_absolute_mode_draws(self, state_pool) -> None:
        """Absolute mode renders without error (no T0 needed)."""
        # fmt: off
        result = (
            SequenceVisualizer.distribution(time_mode="absolute", bin_size="1d", allow_large=True)
            .draw(state_pool, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    def test_single_sequence_draws(self, state_pool_copy) -> None:
        """Drawing from a single Sequence (pool[id]) renders without error."""
        state_pool_copy.set_t0(position=0, anchor="start")
        seq = state_pool_copy[state_pool_copy.unique_ids[0]]
        # fmt: off
        result = (
            SequenceVisualizer.distribution(time_mode="relative", bin_size="1d", allow_large=True)
            .draw(seq, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestDistributionErrors:
    """Expected errors are raised."""

    def test_all_null_t0_raises(self, state_pool_copy) -> None:
        """ValueError when every T0 is null (query=pl.lit(False) matches no rows)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            state_pool_copy.set_t0(query=pl.lit(False), anchor="start")
        with pytest.raises(ValueError, match="null T0"):
            # fmt: off
            SequenceVisualizer.distribution(time_mode="relative", bin_size="1d", allow_large=True) \
                .draw(state_pool_copy, entity_feature="status")
            # fmt: on

    @pytest.mark.parametrize("pool_type", ["interval", "event"])
    def test_incompatible_sequence_type_raises(self, pools_dict, pool_type) -> None:
        """UnsupportedSequenceTypeError when pool type is not 'state'."""
        pool = pools_dict[pool_type]
        with pytest.raises(UnsupportedSequenceTypeError):
            # fmt: off
            SequenceVisualizer.distribution(time_mode="absolute", bin_size="1d", allow_large=True) \
                .draw(pool, entity_feature="status")
            # fmt: on

    def test_numeric_bin_size_on_datetime_pool_raises(self, state_pool_copy) -> None:
        """TypeError when bin_size is numeric but pool is datetime."""
        with pytest.raises(TypeError, match="duration string"):
            # fmt: off
            SequenceVisualizer.distribution(time_mode="absolute", bin_size=1, allow_large=True) \
                .draw(state_pool_copy, entity_feature="status")
            # fmt: on

    def test_string_bin_size_on_timestep_pool_raises(self, state_pool_ts_copy) -> None:
        """TypeError when bin_size is a duration string but pool is timestep."""
        with pytest.raises(TypeError, match="numeric"):
            # fmt: off
            SequenceVisualizer.distribution(time_mode="absolute", bin_size="1d", allow_large=True) \
                .draw(state_pool_ts_copy, entity_feature="status")
            # fmt: on


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestDistributionWarnings:
    """Expected warnings are emitted."""

    def test_timestep_display_unit_warns(self, state_pool_ts_copy) -> None:
        """Explicit display_unit on a timestep pool emits UserWarning."""
        state_pool_ts_copy.set_t0(position=0, anchor="start")
        with pytest.warns(UserWarning, match="ignored"):
            # fmt: off
            SequenceVisualizer.distribution(time_mode="relative", bin_size=1, display_unit="hours", allow_large=True) \
                .prepare_data(state_pool_ts_copy, entity_feature="status")
            # fmt: on
