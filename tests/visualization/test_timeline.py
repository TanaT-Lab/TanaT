#!/usr/bin/env python3
"""
Tests for TimelineVizBuilder.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import polars as pl
import pytest

from tanat.visualization.sequence.core import SequenceVisualizer

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


class TestTimelineSmoke:
    """Basic draw completes without error."""

    # Note: state pools contains None => trigger an error
    @pytest.mark.parametrize("pool_type", ["interval", "event"])
    def test_relative_mode_draws(self, pools_dict, pool_type) -> None:
        """Relative mode (all pool types × datetime/timestep) renders without error."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        anchor = "start" if pool_type != "event" else None
        pool.set_t0(position=0, anchor=anchor)
        result = SequenceVisualizer.timeline(
            time_mode="relative", allow_large=True
        ).draw(pool, entity_feature="status")
        plt.close(result.figure)

    # Note: state pools contains None => trigger an error
    @pytest.mark.parametrize("pool_type", ["interval", "event"])
    def test_absolute_mode_draws(self, pools_dict, pool_type) -> None:
        """Absolute mode (all pool types × datetime/timestep) renders without error."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        # fmt: off
        result = (
            SequenceVisualizer.timeline(time_mode="absolute", allow_large=True)
            .draw(pool, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    # Note: state pools contains None => trigger an error
    @pytest.mark.parametrize("pool_type", ["interval", "event"])
    def test_single_sequence_draws(self, pools_dict, pool_type) -> None:
        """Drawing from a single Sequence (pool[id]) renders without error."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        pool.set_t0(position=0, anchor="start")
        seq = pool[pool.unique_ids[0]]
        # fmt: off
        result = (
            SequenceVisualizer.timeline(time_mode="relative", allow_large=True)
            .draw(seq, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestTimelineErrors:
    """Expected errors are raised."""

    def test_non_categorical_feature_raises(self, interval_pool_copy) -> None:
        """TypeError when entity_feature is not Categorical (no cast_features call)."""
        interval_pool_copy.set_t0(position=0, anchor="start")
        with pytest.raises(TypeError, match="not a categorical"):
            # fmt: off
            SequenceVisualizer.timeline(time_mode="relative", allow_large=True) \
                .draw(interval_pool_copy, entity_feature="status")
            # fmt: on

    def test_all_null_t0_raises(self, interval_pool_copy) -> None:
        """ValueError when every T0 is null (query=pl.lit(False) matches no rows)."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            interval_pool_copy.set_t0(query=pl.lit(False), anchor="start")
        with pytest.raises(ValueError, match="null T0"):
            # fmt: off
            SequenceVisualizer.timeline(time_mode="relative", allow_large=True) \
                .draw(interval_pool_copy, entity_feature="status")
            # fmt: on


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestTimelineWarnings:
    """Expected warnings are emitted."""

    def test_timestep_display_unit_warns(self, interval_pool_ts_copy) -> None:
        """Explicit display_unit on a timestep pool emits UserWarning."""
        interval_pool_ts_copy.cast_features({"status": pl.Categorical})
        interval_pool_ts_copy.set_t0(position=0, anchor="start")
        with pytest.warns(UserWarning, match="ignored"):
            # fmt: off
            SequenceVisualizer.timeline(time_mode="relative", display_unit="hours", allow_large=True) \
                .prepare_data(interval_pool_ts_copy, entity_feature="status")
            # fmt: on
