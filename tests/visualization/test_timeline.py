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

    @pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
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

    @pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
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

    @pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
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


# ---------------------------------------------------------------------------
# NA handling: na_time_index
# ---------------------------------------------------------------------------


class TestTimelineNaTimeIndex:
    """Tests for the na_time_index parameter (state pools have null __END__)."""

    def test_null_end_drop_warns(self, state_pool_copy) -> None:
        """Default na_time_index='drop' emits UserWarning for null __END__ rows."""
        state_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.warns(UserWarning, match="null time index"):
            # fmt: off
            result = (
                SequenceVisualizer.timeline(allow_large=True)
                .draw(state_pool_copy, entity_feature="status")
            )
            # fmt: on
        plt.close(result.figure)

    def test_null_end_raise(self, state_pool_copy) -> None:
        """na_time_index='raise' raises ValueError for null __END__ rows."""
        state_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.raises(ValueError, match="null time index"):
            # fmt: off
            SequenceVisualizer.timeline(na_time_index="raise", allow_large=True) \
                .draw(state_pool_copy, entity_feature="status")
            # fmt: on

    def test_no_nulls_no_warning(self, interval_pool_copy) -> None:
        """Clean data (interval pool) emits no warning."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # fmt: off
            result = (
                SequenceVisualizer.timeline(allow_large=True)
                .draw(interval_pool_copy, entity_feature="status")
            )
            # fmt: on
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# NA handling: na_label
# ---------------------------------------------------------------------------


class TestTimelineNaLabel:
    """Tests for the na_label parameter (inject a feature with actual nulls)."""

    @staticmethod
    def _pool_with_null_label(pool_copy):
        """Return *pool_copy* with a 'nullable_cat' feature containing some nulls."""
        td = pool_copy.temporal_data(fmt="polars")
        n = len(td)
        series = pl.Series(
            "nullable_cat",
            [None if i % 5 == 0 else f"cat_{i % 3}" for i in range(n)],
            dtype=pl.Utf8,
        )
        pool_copy.add_entity_features(pl.DataFrame([series]))
        pool_copy.cast_features({"nullable_cat": pl.Categorical})
        return pool_copy

    def test_null_label_raise(self, interval_pool_copy) -> None:
        """na_label='raise' raises ValueError when null labels exist."""
        pool = self._pool_with_null_label(interval_pool_copy)
        with pytest.raises(ValueError, match="null label"):
            # fmt: off
            SequenceVisualizer.timeline(na_label="raise", allow_large=True) \
                .draw(pool, entity_feature="nullable_cat")
            # fmt: on

    def test_null_label_drop_warns(self, interval_pool_copy) -> None:
        """Default na_label='drop' emits UserWarning for null label rows."""
        pool = self._pool_with_null_label(interval_pool_copy)
        with pytest.warns(UserWarning, match="null label"):
            # fmt: off
            result = (
                SequenceVisualizer.timeline(na_label="drop", allow_large=True)
                .draw(pool, entity_feature="nullable_cat")
            )
            # fmt: on
        plt.close(result.figure)

    def test_null_label_category(self, interval_pool_copy) -> None:
        """na_label='category' replaces nulls with 'N/A' and renders."""
        pool = self._pool_with_null_label(interval_pool_copy)
        # fmt: off
        result = (
            SequenceVisualizer.timeline(na_label="category", allow_large=True)
            .draw(pool, entity_feature="nullable_cat")
        )
        # fmt: on
        plt.close(result.figure)
