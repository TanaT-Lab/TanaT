#!/usr/bin/env python3
"""
Tests for BarplotVizBuilder.
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


class TestBarplotSmoke:
    """Basic draw completes without error."""

    @pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
    def test_count_draws(self, pools_dict, pool_type) -> None:
        """show_as='count' renders for all pool types (datetime/timestep)."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        # fmt: off
        result = (
            SequenceVisualizer.barplot(show_as="count", allow_large=True)
            .draw(pool, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    @pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
    def test_rate_draws(self, pools_dict, pool_type) -> None:
        """show_as='rate' renders for all pool types (datetime/timestep)."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        # fmt: off
        result = (
            SequenceVisualizer.barplot(show_as="rate", allow_large=True)
            .draw(pool, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    @pytest.mark.parametrize("pool_type", ["interval", "state"])
    def test_duration_draws(self, pools_dict, pool_type) -> None:
        """show_as='duration' renders for interval and state pools (datetime/timestep)."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        is_dt = pool.metadata.time_index.is_datetime
        unit = "days" if is_dt else None
        # fmt: off
        result = (
            SequenceVisualizer.barplot(show_as="duration", display_unit=unit, allow_large=True)
            .draw(pool, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    def test_horizontal_orientation_draws(self, interval_pool_copy) -> None:
        """Horizontal orientation renders without error."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        # fmt: off
        result = (
            SequenceVisualizer.barplot(orientation="horizontal", allow_large=True)
            .draw(interval_pool_copy, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestBarplotErrors:
    """Expected errors are raised."""

    def test_duration_on_event_pool_raises(self, event_pool_copy) -> None:
        """UnsupportedShowAsError when show_as='duration' on an event pool."""
        event_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.raises(Exception, match="duration"):
            # fmt: off
            SequenceVisualizer.barplot(show_as="duration", display_unit="days", allow_large=True) \
                .draw(event_pool_copy, entity_feature="status")
            # fmt: on

    def test_non_categorical_feature_raises(self, interval_pool_copy) -> None:
        """TypeError when entity_feature is not Categorical."""
        with pytest.raises(TypeError, match="not a categorical"):
            # fmt: off
            SequenceVisualizer.barplot(allow_large=True) \
                .draw(interval_pool_copy, entity_feature="status")
            # fmt: on


# ---------------------------------------------------------------------------
# NA handling: na_time_index (duration mode only)
# ---------------------------------------------------------------------------


class TestBarplotNaTimeIndex:
    """Tests for na_time_index in duration mode (state pools have null __END__)."""

    def test_null_end_drop_warns(self, state_pool_copy) -> None:
        """Default na_time_index='drop' emits UserWarning for null __END__ rows."""
        state_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.warns(UserWarning, match="null time index"):
            # fmt: off
            result = (
                SequenceVisualizer.barplot(show_as="duration", display_unit="days", allow_large=True)
                .draw(state_pool_copy, entity_feature="status")
            )
            # fmt: on
        plt.close(result.figure)

    def test_null_end_raise(self, state_pool_copy) -> None:
        """na_time_index='raise' raises ValueError for null __END__ rows."""
        state_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.raises(ValueError, match="null time index"):
            # fmt: off
            SequenceVisualizer.barplot(show_as="duration", display_unit="days", na_time_index="raise", allow_large=True) \
                .draw(state_pool_copy, entity_feature="status")
            # fmt: on

    def test_no_nulls_no_warning(self, interval_pool_copy) -> None:
        """Clean data (interval pool, duration mode) emits no warning."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # fmt: off
            result = (
                SequenceVisualizer.barplot(show_as="duration", display_unit="days", allow_large=True)
                .draw(interval_pool_copy, entity_feature="status")
            )
            # fmt: on
        plt.close(result.figure)

    def test_count_mode_ignores_null_time_index(self, state_pool_copy) -> None:
        """show_as='count' does NOT trigger na_time_index handling (no time cols used)."""
        state_pool_copy.cast_features({"status": pl.Categorical})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # fmt: off
            result = (
                SequenceVisualizer.barplot(show_as="count", allow_large=True)
                .draw(state_pool_copy, entity_feature="status")
            )
            # fmt: on
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# NA handling: na_label
# ---------------------------------------------------------------------------


class TestBarplotNaLabel:
    """Tests for the na_label parameter (inject a feature with actual nulls)."""

    @staticmethod
    def _pool_with_null_label(pool_copy):
        """Return *pool_copy* with a 'nullable_cat' feature containing some nulls."""
        td = pool_copy.temporal_data(output_format="polars")
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
            SequenceVisualizer.barplot(na_label="raise", allow_large=True) \
                .draw(pool, entity_feature="nullable_cat")
            # fmt: on

    def test_null_label_drop_warns(self, interval_pool_copy) -> None:
        """Default na_label='drop' emits UserWarning for null label rows."""
        pool = self._pool_with_null_label(interval_pool_copy)
        with pytest.warns(UserWarning, match="null label"):
            # fmt: off
            result = (
                SequenceVisualizer.barplot(na_label="drop", allow_large=True)
                .draw(pool, entity_feature="nullable_cat")
            )
            # fmt: on
        plt.close(result.figure)

    def test_null_label_category(self, interval_pool_copy) -> None:
        """na_label='category' replaces nulls with 'N/A' and renders."""
        pool = self._pool_with_null_label(interval_pool_copy)
        # fmt: off
        result = (
            SequenceVisualizer.barplot(na_label="category", allow_large=True)
            .draw(pool, entity_feature="nullable_cat")
        )
        # fmt: on
        plt.close(result.figure)
