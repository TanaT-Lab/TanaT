#!/usr/bin/env python3
"""
Tests for SpanplotVizBuilder.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import polars as pl
import pytest

from tanat.visualization.sequence.base.exceptions import (
    IncompatibleDisplayUnitError,
    UnsupportedSequenceTypeError,
)
from tanat.visualization.sequence.core import SequenceVisualizer

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


class TestSpanplotSmoke:
    """Basic draw completes without error."""

    @pytest.mark.parametrize("pool_type", ["interval", "state"])
    def test_draws(self, pools_dict, pool_type) -> None:
        """Spanplot renders for interval and state pools (datetime/timestep)."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        is_dt = pool.metadata.time_index.is_datetime
        unit = "days" if is_dt else None
        # fmt: off
        result = (
            SequenceVisualizer.spanplot(display_unit=unit, allow_large=True)
            .draw(pool, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    @pytest.mark.parametrize("kind", ["box", "violin", "strip"])
    def test_all_kinds_draw(self, interval_pool_copy, kind) -> None:
        """All three chart kinds render without error."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        # fmt: off
        result = (
            SequenceVisualizer.spanplot(kind=kind, display_unit="days", allow_large=True)
            .draw(interval_pool_copy, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)

    def test_single_sequence_draws(self, interval_pool_copy) -> None:
        """Drawing from a single Sequence renders without error."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        seq = interval_pool_copy[interval_pool_copy.unique_ids[0]]
        # fmt: off
        result = (
            SequenceVisualizer.spanplot(display_unit="days", allow_large=True)
            .draw(seq, entity_feature="status")
        )
        # fmt: on
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestSpanplotErrors:
    """Expected errors are raised."""

    def test_event_pool_raises(self, event_pool_copy) -> None:
        """UnsupportedSequenceTypeError when pool type is 'event'."""
        event_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.raises(UnsupportedSequenceTypeError):
            # fmt: off
            SequenceVisualizer.spanplot(display_unit="days", allow_large=True) \
                .draw(event_pool_copy, entity_feature="status")
            # fmt: on

    def test_missing_display_unit_on_datetime_raises(self, interval_pool_copy) -> None:
        """IncompatibleDisplayUnitError when display_unit is None on a datetime pool."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.raises(IncompatibleDisplayUnitError):
            # fmt: off
            SequenceVisualizer.spanplot(display_unit=None, allow_large=True) \
                .draw(interval_pool_copy, entity_feature="status")
            # fmt: on

    def test_display_unit_on_timestep_raises(self, interval_pool_ts_copy) -> None:
        """IncompatibleDisplayUnitError when display_unit is set on a timestep pool."""
        interval_pool_ts_copy.cast_features({"status": pl.Categorical})
        with pytest.raises(IncompatibleDisplayUnitError):
            # fmt: off
            SequenceVisualizer.spanplot(display_unit="days", allow_large=True) \
                .draw(interval_pool_ts_copy, entity_feature="status")
            # fmt: on


# ---------------------------------------------------------------------------
# NA handling: na_time_index
# ---------------------------------------------------------------------------


class TestSpanplotNaTimeIndex:
    """Tests for the na_time_index parameter (state pools have null __END__)."""

    def test_null_end_drop_warns(self, state_pool_copy) -> None:
        """Default na_time_index='drop' emits UserWarning for null __END__ rows."""
        state_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.warns(UserWarning, match="null time index"):
            # fmt: off
            result = (
                SequenceVisualizer.spanplot(display_unit="days", allow_large=True)
                .draw(state_pool_copy, entity_feature="status")
            )
            # fmt: on
        plt.close(result.figure)

    def test_null_end_raise(self, state_pool_copy) -> None:
        """na_time_index='raise' raises ValueError for null __END__ rows."""
        state_pool_copy.cast_features({"status": pl.Categorical})
        with pytest.raises(ValueError, match="null time index"):
            # fmt: off
            SequenceVisualizer.spanplot(display_unit="days", na_time_index="raise", allow_large=True) \
                .draw(state_pool_copy, entity_feature="status")
            # fmt: on

    def test_no_nulls_no_warning(self, interval_pool_copy) -> None:
        """Clean data (interval pool) emits no warning."""
        interval_pool_copy.cast_features({"status": pl.Categorical})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # fmt: off
            result = (
                SequenceVisualizer.spanplot(display_unit="days", allow_large=True)
                .draw(interval_pool_copy, entity_feature="status")
            )
            # fmt: on
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# NA handling: na_label
# ---------------------------------------------------------------------------


class TestSpanplotNaLabel:
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
            SequenceVisualizer.spanplot(display_unit="days", na_label="raise", allow_large=True) \
                .draw(pool, entity_feature="nullable_cat")
            # fmt: on

    def test_null_label_drop_warns(self, interval_pool_copy) -> None:
        """Default na_label='drop' emits UserWarning for null label rows."""
        pool = self._pool_with_null_label(interval_pool_copy)
        with pytest.warns(UserWarning, match="null label"):
            # fmt: off
            result = (
                SequenceVisualizer.spanplot(display_unit="days", na_label="drop", allow_large=True)
                .draw(pool, entity_feature="nullable_cat")
            )
            # fmt: on
        plt.close(result.figure)

    def test_null_label_category(self, interval_pool_copy) -> None:
        """na_label='category' replaces nulls with 'N/A' and renders."""
        pool = self._pool_with_null_label(interval_pool_copy)
        # fmt: off
        result = (
            SequenceVisualizer.spanplot(display_unit="days", na_label="category", allow_large=True)
            .draw(pool, entity_feature="nullable_cat")
        )
        # fmt: on
        plt.close(result.figure)
