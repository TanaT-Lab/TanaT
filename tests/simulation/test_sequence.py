#!/usr/bin/env python3
"""
Tests: simulate_events / simulate_intervals / simulate_states.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from tanat import build_events, build_intervals, build_states
from tanat.dataset import simulate_events, simulate_intervals, simulate_states

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_END = np.datetime64(datetime(2025, 1, 1), "us")


# ---------------------------------------------------------------------------
# 1. simulate_events
# ---------------------------------------------------------------------------


class TestSimulateEvents:
    """Tests for simulate_events()."""

    def test_default_returns_dataframe(self) -> None:
        """Default call returns a plain pd.DataFrame (no static)."""
        result = simulate_events(n_ids=50, seed=42)
        assert isinstance(result, pd.DataFrame)

    def test_default_columns(self) -> None:
        """Default columns are id, time, f_0 (float64), f_1 (string)."""
        df = simulate_events(n_ids=50, seed=42)
        assert set(df.columns) == {"id", "time", "f_0", "f_1"}

    def test_default_n_ids(self) -> None:
        """All n_ids distinct IDs appear in the temporal DataFrame."""
        df = simulate_events(n_ids=50, seed=42)
        assert df["id"].nunique() == 50

    def test_with_static_returns_tuple(self) -> None:
        """Returns a (temporal, static) tuple when static_features is set."""
        result = simulate_events(n_ids=50, static_features=2, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        temporal, static = result
        assert isinstance(temporal, pd.DataFrame)
        assert isinstance(static, pd.DataFrame)

    def test_with_static_columns(self) -> None:
        """Static DataFrame has id + auto-named static feature columns."""
        _, static = simulate_events(n_ids=50, static_features=2, seed=42)
        assert set(static.columns) == {"id", "s_0", "s_1"}

    def test_with_static_id_count(self) -> None:
        """Static DataFrame has exactly n_ids rows (one per ID)."""
        _, static = simulate_events(n_ids=50, static_features=2, seed=42)
        assert len(static) == 50
        assert static["id"].nunique() == 50


# ---------------------------------------------------------------------------
# 2. simulate_intervals
# ---------------------------------------------------------------------------


class TestSimulateIntervals:
    """Tests for simulate_intervals()."""

    def test_default_columns(self) -> None:
        """Default columns include id, start, end plus two feature columns."""
        df = simulate_intervals(n_ids=50, seed=42)
        assert {"id", "start", "end", "f_0", "f_1"}.issubset(df.columns)

    def test_start_before_end(self) -> None:
        """Every row must satisfy start < end."""
        df = simulate_intervals(n_ids=50, seed=42)
        assert (df["start"] < df["end"]).all()

    def test_no_overlaps(self) -> None:
        """When allow_overlaps=False, start[i+1] >= end[i] within each ID."""
        df = simulate_intervals(n_ids=30, allow_overlaps=False, seed=0)
        for _, group in df.groupby("id"):
            group = group.sort_values("start")
            if len(group) > 1:
                ends = group["end"].values[:-1]
                next_starts = group["start"].values[1:]
                assert (next_starts >= ends).all()


# ---------------------------------------------------------------------------
# 3. simulate_states
# ---------------------------------------------------------------------------


class TestSimulateStates:
    """Tests for simulate_states()."""

    def test_contiguity(self) -> None:
        """end[i] == start[i+1] for every consecutive pair within each ID."""
        df = simulate_states(n_ids=20, seq_length_range=(3, 6), seed=42)
        for _, group in df.groupby("id"):
            group = group.sort_values("start")
            if len(group) > 1:
                ends = group["end"].values[:-1]
                next_starts = group["start"].values[1:]
                assert (ends == next_starts).all()

    def test_last_end_equals_time_range_end(self) -> None:
        """Last state per ID has end == time_range[1] (default 2025-01-01)."""
        df = simulate_states(n_ids=20, seed=42)
        for _, group in df.groupby("id"):
            group = group.sort_values("start")
            assert group["end"].values[-1] == _DEFAULT_END

    def test_custom_time_range_last_end(self) -> None:
        """Custom time_range: last state end matches the provided end bound."""
        custom_end = datetime(2020, 6, 15)
        df = simulate_states(
            n_ids=10,
            time_range=(datetime(2010, 1, 1), custom_end),
            seed=0,
        )
        expected = np.datetime64(custom_end, "us")
        for _, group in df.groupby("id"):
            group = group.sort_values("start")
            assert group["end"].values[-1] == expected


# ---------------------------------------------------------------------------
# 4. Feature generation
# ---------------------------------------------------------------------------


class TestFeatureGeneration:
    """Tests for feature name resolution and type cycling."""

    def test_named_entity_features_columns(self) -> None:
        """Explicit feature names produce correctly named columns."""
        df = simulate_events(
            n_ids=10,
            entity_features=["score", "status", "flag"],
            seed=0,
        )
        assert "score" in df.columns
        assert "status" in df.columns
        assert "flag" in df.columns

    def test_named_entity_features_types(self) -> None:
        """Type cycling: position 0 -> float64, 1 -> string, 2 -> bool."""
        df = simulate_events(
            n_ids=10,
            entity_features=["score", "status", "flag"],
            seed=0,
        )
        assert df["score"].dtype == np.float64
        assert pd.api.types.is_string_dtype(df["status"])
        assert df["flag"].dtype == bool

    def test_named_static_features(self) -> None:
        """Explicit static feature names appear in the static DataFrame."""
        _, static = simulate_states(
            n_ids=10,
            entity_features=2,
            static_features=["age", "group"],
            seed=0,
        )
        assert set(static.columns) == {"id", "age", "group"}


# ---------------------------------------------------------------------------
# 5. Seed behaviour
# ---------------------------------------------------------------------------


class TestSeedBehaviour:
    """Tests for reproducibility and variation across seeds."""

    def test_same_seed_identical(self) -> None:
        """Two calls with the same seed produce identical DataFrames."""
        df1 = simulate_events(n_ids=20, seed=42)
        df2 = simulate_events(n_ids=20, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        """Two calls with different seeds produce different numeric data."""
        df1 = simulate_events(n_ids=20, seed=42)
        df2 = simulate_events(n_ids=20, seed=99)
        assert not df1["f_0"].equals(df2["f_0"])

    def test_none_seed_non_deterministic(self) -> None:
        """Two calls with seed=None should (almost certainly) differ."""
        df1 = simulate_events(n_ids=50, seed=None)
        df2 = simulate_events(n_ids=50, seed=None)
        assert not df1["f_0"].equals(df2["f_0"])


# ---------------------------------------------------------------------------
# 6. seq_length_range
# ---------------------------------------------------------------------------


class TestSeqLengthRange:
    """Tests for the seq_length_range parameter."""

    def test_length_bounds_events(self) -> None:
        """All IDs have row counts within [min, max]."""
        df = simulate_events(n_ids=50, seq_length_range=(2, 7), seed=42)
        counts = df.groupby("id").size()
        assert (counts >= 2).all()
        assert (counts <= 7).all()

    def test_length_bounds_intervals(self) -> None:
        """All IDs have row counts within [min, max]."""
        df = simulate_intervals(n_ids=50, seq_length_range=(4, 8), seed=42)
        counts = df.groupby("id").size()
        assert (counts >= 4).all()
        assert (counts <= 8).all()

    def test_length_bounds_states(self) -> None:
        """All IDs have row counts within [min, max]."""
        df = simulate_states(n_ids=50, seq_length_range=(1, 5), seed=42)
        counts = df.groupby("id").size()
        assert (counts >= 1).all()
        assert (counts <= 5).all()


# ---------------------------------------------------------------------------
# 7. Pipeline tests (simulate_* -> build_*)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "simulate_fn,build_fn,build_kwargs",
    [
        (
            simulate_events,
            build_events,
            {"id_column": "id", "time_column": "time"},
        ),
        (
            simulate_intervals,
            build_intervals,
            {"id_column": "id", "start_column": "start", "end_column": "end"},
        ),
        (
            simulate_states,
            build_states,
            {"id_column": "id", "start_column": "start", "end_column": "end"},
        ),
    ],
    ids=["events", "intervals", "states"],
)
def test_pipeline(
    simulate_fn: Callable, build_fn: Callable, build_kwargs: dict
) -> None:
    """simulate_* output feeds into build_* without error; pool length equals n_ids."""
    n = 30
    df = simulate_fn(n_ids=n, seed=42)
    pool = build_fn(df, **build_kwargs)
    assert len(pool) == n
