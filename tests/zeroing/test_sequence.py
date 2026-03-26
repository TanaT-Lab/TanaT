#!/usr/bin/env python3
"""
Tests: set_t0() / t0_data() API and T0 propagation to individual Sequence objects.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from tanat.zeroing import _T0, _T0_NEAREST_RANK

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentinel_t0(pool_copy):
    """Return a T0 value compatible with the pool's temporal dtype.

    - datetime pools → datetime(2000, 6, 1)
    - timestep pools → 42.0
    """
    if pool_copy.metadata.is_datetime:
        return datetime(2000, 6, 1)
    return 42.0


def _anchor_for(pool_type: str, anchor: str = "start") -> str | None:
    """Return the anchor value to use, avoiding spurious warnings on event pools.

    Event pools have a single time column, anchor is always ignored.
    To avoid polluting other tests with that warning, return None for event.
    """
    return None if pool_type == "event" else anchor


# ---------------------------------------------------------------------------
# TestSetT0Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSetT0Validation:
    """Validates set_t0() argument dispatch (no strategy data involved)."""

    def test_no_strategy_raises(self, pool_copy) -> None:
        """set_t0() with no strategy keyword → TypeError."""
        with pytest.raises(TypeError):
            pool_copy.set_t0()

    def test_multiple_strategies_raises(self, pool_copy) -> None:
        """set_t0(position=0, direct=1) → TypeError (exactly one strategy required)."""
        with pytest.raises(TypeError):
            pool_copy.set_t0(position=0, direct=_sentinel_t0(pool_copy))

    def test_direct_invalid_type_raises(self, pool_copy) -> None:
        """set_t0(direct="bad") → TypeError (string is not a valid T0Value)."""
        with pytest.raises(TypeError):
            pool_copy.set_t0(direct="bad")

    def test_feature_unknown_column_raises(self, pool_copy) -> None:
        """set_t0(feature="nonexistent") → error (feature validation)."""
        with pytest.raises(Exception):
            pool_copy.set_t0(feature="nonexistent")

    def test_anchor_warning_on_event_pool(self, pool_copy, pool_type: str) -> None:
        """Event pool + anchor="start" → UserWarning (anchor ignored)."""
        if pool_type != "event":
            pytest.skip("Only relevant for event pools.")
        with pytest.warns(UserWarning, match="anchor"):
            pool_copy.set_t0(position=0, anchor="start")

    def test_anchor_default_warning_on_period_pool(
        self, pool_copy, pool_type: str
    ) -> None:
        """Interval/state pool + no anchor= → UserWarning (defaults to 'start')."""
        if pool_type == "event":
            pytest.skip("Only relevant for interval/state pools.")
        with pytest.warns(UserWarning, match="anchor"):
            pool_copy.set_t0(position=0)


# ---------------------------------------------------------------------------
# TestPositionStrategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestPositionStrategy:
    """Position-based T0 strategy tests."""

    def test_position_zero(self, pool_copy, pool_type: str, snapshot) -> None:
        """set_t0(position=0, anchor="start") → t0_data() snapshot.

        Sequences that exist only in static data (no temporal rows) will
        receive _T0_ = null; only sequences with temporal data have non-null.
        """
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        # At least some sequences must have a valid T0
        assert df[_T0].is_not_null().any()
        assert snapshot == df

    def test_position_last_row(self, pool_copy, pool_type: str, snapshot) -> None:
        """set_t0(position=-1, anchor="start") → t0_data() snapshot."""
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(position=-1, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_position_anchor_end(self, pool_copy, pool_type: str, snapshot) -> None:
        """anchor="end" gives values different from anchor="start" (interval/state only)."""
        if pool_type == "event":
            pytest.skip("anchor='end' not applicable to event pools.")
        id_col = pool_copy.settings.id_column

        pool_start = pool_copy.copy()
        pool_start.set_t0(position=0, anchor="start")
        df_start = pool_start.t0_data(output_format="polars").sort(id_col)

        pool_copy.set_t0(position=0, anchor="end")
        df_end = pool_copy.t0_data(output_format="polars").sort(id_col)

        assert not df_end.equals(df_start)
        assert snapshot == df_end

    def test_position_anchor_middle(self, pool_copy, pool_type: str, snapshot) -> None:
        """anchor="middle" (interval/state only) → t0_data() snapshot."""
        if pool_type == "event":
            pytest.skip("anchor='middle' not applicable to event pools.")
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(position=0, anchor="middle")
        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_position_out_of_range_gives_null(self, pool_copy, pool_type: str) -> None:
        """set_t0(position=9999) → all _T0_ = null, UserWarning emitted."""
        with pytest.warns(UserWarning):
            pool_copy.set_t0(position=9999, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars")
        assert df[_T0].is_null().all()

    def test_position_negative_out_of_range(self, pool_copy, pool_type: str) -> None:
        """set_t0(position=-9999) → all _T0_ = null."""
        with pytest.warns(UserWarning):
            pool_copy.set_t0(position=-9999, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars")
        assert df[_T0].is_null().all()


# ---------------------------------------------------------------------------
# TestDirectStrategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestDirectStrategy:
    """Direct T0 strategy tests (scalar and per-sequence dict)."""

    def test_direct_scalar(self, pool_copy, pool_type: str, snapshot) -> None:
        """set_t0(direct=<scalar>) → all _T0_ equal the scalar; snapshot."""
        value = _sentinel_t0(pool_copy)
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(direct=value, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        assert (df[_T0] == value).all()
        assert snapshot == df

    def test_direct_dict_partial(self, pool_copy, pool_type: str) -> None:
        """Dict with first 3 IDs → those 3 have values, rest have _T0_ = null."""
        ids = pool_copy.unique_ids
        partial_ids = ids[:3]
        value = _sentinel_t0(pool_copy)
        mapping = {sid: value for sid in partial_ids}
        pool_copy.set_t0(direct=mapping, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars")
        id_col = pool_copy.settings.id_column

        # The 3 selected IDs have non-null T0
        has_value = df.filter(pl.col(id_col).is_in(partial_ids))
        assert not has_value[_T0].is_null().any()

        # Remaining IDs have null T0 (if there are more than 3 sequences)
        if len(ids) > 3:
            no_value = df.filter(~pl.col(id_col).is_in(partial_ids))
            assert no_value[_T0].is_null().all()

    def test_direct_dict_unknown_keys_warning(self, pool_copy, pool_type: str) -> None:
        """Dict with bogus key → UserWarning (key ignored)."""
        value = _sentinel_t0(pool_copy)
        mapping = {-9999: value}
        with pytest.warns(UserWarning):
            pool_copy.set_t0(direct=mapping, anchor=_anchor_for(pool_type))

    def test_direct_none_gives_type_error(self, pool_copy) -> None:
        """set_t0() with no keyword (direct=None not passed) → TypeError.

        Note: passing direct=None to set_t0() explicitly would be passing a
        kwarg whose value is None.  The strategy dispatcher only sees
        non-None kwargs, so this triggers the "no strategy provided" error.
        """
        with pytest.raises(TypeError):
            pool_copy.set_t0()


# ---------------------------------------------------------------------------
# TestFeatureStrategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestFeatureStrategy:
    """Feature-based T0 strategy tests."""

    def test_feature_from_static_column(self, pool_copy, snapshot) -> None:
        """Cast-aligned static feature → set_t0(feature=...) snapshot."""
        id_col = pool_copy.settings.id_column
        temporal_col = pool_copy.settings.get_time_columns()[0]

        # Build a static feature with the correct temporal dtype:
        # take the first temporal value per sequence from sequence_data.
        # Use the actual Polars dtype from the schema (metadata.time_index.dtype is a str).
        seq_df = pool_copy.temporal_data(output_format="polars")
        actual_temporal_dtype = seq_df.schema[temporal_col]
        t0_static = (
            seq_df.group_by(id_col)
            .agg(pl.col(temporal_col).first().alias("t0_feature"))
            .with_columns(pl.col("t0_feature").cast(actual_temporal_dtype))
        )
        pool_copy.add_static_features(t0_static)
        pool_copy.set_t0(feature="t0_feature")

        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        # IDs with temporal data must have a non-null T0; static-only IDs get null
        assert df[_T0].is_not_null().any()
        assert snapshot == df

    def test_feature_dtype_mismatch_raises(self, pool_copy) -> None:
        """Static feature dtype ≠ temporal dtype → TypeError."""
        # membership_duration is Int64; temporal dtype is Datetime or Float64
        with pytest.raises(TypeError):
            pool_copy.set_t0(feature="membership_duration")

    def test_feature_missing_column_raises(self, pool_copy) -> None:
        """set_t0(feature="nonexistent") → raises (column not found)."""
        with pytest.raises(Exception):
            pool_copy.set_t0(feature="nonexistent")


# ---------------------------------------------------------------------------
# TestQueryStrategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestQueryStrategy:
    """Query-based T0 strategy tests."""

    def test_query_first_match(self, pool_copy, pool_type: str, snapshot) -> None:
        """set_t0(query=..., use_first=True) → snapshot."""
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(
            query=pl.col("status") == "error",
            anchor=_anchor_for(pool_type),
            use_first=True,
        )
        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_query_last_match(self, pool_copy, pool_type: str, snapshot) -> None:
        """set_t0(query=..., use_first=False) → snapshot (different from use_first=True)."""
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(
            query=pl.col("status") == "error",
            anchor=_anchor_for(pool_type),
            use_first=False,
        )
        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_query_no_match_gives_null(self, pool_copy, pool_type: str) -> None:
        """Query matching no rows → all _T0_ = null, UserWarning emitted."""
        with pytest.warns(UserWarning):
            pool_copy.set_t0(
                query=pl.col("value") > 99999,
                anchor=_anchor_for(pool_type),
            )
        df = pool_copy.t0_data(output_format="polars")
        assert df[_T0].is_null().all()

    def test_query_anchor_end(self, pool_copy, pool_type: str, snapshot) -> None:
        """anchor='end' on interval/state → different values from anchor='start'."""
        if pool_type == "event":
            pytest.skip("anchor='end' not applicable to event pools.")
        id_col = pool_copy.settings.id_column

        pool_start = pool_copy.copy()
        pool_start.set_t0(
            query=pl.col("status") == "error",
            anchor="start",
            use_first=True,
        )
        df_start = pool_start.t0_data(output_format="polars").sort(id_col)

        pool_copy.set_t0(
            query=pl.col("status") == "error",
            anchor="end",
            use_first=True,
        )
        df_end = pool_copy.t0_data(output_format="polars").sort(id_col)

        # At least one non-null row must differ
        non_null = df_end[_T0].is_not_null() & df_start[_T0].is_not_null()
        if non_null.any():
            assert not df_end.filter(non_null).equals(df_start.filter(non_null))
        assert snapshot == df_end


# ---------------------------------------------------------------------------
# TestT0DataOutput
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestT0DataOutput:
    """Tests on the t0_data() return value shape and format."""

    def test_columns(self, pool_copy, pool_type: str, snapshot) -> None:
        """t0_data(output_format='polars').columns == [id_col, '_T0_', '_T0_NEAREST_RANK_']."""
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars")
        assert df.columns == [id_col, _T0, _T0_NEAREST_RANK]
        assert snapshot == df.columns

    def test_row_count(self, pool_copy, pool_type: str) -> None:
        """len(t0_data()) == len(pool_copy)."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars")
        assert len(df) == len(pool_copy)

    def test_pandas_output(self, pool_copy, pool_type: str) -> None:
        """t0_data() (default) returns a pd.DataFrame."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        result = pool_copy.t0_data()
        assert isinstance(result, pd.DataFrame)

    def test_polars_output(self, pool_copy, pool_type: str) -> None:
        """t0_data(output_format='polars') returns a pl.DataFrame."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        result = pool_copy.t0_data(output_format="polars")
        assert isinstance(result, pl.DataFrame)

    def test_invalid_format_raises(self, pool_copy, pool_type: str) -> None:
        """t0_data(output_format='numpy') → ValueError."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        with pytest.raises(ValueError):
            pool_copy.t0_data(output_format="numpy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestT0Overwrite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestT0Overwrite:
    """Overwrite behaviour: second set_t0() replaces the first."""

    def test_second_set_t0_replaces(self, pool_copy, pool_type: str, snapshot) -> None:
        """Call set_t0(position=0) then set_t0(position=-1) → matches position=-1 snapshot."""
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        pool_copy.set_t0(position=-1, anchor=_anchor_for(pool_type))
        df = pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_set_t0_returns_self(self, pool_copy, pool_type: str) -> None:
        """set_t0(...) returns the pool itself (fluent API)."""
        ret = pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        assert ret is pool_copy


# ---------------------------------------------------------------------------
# TestDefaultT0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestDefaultT0:
    """Default T0 (lazy trigger without calling set_t0())."""

    def test_default_lazy_trigger(self, pool_copy) -> None:
        """Accessing t0_data() without set_t0() defaults to position=0; result non-empty."""
        df = pool_copy.t0_data(output_format="polars")
        assert len(df) == len(pool_copy)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# TestSequenceT0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequenceT0:
    """T0 propagation from a pool to individual Sequence objects."""

    def test_seq_t0_propagated(self, pool_copy, pool_type: str) -> None:
        """After set_t0(position=0), seq.t0 is not None for at least one sequence."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        seq = pool_copy[pool_copy.unique_ids[0]]
        assert seq.t0 is not None

    def test_seq_t0_nearest_rank_type(self, pool_copy, pool_type: str) -> None:
        """seq.t0_nearest_rank is int or None."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        seq = pool_copy[pool_copy.unique_ids[0]]
        assert seq.t0_nearest_rank is None or isinstance(seq.t0_nearest_rank, int)

    def test_seq_t0_matches_pool(self, pool_copy, pool_type: str) -> None:
        """seq.t0 matches the _T0_ value for that ID in pool_copy.t0_data()."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        id_col = pool_copy.settings.id_column
        sid = pool_copy.unique_ids[0]
        seq = pool_copy[sid]

        pool_df = pool_copy.t0_data(output_format="polars")
        expected_t0 = pool_df.filter(pl.col(id_col) == sid)[_T0][0]
        assert seq.t0 == expected_t0

    def test_seq_t0_nearest_rank_matches_pool(self, pool_copy, pool_type: str) -> None:
        """seq.t0_nearest_rank matches _T0_NEAREST_RANK_ for that ID in pool_copy.t0_data()."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        id_col = pool_copy.settings.id_column
        sid = pool_copy.unique_ids[0]
        seq = pool_copy[sid]

        pool_df = pool_copy.t0_data(output_format="polars")
        expected_rank = pool_df.filter(pl.col(id_col) == sid)[_T0_NEAREST_RANK][0]
        assert seq.t0_nearest_rank == expected_rank

    def test_seq_t0_none_when_no_match(self, pool_copy, pool_type: str) -> None:
        """Query matching no rows → seq.t0 is None and seq.t0_nearest_rank is None."""
        with pytest.warns(UserWarning):
            pool_copy.set_t0(
                query=pl.col("value") > 99999,
                anchor=_anchor_for(pool_type),
            )
        seq = pool_copy[pool_copy.unique_ids[0]]
        assert seq.t0 is None
        assert seq.t0_nearest_rank is None

    def test_seq_standalone_default(self, standalone_seq) -> None:
        """Sequence built outside a pool defaults to position=0."""
        # Accessing t0 triggers the lazy default setter (position=0)
        assert standalone_seq.t0 is not None


# ---------------------------------------------------------------------------
# TestT0AfterSubset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestT0AfterSubset:
    """T0 is preserved after pool_copy.subset() and pool_copy.copy()."""

    def test_t0_preserved_after_subset(self, pool_copy, pool_type: str) -> None:
        """set_t0 → subset(first 3 IDs) → t0_data() has 3 rows, values unchanged."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        id_col = pool_copy.settings.id_column
        ids = pool_copy.unique_ids[:3]

        original_t0 = pool_copy.t0_data(output_format="polars")
        view = pool_copy.subset(ids)
        subset_t0 = view.t0_data(output_format="polars")

        assert len(subset_t0) == 3
        # Values for those 3 IDs are preserved
        expected = original_t0.filter(pl.col(id_col).is_in(ids)).sort(id_col)
        actual = subset_t0.sort(id_col)
        assert actual.equals(expected)

    def test_t0_preserved_after_copy(self, pool_copy, pool_type: str) -> None:
        """set_t0 → copy() → t0_data() matches the original."""
        id_col = pool_copy.settings.id_column
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        original_t0 = pool_copy.t0_data(output_format="polars").sort(id_col)

        copied = pool_copy.copy()
        copied_t0 = copied.t0_data(output_format="polars").sort(id_col)
        assert copied_t0.equals(original_t0)


# ---------------------------------------------------------------------------
# TestT0AfterTrainTestSplit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestT0AfterTrainTestSplit:
    """T0 is preserved after pool_copy.train_test_split()."""

    def test_t0_preserved_after_split(self, pool_copy, pool_type: str) -> None:
        """set_t0 → train_test_split → both halves have t0_data with correct row counts."""
        pool_copy.set_t0(position=0, anchor=_anchor_for(pool_type))
        id_col = pool_copy.settings.id_column
        original_t0 = pool_copy.t0_data(output_format="polars")

        train, test = pool_copy.train_test_split(test_size=0.25, random_state=42)
        train_t0 = train.t0_data(output_format="polars")
        test_t0 = test.t0_data(output_format="polars")

        # Row counts match split sizes
        assert len(train_t0) == len(train)
        assert len(test_t0) == len(test)
        assert len(train_t0) + len(test_t0) == len(original_t0)

        # Values are a subset of the original
        combined = pl.concat([train_t0, test_t0]).sort(id_col)
        original_sorted = original_t0.sort(id_col)
        assert combined.equals(original_sorted)
