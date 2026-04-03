#!/usr/bin/env python3
"""
Tests: TrajectoryPool-level T0 (set_t0 / t0_data) and Trajectory.t0 / t0_nearest_rank.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from tanat import TrajectoryPool
from tanat.trajectory.trajectory import Trajectory
from tanat.zeroing import T0Setter, _T0, _T0_NEAREST_RANK

# Trajectory pool aliases present in the test fixtures
_REF_ALIAS = "events"  # event pool: no anchor needed


def _sentinel_t0(traj_pool_copy):
    """Return a scalar T0 compatible with the pool's temporal dtype."""
    if traj_pool_copy.metadata.time_index.is_datetime:
        return datetime(2000, 6, 1)
    return 42.0


# ---------------------------------------------------------------------------
# TestTrajectorySetT0Validation
# ---------------------------------------------------------------------------


class TestTrajectorySetT0Validation:
    """Argument dispatch: missing on=, unknown alias, conflicting strategies."""

    def test_no_strategy_raises(self, traj_pool_copy) -> None:
        """set_t0() with no strategy keyword → TypeError."""
        with pytest.raises(TypeError):
            traj_pool_copy.set_t0()

    def test_multiple_strategies_raises(self, traj_pool_copy) -> None:
        """set_t0(position=0, direct=...) → TypeError."""
        with pytest.raises(TypeError):
            traj_pool_copy.set_t0(
                position=0, direct=_sentinel_t0(traj_pool_copy), on=_REF_ALIAS
            )

    def test_position_without_on_raises(self, traj_pool_copy) -> None:
        """set_t0(position=0) with no on= → TypeError."""
        with pytest.raises(TypeError, match="on="):
            traj_pool_copy.set_t0(position=0)

    def test_query_without_on_raises(self, traj_pool_copy) -> None:
        """set_t0(query=...) with no on= → TypeError."""
        with pytest.raises(TypeError, match="on="):
            traj_pool_copy.set_t0(query=pl.col("value") > 0)

    def test_unknown_on_alias_raises(self, traj_pool_copy) -> None:
        """set_t0(position=0, on='nonexistent') → KeyError."""
        with pytest.raises(KeyError):
            traj_pool_copy.set_t0(position=0, on="nonexistent_alias")

    def test_direct_with_on_warns(self, traj_pool_copy) -> None:
        """set_t0(direct=..., on=...) → UserWarning (on= ignored)."""
        value = _sentinel_t0(traj_pool_copy)
        with pytest.warns(UserWarning, match="ignored"):
            traj_pool_copy.set_t0(direct=value, on=_REF_ALIAS)

    def test_feature_with_on_warns(self, traj_pool_copy) -> None:
        """set_t0(feature=..., on=...) → UserWarning (on= ignored)."""
        id_col = traj_pool_copy.settings.id_column
        value = _sentinel_t0(traj_pool_copy)
        ids = traj_pool_copy.unique_ids
        t0_df = pl.DataFrame({id_col: ids, "t0_feat": [value] * len(ids)})
        traj_pool_copy.add_static_features(t0_df)
        with pytest.warns(UserWarning, match="ignored"):
            traj_pool_copy.set_t0(feature="t0_feat", on=_REF_ALIAS)

    def test_set_t0_returns_self(self, traj_pool_copy) -> None:
        """set_t0(...) returns the pool itself (fluent API)."""
        ret = traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        assert ret is traj_pool_copy


# ---------------------------------------------------------------------------
# TestTrajectorySetT0Position
# ---------------------------------------------------------------------------


class TestTrajectorySetT0Position:
    """position strategy via on=, snapshot t0_data()."""

    def test_position_zero(self, traj_pool_copy, snapshot) -> None:
        """set_t0(position=0, on='events') → t0_data() snapshot."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert df[_T0].is_not_null().any()
        assert snapshot == df

    def test_position_last(self, traj_pool_copy, snapshot) -> None:
        """set_t0(position=-1, on='events') → t0_data() snapshot."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(position=-1, on=_REF_ALIAS)
        df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_position_out_of_range_gives_null(self, traj_pool_copy) -> None:
        """set_t0(position=9999, on='events') → all _T0_ = null, warning."""
        with pytest.warns(UserWarning):
            traj_pool_copy.set_t0(position=9999, on=_REF_ALIAS)
        df = traj_pool_copy.t0_data(output_format="polars")
        assert df[_T0].is_null().all()


# ---------------------------------------------------------------------------
# TestTrajectorySetT0Direct
# ---------------------------------------------------------------------------


class TestTrajectorySetT0Direct:
    """Scalar and dict direct; no on= needed."""

    def test_direct_scalar(self, traj_pool_copy, snapshot) -> None:
        """set_t0(direct=<scalar>) → all _T0_ equal the scalar."""
        value = _sentinel_t0(traj_pool_copy)
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(direct=value)
        df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert (df[_T0] == value).all()
        assert snapshot == df

    def test_direct_dict_partial(self, traj_pool_copy) -> None:
        """Dict with first 3 IDs → those 3 have values, rest have null."""
        ids = traj_pool_copy.unique_ids
        partial_ids = ids[:3]
        value = _sentinel_t0(traj_pool_copy)
        mapping = {sid: value for sid in partial_ids}
        traj_pool_copy.set_t0(direct=mapping)
        df = traj_pool_copy.t0_data(output_format="polars")
        id_col = traj_pool_copy.settings.id_column

        has_value = df.filter(pl.col(id_col).is_in(partial_ids))
        assert not has_value[_T0].is_null().any()

        if len(ids) > 3:
            no_value = df.filter(~pl.col(id_col).is_in(partial_ids))
            assert no_value[_T0].is_null().all()

    def test_direct_propagates_to_all_aliases(self, traj_pool_copy) -> None:
        """Direct T0 is identical in every alias's nearest-rank column set."""
        value = _sentinel_t0(traj_pool_copy)
        traj_pool_copy.set_t0(direct=value)
        df = traj_pool_copy.t0_data(output_format="polars")
        assert (df[_T0] == value).all()


# ---------------------------------------------------------------------------
# TestTrajectorySetT0Feature
# ---------------------------------------------------------------------------


class TestTrajectorySetT0Feature:
    """Trajectory-level static feature, dtype check."""

    def _add_t0_feature(self, pool):
        """Helper: add a properly-typed 't0_feat' static feature."""
        id_col = pool.settings.id_column
        value = _sentinel_t0(pool)
        ids = pool.unique_ids
        t0_df = pl.DataFrame({id_col: ids, "t0_feat": [value] * len(ids)})
        pool.add_static_features(t0_df)

    def test_feature_strategy(self, traj_pool_copy, snapshot) -> None:
        """set_t0(feature='t0_feat') → t0_data() snapshot; T0 equals feature value."""
        self._add_t0_feature(traj_pool_copy)
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(feature="t0_feat")
        df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert df[_T0].is_not_null().all()
        assert snapshot == df

    def test_feature_dtype_mismatch_raises(self, traj_pool_copy) -> None:
        """Static feature dtype ≠ temporal dtype → TypeError."""
        # 'age' is Int64; temporal dtype is Datetime or Float64
        with pytest.raises(TypeError):
            traj_pool_copy.set_t0(feature="age")

    def test_feature_missing_column_raises(self, traj_pool_copy) -> None:
        """set_t0(feature='nonexistent') → KeyError."""
        with pytest.raises(KeyError):
            traj_pool_copy.set_t0(feature="nonexistent")


# ---------------------------------------------------------------------------
# TestTrajectorySetT0Query
# ---------------------------------------------------------------------------


class TestTrajectorySetT0Query:
    """query strategy via on=, use_first toggle."""

    def test_query_use_first(self, traj_pool_copy, snapshot) -> None:
        """set_t0(query=..., use_first=True, on='events') → snapshot."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(
            query=pl.col("status") == "error",
            use_first=True,
            on=_REF_ALIAS,
        )
        df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_query_use_last(self, traj_pool_copy, snapshot) -> None:
        """set_t0(query=..., use_first=False, on='events') → different snapshot."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(
            query=pl.col("status") == "error",
            use_first=False,
            on=_REF_ALIAS,
        )
        df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_query_no_match_gives_null(self, traj_pool_copy) -> None:
        """Query matching no rows → all _T0_ = null, UserWarning."""
        with pytest.warns(UserWarning):
            traj_pool_copy.set_t0(
                query=pl.col("value") > 99999,
                on=_REF_ALIAS,
            )
        df = traj_pool_copy.t0_data(output_format="polars")
        assert df[_T0].is_null().all()


# ---------------------------------------------------------------------------
# TestTrajectoryT0Data
# ---------------------------------------------------------------------------


class TestTrajectoryT0Data:
    """Column names, per-alias rank columns, output format."""

    def test_columns_after_set_t0(self, traj_pool_copy) -> None:
        """t0_data() columns = [id_col, _T0_, _T0_NEAREST_RANK_<alias>, ...]."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        df = traj_pool_copy.t0_data(output_format="polars")
        expected_rank_cols = [
            f"{alias}{_T0_NEAREST_RANK}"
            for alias in sorted(traj_pool_copy.sequence_pools.keys())
        ]
        assert id_col in df.columns
        assert _T0 in df.columns
        for col in expected_rank_cols:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_row_count(self, traj_pool_copy) -> None:
        """len(t0_data()) == len(traj_pool_copy)."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        df = traj_pool_copy.t0_data(output_format="polars")
        assert len(df) == len(traj_pool_copy)

    def test_pandas_output(self, traj_pool_copy) -> None:
        """t0_data() (default) returns pd.DataFrame."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        result = traj_pool_copy.t0_data()
        assert isinstance(result, pd.DataFrame)

    def test_polars_output(self, traj_pool_copy) -> None:
        """t0_data(output_format='polars') returns pl.DataFrame."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        result = traj_pool_copy.t0_data(output_format="polars")
        assert isinstance(result, pl.DataFrame)

    def test_invalid_format_raises(self, traj_pool_copy) -> None:
        """t0_data(output_format='numpy') → ValueError."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        with pytest.raises(ValueError):
            traj_pool_copy.t0_data(output_format="numpy")


# ---------------------------------------------------------------------------
# TestTrajectoryT0Default
# ---------------------------------------------------------------------------


class TestTrajectoryT0Default:
    """No set_t0() called: lazy trigger picks first alias, t0_data() works."""

    def test_t0_data_without_set_t0(self, traj_pool_copy) -> None:
        """t0_data() works without any set_t0() call (lazy trigger)."""
        df = traj_pool_copy.t0_data(output_format="polars")
        assert isinstance(df, pl.DataFrame)
        assert _T0 in df.columns
        assert len(df) == len(traj_pool_copy)

    def test_default_t0_has_values(self, traj_pool_copy) -> None:
        """Default lazy trigger produces non-null _T0_ for at least one trajectory."""
        df = traj_pool_copy.t0_data(output_format="polars")
        assert df[_T0].is_not_null().any()

    def test_traj_t0_accessible_without_set_t0(self, traj_pool_copy) -> None:
        """traj.t0 works (via pool) without any set_t0() call."""
        tid = traj_pool_copy.unique_ids[0]
        traj = traj_pool_copy[tid]
        _ = traj.t0  # must not raise

    def test_strategy_summary_before_lazy_trigger(self, traj_pool_copy) -> None:
        """Before lazy trigger, strategy_summary shows 'position=0, anchor=start'."""
        # pylint: disable=protected-access
        summary = traj_pool_copy._t0_setter.strategy_summary
        assert "position=0" in summary
        assert "anchor=start" in summary

    def test_strategy_summary_after_lazy_trigger(self, traj_pool_copy) -> None:
        """After lazy trigger, strategy_summary includes on=<first_alias>."""
        traj_pool_copy.t0_data()  # force lazy computation
        # pylint: disable=protected-access
        summary = traj_pool_copy._t0_setter.strategy_summary
        assert "on='" in summary


# ---------------------------------------------------------------------------
# TestTrajectoryT0Override
# ---------------------------------------------------------------------------


class TestTrajectoryT0Override:
    """Second set_t0() replaces first."""

    def test_second_set_t0_replaces(self, traj_pool_copy, snapshot) -> None:
        """set_t0(position=0) then set_t0(position=-1) → matches position=-1."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        traj_pool_copy.set_t0(position=-1, on=_REF_ALIAS)
        df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert snapshot == df

    def test_override_changes_values(self, traj_pool_copy) -> None:
        """Values after second set_t0() differ from first (if pool has >1 row)."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        df_first = traj_pool_copy.t0_data(output_format="polars").sort(id_col)

        traj_pool_copy.set_t0(direct=_sentinel_t0(traj_pool_copy))
        df_second = traj_pool_copy.t0_data(output_format="polars").sort(id_col)

        assert not df_first[_T0].equals(df_second[_T0])


# ---------------------------------------------------------------------------
# TestTrajectoryT0Propagation
# ---------------------------------------------------------------------------


class TestTrajectoryT0Propagation:
    """traj.t0, traj.t0_nearest_rank, per-alias values."""

    def test_traj_t0_not_none(self, traj_pool_copy) -> None:
        """After set_t0(position=0), traj.t0 is not None for at least one trajectory."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        ids = traj_pool_copy.unique_ids
        t0_values = [traj_pool_copy[i].t0 for i in ids[:5]]
        assert any(v is not None for v in t0_values)

    def test_traj_t0_matches_pool(self, traj_pool_copy) -> None:
        """traj.t0 matches the _T0_ value in t0_data() for that ID."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        id_col = traj_pool_copy.settings.id_column
        tid = traj_pool_copy.unique_ids[0]
        traj = traj_pool_copy[tid]

        pool_df = traj_pool_copy.t0_data(output_format="polars")
        expected_t0 = pool_df.filter(pl.col(id_col) == tid)[_T0][0]
        assert traj.t0 == expected_t0

    def test_traj_t0_nearest_rank_is_dict(self, traj_pool_copy) -> None:
        """traj.t0_nearest_rank is a dict keyed by visible alias names."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        traj = traj_pool_copy[traj_pool_copy.unique_ids[0]]
        result = traj.t0_nearest_rank
        assert isinstance(result, dict)
        visible_aliases = set(traj_pool_copy.sequence_pools.keys())
        assert set(result.keys()) <= visible_aliases

    def test_traj_t0_nearest_rank_matches_pool(self, traj_pool_copy) -> None:
        """traj.t0_nearest_rank[alias] matches the _T0_NEAREST_RANK_<alias> column."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        id_col = traj_pool_copy.settings.id_column
        tid = traj_pool_copy.unique_ids[0]
        traj = traj_pool_copy[tid]

        pool_df = traj_pool_copy.t0_data(output_format="polars")
        row = pool_df.filter(pl.col(id_col) == tid)

        for alias in traj.t0_nearest_rank:
            col_name = f"{alias}{_T0_NEAREST_RANK}"
            if col_name in pool_df.columns:
                expected = row[col_name][0]
                assert traj.t0_nearest_rank[alias] == expected

    def test_traj_t0_none_when_no_match(self, traj_pool_copy) -> None:
        """Query matching no rows → traj.t0 is None."""
        with pytest.warns(UserWarning):
            traj_pool_copy.set_t0(
                query=pl.col("value") > 99999,
                on=_REF_ALIAS,
            )
        traj = traj_pool_copy[traj_pool_copy.unique_ids[0]]
        assert traj.t0 is None

    def test_traj_t0_nearest_rank_all_none_when_t0_none(self, traj_pool_copy) -> None:
        """When T0 is None, all nearest-rank values are None."""
        with pytest.warns(UserWarning):
            traj_pool_copy.set_t0(
                query=pl.col("value") > 99999,
                on=_REF_ALIAS,
            )
        traj = traj_pool_copy[traj_pool_copy.unique_ids[0]]
        for val in traj.t0_nearest_rank.values():
            assert val is None


# ---------------------------------------------------------------------------
# TestTrajectoryT0NoPropagation
# ---------------------------------------------------------------------------


class TestTrajectoryT0NoPropagation:
    """T0 is NOT propagated to child SequencePool or Sequence objects."""

    def test_sub_pool_has_independent_setter(self, traj_pool_copy) -> None:
        """After tpool.set_t0(position=-1, on='events'), sub-pool setter is unchanged."""
        traj_pool_copy.set_t0(position=-1, on=_REF_ALIAS)
        # pylint: disable=protected-access
        for pool in traj_pool_copy.sequence_pools.values():
            assert pool._t0_setter is not traj_pool_copy._t0_setter

    def test_sub_sequence_t0_uses_sub_pool_setter(self, traj_pool_copy) -> None:
        """traj['events'].t0 uses the sub-pool's own setter, not trajectory T0.

        Trajectory T0 uses position=-1 (last row).
        Sub-pool default is position=0 (first row).
        With more than 1 row per sequence the values should differ.
        """
        traj_pool_copy.set_t0(position=-1, on=_REF_ALIAS)
        id_col = traj_pool_copy.settings.id_column
        traj_df = traj_pool_copy.t0_data(output_format="polars").sort(id_col)

        # Sub-pool computes its own T0 (default: position=0).
        events_pool = traj_pool_copy.sequence_pools[_REF_ALIAS]
        sub_df = events_pool.t0_data(output_format="polars").sort(id_col)

        # At least some T0 values should differ when sequences have > 1 row.
        non_null_mask = traj_df[_T0].is_not_null() & sub_df[_T0].is_not_null()
        if non_null_mask.sum() > 0:
            assert not traj_df.filter(non_null_mask)[_T0].equals(
                sub_df.filter(non_null_mask)[_T0]
            ), (
                "Expected trajectory T0 (position=-1) to differ from sub-pool T0 "
                "(position=0 default). Passes vacuously if all sequences have 1 row."
            )


# ---------------------------------------------------------------------------
# TestTrajectoryT0SubsetCopy
# ---------------------------------------------------------------------------


class TestTrajectoryT0SubsetCopy:
    """T0 preserved after subset() and copy()."""

    def test_t0_preserved_after_subset(self, traj_pool_copy) -> None:
        """set_t0 → subset(first 3 IDs) → t0_data() has 3 rows, values unchanged."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        id_col = traj_pool_copy.settings.id_column
        ids = traj_pool_copy.unique_ids[:3]

        original_t0 = traj_pool_copy.t0_data(output_format="polars")
        view = traj_pool_copy.subset(ids)
        subset_t0 = view.t0_data(output_format="polars")

        assert len(subset_t0) == 3
        expected = original_t0.filter(pl.col(id_col).is_in(ids)).sort(id_col)
        actual = subset_t0.sort(id_col)
        assert actual[_T0].equals(expected[_T0])

    def test_t0_preserved_after_copy(self, traj_pool_copy) -> None:
        """set_t0 → copy() → t0_data() matches the original."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        original_t0 = traj_pool_copy.t0_data(output_format="polars").sort(id_col)

        copied = traj_pool_copy.copy()
        copied_t0 = copied.t0_data(output_format="polars").sort(id_col)
        assert copied_t0[_T0].equals(original_t0[_T0])

    def test_t0_preserved_after_drop_and_rebuild(self, traj_pool_copy) -> None:
        """set_t0 → drop_sequence_pools → _T0_ unchanged, dropped alias rank absent."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        id_col = traj_pool_copy.settings.id_column

        original_t0 = traj_pool_copy.t0_data(output_format="polars").sort(id_col)

        non_ref_alias = next(
            a for a in traj_pool_copy.sequence_pools if a != _REF_ALIAS
        )
        traj_pool_copy.drop_sequence_pools(non_ref_alias)

        after_drop = traj_pool_copy.t0_data(output_format="polars").sort(id_col)
        assert after_drop[_T0].equals(original_t0[_T0])
        dropped_col = f"{non_ref_alias}{_T0_NEAREST_RANK}"
        assert dropped_col not in after_drop.columns


# ---------------------------------------------------------------------------
# TestTrajectoryT0Standalone
# ---------------------------------------------------------------------------


class TestTrajectoryT0Standalone:
    """Standalone Trajectory (no parent pool) computes default T0 lazily."""

    def test_standalone_traj_t0_accessible(self, traj_store) -> None:
        """Standalone Trajectory: traj.t0 is accessible without raising."""
        pool = TrajectoryPool(store=traj_store)
        tid = pool.unique_ids[0]
        standalone = Trajectory(
            id_value=tid,
            store=traj_store,
            id_column=pool.settings.id_column,
        )
        _ = standalone.t0  # must not raise

    def test_standalone_traj_t0_nearest_rank_is_dict(self, traj_store) -> None:
        """Standalone Trajectory: t0_nearest_rank is a dict."""
        pool = TrajectoryPool(store=traj_store)
        tid = pool.unique_ids[0]
        standalone = Trajectory(
            id_value=tid,
            store=traj_store,
            id_column=pool.settings.id_column,
        )
        result = standalone.t0_nearest_rank
        assert isinstance(result, dict)

    def test_standalone_traj_has_t0_when_data_present(self, traj_store) -> None:
        """Standalone Trajectory with data in first alias: t0 is not None."""
        pool = TrajectoryPool(store=traj_store)
        # pylint: disable=protected-access
        first_alias = pool._store_aliases[0]
        first_pool = pool.sequence_pools[first_alias]
        tid = first_pool.unique_ids[0]

        standalone = Trajectory(
            id_value=tid,
            store=traj_store,
            id_column=pool.settings.id_column,
        )
        assert standalone.t0 is not None

    def test_standalone_traj_nearest_rank_keyed_by_aliases(self, traj_store) -> None:
        """Standalone Trajectory: t0_nearest_rank keys match visible aliases."""
        pool = TrajectoryPool(store=traj_store)
        # pylint: disable=protected-access
        first_alias = pool._store_aliases[0]
        first_pool = pool.sequence_pools[first_alias]
        tid = first_pool.unique_ids[0]

        standalone = Trajectory(
            id_value=tid,
            store=traj_store,
            id_column=pool.settings.id_column,
        )
        ranks = standalone.t0_nearest_rank
        # Keys must be a subset of visible aliases.
        assert set(ranks.keys()) <= set(standalone._store_aliases)


# ---------------------------------------------------------------------------
# TestStrategySummaryOn
# ---------------------------------------------------------------------------


class TestStrategySummaryOn:
    """strategy_summary includes on='alias' after compute_from_trajectory."""

    def test_summary_before_compute(self) -> None:
        """A freshly created default T0 setter has no on= suffix."""
        fresh_setter = T0Setter.default(is_event=False)
        assert fresh_setter._on is None
        assert "on='" not in fresh_setter.strategy_summary

    def test_summary_after_set_t0_position(self, traj_pool_copy) -> None:
        """set_t0(position=0, on='events') → summary includes on='events'."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        # pylint: disable=protected-access
        summary = traj_pool_copy._t0_setter.strategy_summary
        assert f"on='{_REF_ALIAS}'" in summary

    def test_summary_after_lazy_trigger(self, traj_pool_copy) -> None:
        """Lazy trigger resolves first alias: summary includes on=<first_alias>."""
        first_alias = next(iter(traj_pool_copy.sequence_pools.keys()))
        traj_pool_copy.t0_data()  # force lazy trigger
        # pylint: disable=protected-access
        summary = traj_pool_copy._t0_setter.strategy_summary
        assert f"on='{first_alias}'" in summary

    def test_summary_feature_has_no_on(self, traj_pool_copy) -> None:
        """feature= strategy: summary has no on= suffix (trajectory-level static)."""
        id_col = traj_pool_copy.settings.id_column
        value = _sentinel_t0(traj_pool_copy)
        ids = traj_pool_copy.unique_ids
        t0_df = pl.DataFrame({id_col: ids, "t0_feat": [value] * len(ids)})
        traj_pool_copy.add_static_features(t0_df)
        traj_pool_copy.set_t0(feature="t0_feat")
        # pylint: disable=protected-access
        summary = traj_pool_copy._t0_setter.strategy_summary
        assert "on='" not in summary

    def test_summary_direct_includes_on_first_alias(self, traj_pool_copy) -> None:
        """direct= strategy: summary includes on=<first_alias> (ID dtype resolution)."""
        first_alias = next(iter(traj_pool_copy.sequence_pools.keys()))
        traj_pool_copy.set_t0(direct=_sentinel_t0(traj_pool_copy))
        # pylint: disable=protected-access
        summary = traj_pool_copy._t0_setter.strategy_summary
        assert f"on='{first_alias}'" in summary
