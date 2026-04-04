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

    def test_t0_data_schema(self, traj_pool_copy) -> None:
        """t0_data() has expected columns and one row per trajectory."""
        id_col = traj_pool_copy.settings.id_column
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        df = traj_pool_copy.t0_data(output_format="polars")
        assert len(df) == len(traj_pool_copy)
        assert id_col in df.columns
        assert _T0 in df.columns
        for alias in traj_pool_copy.sequence_pools:
            assert f"{alias}{_T0_NEAREST_RANK}" in df.columns

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

    def test_lazy_trigger_produces_valid_t0(self, traj_pool_copy) -> None:
        """t0_data() works without set_t0(); produces non-null values."""
        df = traj_pool_copy.t0_data(output_format="polars")
        assert isinstance(df, pl.DataFrame)
        assert _T0 in df.columns
        assert len(df) == len(traj_pool_copy)
        assert df[_T0].is_not_null().any()

    def test_traj_t0_accessible_without_set_t0(self, traj_pool_copy) -> None:
        """traj.t0 works (via pool) without any set_t0() call."""
        tid = traj_pool_copy.unique_ids[0]
        traj = traj_pool_copy[tid]
        _ = traj.t0  # must not raise


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

    def test_traj_t0_nearest_rank(self, traj_pool_copy) -> None:
        """t0_nearest_rank is a dict keyed by alias; values match t0_data() columns."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        id_col = traj_pool_copy.settings.id_column
        tid = traj_pool_copy.unique_ids[0]
        traj = traj_pool_copy[tid]
        ranks = traj.t0_nearest_rank

        assert isinstance(ranks, dict)
        assert set(ranks.keys()) <= set(traj_pool_copy.sequence_pools.keys())

        pool_df = traj_pool_copy.t0_data(output_format="polars")
        row = pool_df.filter(pl.col(id_col) == tid)
        for alias, rank in ranks.items():
            col_name = f"{alias}{_T0_NEAREST_RANK}"
            if col_name in pool_df.columns:
                assert rank == row[col_name][0]

    def test_null_t0_when_no_match(self, traj_pool_copy) -> None:
        """Query matching no rows: traj.t0 is None; all ranks are None."""
        with pytest.warns(UserWarning):
            traj_pool_copy.set_t0(
                query=pl.col("value") > 99999,
                on=_REF_ALIAS,
            )
        traj = traj_pool_copy[traj_pool_copy.unique_ids[0]]
        assert traj.t0 is None
        for val in traj.t0_nearest_rank.values():
            assert val is None


# ---------------------------------------------------------------------------
# TestTrajectoryT0SubPoolPropagation
# ---------------------------------------------------------------------------


class TestTrajectoryT0SubPoolPropagation:
    """After set_t0(), sub-pool T0 equals trajectory T0; nearest rank is per-pool."""

    def test_sub_pool_nearest_rank_differs(self, traj_pool_copy) -> None:
        """Each sub-pool computes its own nearest rank against its own temporal grid."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        id_col = traj_pool_copy.settings.id_column
        tid = traj_pool_copy.unique_ids[0]
        traj = traj_pool_copy[tid]
        # Each sub-pool has an independent _T0_NEAREST_RANK_ column computed
        # against its own temporal index.  Verify the column is present and
        # the rank is a non-negative integer for IDs that have sequence data.
        for alias, pool in traj_pool_copy.sequence_pools.items():
            pool_df = pool.t0_data(output_format="polars")
            assert _T0_NEAREST_RANK in pool_df.columns
            row = pool_df.filter(pl.col(id_col) == tid)
            if row.height > 0 and row[_T0][0] is not None:
                # Nearest rank must be an integer ≥ 0
                assert traj[alias].t0_nearest_rank is not None
                assert traj[alias].t0_nearest_rank >= 0

    def test_sub_pool_parent_pool(self, traj_pool_copy) -> None:
        """Sub-pools reference parent via _parent_pool; after set_t0 display shows 'from trajectory'."""
        for pool in traj_pool_copy.sequence_pools.values():
            # pylint: disable=protected-access
            assert pool._parent_pool is traj_pool_copy

        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        for pool in traj_pool_copy.sequence_pools.values():
            # pylint: disable=protected-access
            assert "(from trajectory)" in pool._t0_display_label()

    def test_sub_pool_lazy_trigger_before_set_t0(self, traj_pool_copy) -> None:
        """Before set_t0(), accessing t0_data() on a sub-pool triggers lazy parent
        computation and returns a valid DataFrame (no RuntimeError / None crash)."""
        for pool in traj_pool_copy.sequence_pools.values():
            df = pool.t0_data(output_format="polars")
            assert isinstance(df, pl.DataFrame)
            assert _T0 in df.columns

    def test_sub_pool_t0_updates_after_second_set_t0(self, traj_pool_copy) -> None:
        """After a second set_t0(), sub-pool T0 reflects the new strategy."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        tid = traj_pool_copy.unique_ids[0]
        t0_first = {
            alias: pool.t0_data(output_format="polars").filter(
                pl.col(traj_pool_copy.settings.id_column) == tid
            )[_T0][0]
            for alias, pool in traj_pool_copy.sequence_pools.items()
        }

        traj_pool_copy.set_t0(direct=_sentinel_t0(traj_pool_copy))
        t0_second = {
            alias: pool.t0_data(output_format="polars").filter(
                pl.col(traj_pool_copy.settings.id_column) == tid
            )[_T0][0]
            for alias, pool in traj_pool_copy.sequence_pools.items()
        }

        # At least one alias must have a different T0 after the second set_t0().
        assert any(t0_first[a] != t0_second[a] for a in t0_first)

    def test_propagation_survives_subset(self, traj_pool_copy) -> None:
        """T0 is consistent after subset(): sub-pool T0 still matches traj.t0."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        ids = traj_pool_copy.unique_ids[:3]
        view = traj_pool_copy.subset(ids)
        for tid in view.unique_ids:
            traj = view[tid]
            expected_t0 = traj.t0
            for alias in view.sequence_pools:
                assert traj[alias].t0 == expected_t0, (
                    f"After subset(), traj['{alias}'].t0 = {traj[alias].t0!r} "
                    f"!= traj.t0 = {expected_t0!r} for id={tid!r}"
                )

    def test_sub_pool_set_t0_locked(self, traj_pool_copy) -> None:
        """set_t0() on a managed sub-pool raises RuntimeError."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        pool = next(iter(traj_pool_copy.sequence_pools.values()))
        with pytest.raises(RuntimeError, match="set_t0"):
            pool.set_t0(position=0)

    def test_sequence_pool_to_sequence_t0_matches_traj(self, traj_pool_copy) -> None:
        """TrajectoryPool → SequencePool → Sequence: seq.t0 matches traj.t0."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        tid = traj_pool_copy.unique_ids[0]
        expected_t0 = traj_pool_copy[tid].t0
        for alias, pool in traj_pool_copy.sequence_pools.items():
            if tid in pool.unique_ids:
                seq = pool[tid]
                assert seq.t0 == expected_t0, (
                    f"sequence_pools['{alias}'][{tid!r}].t0 = {seq.t0!r} "
                    f"!= traj.t0 = {expected_t0!r}"
                )

    def test_trajectory_to_sequence_t0_matches_traj(self, traj_pool_copy) -> None:
        """TrajectoryPool → Trajectory → Sequence: traj[alias].t0 matches traj.t0."""
        traj_pool_copy.set_t0(position=0, on=_REF_ALIAS)
        tid = traj_pool_copy.unique_ids[0]
        traj = traj_pool_copy[tid]
        expected_t0 = traj.t0
        for alias in traj:
            seq = traj[alias]
            assert (
                seq.t0 == expected_t0
            ), f"traj['{alias}'].t0 = {seq.t0!r} != traj.t0 = {expected_t0!r}"


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

    def test_standalone_t0_properties(self, traj_store) -> None:
        """Standalone Trajectory: t0 is not None, nearest_rank is a dict keyed by aliases."""
        pool = TrajectoryPool(store=traj_store)
        # pylint: disable=protected-access
        tid = pool.sequence_pools[pool._store_aliases[0]].unique_ids[0]
        standalone = Trajectory(
            id_value=tid,
            store=traj_store,
            id_column=pool.settings.id_column,
        )
        assert standalone.t0 is not None
        ranks = standalone.t0_nearest_rank
        assert isinstance(ranks, dict)
        assert set(ranks.keys()) <= set(standalone._store_aliases)

    def test_standalone_sequence_inherits_t0(self, traj_store) -> None:
        """Standalone traj['alias'].t0 matches traj.t0 via ephemeral pool delegation."""
        pool = TrajectoryPool(store=traj_store)
        # pylint: disable=protected-access
        tid = pool.sequence_pools[pool._store_aliases[0]].unique_ids[0]
        standalone = Trajectory(
            id_value=tid,
            store=traj_store,
            id_column=pool.settings.id_column,
        )
        expected_t0 = standalone.t0
        for alias in standalone:
            seq = standalone[alias]
            assert seq.t0 == expected_t0


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
