#!/usr/bin/env python3
"""Tests: strict parameter on SequencePool.cast_features.

Each test class is parametrized over the three pool types (interval, event,
state) following the project's standard pattern.
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.cast.base import CastStep, ColumnMapCast
from tanat.criterion.type.entity import EntityCriterion

# ---------------------------------------------------------------------------
# strict=True (default)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestCastFeaturesStrictDefault:
    """Default strict=True raises at probe time when values cannot be cast."""

    def test_raises_on_bad_values(self, cast_pools_dict: dict, pool_type: str) -> None:
        """probe raises TypeError when any value is incompatible with the target type."""
        pool = cast_pools_dict[pool_type].copy()
        with pytest.raises(TypeError):
            pool.cast_features({"age_str": pl.Int64})  # "bad" → Int64 raises


# ---------------------------------------------------------------------------
# strict=False (lenient)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestCastFeaturesLenient:
    """strict=False silently converts non-castable values to null."""

    def test_returns_nulls(self, cast_pools_dict: dict, pool_type: str) -> None:
        """Non-castable rows become null; no exception is raised."""
        pool = cast_pools_dict[pool_type].copy()
        pool.cast_features({"age_str": pl.Int64}, strict=False)

        df = pool.temporal_data(fmt="polars")
        assert df["age_str"].dtype == pl.Int64
        assert df["age_str"].is_null().sum() == 1  # only "bad" (id=2, row 0)

    def test_convertible_values_are_correct(
        self, cast_pools_dict: dict, pool_type: str
    ) -> None:
        """Castable values convert correctly; only the non-castable row becomes null."""
        pool = cast_pools_dict[pool_type].copy()
        pool.cast_features({"age_str": pl.Int64}, strict=False)

        assert pool.subset([1]).temporal_data(fmt="polars")["age_str"].to_list() == [
            10,
            20,
        ]
        assert pool.subset([2]).temporal_data(fmt="polars")["age_str"].to_list() == [
            None,
            40,
        ]

    def test_which_identifies_null_producing_ids(
        self, cast_pools_dict: dict, pool_type: str
    ) -> None:
        """which() on the cast column correctly surfaces IDs that produced nulls."""
        pool = cast_pools_dict[pool_type].copy()
        pool.cast_features({"age_str": pl.Int64}, strict=False)

        ids_with_null = pool.which(EntityCriterion(query=pl.col("age_str").is_null()))
        assert ids_with_null == {2}

        clean = pool.subset(list(set(pool.unique_ids) - ids_with_null))
        assert clean.temporal_data(fmt="polars")["age_str"].is_not_null().all()


# ---------------------------------------------------------------------------
# Multi-step chained recipe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestCastFeaturesChained:
    """Each cast_features call appends an independent CastStep to the recipe."""

    def test_recipe_stores_both_steps(
        self, cast_pools_dict: dict, pool_type: str
    ) -> None:
        """Two successive calls append CastStep entries with independent strict flags."""
        pool = cast_pools_dict[pool_type].copy()
        pool.cast_features({"age_str": pl.String}, strict=True)  # step 1
        pool.cast_features({"age_str": pl.Int64}, strict=False)  # step 2

        recipe = pool._casts.entity["age_str"]  # pylint: disable=protected-access
        assert recipe == [CastStep(pl.String, True), CastStep(pl.Int64, False)]

    def test_runtime_produces_correct_nulls(
        self, cast_pools_dict: dict, pool_type: str
    ) -> None:
        """The lenient step 2 produces nulls at runtime; step 1 is transparent here."""
        pool = cast_pools_dict[pool_type].copy()
        pool.cast_features({"age_str": pl.String}, strict=True)
        pool.cast_features({"age_str": pl.Int64}, strict=False)

        assert pool.subset([2]).temporal_data(fmt="polars")["age_str"].to_list() == [
            None,
            40,
        ]


# ---------------------------------------------------------------------------
# Direct probe_lf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestCastFeaturesProbe:
    """Direct probe_lf validation with strict=False."""

    def test_strict_false_does_not_raise_on_bad_sample(
        self, cast_pools_dict: dict, pool_type: str
    ) -> None:
        """probe_lf with strict=False completes silently even when sample contains bad values."""
        col_map = ColumnMapCast().append({"age_str": pl.Int64}, strict=False)
        lf = cast_pools_dict[pool_type].temporal_data(fmt="polars").lazy()
        col_map.probe_lf(lf)  # must not raise
