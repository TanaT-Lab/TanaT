#!/usr/bin/env python3
"""Tests: SequencePool.save() persistence of mutations to disk.

Pattern: copy session pool → mutate → save(store_name) → ws[store_name] → assert.

``tmp_path.name`` gives a pytest-generated unique identifier per test
invocation (function-scoped), used as the workspace store name so that:

- saves land in the shared session workspace without collision,
- reload goes through ``ws[store_name]``, the same path a real user takes.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tanat import get_workspace


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolSave:
    """save() materialises in-memory mutations and resets pool state."""

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def test_save_clears_dirty(
        self, pools_dict: dict, pool_type: str, tmp_path: Path
    ) -> None:
        """is_dirty is False after a successful save."""
        pool = pools_dict[pool_type].copy()
        n = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"tmp": [0.0] * n}))
        assert pool.is_dirty
        pool.save(tmp_path.name, overwrite=True)
        assert not pool.is_dirty

    # ------------------------------------------------------------------
    # Entity features
    # ------------------------------------------------------------------

    def test_save_persists_entity_feature(
        self, pools_dict: dict, pool_type: str, tmp_path: Path
    ) -> None:
        """Column added via add_entity_features is present in the reloaded pool."""
        pool = pools_dict[pool_type].copy()
        n = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"saved_feat": [1.0] * n}))
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        assert "saved_feat" in reloaded.sequence_data(output_format="polars").columns

    def test_save_persists_entity_feature_values(
        self, pools_dict: dict, pool_type: str, tmp_path: Path
    ) -> None:
        """Values of a persisted entity feature survive the save / reload cycle."""
        pool = pools_dict[pool_type].copy()
        n = pool.sequence_data(output_format="polars").height
        pool.add_entity_features(pl.DataFrame({"constant": [42.0] * n}))
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        col = reloaded.sequence_data(output_format="polars")["constant"]
        assert col.drop_nulls().min() == 42.0
        assert col.drop_nulls().max() == 42.0

    # ------------------------------------------------------------------
    # Static features
    # ------------------------------------------------------------------

    def test_save_persists_static_feature(
        self, pools_dict: dict, pool_type: str, tmp_path: Path
    ) -> None:
        """Column added via add_static_features is present in the reloaded pool."""
        pool = pools_dict[pool_type].copy()
        summary = pool.apply(
            pl.col("value").mean().alias("v_mean"),
            by_id=True,
            output_format="polars",
        )
        pool.add_static_features(summary)
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        assert "v_mean" in reloaded.static_data(output_format="polars").columns

    # ------------------------------------------------------------------
    # Drop features
    # ------------------------------------------------------------------

    def test_save_materialises_entity_drop(
        self, pools_dict: dict, pool_type: str, tmp_path: Path
    ) -> None:
        """Soft-dropped entity column is absent from the reloaded pool.

        ``save()`` filters the written frame to ``settings.entity_features``,
        so a soft-dropped column is materialised (excluded from disk) on save.
        """
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["flag_valid"], is_static=False)
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        assert (
            "flag_valid" not in reloaded.sequence_data(output_format="polars").columns
        )

    def test_save_materialises_static_drop(
        self, pools_dict: dict, pool_type: str, tmp_path: Path
    ) -> None:
        """Soft-dropped static column is absent from the reloaded pool.

        ``save()`` filters the written frame to ``settings.static_features``,
        so a soft-dropped column is materialised (excluded from disk) on save.
        """
        pool = pools_dict[pool_type].copy()
        pool.drop_features(["age"], is_static=True)
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        assert "age" not in reloaded.static_data(output_format="polars").columns

    # ------------------------------------------------------------------
    # Cast features
    # ------------------------------------------------------------------

    def test_save_persists_cast(
        self, pools_dict: dict, pool_type: str, tmp_path: Path
    ) -> None:
        """Dtype cast applied before save is reflected in the reloaded pool schema."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        assert (
            reloaded.sequence_data(output_format="polars").schema["status"]
            == pl.Categorical
        )
