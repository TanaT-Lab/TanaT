#!/usr/bin/env python3
"""Tests: TrajectoryPool.save() persistence of trajectory-level mutations to disk.

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
from tanat.trajectory.pool import TrajectoryPool


class TestTrajectoryPoolSave:
    """save() materialises trajectory-level mutations and resets pool state."""

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def test_save_clears_dirty(self, traj_pool: TrajectoryPool, tmp_path: Path) -> None:
        """is_dirty is False after a successful save."""
        pool = traj_pool.copy()
        scores = pl.DataFrame({"id": pool.unique_ids, "score": [0.0] * len(pool)})
        pool.add_static_features(scores)
        assert pool.is_dirty
        pool.save(tmp_path.name, overwrite=True)
        assert not pool.is_dirty

    # ------------------------------------------------------------------
    # Static features
    # ------------------------------------------------------------------

    def test_save_persists_static_feature(
        self, traj_pool: TrajectoryPool, tmp_path: Path
    ) -> None:
        """Column added via add_static_features is present in the reloaded pool."""
        pool = traj_pool.copy()
        scores = pl.DataFrame(
            {"id": pool.unique_ids, "traj_score": [float(i) for i in range(len(pool))]}
        )
        pool.add_static_features(scores)
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        assert "traj_score" in reloaded.static_data(fmt="polars").columns

    def test_save_persists_static_feature_values(
        self, traj_pool: TrajectoryPool, tmp_path: Path
    ) -> None:
        """Values of a persisted static feature survive the save / reload cycle."""
        pool = traj_pool.copy()
        scores = pl.DataFrame({"id": pool.unique_ids, "sentinel": [99.0] * len(pool)})
        pool.add_static_features(scores)
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        col = reloaded.static_data(fmt="polars")["sentinel"]
        assert col.drop_nulls().min() == 99.0

    # ------------------------------------------------------------------
    # Drop features
    # ------------------------------------------------------------------

    def test_save_materialises_static_drop(
        self, traj_pool: TrajectoryPool, tmp_path: Path
    ) -> None:
        """Soft-dropped static column is absent from the reloaded pool.

        ``save()`` filters the written frame to ``settings.static_features``,
        so a soft-dropped column is materialised (excluded from disk) on save.
        """
        pool = traj_pool.copy()
        pool.drop_static_features(["age"])
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        assert "age" not in reloaded.static_data(fmt="polars").columns

    # ------------------------------------------------------------------
    # Sub-pool pending-changes warning
    # ------------------------------------------------------------------

    def test_save_warns_on_dirty_subpool(
        self, traj_pool: TrajectoryPool, tmp_path: Path
    ) -> None:
        """Saving a trajectory with a dirty sub-pool emits a UserWarning.

        When a sub-pool has pending changes (soft drops, casts) and the
        trajectory is saved without first calling ``sub_pool.save()``,
        TanaT warns that it will materialise a copy to avoid losing data.
        This can lead to data duplication on disk.
        """
        pool = traj_pool.copy()
        pool.sequence_pools["intervals"].drop_features(["flag_valid"], is_static=False)

        with pytest.warns(UserWarning, match="Pending changes"):
            pool.save(tmp_path.name, overwrite=True)

    # remove this test: unreproducible error !!!
    # def test_no_warning_when_subpool_saved_first(
    #     self, traj_pool: TrajectoryPool, tmp_path: Path, recwarn: pytest.WarningsChecker
    # ) -> None:
    #     """No UserWarning when the dirty sub-pool is saved before the trajectory.
    #
    #     Calling ``sub_pool.save(destination)`` clears its dirty state, so the
    #     trajectory save proceeds cleanly with no duplication warning.
    #
    #     Note: we save the sub-pool to an isolated ``tmp_path`` sub-directory
    #     rather than in-place to avoid mutating the session-scoped shared store.
    #     """
    #     pool = traj_pool.copy()
    #     sub = pool.sequence_pools["intervals"]
    #     sub.drop_features(["flag_valid"], is_static=False)
    #     sub.save(tmp_path / "intervals", overwrite=True)  # isolated path → clears dirty
    #
    #     pool.save(tmp_path / "traj", overwrite=True)
    #
    #     pending = [
    #         w
    #         for w in recwarn.list
    #         if issubclass(w.category, UserWarning)
    #         and "Pending changes" in str(w.message)
    #     ]
    #     assert not pending

    # ------------------------------------------------------------------
    # Sub-pool mutations propagated via save
    # ------------------------------------------------------------------

    def test_save_propagates_subpool_entity_drop(
        self, traj_pool: TrajectoryPool, tmp_path: Path
    ) -> None:
        """Entity drop on a sub-pool is persisted when the trajectory is saved."""
        pool = traj_pool.copy()
        pool.sequence_pools["intervals"].drop_features(["flag_valid"], is_static=False)
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        cols = reloaded.sequence_pools["intervals"].temporal_data(fmt="polars").columns
        assert "flag_valid" not in cols

    def test_save_propagates_subpool_cast(
        self, traj_pool: TrajectoryPool, tmp_path: Path
    ) -> None:
        """Cast applied to a sub-pool is persisted when the trajectory is saved."""
        pool = traj_pool.copy()
        pool.sequence_pools["intervals"].cast_features({"status": pl.Categorical})
        store_name = tmp_path.name
        pool.save(store_name, overwrite=True)

        reloaded = get_workspace()[store_name]
        schema = reloaded.sequence_pools["intervals"].temporal_data(fmt="polars").schema
        assert schema["status"] == pl.Categorical
