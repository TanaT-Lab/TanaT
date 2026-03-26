#!/usr/bin/env python3
"""Tests: trajectory pool metadata structure, cast operations, and standalone Trajectory."""

from __future__ import annotations

import pytest
import polars as pl

from tanat.metadata.trajectory import TrajectoryMetadata
from tanat.trajectory.pool import TrajectoryPool

# ---------------------------------------------------------------------------
# Trajectory pool: full metadata snapshot
# ---------------------------------------------------------------------------


class TestTrajectoryMetadata:
    """TrajectoryMetadata: type guard, static feature names, full snapshot."""

    def test_type(self, traj_pool: TrajectoryPool) -> None:
        """tpool.metadata is a TrajectoryMetadata instance."""
        assert isinstance(traj_pool.metadata, TrajectoryMetadata)

    def test_metadata(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """Full metadata (traj_id, temporal range, static features) matches snapshot."""
        assert snapshot == traj_pool.metadata.to_json_dict()


# ---------------------------------------------------------------------------
# Trajectory pool: cast operations (metadata reflects the new dtype)
# ---------------------------------------------------------------------------


class TestTrajectoryPoolCast:
    """Cast operations on a trajectory pool copy: dtype changes reflected in metadata + dirty flag."""

    def test_cast_id_dtype(self, traj_pool: TrajectoryPool) -> None:
        """After cast_id(pl.String), metadata.traj_id is String."""
        pool = traj_pool.copy()
        pool.cast_id(pl.String)
        assert pool.metadata.traj_id == pl.String

    def test_cast_id_marks_dirty(self, traj_pool: TrajectoryPool) -> None:
        """cast_id marks the pool as dirty."""
        pool = traj_pool.copy()
        pool.cast_id(pl.String)
        assert pool.is_dirty

    def test_cast_static_metadata(self, traj_pool: TrajectoryPool) -> None:
        """After cast_static_features({'group': Categorical}), is_categorical_feature returns True."""
        pool = traj_pool.copy()
        pool.cast_static_features({"group": pl.Categorical})
        assert pool.metadata.is_categorical_feature("group")

    def test_cast_to_timestep_on_datetime_raises(
        self, traj_pool: TrajectoryPool
    ) -> None:
        """cast_to_timestep raises TypeError on a datetime pool."""
        if not traj_pool.metadata.time_index.is_datetime:
            pytest.skip("only applies to datetime pools")
        with pytest.raises(TypeError):
            traj_pool.copy().cast_to_timestep(pl.Int64)


# ---------------------------------------------------------------------------
# Trajectory pool: cast propagation to child sequence pools
# ---------------------------------------------------------------------------


class TestTrajectoryPoolCastPropagation:
    """Casts on TrajectoryPool propagate to child SequencePool objects.

    Two propagation paths are exercised:
    - **lazy**: pools not yet built → _build_pools applies the cast at construction
    - **eager**: pools already accessed → _sync_pool_casts updates them in place
    """

    def test_cast_id_propagates_lazy(self, traj_pool: TrajectoryPool) -> None:
        """cast_id before any pool access → child pools built with the new seq_id dtype."""
        pool = traj_pool.copy()
        pool.cast_id(pl.String)
        # Access AFTER cast: lazy path (_build_pools applies _casts.id)
        for seq_pool in pool.sequence_pools.values():
            assert seq_pool.metadata.seq_id == pl.String

    def test_cast_id_propagates_eager(self, traj_pool: TrajectoryPool) -> None:
        """cast_id after pool access → _sync_pool_casts updates existing sub-pools in place."""
        pool = traj_pool.copy()
        _ = pool.sequence_pools  # materialise pools first (eager path)
        pool.cast_id(pl.String)
        for seq_pool in pool.sequence_pools.values():
            assert seq_pool.metadata.seq_id == pl.String

    def test_cast_to_datetime_propagates(self, traj_pool: TrajectoryPool) -> None:
        """cast_to_datetime('ms') propagates temporal unit to all child sequence pools."""
        if not traj_pool.metadata.time_index.is_datetime:
            pytest.skip("only applies to datetime pools")
        pool = traj_pool.copy()
        pool.cast_to_datetime("ms")
        for seq_pool in pool.sequence_pools.values():
            assert seq_pool.metadata.is_datetime
            assert seq_pool.metadata.time_index.unit == "ms"

    def test_cast_to_timestep_propagates(self, traj_pool: TrajectoryPool) -> None:
        """cast_to_timestep propagates non-datetime flag to all child sequence pools."""
        if traj_pool.metadata.time_index.is_datetime:
            pytest.skip("only applies to timestep pools")
        pool = traj_pool.copy()
        pool.cast_to_timestep(pl.Int64)
        for seq_pool in pool.sequence_pools.values():
            assert not seq_pool.metadata.is_datetime


# ---------------------------------------------------------------------------
# Standalone Trajectory: metadata without cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("traj_case", ["partial", "complete"])
class TestStandaloneTrajectoryMetadata:
    """A Trajectory built directly from a store infers its own metadata.

    partial  (id=1)  : absent from static.csv, static stats are null.
    complete (id=11) : present in both trajectory and static data, real stats.
    """

    def test_type(self, standalone_trajs, traj_case: str) -> None:
        """metadata is a TrajectoryMetadata instance."""
        assert isinstance(standalone_trajs[traj_case].metadata, TrajectoryMetadata)

    def test_traj_id_dtype(self, standalone_trajs, traj_case: str) -> None:
        """metadata.traj_id matches the stored dtype (Int64)."""
        assert standalone_trajs[traj_case].metadata.traj_id == pl.Int64

    def test_no_parent_pool(self, standalone_trajs, traj_case: str) -> None:
        """Standalone trajectory has no parent pool; it infers its own metadata."""
        assert (  # pylint: disable=protected-access
            standalone_trajs[traj_case]._parent_pool is None
        )

    def test_metadata(self, standalone_trajs, traj_case: str, snapshot) -> None:
        """Full metadata snapshot: static stats are null (partial) or real (complete)."""
        assert snapshot == standalone_trajs[traj_case].metadata.to_json_dict()


# ---------------------------------------------------------------------------
# Standalone Trajectory: cast_recipe passed at construction
# ---------------------------------------------------------------------------


class TestPoolCastToTrajectory:
    """Cast applied at pool level is reflected in child trajectory metadata."""

    def test_cast_id_reflected(self, traj_pool: TrajectoryPool) -> None:
        """cast_id(pl.String) on pool → trajectory.metadata.traj_id == pl.String."""
        pool = traj_pool.copy()
        pool.cast_id(pl.String)
        traj = pool[str(pool.unique_ids[0])]
        assert traj.metadata.traj_id == pl.String

    def test_cast_static_feature_reflected(self, traj_pool: TrajectoryPool) -> None:
        """cast_static_features on pool → trajectory reflects the cast."""
        pool = traj_pool.copy()
        pool.cast_static_features({"group": pl.Categorical})
        traj = pool[pool.unique_ids[0]]
        assert traj.metadata.is_categorical_feature("group")


# ---------------------------------------------------------------------------
# Trajectory pool: static feature scoping propagation to child Trajectory
# ---------------------------------------------------------------------------


class TestTrajectoryPoolScoping:
    """Static feature subset via get_trajectories() → traj.metadata reflects only visible features."""

    def test_default_has_all_static(self, traj_pool: TrajectoryPool) -> None:
        """Trajectory built via tpool[id] (no subset) exposes all pool static features."""
        if not traj_pool.settings.static_features:
            pytest.skip("no static features")
        traj = traj_pool[traj_pool.unique_ids[0]]
        assert len(traj.metadata.static_features) == len(
            traj_pool.settings.static_features
        )

    def test_static_subset_reduces_metadata(self, traj_pool: TrajectoryPool) -> None:
        """get_trajectories(static_features=subset) → traj.metadata contains only those features."""
        if not traj_pool.settings.static_features:
            pytest.skip("no static features")
        subset = traj_pool.settings.static_features[:1]
        trajs = traj_pool.get_trajectories(static_features=subset)
        traj = next(iter(trajs.values()))
        assert [f.name for f in traj.metadata.static_features] == subset

    def test_pool_metadata_unchanged(self, traj_pool: TrajectoryPool) -> None:
        """Scoping a child trajectory must not mutate the parent pool's metadata."""
        if not traj_pool.settings.static_features:
            pytest.skip("no static features")
        n_before = len(traj_pool.metadata.static_features)
        _ = traj_pool.get_trajectories(
            static_features=traj_pool.settings.static_features[:1]
        )
        assert len(traj_pool.metadata.static_features) == n_before

    def test_settings_static_features_sorted(self, traj_pool: TrajectoryPool) -> None:
        """settings.static_features is always in alphabetical order (normalised at construction)."""
        if not traj_pool.settings.static_features:
            pytest.skip("no static features")
        names = traj_pool.settings.static_features
        assert names == sorted(names)

    def test_metadata_static_features_sorted(self, traj_pool: TrajectoryPool) -> None:
        """metadata.static_features is always in alphabetical order (built by build_feature_metadata)."""
        if not traj_pool.settings.static_features:
            pytest.skip("no static features")
        names = [f.name for f in traj_pool.metadata.static_features]
        assert names == sorted(names)
