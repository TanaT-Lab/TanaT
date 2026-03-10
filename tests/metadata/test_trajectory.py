#!/usr/bin/env python3
"""Tests: trajectory pool metadata structure, cast operations, and standalone Trajectory."""

from __future__ import annotations

import pytest
import polars as pl

from tanat.metadata.trajectory import TrajectoryMetadata
from tanat.trajectory.pool import TrajectoryPool
from tanat.trajectory.trajectory import Trajectory

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
        if not traj_pool.metadata.temporal.is_datetime:
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
        if not traj_pool.metadata.temporal.is_datetime:
            pytest.skip("only applies to datetime pools")
        pool = traj_pool.copy()
        pool.cast_to_datetime("ms")
        for seq_pool in pool.sequence_pools.values():
            assert seq_pool.metadata.is_datetime
            assert seq_pool.metadata.temporal.unit == "ms"

    def test_cast_to_timestep_propagates(self, traj_pool: TrajectoryPool) -> None:
        """cast_to_timestep propagates non-datetime flag to all child sequence pools."""
        if traj_pool.metadata.temporal.is_datetime:
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

    def test_no_parent_metadata(self, standalone_trajs, traj_case: str) -> None:
        """Standalone trajectory has no parent_metadata; it infers its own."""
        assert (  # pylint: disable=protected-access
            standalone_trajs[traj_case]._parent_metadata is None
        )

    def test_metadata(self, standalone_trajs, traj_case: str, snapshot) -> None:
        """Full metadata snapshot: static stats are null (partial) or real (complete)."""
        assert snapshot == standalone_trajs[traj_case].metadata.to_json_dict()


# ---------------------------------------------------------------------------
# Standalone Trajectory: cast_recipe passed at construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("traj_case", ["partial", "complete"])
class TestStandaloneTrajectoryCast:
    """cast_recipe passed at construction is reflected in metadata."""

    def test_cast_id_reflected(
        self, standalone_trajs, traj_case: str, traj_store
    ) -> None:
        """cast_recipe={'id': pl.String} → metadata.traj_id == pl.String."""
        orig = standalone_trajs[traj_case]
        traj = Trajectory(
            id_value=str(orig.id_value),  # id_value must match the cast dtype
            store=traj_store,
            cast_recipe={"id": pl.String},
        )
        assert traj.metadata.traj_id == pl.String

    def test_cast_static_feature_reflected(
        self, standalone_trajs, traj_case: str, traj_store
    ) -> None:
        """cast_recipe with static cast → is_categorical_feature returns True."""
        orig = standalone_trajs[traj_case]
        traj = Trajectory(
            id_value=orig.id_value,
            store=traj_store,
            cast_recipe={"static": {"group": pl.Categorical}},
        )
        assert traj.metadata.is_categorical_feature("group")
