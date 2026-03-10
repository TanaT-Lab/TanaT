#!/usr/bin/env python3
"""Tests: explore a pool, navigate to sequences and entities."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from tanat.trajectory.pool import TrajectoryPool
from tanat.trajectory.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Sequence pool: pool-level read API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePool:
    """Pool-level read API: size, unique IDs, sequence_data and static_data schemas."""

    def test_len(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """Pool size matches snapshot."""
        assert len(pools_dict[pool_type]) == snapshot

    def test_unique_ids(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """First five unique IDs match snapshot (verifies ordering and ID type)."""
        assert snapshot == pools_dict[pool_type].unique_ids[:5]

    def test_sequence_data_schema(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """sequence_data() column schema matches snapshot."""
        df = pools_dict[pool_type].sequence_data(output_format="polars")
        assert snapshot == dict(df.schema)

    def test_static_data_schema(
        self, pools_dict: dict, pool_type: str, snapshot
    ) -> None:
        """static_data() column schema matches snapshot."""
        sd = pools_dict[pool_type].static_data(output_format="polars")
        assert sd is not None
        assert snapshot == dict(sd.schema)


# ---------------------------------------------------------------------------
# Sequence: single-sequence navigation via pool[id]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequence:
    """Navigate from pool to a single sequence via pool[id]."""

    def test_access_by_id(self, pools_dict: dict, pool_type: str) -> None:
        """pool[id].id_value matches the requested ID."""
        pool = pools_dict[pool_type]
        first_id = pool.unique_ids[0]
        assert pool[first_id].id_value == first_id

    def test_len(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """Row count for the first sequence matches snapshot."""
        pool = pools_dict[pool_type]
        assert len(pool[pool.unique_ids[0]]) == snapshot

    def test_sequence_data(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """sequence_data() for the first sequence matches snapshot."""
        pool = pools_dict[pool_type]
        seq = pool[pool.unique_ids[0]]
        df = seq.sequence_data(output_format="polars")
        assert snapshot == df.select(sorted(df.columns))


# ---------------------------------------------------------------------------
# Entity: individual event / state navigation via seq[index]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestEntity:
    """Navigate from sequence to individual events / states via seq[index]."""

    def test_data(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """entity.data() for the first event of the first sequence matches snapshot."""
        pool = pools_dict[pool_type]
        entity = pool[pool.unique_ids[0]][0]
        assert snapshot == entity.data()

    def test_temporal_extent(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """entity.temporal_extent for the first event matches snapshot."""
        pool = pools_dict[pool_type]
        entity = pool[pool.unique_ids[0]][0]
        assert snapshot == entity.temporal_extent


# ---------------------------------------------------------------------------
# Trajectory pool: pool-level read API
# ---------------------------------------------------------------------------


class TestTrajectoryPool:
    """Trajectory pool: size and unique IDs."""

    def test_len(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """Trajectory pool size matches snapshot."""
        assert len(traj_pool) == snapshot

    def test_unique_ids(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """First five trajectory IDs match snapshot."""
        assert snapshot == traj_pool.unique_ids[:5]

    def test_sequence_pools_type(self, traj_pool: TrajectoryPool) -> None:
        """sequence_pools returns a MappingProxyType (read-only mapping)."""
        assert isinstance(traj_pool.sequence_pools, MappingProxyType)

    def test_sequence_pools_read_only(self, traj_pool: TrajectoryPool) -> None:
        """Direct item assignment on sequence_pools raises TypeError."""
        alias = next(iter(traj_pool.sequence_pools))
        with pytest.raises(TypeError):
            traj_pool.sequence_pools[alias] = None  # type: ignore[index]


# ---------------------------------------------------------------------------
# Trajectory: single-trajectory navigation via tpool[id]
# ---------------------------------------------------------------------------


class TestTrajectory:
    """Navigate from trajectory pool to a single trajectory via tpool[id]."""

    def test_access_by_id(self, traj_pool: TrajectoryPool) -> None:
        """tpool[id].id_value matches the requested ID."""
        first_id = traj_pool.unique_ids[0]
        traj: Trajectory = traj_pool[first_id]
        assert traj.id_value == first_id

    def test_sub_sequence_data(self, traj_pool: TrajectoryPool, snapshot) -> None:
        """sequence_data() on the 'intervals' sub-sequence of the first trajectory matches snapshot."""
        traj: Trajectory = traj_pool[traj_pool.unique_ids[0]]
        df = traj["intervals"].sequence_data(output_format="polars")
        assert snapshot == df.select(sorted(df.columns))
