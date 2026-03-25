#!/usr/bin/env python3
"""
Unit tests for SequenceMetadata.scope() and TrajectoryMetadata.scope().
"""

from __future__ import annotations

import polars as pl
import pytest

from tanat.metadata.feature import NumericalInfo
from tanat.metadata.sequence import SequenceMetadata, TemporalIndexInfo
from tanat.metadata.trajectory import TrajectoryMetadata

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temporal() -> TemporalIndexInfo:
    """Minimal timestep TemporalIndexInfo (no I/O)."""
    return TemporalIndexInfo(dtype="Int64", is_datetime=False, min=0, max=100)


@pytest.fixture
def seq_metadata(temporal: TemporalIndexInfo) -> SequenceMetadata:
    """SequenceMetadata with 3 entity features and 2 static features."""
    entity = [
        NumericalInfo(name="heart_rate", dtype="Float64", min=45.0, max=180.0),
        NumericalInfo(name="blood_pressure", dtype="Float64", min=60.0, max=200.0),
        NumericalInfo(name="temperature", dtype="Float64", min=35.5, max=42.0),
    ]
    static = [
        NumericalInfo(name="age", dtype="Float64", min=18.0, max=95.0),
        NumericalInfo(name="weight", dtype="Float64", min=40.0, max=120.0),
    ]
    return SequenceMetadata(
        seq_id=pl.Int64,
        temporal=temporal,
        entity_features=entity,
        static_features=static,
    )


@pytest.fixture
def seq_metadata_no_static(temporal: TemporalIndexInfo) -> SequenceMetadata:
    """SequenceMetadata without any static features (static_features=None)."""
    entity = [
        NumericalInfo(name="heart_rate", dtype="Float64", min=45.0, max=180.0),
    ]
    return SequenceMetadata(
        seq_id=pl.Int64,
        temporal=temporal,
        entity_features=entity,
        static_features=None,
    )


@pytest.fixture
def traj_metadata(temporal: TemporalIndexInfo) -> TrajectoryMetadata:
    """TrajectoryMetadata with 3 static features."""
    static = [
        NumericalInfo(name="age", dtype="Float64", min=18.0, max=95.0),
        NumericalInfo(name="weight", dtype="Float64", min=40.0, max=120.0),
        NumericalInfo(name="score", dtype="Float64", min=0.0, max=100.0),
    ]
    return TrajectoryMetadata(
        traj_id=pl.Int64,
        temporal=temporal,
        static_features=static,
    )


# ---------------------------------------------------------------------------
# SequenceMetadata.scope()
# ---------------------------------------------------------------------------


class TestSequenceMetadataScope:
    """Invariants of SequenceMetadata.scope() — no I/O, purely in-memory."""

    def test_no_args_returns_self(self, seq_metadata: SequenceMetadata) -> None:
        """scope() with no arguments must return the exact same instance (fast-path)."""
        assert seq_metadata.scope() is seq_metadata

    def test_all_entity_features_returns_self(
        self, seq_metadata: SequenceMetadata
    ) -> None:
        """scope(entity_features=<all names>) must return self (length fast-path)."""
        all_names = [f.name for f in seq_metadata.entity_features]
        assert seq_metadata.scope(entity_features=all_names) is seq_metadata

    def test_entity_subset_filters(self, seq_metadata: SequenceMetadata) -> None:
        """scope(entity_features=[...]) must reduce entity_features to the requested names."""
        scoped = seq_metadata.scope(entity_features=["heart_rate", "temperature"])
        assert [f.name for f in scoped.entity_features] == ["heart_rate", "temperature"]

    def test_entity_subset_preserves_original_order(
        self, seq_metadata: SequenceMetadata
    ) -> None:
        """Order must follow the original metadata, not the input list order."""
        scoped = seq_metadata.scope(entity_features=["temperature", "heart_rate"])
        assert [f.name for f in scoped.entity_features] == ["heart_rate", "temperature"]

    def test_static_empty_list_gives_none(self, seq_metadata: SequenceMetadata) -> None:
        """scope(static_features=[]) must produce static_features=None (no-static convention)."""
        scoped = seq_metadata.scope(static_features=[])
        assert scoped.static_features is None

    def test_entity_unchanged_when_only_static_filtered(
        self, seq_metadata: SequenceMetadata
    ) -> None:
        """Filtering only static features must leave entity_features as the exact same list object."""
        scoped = seq_metadata.scope(static_features=["age"])
        assert scoped.entity_features is seq_metadata.entity_features

    def test_static_unchanged_when_only_entity_filtered(
        self, seq_metadata: SequenceMetadata
    ) -> None:
        """Filtering only entity features must leave static_features as the exact same list object."""
        scoped = seq_metadata.scope(entity_features=["heart_rate"])
        assert scoped.static_features is seq_metadata.static_features

    def test_temporal_and_id_preserved(self, seq_metadata: SequenceMetadata) -> None:
        """scope() must propagate seq_id and temporal unchanged."""
        scoped = seq_metadata.scope(entity_features=["heart_rate"])
        assert scoped.seq_id == seq_metadata.seq_id
        assert scoped.temporal is seq_metadata.temporal

    def test_scope_on_metadata_without_static(
        self, seq_metadata_no_static: SequenceMetadata
    ) -> None:
        """scope(static_features=[...]) on metadata with no static must return self unchanged."""
        scoped = seq_metadata_no_static.scope(static_features=["anything"])
        assert scoped is seq_metadata_no_static

    def test_combined_subset(self, seq_metadata: SequenceMetadata) -> None:
        """scope() with both axes filtered must reduce both feature lists independently."""
        scoped = seq_metadata.scope(
            entity_features=["heart_rate"], static_features=["age"]
        )
        assert len(scoped.entity_features) == 1
        assert len(scoped.static_features) == 1

    def test_idempotent(self, seq_metadata: SequenceMetadata) -> None:
        """scope(x).scope(x) must equal scope(x) (fast-path on the already-filtered result)."""
        names = ["heart_rate"]
        once = seq_metadata.scope(entity_features=names)
        twice = once.scope(entity_features=names)
        assert twice is once


# ---------------------------------------------------------------------------
# TrajectoryMetadata.scope()
# ---------------------------------------------------------------------------


class TestTrajectoryMetadataScope:
    """Invariants of TrajectoryMetadata.scope() — no I/O, purely in-memory."""

    def test_no_args_returns_self(self, traj_metadata: TrajectoryMetadata) -> None:
        """scope() with no arguments must return the exact same instance (fast-path)."""
        assert traj_metadata.scope() is traj_metadata

    def test_static_subset_filters(self, traj_metadata: TrajectoryMetadata) -> None:
        """scope(static_features=['age']) must reduce static_features to that single feature."""
        scoped = traj_metadata.scope(static_features=["age"])
        assert len(scoped.static_features) == 1
        assert scoped.static_features[0].name == "age"

    def test_all_static_features_returns_self(
        self, traj_metadata: TrajectoryMetadata
    ) -> None:
        """scope(static_features=<all names>) must return self (length fast-path)."""
        all_names = [f.name for f in traj_metadata.static_features]
        assert traj_metadata.scope(static_features=all_names) is traj_metadata

    def test_temporal_and_id_preserved(self, traj_metadata: TrajectoryMetadata) -> None:
        """scope() must propagate traj_id and temporal unchanged."""
        scoped = traj_metadata.scope(static_features=["age"])
        assert scoped.traj_id == traj_metadata.traj_id
        assert scoped.temporal is traj_metadata.temporal

    def test_idempotent(self, traj_metadata: TrajectoryMetadata) -> None:
        """scope(x).scope(x) must equal scope(x) (fast-path on the already-filtered result)."""
        names = ["age"]
        once = traj_metadata.scope(static_features=names)
        twice = once.scope(static_features=names)
        assert twice is once
