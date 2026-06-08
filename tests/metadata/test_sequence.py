#!/usr/bin/env python3
"""Tests: sequence pool metadata structure, cast operations, and standalone Sequence."""

from __future__ import annotations

import pytest
import polars as pl

from tanat.cast.base import CastStep
from tanat.metadata.feature import (
    ArrayInfo,
    BooleanInfo,
    CategoricalInfo,
    NumericalInfo,
    StringInfo,
    TemporalInfo,
)
from tanat.metadata.sequence import SequenceMetadata
from tanat.sequence.type.interval.sequence import IntervalSequence
from tanat.sequence.type.event.sequence import EventSequence
from tanat.sequence.type.state.sequence import StateSequence

SEQ_CLS: dict = {
    "interval": IntervalSequence,
    "event": EventSequence,
    "state": StateSequence,
}

# ---------------------------------------------------------------------------
# Sequence pool: full metadata snapshot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolMetadata:
    """SequenceMetadata: type guard and full serialized snapshot."""

    def test_type(self, pools_dict: dict, pool_type: str) -> None:
        """pool.metadata is a SequenceMetadata instance."""
        assert isinstance(pools_dict[pool_type].metadata, SequenceMetadata)

    def test_metadata(self, pools_dict: dict, pool_type: str, snapshot) -> None:
        """Full metadata (seq_id, temporal, all features) matches snapshot."""
        assert snapshot == pools_dict[pool_type].metadata.to_json_dict()


# ---------------------------------------------------------------------------
# Sequence pool: feature introspection helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequenceMetadataHelpers:
    """feature_info and is_numeric_feature: behavioral checks independent of data values."""

    def test_feature_info_numeric(self, pools_dict: dict, pool_type: str) -> None:
        """feature_info('value') returns a NumericalInfo (Float64)."""
        assert isinstance(
            pools_dict[pool_type].metadata.feature_info("value"), NumericalInfo
        )

    def test_feature_info_string(self, pools_dict: dict, pool_type: str) -> None:
        """feature_info('status') returns a StringInfo (raw String, not yet cast)."""
        assert isinstance(
            pools_dict[pool_type].metadata.feature_info("status"), StringInfo
        )

    def test_feature_info_boolean(self, pools_dict: dict, pool_type: str) -> None:
        """feature_info('flag_valid') returns a BooleanInfo."""
        assert isinstance(
            pools_dict[pool_type].metadata.feature_info("flag_valid"), BooleanInfo
        )

    def test_feature_info_categorical(self, pools_dict: dict, pool_type: str) -> None:
        """After cast_features({'status': Categorical}), feature_info returns CategoricalInfo."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        assert isinstance(pool.metadata.feature_info("status"), CategoricalInfo)

    def test_feature_info_missing(self, pools_dict: dict, pool_type: str) -> None:
        """feature_info on an unknown name returns None."""
        assert pools_dict[pool_type].metadata.feature_info("nonexistent") is None

    def test_is_numeric_true(self, pools_dict: dict, pool_type: str) -> None:
        """is_numeric_feature('value') is True for a Float64 feature."""
        assert pools_dict[pool_type].metadata.is_numeric_feature("value")

    def test_is_numeric_false(self, pools_dict: dict, pool_type: str) -> None:
        """is_numeric_feature('status') is False for a String feature."""
        assert not pools_dict[pool_type].metadata.is_numeric_feature("status")

    def test_feature_info_array(self, pools_dict: dict, pool_type: str) -> None:
        """feature_info('token_emb') returns an ArrayInfo with dimension=32."""
        fi = pools_dict[pool_type].metadata.feature_info("token_emb")
        assert isinstance(fi, ArrayInfo)
        assert fi.dimension == 32

    def test_feature_info_temporal_duration(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """feature_info('observed_at') returns a TemporalInfo (datetime pools only).

        'observed_at' is a native Datetime[us] column from sequence_main.parquet;
        it is absent from timestep pools where no datetime columns are ingested.
        """
        if not pools_dict[pool_type].metadata.time_index.is_datetime:
            pytest.skip("observed_at only present in datetime pools")
        fi = pools_dict[pool_type].metadata.feature_info("observed_at")
        assert isinstance(fi, TemporalInfo)
        assert not fi.is_duration


# ---------------------------------------------------------------------------
# Sequence pool: cast operations (metadata reflects the new dtype)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolCast:
    """Cast operations on a pool copy: dtype changes reflected in metadata + dirty flag."""

    def test_cast_entity_metadata(self, pools_dict: dict, pool_type: str) -> None:
        """After cast_features({'status': Categorical}), feature_info returns CategoricalInfo."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        assert isinstance(pool.metadata.feature_info("status"), CategoricalInfo)

    def test_cast_entity_marks_dirty(self, pools_dict: dict, pool_type: str) -> None:
        """cast_features marks the pool as dirty."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        assert pool.is_dirty

    def test_cast_static_metadata(self, pools_dict: dict, pool_type: str) -> None:
        """After cast_features({'group': Categorical}, is_static=True), feature_info returns CategoricalInfo."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"group": pl.Categorical}, is_static=True)
        assert isinstance(
            pool.metadata.feature_info("group", is_static=True), CategoricalInfo
        )

    def test_cast_id_dtype(self, pools_dict: dict, pool_type: str) -> None:
        """After cast_id(pl.String), metadata.seq_id is String."""
        pool = pools_dict[pool_type].copy()
        pool.cast_id(pl.String)
        assert pool.metadata.seq_id == pl.String

    def test_cast_to_timestep_on_datetime_raises(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """cast_to_timestep raises TypeError on a datetime pool."""
        pool = pools_dict[pool_type]
        if not pool.metadata.time_index.is_datetime:
            pytest.skip("only applies to datetime pools")
        with pytest.raises(TypeError):
            pool.copy().cast_to_timestep(pl.Int64)

    def test_cast_duration_feature_unit(self, pools_dict: dict, pool_type: str) -> None:
        """cast_features({'duration': pl.Duration('ms')}) from numeric → TemporalInfo with ms unit.

        'duration' is stored as Int64 in both datetime and timestep pools (CSV source).
        Casting it to Duration promotes the feature to a TemporalInfo.
        """
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"duration": pl.Duration("ms")})
        fi = pool.metadata.feature_info("duration")
        assert isinstance(fi, TemporalInfo)
        assert fi.is_duration
        assert "ms" in fi.dtype

    def test_cast_chained_recipe_accumulates(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Two successive cast_features() build a recipe instead of overwriting.

        Float64 → [Float32] → [Float32, Float64]: the second call appends to the
        existing recipe rather than replacing it.
        """
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"value": pl.Float32})
        pool.cast_features({"value": pl.Float64})
        recipe = pool._casts.entity["value"]  # pylint: disable=protected-access
        assert recipe == [CastStep(pl.Float32), CastStep(pl.Float64)]

    def test_cast_chained_metadata_reflects_final_step(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """After Float32 → Float64 chain, metadata exposes the final dtype (Float64).

        The intermediate Float32 step is part of the recipe but the observable
        dtype in metadata must reflect the last cast in the chain.
        """
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"value": pl.Float32})
        pool.cast_features({"value": pl.Float64})
        fi = pool.metadata.feature_info("value")
        assert isinstance(fi, NumericalInfo)
        assert fi.dtype == str(pl.Float64)


# ---------------------------------------------------------------------------
# Sequence pool: cast propagation to child Sequence objects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolCastPropagation:
    """Casts on SequencePool propagate to Sequence children built from it.

    Sequences are created on demand: each pool[id] call passes the pool's
    (recomputed) metadata as parent_metadata to the new Sequence instance.
    """

    def test_cast_id_propagates(self, pools_dict: dict, pool_type: str) -> None:
        """cast_id → Sequence built from the pool reflects the new seq_id dtype."""
        pool = pools_dict[pool_type].copy()
        pool.cast_id(pl.String)
        seq = pool[pool.unique_ids[0]]
        assert seq.metadata.seq_id == pl.String

    def test_cast_features_propagates(self, pools_dict: dict, pool_type: str) -> None:
        """cast_features({'status': Categorical}) → Sequence metadata shows CategoricalInfo."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        seq = pool[pool.unique_ids[0]]
        assert isinstance(seq.metadata.feature_info("status"), CategoricalInfo)

    def test_cast_to_datetime_propagates(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """cast_to_datetime('ms') → Sequence metadata reflects the new unit."""
        if not pools_dict[pool_type].metadata.time_index.is_datetime:
            pytest.skip("only applies to datetime pools")
        pool = pools_dict[pool_type].copy()
        pool.cast_to_datetime("ms")
        seq = pool[pool.unique_ids[0]]
        assert seq.metadata.time_index.is_datetime
        assert seq.metadata.time_index.unit == "ms"

    def test_cast_to_timestep_propagates(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """cast_to_timestep → Sequence metadata reflects non-datetime temporal type."""
        if pools_dict[pool_type].metadata.time_index.is_datetime:
            pytest.skip("only applies to timestep pools")
        pool = pools_dict[pool_type].copy()
        pool.cast_to_timestep(pl.Int64)
        seq = pool[pool.unique_ids[0]]
        assert not seq.metadata.time_index.is_datetime

    def test_chained_features_propagates(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """Chained recipe propagates to child Sequence: seq.metadata shows the final dtype.

        After Float64 → Float32 → Float64, the Sequence built from the pool must
        expose Float64 as the effective dtype for 'value' in its own metadata.
        """
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"value": pl.Float32})
        pool.cast_features({"value": pl.Float64})
        seq = pool[pool.unique_ids[0]]
        fi = seq.metadata.feature_info("value")
        assert isinstance(fi, NumericalInfo)
        assert fi.dtype == str(pl.Float64)


# ---------------------------------------------------------------------------
# Standalone Sequence: metadata without cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_case", ["partial", "complete"])
class TestStandaloneSequenceMetadata:
    """A Sequence built directly from a store infers its own metadata.

    partial  (id=1)  : absent from static.csv, static stats are null.
    complete (id=11) : present in both seq and static data, real stats.
    """

    def test_type(self, standalone_seqs, seq_case) -> None:
        """metadata is a SequenceMetadata instance."""
        assert isinstance(standalone_seqs[seq_case].metadata, SequenceMetadata)

    def test_seq_id_dtype(self, standalone_seqs, seq_case) -> None:
        """metadata.seq_id matches the stored dtype (Int64)."""
        assert standalone_seqs[seq_case].metadata.seq_id == pl.Int64

    def test_no_parent_pool(self, standalone_seqs, seq_case) -> None:
        """Standalone sequence has no parent pool; it infers its own metadata."""
        assert (
            standalone_seqs[seq_case]._parent_pool is None
        )  # pylint: disable=protected-access

    def test_metadata(self, standalone_seqs, seq_case, snapshot) -> None:
        """Full metadata snapshot: static stats are null (partial) or real (complete)."""
        assert snapshot == standalone_seqs[seq_case].metadata.to_json_dict()


# ---------------------------------------------------------------------------
# Pool cast propagation to child Sequence via _parent_pool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestPoolCastToSequence:
    """Casts applied on a pool are reflected in Sequences built from it.

    cast_recipe is a pool-level concern: ``pool.cast_id()`` /
    ``pool.cast_features()`` are the canonical entry points.
    Sequences read casts lazily from ``_parent_pool._casts``.
    """

    def test_cast_id_reflected(self, pools_dict: dict, pool_type: str) -> None:
        """pool.cast_id(pl.String) → pool[id].metadata.seq_id == pl.String."""
        pool = pools_dict[pool_type].copy()
        pool.cast_id(pl.String)
        id_value = pool.unique_ids[0]
        seq = pool[id_value]
        assert seq.metadata.seq_id == pl.String

    def test_cast_entity_feature_reflected(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """pool.cast_features({'status': Categorical}) → CategoricalInfo in seq.metadata."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        seq = pool[pool.unique_ids[0]]
        assert isinstance(seq.metadata.feature_info("status"), CategoricalInfo)


# ---------------------------------------------------------------------------
# Sequence pool: feature scoping propagation to child Sequence objects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestSequencePoolScoping:
    """Feature subset via get_sequences() → seq.metadata reflects only visible features."""

    def test_default_has_all_entity_features(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """seq built via pool[id] (no subset) exposes all pool entity features."""
        pool = pools_dict[pool_type]
        seq = pool[pool.unique_ids[0]]
        assert len(seq.metadata.entity_features) == len(pool.settings.entity_features)

    def test_entity_subset_reduces_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """get_sequences(entity_features=subset) → seq.metadata contains only the requested features."""
        pool = pools_dict[pool_type]
        subset = pool.settings.entity_features[:2]
        seqs = pool.get_sequences(entity_features=subset)
        seq = next(iter(seqs.values()))
        assert [f.name for f in seq.metadata.entity_features] == subset

    def test_static_subset_reduces_metadata(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """get_sequences(static_features=subset) → seq.metadata.static_features reflects the subset."""
        pool = pools_dict[pool_type]
        if not pool.settings.static_features:
            pytest.skip("no static features")
        subset = pool.settings.static_features[:1]
        seqs = pool.get_sequences(static_features=subset)
        seq = next(iter(seqs.values()))
        assert [f.name for f in seq.metadata.static_features] == subset

    def test_pool_metadata_unchanged(self, pools_dict: dict, pool_type: str) -> None:
        """Scoping a child sequence must not mutate the parent pool's metadata."""
        pool = pools_dict[pool_type]
        n_before = len(pool.metadata.entity_features)
        _ = pool.get_sequences(entity_features=pool.settings.entity_features[:1])
        assert len(pool.metadata.entity_features) == n_before

    def test_settings_entity_features_sorted(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """settings.entity_features is always in alphabetical order (normalised at construction)."""
        pool = pools_dict[pool_type]
        names = pool.settings.entity_features
        assert names == sorted(names)

    def test_metadata_entity_features_sorted(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """metadata.entity_features is always in alphabetical order (built by build_feature_metadata)."""
        pool = pools_dict[pool_type]
        names = [f.name for f in pool.metadata.entity_features]
        assert names == sorted(names)

    def test_settings_static_features_sorted(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """settings.static_features is always in alphabetical order (normalised at construction)."""
        pool = pools_dict[pool_type]
        names = pool.settings.static_features
        assert names == sorted(names)

    def test_metadata_static_features_sorted(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """metadata.static_features is always in alphabetical order (built by build_feature_metadata)."""
        pool = pools_dict[pool_type]
        if not pool.settings.static_features:
            pytest.skip("no static features")
        names = [f.name for f in pool.metadata.static_features]
        assert names == sorted(names)
