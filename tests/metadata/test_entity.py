#!/usr/bin/env python3
"""Tests: Entity metadata, from a sequence (parent_metadata path) and standalone."""

from __future__ import annotations

import pytest
import polars as pl

from tanat.metadata.feature import (
    ArrayInfo,
    CategoricalInfo,
    FeatureInfo,
    NumericalInfo,
)

# ---------------------------------------------------------------------------
# Entity from a Sequence (parent_metadata reuse path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_type", ["interval", "event", "state"])
class TestEntityFromSequence:
    """Entity obtained via seq[rank] inherits parent_metadata from the pool.

    No extra I/O: metadata keys and types come from the parent SequenceMetadata.
    """

    def _get_entity(self, pools_dict: dict, pool_type: str):
        pool = pools_dict[pool_type]
        seq = pool[pool.unique_ids[0]]
        return seq[0]

    def test_parent_metadata_set(self, pools_dict: dict, pool_type: str) -> None:
        """Entity obtained from a sequence carries parent_metadata."""
        entity = self._get_entity(pools_dict, pool_type)
        assert entity._parent_metadata is not None  # pylint: disable=protected-access

    def test_metadata_is_dict_of_feature_info(
        self, pools_dict: dict, pool_type: str
    ) -> None:
        """entity.metadata returns a dict mapping names to FeatureInfo instances."""
        entity = self._get_entity(pools_dict, pool_type)
        meta = entity.metadata
        assert isinstance(meta, dict)
        assert all(isinstance(v, FeatureInfo) for v in meta.values())

    def test_feature_info_numeric(self, pools_dict: dict, pool_type: str) -> None:
        """entity.metadata['value'] is a NumericalInfo (Float64)."""
        entity = self._get_entity(pools_dict, pool_type)
        assert isinstance(entity.metadata["value"], NumericalInfo)

    def test_feature_info_array(self, pools_dict: dict, pool_type: str) -> None:
        """entity.metadata['token_emb'] is an ArrayInfo with dimension=32."""
        entity = self._get_entity(pools_dict, pool_type)
        fi = entity.metadata["token_emb"]
        assert isinstance(fi, ArrayInfo)
        assert fi.dimension == 32

    def test_cast_reflected(self, pools_dict: dict, pool_type: str) -> None:
        """cast_features({'status': Categorical}) on the pool propagates to entity metadata."""
        pool = pools_dict[pool_type].copy()
        pool.cast_features({"status": pl.Categorical})
        seq = pool[pool.unique_ids[0]]
        assert isinstance(seq[0].metadata["status"], CategoricalInfo)

    def test_scoped_entity_features(self, pools_dict: dict, pool_type: str) -> None:
        """Entity from a scoped sequence only sees the requested feature subset."""
        pool = pools_dict[pool_type]
        subset = pool.settings.entity_features[:2]
        seqs = pool.get_sequences(entity_features=subset)
        seq = next(iter(seqs.values()))
        entity = seq[0]
        assert set(entity.metadata.keys()) == set(subset)


# ---------------------------------------------------------------------------
# Standalone Entity (no parent; metadata inferred from store)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entity_case", ["partial", "complete"])
class TestStandaloneEntity:
    """Standalone Entity (no parent sequence): metadata inferred directly from store.

    partial  (id=1)  : absent from static.csv, static stats are null.
    complete (id=11) : present in both seq and static data, real static stats.

    Parametrized over entity_case × seq_type (interval/event/state) ×
    stores_dict (datetime/timestep) = 12 combinations per test.
    """

    def test_no_parent_metadata(self, standalone_entities, entity_case: str) -> None:
        """Standalone entity has no parent_metadata; it infers its own."""
        assert (  # pylint: disable=protected-access
            standalone_entities[entity_case]._parent_metadata is None
        )

    def test_metadata_is_dict_of_feature_info(
        self, standalone_entities, entity_case: str
    ) -> None:
        """entity.metadata returns a dict of FeatureInfo instances (inferred from store)."""
        meta = standalone_entities[entity_case].metadata
        assert isinstance(meta, dict)
        assert all(isinstance(v, FeatureInfo) for v in meta.values())

    def test_feature_info_numeric(self, standalone_entities, entity_case: str) -> None:
        """entity.metadata['value'] is a NumericalInfo."""
        assert isinstance(
            standalone_entities[entity_case].metadata["value"], NumericalInfo
        )

    def test_feature_info_array(self, standalone_entities, entity_case: str) -> None:
        """entity.metadata['token_emb'] is an ArrayInfo with dimension=32."""
        fi = standalone_entities[entity_case].metadata["token_emb"]
        assert isinstance(fi, ArrayInfo)
        assert fi.dimension == 32
