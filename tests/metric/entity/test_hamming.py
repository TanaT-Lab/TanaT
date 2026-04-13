#!/usr/bin/env python3
"""
Tests: HammingEntityMetric
"""

from __future__ import annotations

import pytest
import numpy as np
from numba.typed import List as NumbaList
from syrupy.assertion import SnapshotAssertion

from tanat.metric.entity import EntityMetric, HammingEntityMetric

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


class TestHammingCompute:
    """Core distance computation with real Entity objects."""

    def test_equal_entities_returns_zero(self, cat_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = cat_pool[cat_pool.unique_ids[0]]
        ent = seq[0]
        h = HammingEntityMetric(entity_feature="status")
        assert h(ent, ent) == 0.0

    def test_different_ids_possibly_nonzero(self, cat_pool) -> None:
        """Distance between two distinct entities is a non-negative float."""
        ids = cat_pool.unique_ids
        h = HammingEntityMetric(entity_feature="status")
        result = h(cat_pool[ids[0]][0], cat_pool[ids[1]][0])
        assert isinstance(result, float)
        assert result >= 0.0

    def test_default_feature_uses_first(self, cat_pool_status_only) -> None:
        """entity_feature=None → first entity feature in metadata (here: 'status')."""
        pool = cat_pool_status_only
        h_default = HammingEntityMetric()  # entity_feature=None
        h_explicit = HammingEntityMetric(entity_feature="status")
        ent = pool[pool.unique_ids[0]][0]
        assert h_default(ent, ent) == h_explicit(ent, ent)

    def test_custom_cost(self, cat_pool) -> None:
        """cost dict is respected for matching pairs."""
        ids = cat_pool.unique_ids
        ent_a = cat_pool[ids[0]][0]
        ent_b = cat_pool[ids[1]][0]
        val_a, val_b = ent_a["status"], ent_b["status"]
        if val_a == val_b:
            pytest.skip("Need two entities with different status values")
        h = HammingEntityMetric(entity_feature="status", cost={(val_a, val_b): 0.25})
        assert h(ent_a, ent_b) == 0.25

    def test_mismatch_cost_fallback(self, cat_pool) -> None:
        """When pair not in cost dict, mismatch_cost is used."""
        ids = cat_pool.unique_ids
        ent_a = cat_pool[ids[0]][0]
        ent_b = cat_pool[ids[1]][0]
        if ent_a["status"] == ent_b["status"]:
            pytest.skip("Need different values")
        h = HammingEntityMetric(
            entity_feature="status", cost={("X", "Y"): 0.1}, mismatch_cost=0.8
        )
        assert h(ent_a, ent_b) == 0.8


# ---------------------------------------------------------------------------
# Shadow dispatch (temporary override via kwargs)
# ---------------------------------------------------------------------------


class TestShadowDispatch:
    """Temporary kwarg overrides via shadow dispatch."""

    def test_shadow_override_mismatch_cost(self, cat_pool) -> None:
        """Kwarg override applies only to the call; stored settings remain unchanged."""
        ids = cat_pool.unique_ids
        ent_a = cat_pool[ids[0]][0]
        ent_b = cat_pool[ids[1]][0]
        if ent_a["status"] == ent_b["status"]:
            pytest.skip("Need different values")

        h = HammingEntityMetric(entity_feature="status", mismatch_cost=1.0)
        result_override = h(ent_a, ent_b, mismatch_cost=0.5)
        result_original = h(ent_a, ent_b)

        assert result_override == 0.5
        assert result_original == 1.0
        assert h.settings.mismatch_cost == 1.0  # unchanged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestHammingValidation:
    """Validation errors on wrong feature dtype or argument type."""

    def test_string_feature_raises_type_error(self, state_pool) -> None:
        """A String feature (not yet cast) raises TypeError mentioning pl.Categorical."""
        ent = state_pool[state_pool.unique_ids[0]][0]
        h = HammingEntityMetric(entity_feature="status")
        with pytest.raises(TypeError, match="pl.Categorical"):
            h(ent, ent)

    def test_numerical_feature_raises(self, state_pool) -> None:
        """A numerical feature raises TypeError mentioning 'Categorical'."""
        ent = state_pool[state_pool.unique_ids[0]][0]
        h = HammingEntityMetric(entity_feature="value")
        with pytest.raises(TypeError, match="Categorical"):
            h(ent, ent)

    def test_non_entity_raises(self, cat_pool) -> None:
        """Passing a non-Entity raises TypeError mentioning 'Entity'."""
        ent = cat_pool[cat_pool.unique_ids[0]][0]
        h = HammingEntityMetric(entity_feature="status")
        with pytest.raises(TypeError, match="Entity"):
            h("not_an_entity", ent)

    def test_missing_feature_raises_key_error(self, cat_pool) -> None:
        """A feature absent from entity metadata raises KeyError."""
        ent = cat_pool[cat_pool.unique_ids[0]][0]
        h = HammingEntityMetric(entity_feature="nonexistent_feature")
        with pytest.raises(KeyError, match="nonexistent_feature"):
            h(ent, ent)

    def test_asymmetric_cost_raises(self) -> None:
        """Conflicting asymmetric cost entries raise ValueError."""
        with pytest.raises((ValueError, Exception), match="Asymmetric"):
            HammingEntityMetric(cost={("A", "B"): 0.3, ("B", "A"): 0.7})

    def test_symmetric_cost_accepted(self) -> None:
        """Same value for both orderings is accepted."""
        h = HammingEntityMetric(cost={("A", "B"): 0.5, ("B", "A"): 0.5})
        assert h.settings.cost == {("A", "B"): 0.5, ("B", "A"): 0.5}

    def test_single_direction_cost_accepted(self) -> None:
        """Only one ordering defined is accepted (symmetric by fallback)."""
        h = HammingEntityMetric(cost={("A", "B"): 0.3})
        assert h.settings.cost == {("A", "B"): 0.3}


# ---------------------------------------------------------------------------
# Config serialisation round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundtrip:
    """Serialization round-trips and registry dispatch via to_config / from_config."""

    def test_to_config_structure(self) -> None:
        """to_config() contains the correct type key and settings fields."""
        h = HammingEntityMetric(entity_feature="status", mismatch_cost=0.7)
        config = h.to_config()
        assert config["type"] == "hamming"
        assert config["settings"]["entity_feature"] == "status"
        assert config["settings"]["mismatch_cost"] == 0.7

    def test_from_config_roundtrip(self) -> None:
        """Settings survive a to_config / from_config round-trip."""
        h = HammingEntityMetric(entity_feature="status", mismatch_cost=0.7)
        config = h.to_config()
        h2 = HammingEntityMetric.from_config(config)
        assert h2.settings.entity_feature == "status"
        assert h2.settings.mismatch_cost == 0.7

    def test_from_config_entity_metric_registry(self) -> None:
        """EntityMetric.from_config dispatches to the correct subclass via registry."""
        h = EntityMetric.from_config({"type": "hamming"})
        assert isinstance(h, HammingEntityMetric)

    def test_to_config_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Full config dict matches snapshot (regression guard for serialization format)."""
        h = HammingEntityMetric(
            entity_feature="status",
            mismatch_cost=0.7,
            cost={("A", "B"): 0.3},
        )
        assert h.to_config() == snapshot


# ---------------------------------------------------------------------------
# Unit tests: prepare_batch_data
# ---------------------------------------------------------------------------


class TestPrepareBatchData:
    """Validate the extraction / encoding step of HammingEntityMetric."""

    def test_returns_correct_types(self, cat_pool) -> None:
        """arrays is NumbaList, lengths is int32 array, context is tuple."""
        em = HammingEntityMetric(entity_feature="status")
        arrays, lengths, context = em.prepare_batch_data(cat_pool)
        assert isinstance(arrays, NumbaList)
        assert len(arrays) == len(cat_pool)
        assert lengths.dtype == np.int32
        assert isinstance(context, tuple)

    def test_lengths_match_sequences(self, cat_pool) -> None:
        """Each length matches len(pool[sid])."""
        em = HammingEntityMetric(entity_feature="status")
        arrays, lengths, _ = em.prepare_batch_data(cat_pool)
        for i, sid in enumerate(cat_pool.unique_ids):
            assert lengths[i] == len(cat_pool[sid]), (
                f"Length mismatch for ID {sid}: "
                f"got {lengths[i]}, expected {len(cat_pool[sid])}"
            )

    def test_context_empty_without_cost(self, cat_pool) -> None:
        """Without a cost dict, context is an empty tuple."""
        em = HammingEntityMetric(entity_feature="status")
        _, _, context = em.prepare_batch_data(cat_pool)
        assert context == ()

    def test_cost_matrix_shape(self, cat_pool) -> None:
        """With a cost dict, context contains a (V × V) float32 matrix."""
        em = HammingEntityMetric(
            entity_feature="status",
            cost={("f_0", "f_1"): 0.3},
            mismatch_cost=0.8,
        )
        _, _, context = em.prepare_batch_data(cat_pool)
        assert len(context) == 1
        cost_matrix = context[0]
        V = cost_matrix.shape[0]
        assert cost_matrix.shape == (V, V)
        assert cost_matrix.dtype == np.float32

    def test_cost_matrix_diagonal_zero(self, cat_pool) -> None:
        """Diagonal of cost matrix is 0.0 (no self-distance cost)."""
        em = HammingEntityMetric(
            entity_feature="status",
            cost={("f_0", "f_1"): 0.5},
            mismatch_cost=0.8,
        )
        _, _, context = em.prepare_batch_data(cat_pool)
        cost_matrix = context[0]
        np.testing.assert_array_equal(np.diag(cost_matrix), 0.0)

    def test_arrays_contain_int32(self, cat_pool) -> None:
        """Each element of arrays is a numpy int32 array."""
        em = HammingEntityMetric(entity_feature="status")
        arrays, _, _ = em.prepare_batch_data(cat_pool)
        for i, _ in enumerate(arrays):
            assert arrays[i].dtype == np.int32
