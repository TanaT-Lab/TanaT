#!/usr/bin/env python3
"""
Tests: HammingEntityMetric
"""

from __future__ import annotations

import pytest

from tanat.metric.entity import EntityMetric, L2EntityMetric

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


class TestL2Compute:
    """Core distance computation with real Entity objects."""

    def test_equal_entities_returns_zero(self, num_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = num_pool[num_pool.unique_ids[0]]
        ent = seq[0]
        h = L2EntityMetric(entity_feature="value")
        assert h(ent, ent) == 0.0

    def test_find_numerical_attribute(self, cat_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = cat_pool[cat_pool.unique_ids[0]]
        ent = seq[0]
        h = L2EntityMetric()
        assert h(ent, ent) == 0.0

    def test_save_numerical_attribute(self, cat_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = cat_pool[cat_pool.unique_ids[0]]
        ent = seq[0]
        h = L2EntityMetric()
        h(ent, ent)
        assert h.settings.entity_feature is not None

    def test_returns_correct_value(self, num_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = num_pool[num_pool.unique_ids[0]]
        ent = seq[0]
        ent2 = seq[2]
        h = L2EntityMetric(entity_feature="value", normalize=False)
        assert h(ent, ent2) == (ent["value"] - ent2["value"]) ** 2

    def test_returns_correct_value_normalize(self, num_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = num_pool[num_pool.unique_ids[0]]
        ent = seq[0]
        ent2 = seq[2]
        h = L2EntityMetric(entity_feature="value")
        assert h(ent, ent2) == (ent["value"] - ent2["value"]) ** 2 / (
            ent["value"] ** 2 + ent2["value"] ** 2
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestHammingValidation:
    """Validation errors on wrong feature dtype or argument type."""

    def test_categorical_feature_raises(self, state_pool) -> None:
        """A categorical feature raises TypeError mentioning 'Numerical'."""
        ent = state_pool[state_pool.unique_ids[0]][0]
        h = L2EntityMetric(entity_feature="status")
        with pytest.raises(TypeError, match="Numerical"):
            h(ent, ent)

    def test_non_entity_raises(self, num_pool) -> None:
        """Passing a non-Entity raises TypeError mentioning 'Entity'."""
        ent = num_pool[num_pool.unique_ids[0]][0]
        h = L2EntityMetric(entity_feature="value")
        with pytest.raises(TypeError, match="Entity"):
            h("not_an_entity", ent)

    def test_missing_feature_raises_key_error(self, num_pool) -> None:
        """A feature absent from entity metadata raises KeyError."""
        ent = num_pool[num_pool.unique_ids[0]][0]
        h = L2EntityMetric(entity_feature="nonexistent_feature")
        with pytest.raises(KeyError, match="nonexistent_feature"):
            h(ent, ent)


# ---------------------------------------------------------------------------
# Config serialisation round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundtrip:
    """Serialization round-trips and registry dispatch via to_config / from_config."""

    def test_to_config_structure(self) -> None:
        """to_config() contains the correct type key and settings fields."""
        h = L2EntityMetric(entity_feature="value", nan_cost=0.7)
        config = h.to_config()
        assert config["type"] == "l2entity"
        assert config["settings"]["entity_feature"] == "value"
        assert config["settings"]["nan_cost"] == 0.7

    def test_from_config_roundtrip(self) -> None:
        """Settings survive a to_config / from_config round-trip."""
        h = L2EntityMetric(entity_feature="value", nan_cost=0.7)
        config = h.to_config()
        h2 = L2EntityMetric.from_config(config)
        assert h2.settings.entity_feature == "value"
        assert h2.settings.nan_cost == 0.7

    def test_from_config_entity_metric_registry(self) -> None:
        """EntityMetric.from_config dispatches to the correct subclass via registry."""
        h = EntityMetric.from_config({"type": "l2entity"})
        assert isinstance(h, L2EntityMetric)
