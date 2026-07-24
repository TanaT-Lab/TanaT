#!/usr/bin/env python3
"""
Tests: HammingEntityMetric
"""

from __future__ import annotations

import pytest

from tanat.metric.entity import (
    EntityMetric,
    CombinedEntityMetric,
    L2EntityMetric,
    HammingEntityMetric,
)

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


class TestCombineCompute:
    """Core distance computation with real Entity objects."""

    def test_equal_entities_returns_zero(self, cat_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = cat_pool[cat_pool.unique_ids[0]]
        ent = seq[0]
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ]
        )
        assert h(ent, ent) == 0.0

    def test_weights_values(self, cat_pool) -> None:
        """Distance from an entity to itself is 0."""
        seq = cat_pool[cat_pool.unique_ids[0]]
        ent = seq[0]
        ent2 = seq[2]
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ],
            weights=[0.5, 0.5],
        )
        h2 = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ],
            weights=[1, 1],
        )
        assert pytest.approx(2 * h(ent, ent2) - h2(ent, ent2), rel=1e-3) == 0.0

    def test_weights_size_error(self) -> None:
        """Invalid weights size raises ValueError."""
        with pytest.raises(ValueError, match="Weights"):
            h = CombinedEntityMetric(
                metrics_config=[
                    L2EntityMetric(entity_feature="value").to_config(),
                    HammingEntityMetric(entity_feature="status").to_config(),
                ],
                weights=[1],
            )

    def test_unknown_agg_error(self) -> None:
        """Invalid agg function raises ValueError."""
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ],
            agg="sum",
        )
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ],
            agg="mean",
        )
        with pytest.raises(ValueError, match="aggregation"):
            h = CombinedEntityMetric(
                metrics_config=[
                    L2EntityMetric(entity_feature="value").to_config(),
                    HammingEntityMetric(entity_feature="status").to_config(),
                ],
                agg="other",
            )

    def test_wrong_setting_type_error(self) -> None:
        """Invalid metric settings raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            h = CombinedEntityMetric(
                metrics_config=[
                    L2EntityMetric(entity_feature="value"),
                    HammingEntityMetric(entity_feature="status"),
                ]
            )

    def test_wrong_setting_dict_error(self) -> None:
        """Invalid metric settings raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            h = CombinedEntityMetric(
                metrics_config=[
                    {"key": "value"},
                    HammingEntityMetric(entity_feature="status"),
                ]
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestCombinedValidation:
    """Validation errors on wrong feature dtype or argument type."""

    def test_submetric_feature_error_raises(self, state_pool) -> None:
        """A categorical feature raises TypeError mentioning 'Numerical'."""
        ent = state_pool[state_pool.unique_ids[0]][0]
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="status").to_config(),
                HammingEntityMetric(entity_feature="value").to_config(),
            ]
        )
        with pytest.raises(TypeError, match="Numerical"):
            h(ent, ent)

    def test_non_entity_raises(self, num_pool) -> None:
        """Passing a non-Entity raises TypeError mentioning 'Entity'."""
        ent = num_pool[num_pool.unique_ids[0]][0]
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ]
        )
        with pytest.raises(TypeError, match="Entity"):
            h("not_an_entity", ent)

    def test_missing_feature_raises_key_error(self, num_pool) -> None:
        """A feature absent from entity metadata raises KeyError."""
        ent = num_pool[num_pool.unique_ids[0]][0]
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="nonexistent_feature").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ]
        )
        with pytest.raises(KeyError, match="nonexistent_feature"):
            h(ent, ent)


# ---------------------------------------------------------------------------
# Config serialisation round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundtrip:
    """Serialization round-trips and registry dispatch via to_config / from_config."""

    def test_to_config_structure(self) -> None:
        """to_config() contains the correct type key and settings fields."""
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ]
        )
        config = h.to_config()
        assert config["type"] == "combinedentity"
        assert isinstance(config["settings"]["metrics_config"], list)
        assert config["settings"]["agg"] == "sum"

    def test_from_config_roundtrip(self) -> None:
        """Settings survive a to_config / from_config round-trip."""
        h = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ],
            weights=[0.5, 0.6],
        )
        config = h.to_config()
        h2 = EntityMetric.from_config(config)
        assert isinstance(h2, CombinedEntityMetric)
        assert len(h2.settings.metrics_config) == 2
        assert h2.settings.weights[0] == 0.5
