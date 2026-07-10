#!/usr/bin/env python3
"""
CombinedEntityMetric: a metric that combines several other metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import field

from tanat_utils import settings_dataclass as dataclass

from ...base import EntityMetric, EntityMetricSettings

if TYPE_CHECKING:
    from .....sequence.base.entity import Entity


@dataclass
class CombinedEntityMetricSettings(EntityMetricSettings):
    """Settings for :class:`CombinedEntityMetric`. The settings defines
    the metrics, their weights and how to aggregate them.

    Args:
        metrics_config: List of metric configuration (dictionaries).
            see the example below to define easily the configuration from other
            existing classes.
        agg: Aggregate function name (default: 'sum'). At the time, we only implemented
            the sum aggregate function.
        weights: List of weights for the aggregation function (optional, default: None).
            If defined, this list must contains as much real values as the number of
            metrics

    Example::

        settings = CombinedEntityMetricSettings(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ],
            weights=[0.7, 0.3],
        )


    warning::

        It is not possible to use the combined entity metric as an element of the
        metrics to combine.
        This prevent recursive definition of metrics that may be problematics.
    """

    metrics_config: list = field(default_factory=list[dict[str, Any]])
    weights: list | None = None
    agg: str = "sum"

    def __post_init__(self):
        assert self.weights is None or len(self.weights) == len(self.metrics_config)


class CombinedEntityMetric(EntityMetric, register_name="combinedentity"):
    """Metric between entities that involves several entity metrics to
    aggregate. The entity metrics can involve several entity features but
    also combine different manner to compare the same entity feature.

    Args:
        metrics_config: List of metric configuration (dictionaries).
            see the example below to define easily the configuration from other
            existing classes.
        agg: Aggregate function name (default: 'sum'). At the time, we only implemented
            the sum aggregate function.
        weights: List of weights for the aggregation function (optional, default: None).
            If defined, this list must contains as much real values as the number of
            metrics

    Example::

        metric = CombinedEntityMetric(
            metrics_config=[
                L2EntityMetric(entity_feature="value").to_config(),
                HammingEntityMetric(entity_feature="status").to_config(),
            ],
            weights=[0.7, 0.3],
        )

        metric(ent_a, ent_b)

    warning::

        It is not possible to use the combined entity metric as an element of the
        metrics to combine.
        This prevent recursive definition of metrics that may be problematics.

    """

    SETTINGS_CLASS = CombinedEntityMetricSettings
    NUMBA_OPTIM: bool = False

    def __init__(
        self,
        metrics_config: list[dict[str, Any]],
        weights: list | None = None,
        agg: str = "sum",
    ) -> None:
        super().__init__(
            settings=CombinedEntityMetricSettings(
                metrics_config=metrics_config,
                weights=weights,
                agg=agg,
            )
        )

        if weights is not None and len(weights) != len(metrics_config):
            raise ValueError(
                "Weights must contains the same number of elements as the number of metrics."
            )
        if agg not in ["sum"]:
            raise ValueError(
                f"Unknown aggregation function '{agg}' in metric configuration."
            )
        # --------------
        # create metrics from the configurations
        # --------------
        self.metrics = []
        self._prepate_metrics_from_settings()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _prepate_metrics_from_settings(self):
        """The function creates instances of metrics from metrics'
        configurations in the settings"""

        for metric_config in self.settings.metrics_config:
            if metric_config["type"] is None:
                raise KeyError(
                    "Invalid metric configuration.",
                )
            if metric_config["type"] == self.get_registration_name():
                raise KeyError(
                    "Recursive usage of CombinedEntityMetric is prohibited.",
                )

            local_metric = EntityMetric.from_config(metric_config)
            self.metrics.append(local_metric)

    def validate_entity(self, ent_a: Entity, ent_b: Entity | None = None) -> None:
        """Verify the configured feature exists and validate the requirements
        for each underlined metrics."""
        self._validate_entity_instance(ent_a, ent_b)

        for metric in self.metrics:
            metric.validate_entity(ent_a, ent_b)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, ent_a: Entity, ent_b: Entity) -> float:
        """Compute the distance between two validated entities."""

        # compute each individual metric value
        values = []
        for metric in self.metrics:
            values.append(metric(ent_a, ent_b))

        if self.settings.agg == "sum":
            if self.settings.weights is None:
                return sum(values)
            return sum([v * w for v, w in zip(values, self.settings.weights)])

        return 0
