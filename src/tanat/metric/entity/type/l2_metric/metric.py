#!/usr/bin/env python3
"""
L2EntityMetric: numerical feature distance by L2 distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from tanat_utils import settings_dataclass as dataclass

from .....metadata.feature import NumericalInfo, FeatureInfo
from ...base import EntityMetric

if TYPE_CHECKING:
    from .....sequence.base.entity import Entity


@dataclass
class L2Settings:
    """Settings for :class:`L2EntityMetric`.

    Args:
        entity_feature: Name of the numerical feature to compare.
            ``None`` - first entity feature from the pool/entity metadata.
        nan_cost: Default cost applied when at least one value is NaN
            (default: ``1.0``).
        normalize: Compute the absolute relative difference of the values to ensure
            to have a value between 0 and 1.
    """

    entity_feature: str | None = None
    nan_cost: float = 1.0
    normalize: bool = True


class L2EntityMetric(EntityMetric, register_name="l2entity"):
    """Numerical distance between two entities evaluated as the squared
    difference of values (no square root applied).

    Returns the L2 distance between feature values when both are
    defined, and ``nan_cost`` value in case there is a ``NaN``.

    If no ``entity_feature`` provided when the metric is created, a
    feature will be defined automatically as the first numerical attribute
    found when the metric is applied a first time to an entity.
    The updated feature name is then frozen for future usages.
    If there is no numerical attribute, an error is raised.

    Example::

        metric = L2EntityMetric(
            entity_feature="value",
            nan_cost=0.8,
        )
        metric(ent_a, ent_b)


    The normalized version of the L2 metric evaluate the quantity $\frac{(f_1-f_2)^2}{f_1^2+f_2^2}$
    that is between 0 (when $f_1$ equals $f_2$) and 1 (when $f_1$ is null for instance).
    """

    SETTINGS_CLASS = L2Settings
    NUMBA_OPTIM: bool = False

    def __init__(
        self,
        entity_feature: str | None = None,
        nan_cost: float = 1.0,
        normalize: bool = True,
    ) -> None:
        super().__init__(
            settings=L2Settings(
                entity_feature=entity_feature,
                nan_cost=nan_cost,
                normalize=normalize,
            )
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_entity(self, ent_a: Entity, ent_b: Entity | None = None) -> None:
        """Verify the configured feature exists and is numerical."""
        self._validate_entity_instance(ent_a, ent_b)

        feature = self.settings.entity_feature
        fid = 0
        while feature is None and fid < len(ent_a.feature_names):
            if self._check_numerical(ent_a.metadata.get(ent_a.feature_names[fid])):
                feature = ent_a.feature_names[fid]
                break

        if feature is None:
            raise TypeError(
                "L2EntityMetric requires at least one Numerical feature, "
                "none has been found. "
                "Provide at least one attribute of type pl.Float32, pl.Float16 or pl.Int32, ..."
            )
        elif self.settings.entity_feature is not None:
            # when en entity has been defined, it must be validated
            self._validate_numerical(ent_a.metadata.get(feature), feature)

        if ent_b is not None:
            self._validate_numerical(ent_b.metadata.get(feature), feature)

        if self.settings.entity_feature is None:
            # the setting is not complete and we validate its consistency with
            # an entity, then, we use this setting for future usage of the metric
            self.update_settings(entity_feature=feature)

    def _check_numerical(self, info: FeatureInfo | None) -> None:
        """Return True if `feature` exists and is numerical and False otherwise."""
        if info is None or not isinstance(info, NumericalInfo):
            return False
        return True

    def _validate_numerical(self, info: FeatureInfo | None, feature: str) -> None:
        """Assert that *feature* metadata is numerical."""
        if info is None:
            raise KeyError(f"Feature '{feature}' not found in entity features.")
        if not isinstance(info, NumericalInfo):
            raise TypeError(
                f"L2EntityMetric requires a Numerical feature, "
                f"got '{feature}' ({type(info).__name__}). "
                f"Cast it first to pl.Float32, pl.Float16 or pl.Int32, ..."
            )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, ent_a: Entity, ent_b: Entity) -> float:
        """Compute L2 distance between two validated entities."""
        feature = self.settings.entity_feature or ent_a.feature_names[0]

        val_a = ent_a[feature]
        val_b = ent_b[feature]

        if val_a == pl.Null or val_b == pl.Null:
            return self.settings.nan_cost

        if val_a == val_b:
            # include the case val_a == val_b == 0.0 ... which may problematic
            return float(0.0)

        return (
            float((val_a - val_b) ** 2)
            if not self.settings.normalize
            else ((val_a - val_b) ** 2) / (val_a**2 + val_b**2)
        )
