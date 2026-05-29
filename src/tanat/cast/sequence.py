#!/usr/bin/env python3
"""Sequence-level cast recipes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar

import polars as pl

from .base import BaseCastRecipe, ColumnMapCast, StructuralCasts


@dataclass(frozen=True)
class SequenceFeatureCasts:
    """Feature casts consumed by sequence views."""

    entity: ColumnMapCast = field(default_factory=ColumnMapCast)
    static: ColumnMapCast = field(default_factory=ColumnMapCast)

    def is_empty(self) -> bool:
        """Return ``True`` when no entity and no static cast is registered."""
        return self.entity.is_empty() and self.static.is_empty()

    def copy(self) -> SequenceFeatureCasts:
        """Return a deep copy of the feature casts."""
        return replace(self, entity=self.entity.copy(), static=self.static.copy())

    def apply(self, lf: pl.LazyFrame, *, is_static: bool) -> pl.LazyFrame:
        """Apply entity or static feature casts to *lf* (no-op when empty)."""
        return (self.static if is_static else self.entity).apply(lf)

    def probe_lf(self, lf: pl.LazyFrame | None, *, is_static: bool) -> None:
        """Force-evaluate the selected bucket on a view LazyFrame."""
        if lf is None:
            return
        (self.static if is_static else self.entity).probe_lf(lf)

    def probe(self, view, *, is_static: bool) -> None:
        """Probe this candidate recipe on top of the current sequence view."""
        bucket = self.static if is_static else self.entity
        if bucket.is_empty():
            return
        if is_static:
            lf = view._frames.static()  # pylint: disable=protected-access
        else:
            lf = view._frames.temporal()  # pylint: disable=protected-access
        if lf is None:
            return
        bucket.probe_lf(lf)


@dataclass(frozen=True)
class SequenceCastRecipe(BaseCastRecipe):
    """Type-cast overrides for a sequence view."""

    structural: StructuralCasts = field(default_factory=StructuralCasts)
    features: SequenceFeatureCasts = field(default_factory=SequenceFeatureCasts)

    _FEATURES_CLS: ClassVar[type] = SequenceFeatureCasts

    # ------------------------------------------------------------------
    # Sequence-specific shortcuts
    # ------------------------------------------------------------------

    @property
    def entity(self) -> dict[str, list[pl.DataType]]:
        """Per-column dtype steps of the entity feature recipes."""
        return self.features.entity.recipes

    def entity_caster(self, col: str) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster for entity feature *col*, or ``None``."""
        return self.features.entity.caster(col)

    def feature_exprs(self, is_static: bool = False) -> list[pl.Expr]:
        """Return the ``with_columns`` expressions for entity or static features."""
        return (self.features.static if is_static else self.features.entity).exprs()

    # ------------------------------------------------------------------
    # Functional builder
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        id: pl.DataType | None = None,  # pylint: disable=redefined-builtin
        time_index: pl.DataType | None = None,
        entity: dict[str, pl.DataType] | None = None,
        static: dict[str, pl.DataType] | None = None,
    ) -> SequenceCastRecipe:
        """Return a new recipe extended with the given structural and feature casts."""
        structural = self.structural
        if id is not None:
            structural = structural.append_id(id)
        if time_index is not None:
            structural = structural.append_time_index(time_index)

        features = self.features
        if entity:
            features = replace(features, entity=features.entity.append(entity))
        if static:
            features = replace(features, static=features.static.append(static))
        return replace(self, structural=structural, features=features)
