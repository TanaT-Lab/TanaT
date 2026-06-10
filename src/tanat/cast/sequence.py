#!/usr/bin/env python3
"""Sequence-level cast recipes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar, NamedTuple

import polars as pl

from .base import BaseCastRecipe, ColumnMapCast, StructuralCasts, _require_single_field


class _ForkCasts(NamedTuple):
    """Precompiled casts for a fork operation. Internal use only."""

    time_index: Callable[[pl.Expr], pl.Expr] | None
    # Callable: the store applies it to its own SCH columns.
    # None = no time-index cast registered.

    feature: list[pl.Expr]
    # Pre-built expressions for the entity feature column (e.g. duration).
    # Empty = no cast or scalar duration.

    static: list[pl.Expr]
    # Pre-built expressions for the static feature column (e.g. end_value).
    # Empty = no cast or scalar end_value.

    time_index_dtype: pl.DataType | None


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
        """Probe this candidate recipe on pre-scope raw data.

        Feature casts run **before** entity scopes in the view pipeline.
        The probe must therefore use the same pre-scope data the cast will
        see at runtime.
        """
        bucket = self.static if is_static else self.entity
        if bucket.is_empty():
            return
        # pylint: disable=protected-access
        frames = view._frames
        if is_static:
            lf = frames._fetch(is_static=True)
            if lf is None:
                return
            lf = frames._rename(lf, is_static=True)
            lf = view._casts.structural.apply(lf, id_col=view.settings.id_column)
            lf = view._casts.features.apply(lf, is_static=True)
        else:
            lf = frames._fetch(is_static=False)
            lf = frames._rename(lf, is_static=False)
            lf = view._casts.structural.apply(
                lf,
                id_col=view.settings.id_column,
                time_cols=view.settings.get_time_columns(),
            )
            lf = view._casts.features.apply(lf, is_static=False)
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

    def fork_casts(
        self,
        *,
        feature_col: str | None = None,
        static_col: str | None = None,
    ) -> _ForkCasts:
        """Return precompiled casts for a fork operation."""
        fc = self.entity_caster(feature_col) if feature_col else None
        sc = self.static_caster(static_col) if static_col else None
        return _ForkCasts(
            time_index=self.time_index_caster(),
            feature=[fc(pl.col(feature_col))] if fc else [],
            static=[sc(pl.col(static_col))] if sc else [],
            time_index_dtype=self.time_index_dtype,
        )

    def feature_exprs(self, is_static: bool = False) -> list[pl.Expr]:
        """Return the ``with_columns`` expressions for entity or static features."""
        return (self.features.static if is_static else self.features.entity).exprs()

    # ------------------------------------------------------------------
    # Functional builder
    # ------------------------------------------------------------------

    def probe(self, view) -> None:
        """Validate structural and all feature casts against *view*."""
        self.structural.probe(view)
        self.features.probe(view, is_static=False)
        self.features.probe(view, is_static=True)

    def append(
        self,
        *,
        id: pl.DataType | None = None,  # pylint: disable=redefined-builtin
        time_index: pl.DataType | None = None,
        entity: dict[str, pl.DataType] | None = None,
        static: dict[str, pl.DataType] | None = None,
        strict: bool = True,
    ) -> SequenceCastRecipe:
        """Return a new recipe extended with **one** structural or feature cast.

        Exactly one of *id*, *time_index*, *entity*, or *static* must be
        provided per call.  Bundling multiple fields in a single call is
        intentionally rejected to keep each recipe step traceable.

        Args:
            strict: Forwarded to :meth:`ColumnMapCast.append` for feature
                buckets only (``entity`` and ``static``).  Structural columns
                (``id``, ``time_index``) are always cast strictly.

        Raises:
            ValueError: If zero or more than one field argument is provided.
        """
        _require_single_field(
            {"id": id, "time_index": time_index, "entity": entity, "static": static}
        )

        structural = self.structural
        if id is not None:
            structural = structural.append_id(id)
        if time_index is not None:
            structural = structural.append_time_index(time_index)

        features = self.features
        if entity:
            features = replace(features, entity=features.entity.append(entity, strict))
        if static:
            features = replace(features, static=features.static.append(static, strict))
        return replace(self, structural=structural, features=features)
