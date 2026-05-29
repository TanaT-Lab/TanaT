#!/usr/bin/env python3
"""Trajectory-level cast recipes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import polars as pl

from .base import BaseCastRecipe, ColumnMapCast, StructuralCasts


@dataclass(frozen=True)
class TrajectoryFeatureCasts:
    """Feature casts consumed by trajectory views (static only)."""

    static: ColumnMapCast = field(default_factory=ColumnMapCast)

    def is_empty(self) -> bool:
        """Return ``True`` when no static cast is registered."""
        return self.static.is_empty()

    def copy(self) -> TrajectoryFeatureCasts:
        """Return a deep copy of the feature casts."""
        return replace(self, static=self.static.copy())

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply static feature casts to *lf* (no-op when empty)."""
        return self.static.apply(lf)

    def probe_lf(self, lf: pl.LazyFrame | None) -> None:
        """Force-evaluate the static bucket on a view LazyFrame."""
        if lf is not None:
            self.static.probe_lf(lf)

    def probe(self, view) -> None:
        """Probe this candidate recipe on top of the current trajectory view."""
        if self.static.is_empty():
            return
        lf = view._frames.static()  # pylint: disable=protected-access
        if lf is not None:
            self.static.probe_lf(lf)


@dataclass(frozen=True)
class TrajectoryCastRecipe(BaseCastRecipe):
    """Type-cast overrides for a trajectory view.

    Entity-level feature casts live in the trajectory's sub-pools (see
    :class:`~tanat.cast.sequence.SequenceCastRecipe`); at the trajectory
    layer only ``static`` features are observable directly.
    """

    structural: StructuralCasts = field(default_factory=StructuralCasts)
    features: TrajectoryFeatureCasts = field(default_factory=TrajectoryFeatureCasts)

    _FEATURES_CLS: ClassVar[type] = TrajectoryFeatureCasts

    # ------------------------------------------------------------------
    # Trajectory-specific shortcut
    # ------------------------------------------------------------------

    def feature_exprs(self) -> list[pl.Expr]:
        """Return the ``with_columns`` expressions for static features."""
        return self.features.static.exprs()

    # ------------------------------------------------------------------
    # Functional builder
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        id: pl.DataType | None = None,  # pylint: disable=redefined-builtin
        time_index: pl.DataType | None = None,
        static: dict[str, pl.DataType] | None = None,
    ) -> TrajectoryCastRecipe:
        """Return a new recipe extended with the given structural and static casts."""
        structural = self.structural
        if id is not None:
            structural = structural.append_id(id)
        if time_index is not None:
            structural = structural.append_time_index(time_index)
        features = self.features
        if static:
            features = replace(features, static=features.static.append(static))
        return replace(self, structural=structural, features=features)
