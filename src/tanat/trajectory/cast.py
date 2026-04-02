#!/usr/bin/env python3
"""Cast recipe for trajectory pool/trajectory views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable

import polars as pl

if TYPE_CHECKING:
    from ..store.trajectory.store import TrajectoryStore


@dataclass(frozen=True)
class TrajectoryCastRecipe:
    """Type-cast overrides for a trajectory view (TrajectoryPool / Trajectory).

    Each field is an **ordered recipe** of target types, applied
    left-to-right at read time.  Empty lists mean "no cast".

    Attributes:
        id: Cast recipe for the trajectory ID column.
        time_index: Cast recipe for time-index columns in linked stores.
        static: Per-column cast recipes for trajectory-level static features.
    """

    id: list[pl.DataType] = field(default_factory=list)
    time_index: list[pl.DataType] = field(default_factory=list)
    static: dict[str, list[pl.DataType]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Returns ``True`` if no cast is defined."""
        return not self.id and not self.time_index and not self.static

    @property
    def id_dtype(self) -> pl.DataType | None:
        """Final dtype of the ID recipe, or ``None`` if empty."""
        return self.id[-1] if self.id else None

    @property
    def time_index_dtype(self) -> pl.DataType | None:
        """Final dtype of the time-index recipe, or ``None`` if empty."""
        return self.time_index[-1] if self.time_index else None

    def replace(self, **kwargs) -> TrajectoryCastRecipe:
        """Return a copy with the specified fields replaced.

        Example::

            new = recipe.replace(id=[pl.Utf8])
        """
        return replace(self, **kwargs)

    def copy(self) -> TrajectoryCastRecipe:
        """Returns a deep copy (all nested containers are new objects)."""
        return self.replace(
            id=list(self.id),
            time_index=list(self.time_index),
            static={k: list(v) for k, v in self.static.items()},
        )

    @staticmethod
    def _build_caster(
        recipe: list[pl.DataType],
    ) -> Callable[[pl.Expr], pl.Expr]:
        """Return a column-agnostic transformer that applies *recipe* in order."""

        def caster(expr: pl.Expr) -> pl.Expr:
            for dtype in recipe:
                expr = expr.cast(dtype)
            return expr

        return caster

    def id_caster(self) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return a transformer for the ID column, or ``None`` if no casts are set."""
        if not self.id:
            return None
        return self._build_caster(self.id)

    def time_index_caster(self) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return a transformer for time-index columns, or ``None`` if no casts are set."""
        if not self.time_index:
            return None
        return self._build_caster(self.time_index)

    def static_caster(self, col: str) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return a transformer for a single static feature column, or ``None``."""
        recipe = self.static.get(col)
        if not recipe:
            return None
        return self._build_caster(recipe)

    def feature_exprs(self) -> list[pl.Expr]:
        """Cast expressions for static features.

        Returns expressions ready for ``lf.with_columns(exprs)``.
        Empty list when no casts are defined.
        """
        return [
            self._build_caster(recipe)(pl.col(col))
            for col, recipe in self.static.items()
        ]

    @classmethod
    def coerce(cls, value: TrajectoryCastRecipe | dict | None) -> TrajectoryCastRecipe:
        """Normalise *value* to a ``TrajectoryCastRecipe``.

        Accepts ``None`` (→ empty), a ``dict``, or an existing instance.
        """
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, cls):
            return value
        raise TypeError(
            f"'cast_recipe' must be a {cls.__name__} instance or dict,"
            f" got {type(value).__name__}"
        )

    def probe(self, store: TrajectoryStore) -> None:
        """Validate every recipe on a small sample from *store*.

        Raises:
            TypeError: If any step is incompatible with the data.
        """
        if self.id:
            store.probe_id_cast_recipe(self.id)
        if self.time_index:
            store.probe_time_cast_recipe(self.time_index)
        if self.static:
            store.probe_static_cast_recipe(self.static)
