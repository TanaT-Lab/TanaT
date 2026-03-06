#!/usr/bin/env python3
"""Cast recipe for trajectory pool/trajectory views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ..store.trajectory.store import TrajectoryStore


@dataclass(frozen=True)
class TrajectoryCastRecipe:
    """Holds type-cast overrides for a trajectory view (TrajectoryPool or Trajectory).

    Attributes:
        id: Target type for the trajectory ID column.
        temporal: Target type for temporal columns in linked sequence stores.
        static: Per-column casts for trajectory-level static features.
    """

    id: pl.DataType | None = None
    temporal: pl.DataType | None = None
    static: dict[str, pl.DataType] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Returns ``True`` if no cast is defined."""
        return self.id is None and not self.static and self.temporal is None

    def with_fields(self, **kwargs) -> TrajectoryCastRecipe:
        """Return a new recipe with the given fields replaced.

        Thin wrapper around :func:`dataclasses.replace`.  Dict fields
        (*static*) are passed through as-is - copy them explicitly
        when you need independence from the original.

        Example::

            new = recipe.with_fields(id=pl.Utf8)
            merged = recipe.with_fields(static={**recipe.static, "group": pl.Categorical})
        """
        return replace(self, **kwargs)

    def copy(self) -> TrajectoryCastRecipe:
        """Returns a copy (dict fields are new dicts)."""
        return self.with_fields(static=dict(self.static))

    @classmethod
    def coerce(cls, value: TrajectoryCastRecipe | dict | None) -> TrajectoryCastRecipe:
        """Normalise *value* to a :class:`TrajectoryCastRecipe`.

        Accepted inputs:

        - ``None``                       → empty recipe
        - ``dict``                       → ``TrajectoryCastRecipe(**value)``
        - :class:`TrajectoryCastRecipe`  → returned as-is (frozen: safe to share)

        Args:
            value: Raw cast recipe input from a constructor or public API.

        Raises:
            TypeError: If *value* is not one of the accepted types.
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
        """Validate all non-empty fields against a small sample of *store* data.

        Delegates to the store's individual probe helpers.

        Args:
            store: The trajectory store to validate against.

        Raises:
            TypeError: If any cast is incompatible with the underlying data.
        """
        if self.id is not None:
            store.probe_id_cast(self.id)
        if self.static:
            store.probe_static_cast(self.static)
