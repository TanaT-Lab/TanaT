#!/usr/bin/env python3
"""Cast recipe for sequence pool/sequence/entity views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ...store.sequence.store import SequenceStore


@dataclass(frozen=True)
class SequenceCastRecipe:
    """Holds all type-cast overrides for a view (Pool, Sequence, or Entity).

    Applied at read time only.

    Attributes:
        id: Target type for the sequence ID column.
        time_index: Target type for time index columns.
        entity: Per-column casts for entity features.
        static: Per-column casts for static features (unused at Entity level).
    """

    id: pl.DataType | None = None
    time_index: pl.DataType | None = None
    entity: dict[str, pl.DataType] = field(default_factory=dict)
    static: dict[str, pl.DataType] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Returns ``True`` if no cast is defined."""
        return (
            self.id is None
            and self.time_index is None
            and not self.entity
            and not self.static
        )

    def with_fields(self, **kwargs) -> SequenceCastRecipe:
        """Return a new recipe with the given fields replaced.

        Thin wrapper around :func:`dataclasses.replace`.
        Example::

            new = recipe.with_fields(id=pl.Utf8)
            merged = recipe.with_fields(entity={**recipe.entity, "age": pl.Float32})
        """
        return replace(self, **kwargs)

    def copy(self) -> SequenceCastRecipe:
        """Returns a copy (dict fields are new dicts)."""
        return self.with_fields(entity=dict(self.entity), static=dict(self.static))

    @classmethod
    def coerce(cls, value: SequenceCastRecipe | dict | None) -> SequenceCastRecipe:
        """Normalise *value* to a :class:`SequenceCastRecipe`.

        Accepted inputs:

        - ``None``                    → empty recipe
        - ``dict``                    → ``SequenceCastRecipe(**value)``
        - :class:`SequenceCastRecipe` → returned as-is (frozen: safe to share)

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

    def probe(self, store: SequenceStore) -> None:
        """Validate all non-empty fields against a small sample of *store* data.

        Delegates to the store's individual probe helpers.

        Args:
            store: The sequence store to validate against.

        Raises:
            TypeError: If any cast is incompatible with the underlying data.
        """
        if self.id is not None:
            store.probe_id_cast(self.id)
        if self.time_index is not None:
            store.probe_time_cast(self.time_index)
        if self.entity:
            store.probe_entity_cast(self.entity)
        if self.static:
            store.probe_static_cast(self.static)
