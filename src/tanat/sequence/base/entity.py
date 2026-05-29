#!/usr/bin/env python3
"""
Entity: Flyweight object representing a single row in a Sequence (Event, State, etc.).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from tanat_utils import Registrable
from tanat_utils.pretty_format import format_header, format_section, format_kv

from ...metadata.feature import FeatureInfo, build_feature_metadata
from ...store.sequence.store import SequenceStore
from ...store.sequence.schema import StoreSchema as SCH
from ...cast import SequenceCastRecipe
from ._utils import resolve_store

if TYPE_CHECKING:
    from ...metadata.sequence import SequenceMetadata
    from .sequence import Sequence


def _collapse_temporal(values: list[Any]) -> Any:
    """Collapse a list of temporal values to a scalar when there is only one."""
    return values[0] if len(values) == 1 else values


@dataclass(frozen=True)
class PrefetchedEntityData:
    """One entity row pre-collected by the parent Sequence during iteration.

    Owns the view-schema row and knows how to expose it the way Entity does,
    so Entity never slices the raw row itself.
    """

    _row: dict[str, Any]
    _time_columns: tuple[str, ...]
    _feature_columns: tuple[str, ...]

    def temporal_extent(self) -> Any:
        """Return the temporal extent from the prefetched row."""
        return _collapse_temporal([self._row[c] for c in self._time_columns])

    def features(self) -> dict[str, Any]:
        """Return the feature values from the prefetched row."""
        return {c: self._row[c] for c in self._feature_columns}


class Entity(Registrable, ABC):
    """
    Abstract Flyweight object acting as a proxy to a specific row in a Sequence.
    """

    _REGISTER = {}

    def __init__(
        self,
        id_value,
        store: str | Path | SequenceStore,
        features: list[str] | None = None,
        *,
        rank: int,
        store_index: int,
        cast_recipe: SequenceCastRecipe | dict | None = None,
        virtual_id: str | None = None,
        parent_metadata: SequenceMetadata | None = None,
    ) -> None:
        """Create an entity proxy for a single row in a sequence.

        Args:
            id_value: Sequence identifier this entity belongs to.
            store: Store path, name, or :class:`SequenceStore` instance.
            features: Visible feature names propagated from the parent
                :class:`Sequence`.  ``None`` → all store features.
            rank: 0-based position within the sequence as seen by the user
                (accounts for filtering/masking).
            store_index: Absolute physical row index in the store (``SCH.STORE_INDEX``).
            cast_recipe: Cast recipe propagated from the parent
                :class:`Sequence`.  Normalised via
                :meth:`SequenceCastRecipe.coerce`.
            virtual_id: Virtual context UUID from the parent
                :class:`Sequence`.
            parent_metadata: Pre-computed
                :class:`~tanat.metadata.sequence.SequenceMetadata` from the
                parent pool.  When provided, ``metadata`` returns this
                directly (no extra I/O).
        """
        self._id_value = id_value
        self._rank = rank
        self._store_index = store_index
        self._store = resolve_store(store)
        # Resolve features if None. Assume provided features are valid for performances
        self._features: list[str] = (
            features if features is not None else self._store.entity_features()
        )
        self._virtual_id: str | None = virtual_id
        self._casts: SequenceCastRecipe = SequenceCastRecipe.coerce(cast_recipe)
        self._parent_metadata: SequenceMetadata | None = parent_metadata
        self._prefetched: PrefetchedEntityData | None = None

    @classmethod
    def from_parent(
        cls,
        parent: Sequence,
        *,
        rank: int,
        store_index: int,
        prefetched: PrefetchedEntityData | None = None,
    ) -> "Entity":
        """Build a sequence-managed entity.  **Not part of the public API.**

        Snapshots the shared context from *parent* (store, features, casts,
        virtual_id, metadata), bypassing store and feature resolution. The
        per-row *prefetched* is optional: present during iteration, absent
        for point access.
        """
        ent = object.__new__(cls)
        # pylint: disable=protected-access
        ent._id_value = parent._id_value
        ent._store = parent._store
        ent._features = parent.settings.entity_features
        ent._rank = rank
        ent._store_index = store_index
        ent._casts = parent._casts
        ent._virtual_id = parent._virtual_id
        ent._parent_metadata = parent.metadata
        ent._prefetched = prefetched
        return ent

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(id={self._id_value}, rank={self._rank})"

    def __str__(self) -> str:
        cls = type(self).__name__
        feature_values = self.data()
        overview = [
            format_kv("Sequence ID", str(self._id_value)),
            format_kv("Rank", str(self._rank)),
        ]
        feature_lines = [
            format_kv(name, str(value)) for name, value in feature_values.items()
        ]

        parts = [
            format_header(f"{cls} Summary"),
            "",
            format_section("Overview", overview),
            "",
            format_section("Entity Features", feature_lines),
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def id_value(self):
        """The sequence identifier this entity belongs to."""
        return self._id_value

    @property
    def rank(self) -> int:
        """0-based position of this entity within its sequence.

        Always matches the index used to retrieve it::

            entity = seq[3]
            entity.rank  # 3
        """
        return self._rank

    @property
    def feature_names(self) -> list[str]:
        """Visible feature names."""
        return list(self._features)

    @property
    def metadata(self) -> dict[str, FeatureInfo]:
        """
        Feature metadata for this entity.

        Returns a dictionary mapping each visible feature name to its
        ``FeatureInfo`` descriptor (type, stats, …).
        When created from a Sequence (which itself comes from a Pool),
        the pool-level metadata is reused directly. Stats are consistent
        across all entities in the pool, no extra I/O.
        """
        if self._parent_metadata is not None:
            scoped = self._parent_metadata.scope(entity_features=self._features)
            return {f.name: f for f in scoped.entity_features}

        # Fallback for standalone Entity: infer directly from store
        entity_lf = self._store.entity(virtual_id=self._virtual_id)
        entity_lf = self._casts.features.apply(entity_lf, is_static=False)
        if self._features is not None:
            entity_lf = entity_lf.select(self._features)
        infos = build_feature_metadata(entity_lf)
        return {f.name: f for f in infos}

    @property
    def temporal_extent(self) -> Any:
        """
        The temporal extent of this entity.

        Returns:
            A ``list`` of two values ``[start, end]`` for interval-based
            sequences, or a single scalar for event sequences.
        """
        if self._prefetched is not None:
            return self._prefetched.temporal_extent()
        lf = self._store.time_index(self._virtual_id, with_store_index=True)
        values = list(
            self._row_at_store_index(
                self._casts.structural.apply(lf, time_cols=SCH.time_index_columns())
            ).values()
        )
        return _collapse_temporal(values)

    def data(
        self,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Access the feature values for this entity as a dictionary.

        Only feature columns are returned; the sequence identifier and
        time columns are excluded (use :pyattr:`id_value` and
        :pyattr:`temporal_extent` instead).

        Args:
            features: Feature name(s) to include (``None`` → all visible
                entity features).

        Returns:
            A ``dict`` mapping feature names to their scalar values.
        """
        if self._prefetched is not None:
            row = self._prefetched.features()
        else:
            lf = self._store.entity(self._virtual_id, with_store_index=True)
            row = self._row_at_store_index(
                self._casts.features.apply(lf, is_static=False)
            )
        # Resolve + validate feature scope
        effective = self._resolve_features(features, available=list(row.keys()))
        return {k: v for k, v in row.items() if k in set(effective)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_at_store_index(self, lf: pl.LazyFrame) -> dict:
        """Collect the single row matching ``self._store_index`` from *lf*.

        *lf* must contain ``SCH.STORE_INDEX``; it is dropped from the result.
        """
        return (
            lf.filter(pl.col(SCH.STORE_INDEX) == self._store_index)
            .drop(SCH.STORE_INDEX)
            .collect()
            .row(0, named=True)
        )

    def _resolve_features(
        self,
        features: list[str] | None,
        available: list[str] | None = None,
    ) -> list[str]:
        """
        Resolve the effective feature list and validate names.

        * ``features=None`` → ``self._features`` (always a concrete list after init).
        * ``features`` provided → validated against *available*.

        Args:
            features: User-requested feature names (``None`` → all visible).
            available: Known feature names (e.g. row keys) used to
                validate the resolved list.  When provided, a
                ``KeyError`` is raised for any unknown name.

        Raises:
            KeyError: If a resolved feature is not in *available*.
        """
        if isinstance(features, str):
            features = [features]

        # No explicit request → use scope (always set after __init__)
        if features is None:
            return self._features

        # Validate requested features against available keys
        if available is not None:
            available_set = set(available)
            for f in features:
                if f not in available_set:
                    raise KeyError(
                        f"Feature '{f}' not found in entity features. "
                        f"Available: {sorted(available_set)}"
                    )

        return features

    def __getitem__(self, name: str) -> Any:
        """
        Access a single feature value by name.

        Args:
            name: The feature name.

        Returns:
            The scalar value for this entity / feature.

        Raises:
            KeyError: If the feature does not exist.
        """
        row = self.data(features=[name])
        if name not in row:
            raise KeyError(f"Feature '{name}' not found in {type(self).__name__}.")
        return row[name]
