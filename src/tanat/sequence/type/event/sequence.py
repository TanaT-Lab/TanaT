#!/usr/bin/env python3
"""Event sequence implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...base.sequence import Sequence
from .settings import EventSequenceSettings

if TYPE_CHECKING:
    from ....store.sequence.store import SequenceStore


class EventSequence(Sequence, register_name="event"):
    """A single event sequence (one timestamp per entity row)."""

    SETTINGS_CLASS = EventSequenceSettings

    def __init__(
        self,
        id_value,
        store: str | Path | SequenceStore,
        *,
        id_column: str = "id",
        time_column: str = "time",
        entity_features: list[str] | None = None,
        static_features: list[str] | None = None,
    ) -> None:
        """Create an event sequence for *id_value*.

        Args:
            id_value: Sequence identifier.
            store: Store path, name, or :class:`SequenceStore` instance.
            id_column: User-facing name for the sequence ID column.
            time_column: User-facing name for the event timestamp column.
            entity_features: Subset of entity feature names to expose.
                ``None`` → all available from the store.
            static_features: Static feature names to expose.
                ``None`` → all available.  ``[]`` → none.
        """
        _store = self._resolve_store(store)
        ef, sf = self._resolve_features(_store, entity_features, static_features)
        super().__init__(
            id_value,
            _store,
            EventSequenceSettings(
                id_column=id_column,
                time_column=time_column,
                entity_features=ef,
                static_features=sf,
            ),
        )
