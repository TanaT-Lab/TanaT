#!/usr/bin/env python3
"""Interval sequence implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...base.sequence import Sequence
from .settings import IntervalSequenceSettings

if TYPE_CHECKING:
    from ....store.sequence.store import SequenceStore


class IntervalSequence(Sequence, register_name="interval"):
    """A single interval sequence.

    Unlike state sequences, intervals are **not** required to be
    contiguous: gaps between intervals are allowed, and two intervals
    may overlap in time.
    """

    SETTINGS_CLASS = IntervalSequenceSettings

    def __init__(
        self,
        id_value,
        store: str | Path | SequenceStore,
        *,
        id_column: str = "id",
        start_column: str = "start",
        end_column: str = "end",
        entity_features: list[str] | None = None,
        static_features: list[str] | None = None,
    ) -> None:
        """Create an interval sequence for *id_value*.

        Args:
            id_value: Sequence identifier.
            store: Store path, name, or :class:`SequenceStore` instance.
            id_column: User-facing name for the sequence ID column.
            start_column: User-facing name for the interval start column.
            end_column: User-facing name for the interval end column.
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
            IntervalSequenceSettings(
                id_column=id_column,
                start_column=start_column,
                end_column=end_column,
                entity_features=ef,
                static_features=sf,
            ),
        )
