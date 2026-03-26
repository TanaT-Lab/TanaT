#!/usr/bin/env python3
"""State sequence implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ...base.sequence import Sequence
from ...base._describe import (
    _n_unique_entities_expr,
    _temporal_span_expr,
    _duration_stats_exprs,
    _n_transitions_expr,
)
from .settings import StateSequenceSettings

if TYPE_CHECKING:
    from ....store.sequence.store import SequenceStore


class StateSequence(Sequence, register_name="state"):
    """A single state sequence.

    States are **contiguous and non-overlapping**: the end of one state
    is always the start of the next, with no gaps in between.
    """

    SETTINGS_CLASS = StateSequenceSettings

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
        """Create a state sequence for *id_value*.

        Args:
            id_value: Sequence identifier.
            store: Store path, name, or :class:`SequenceStore` instance.
            id_column: User-facing name for the sequence ID column.
            start_column: User-facing name for the state start column.
            end_column: User-facing name for the state end column.
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
            StateSequenceSettings(
                id_column=id_column,
                start_column=start_column,
                end_column=end_column,
                entity_features=ef,
                static_features=sf,
            ),
        )

    @classmethod
    def _exprs_for_describe(cls, settings: StateSequenceSettings) -> list[pl.Expr]:
        """Polars expressions for state-specific describe stats.

        Columns: ``length``, ``n_unique_entities``, ``temporal_span``,
        ``mean_duration``, ``median_duration``, ``duration_std``,
        ``n_transitions``.
        """
        return [
            pl.len().alias("length"),
            _n_unique_entities_expr(settings.entity_features),
            _temporal_span_expr(settings.get_time_columns()),
            *_duration_stats_exprs(settings.start_column, settings.end_column),
            _n_transitions_expr(settings.entity_features),
        ]
