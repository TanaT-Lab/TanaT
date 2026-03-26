#!/usr/bin/env python3
"""Interval sequence implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ...base.sequence import Sequence
from ...base._describe import (
    _n_unique_entities_expr,
    _temporal_span_expr,
    _duration_stats_exprs,
)
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

    @classmethod
    def _exprs_for_describe(cls, settings: IntervalSequenceSettings) -> list[pl.Expr]:
        """Polars expressions for interval-specific describe stats.

        Columns: ``length``, ``n_unique_entities``, ``temporal_span``,
        ``mean_duration``, ``median_duration``, ``duration_std``.
        """
        return [
            pl.len().alias("length"),
            _n_unique_entities_expr(settings.entity_features),
            _temporal_span_expr(settings.get_time_columns()),
            *_duration_stats_exprs(settings.start_column, settings.end_column),
        ]
