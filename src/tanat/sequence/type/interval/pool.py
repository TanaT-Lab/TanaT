#!/usr/bin/env python3
"""Interval sequence pool implementation."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn

import polars as pl

from ...base.pool import SequencePool
from .settings import IntervalSequenceSettings

if TYPE_CHECKING:
    from ....store.sequence.store import SequenceStore
    from ...base.cast import SequenceCastRecipe
    from ....store.sequence.builder.type.interval import IntervalSequenceStoreBuilder
    from ..event.pool import EventSequencePool


LOGGER = logging.getLogger(__name__)


class IntervalSequencePool(SequencePool, register_name="interval"):
    """Pool of interval sequences.

    Unlike state sequences, intervals are **not** required to be
    contiguous: gaps between intervals are allowed, and two intervals
    may overlap in time.
    """

    SETTINGS_CLASS = IntervalSequenceSettings

    # ------------------------------------------------------------------
    # Init (load from existing store)
    # ------------------------------------------------------------------

    def __init__(
        self,
        store: str | Path | SequenceStore,
        *,
        id_column: str = "id",
        start_column: str = "start",
        end_column: str = "end",
        entity_features: list[str] | None = None,
        static_features: list[str] | None = None,
        cast_recipe: SequenceCastRecipe | dict | None = None,
    ) -> None:
        """Create an interval sequence pool backed by *store*.

        Args:
            store: Store path, name, or :class:`SequenceStore` instance.
            id_column: User-facing name for the sequence ID column.
            start_column: User-facing name for the interval start column.
            end_column: User-facing name for the interval end column.
            entity_features: Subset of entity feature names to expose.
                ``None`` → all available from the store.
            static_features: Static feature names to expose.
                ``None`` → all available.  ``[]`` → none.
            cast_recipe: Optional cast recipe (or dict) applied at read time.
                Normalised via :meth:`SequenceCastRecipe.coerce` and probed
                eagerly.

        Raises:
            TypeError: If *cast_recipe* is not a :class:`SequenceCastRecipe`,
                ``dict``, or ``None``.
        """
        _store = self._resolve_store(store)
        ef, sf = self._resolve_features(_store, entity_features, static_features)
        super().__init__(
            _store,
            IntervalSequenceSettings(
                id_column=id_column,
                start_column=start_column,
                end_column=end_column,
                entity_features=ef,
                static_features=sf,
            ),
            cast_recipe=cast_recipe,
        )

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def builder(
        cls, *, sort_anchor: Literal["start", "end", "middle"] = "start"
    ) -> IntervalSequenceStoreBuilder:
        """Return a fluent builder for constructing an interval sequence store.

        Args:
            sort_anchor: Intra-sequence sort column - ``"start"`` (default),
                         ``"end"`` for right-censored datasets, or ``"middle"``
                         to sort by the interval midpoint
                         ``(T_START + T_END) / 2``.
        """
        return cls._make_builder(sort_anchor=sort_anchor)  # type: ignore[return-value]

    def _assign_bins(
        self,
        lf: pl.LazyFrame,
        t_min,
        bin_size_native: int | float,
        is_datetime: bool,
        bin_col: str = "__bin__",
    ) -> pl.LazyFrame:
        """Explodes each interval across all bins it overlaps: [bin_start, bin_end] inclusive."""
        start_col, end_col = self.settings.get_temporal_columns()
        return (
            lf.with_columns(
                [
                    self._to_bin_expr(
                        start_col, t_min, bin_size_native, is_datetime
                    ).alias(f"__{bin_col}_start__"),
                    self._to_bin_expr(
                        end_col, t_min, bin_size_native, is_datetime
                    ).alias(f"__{bin_col}_end__"),
                ]
            )
            .filter(
                pl.col(f"__{bin_col}_end__") >= 0
            )  # drop intervals entirely before origin
            .with_columns(
                pl.int_ranges(
                    pl.max_horizontal(pl.col(f"__{bin_col}_start__"), pl.lit(0)),
                    pl.col(f"__{bin_col}_end__") + 1,
                ).alias(bin_col)
            )
            .explode(bin_col)
            .drop([f"__{bin_col}_start__", f"__{bin_col}_end__"])
        )

    # ------------------------------------------------------------------
    # Type conversions
    # ------------------------------------------------------------------

    def as_interval(self) -> IntervalSequencePool:
        """Return this pool unchanged - source and target types are identical.

        A warning is emitted to signal the no-op conversion.

        Returns:
            ``self`` (no copy, no I/O).
        """
        warnings.warn(
            "as_interval() called on a pool that is already an interval pool - "
            "this is a no-op. Check that you have the correct pool type.",
            UserWarning,
            stacklevel=2,
        )
        return self

    def as_state(self) -> NoReturn:
        """Not supported: interval → state conversion is ambiguous.

        Intervals may overlap or contain gaps; neither property can be resolved
        into contiguous non-overlapping states without domain-specific
        merge / fill logic. Apply a manual Polars transformation instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "interval → state conversion is not supported: intervals may overlap "
            "or contain gaps, which cannot be resolved into contiguous states "
            "automatically. Apply a manual transformation instead."
        )

    def as_event(
        self,
        anchor: Literal["start", "end", "middle"],
        *,
        time_column: str = "time",
        destination: str | Path | None = None,
        overwrite: bool = False,
    ) -> EventSequencePool:
        """Convert this interval pool to an event pool by anchoring to one timestamp.

        Args:
            anchor: ``"start"``, ``"end"``, or ``"middle"`` - selects which
                timestamp (or their midpoint) becomes the event timestamp.
            time_column: User-facing name for the event timestamp. Defaults to ``"time"``.
            destination: ``None`` → ephemeral result; path → new persistent store.
            overwrite: Replace *destination* if it already exists.

        Returns:
            A new :class:`EventSequencePool`.
        """
        return self._as_event_from_period(
            anchor=anchor,
            time_column=time_column,
            destination=destination,
            overwrite=overwrite,
        )
