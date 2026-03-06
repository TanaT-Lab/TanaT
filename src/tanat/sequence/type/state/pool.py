#!/usr/bin/env python3
"""State sequence pool implementation."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from ...base.pool import SequencePool
from .settings import StateSequenceSettings

if TYPE_CHECKING:
    from datetime import datetime
    from ....store.sequence.store import SequenceStore
    from ...base.cast import SequenceCastRecipe
    from ....store.sequence.builder.type.state import StateSequenceStoreBuilder
    from ..event.pool import EventSequencePool
    from ..interval.pool import IntervalSequencePool


LOGGER = logging.getLogger(__name__)


class StateSequencePool(SequencePool, register_name="state"):
    """Pool of state sequences.

    States are **contiguous and non-overlapping**: the end of one state
    is always the start of the next, with no gaps in between.
    """

    SETTINGS_CLASS = StateSequenceSettings

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
        """Create a state sequence pool backed by *store*.

        Args:
            store: Store path, name, or :class:`SequenceStore` instance.
            id_column: User-facing name for the sequence ID column.
            start_column: User-facing name for the state start column.
            end_column: User-facing name for the state end column.
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
            StateSequenceSettings(
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
        cls,
        *,
        end_value: datetime | int | float | None = None,
        validate_continuity: bool = True,
    ) -> StateSequenceStoreBuilder:
        """Return a fluent builder for constructing a state sequence store.

        Args:
            end_value: Sentinel for ``T_END`` of the last state in each sequence
                       when ``end_column`` is not provided at source registration
                       time.  ``None`` → leaves the last ``T_END`` as ``null``.
            validate_continuity: When ``end_column`` *is* provided, verify that
                       states are truly contiguous (``T_END[i] == T_START[i+1]``)
                       before writing.  Defaults to ``True``.  Set to ``False``
                       on large datasets where the cost of a full ``collect()``
                       is unacceptable.
        """
        return cls._make_builder(  # type: ignore[return-value]
            end_value=end_value,
            validate_continuity=validate_continuity,
        )

    def _assign_bins(
        self,
        lf: pl.LazyFrame,
        t_min,
        bin_size_native: int | float,
        is_datetime: bool,
        bin_col: str = "__bin__",
    ) -> pl.LazyFrame:
        """Explodes each state across all bins it spans: [bin_start, bin_end] inclusive."""
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
            )  # drop states entirely before origin
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

    def as_state(self) -> StateSequencePool:
        """Return this pool unchanged - source and target types are identical.

        A warning is emitted to signal the no-op conversion.

        Returns:
            ``self`` (no copy, no I/O).
        """
        warnings.warn(
            "as_state() called on a pool that is already a state pool - "
            "this is a no-op. Check that you have the correct pool type.",
            UserWarning,
            stacklevel=2,
        )
        return self

    def as_interval(
        self,
        *,
        start_column: str | None = None,
        end_column: str | None = None,
        destination: str | Path | None = None,
        overwrite: bool = False,
    ) -> IntervalSequencePool:
        """Convert this state pool to an interval pool.

        States and intervals share the same ``(_t_start, _t_end)`` physical
        layout - no temporal recomputation needed.

        Args:
            start_column: User-facing name for the start column.
                ``None`` inherits this pool's current setting.
            end_column: User-facing name for the end column.
                ``None`` inherits this pool's current setting.
            destination: ``None`` → ephemeral result; path → new persistent store.
            overwrite: Replace *destination* if it already exists.

        Returns:
            A new :class:`IntervalSequencePool`.
        """
        interval_cls = SequencePool.get_registered("interval")
        new_settings = {
            "id_column": self.settings.id_column,
            "start_column": start_column or self.settings.start_column,
            "end_column": end_column or self.settings.end_column,
            "entity_features": list(self.settings.entity_features),
            "static_features": list(self.settings.static_features),
        }
        new_uuid = self._store.fork_virtual_context(self._virtual_id)
        if destination is None:
            return self._reinterpret_as(interval_cls, new_settings, new_uuid)
        return self._persist_as(
            interval_cls, new_settings, new_uuid, destination, overwrite
        )

    def as_event(
        self,
        anchor: Literal["start", "end", "middle"],
        *,
        time_column: str = "time",
        destination: str | Path | None = None,
        overwrite: bool = False,
    ) -> EventSequencePool:
        """Convert this state pool to an event pool by anchoring to one timestamp.

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
