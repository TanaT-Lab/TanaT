#!/usr/bin/env python3
"""Event sequence pool implementation."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ...base.pool import SequencePool
from .settings import EventSequenceSettings

if TYPE_CHECKING:
    from ....store.sequence.store import SequenceStore
    from ...base.cast import SequenceCastRecipe
    from ....store.sequence.builder.type.event import EventSequenceStoreBuilder
    from ..interval.pool import IntervalSequencePool
    from ..state.pool import StateSequencePool


LOGGER = logging.getLogger(__name__)

#: Type alias for the ``duration`` argument in :meth:`EventSequencePool.as_interval`.
#: Can be a :class:`~datetime.timedelta` / numeric scalar (applied uniformly)
#: or a ``str`` naming an entity feature column (per-row duration).
Duration = timedelta | int | float | str


class EventSequencePool(SequencePool, register_name="event"):
    """Pool of event sequences (single timestamp per entity row)."""

    SETTINGS_CLASS = EventSequenceSettings

    # ------------------------------------------------------------------
    # Init (load from existing store)
    # ------------------------------------------------------------------

    def __init__(
        self,
        store: str | Path | SequenceStore,
        *,
        id_column: str = "id",
        time_column: str = "time",
        entity_features: list[str] | None = None,
        static_features: list[str] | None = None,
        cast_recipe: SequenceCastRecipe | dict | None = None,
    ) -> None:
        """Create an event sequence pool backed by *store*.

        Args:
            store: Store path, name, or :class:`SequenceStore` instance.
            id_column: User-facing name for the sequence ID column.
            time_column: User-facing name for the event timestamp column.
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
            EventSequenceSettings(
                id_column=id_column,
                time_column=time_column,
                entity_features=ef,
                static_features=sf,
            ),
            cast_recipe=cast_recipe,
        )

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def builder(cls) -> EventSequenceStoreBuilder:
        """Return a fluent builder for constructing an event sequence store."""
        return cls._make_builder()  # type: ignore[return-value]

    def _assign_bins(
        self,
        lf: pl.LazyFrame,
        t_min,
        bin_size_native: int | float,
        is_datetime: bool,
        bin_col: str = "__bin__",
    ) -> pl.LazyFrame:
        """One bin per row (single event timestamp)."""
        time_col = self.settings.get_time_columns()[0]
        return lf.with_columns(
            self._to_bin_expr(time_col, t_min, bin_size_native, is_datetime).alias(
                bin_col
            )
        ).filter(pl.col(bin_col) >= 0)

    # ------------------------------------------------------------------
    # Type conversions
    # ------------------------------------------------------------------

    def as_event(self) -> EventSequencePool:
        """Return this pool unchanged . Source and target types are identical.

        A warning is emitted to signal the no-op conversion.

        Returns:
            ``self`` (no copy, no I/O).
        """
        warnings.warn(
            "as_event() called on a pool that is already an event pool - "
            "this is a no-op. Check that you have the correct pool type.",
            UserWarning,
            stacklevel=2,
        )
        return self

    def as_interval(
        self,
        duration: Duration,
        *,
        start_column: str = "start",
        end_column: str = "end",
        destination: str | Path | None = None,
        overwrite: bool = False,
    ) -> IntervalSequencePool:
        """Convert this event pool to an interval pool by computing ``_t_end``.

        Each event timestamp becomes ``_t_start``; ``_t_end`` is computed as
        ``_t_start + duration``. The resulting time index is stored as a
        virtual override (ephemeral) or written to a new persistent store.

        Args:
            duration: Interval length added to each event timestamp. Can be:

                - A :class:`~datetime.timedelta` or numeric scalar: applied
                  uniformly to every event.
                - A ``str``: name of an entity feature column whose values
                  provide per-row durations.
            start_column: User-facing name for the start column. Defaults to ``"start"``.
            end_column: User-facing name for the end column. Defaults to ``"end"``.
            destination: ``None`` → ephemeral result; path → new persistent store.
            overwrite: Replace *destination* if it already exists.

        Returns:
            A new :class:`IntervalSequencePool`.
        """
        # validate duration
        if isinstance(duration, str):
            self.settings.validate_features([duration], is_static=False)
            # Type check: the feature column must be compatible with the time index
            if self.metadata.time_index.is_datetime:
                if not self.metadata.is_duration_feature(duration):
                    got = self.metadata.feature_info(duration).dtype
                    raise TypeError(
                        f"Duration column {duration!r} must be of type pl.Duration "
                        "when the time index is Datetime. "
                        f"Got: {got}."
                    )
            else:
                if not self.metadata.is_numeric_feature(duration):
                    got = self.metadata.feature_info(duration).dtype
                    raise TypeError(
                        f"Duration column {duration!r} must be numeric "
                        "when the time index is a timestep (non-Datetime). "
                        f"Got: {got}."
                    )
        elif isinstance(duration, (int, float)):
            if self.metadata.time_index.is_datetime:
                raise ValueError(
                    "Numeric duration is not valid for datetime time index. "
                    "Use a timedelta or an entity feature column instead."
                )
        elif isinstance(duration, timedelta):
            if not self.metadata.time_index.is_datetime:
                raise ValueError(
                    "Timedelta duration is not valid for numeric time index. "
                    "Use a numeric scalar or an entity feature column instead."
                )
        else:
            raise TypeError(
                f"duration must be a timedelta, int, float, or str (feature column name), "
                f"got {type(duration).__name__!r}."
            )

        return self._as_interval_impl(
            duration, start_column, end_column, destination, overwrite
        )

    def as_state(
        self,
        *,
        end_value: datetime | int | float | str | None = None,
        start_column: str = "start",
        end_column: str = "end",
        destination: str | Path | None = None,
        overwrite: bool = False,
    ) -> StateSequencePool:
        """Convert this event pool to a state pool by computing ``_t_end``.

        Each event timestamp becomes ``_t_start``; ``_t_end`` is taken from
        the *next* event in the same sequence (``shift(-1).over(_seq_id)``).

        Args:
            end_value: Sentinel for ``_t_end`` of the last event per sequence.
                ``None`` leaves the last row with ``_t_end = null``.
                A ``str`` names a *static* feature column whose per-sequence
                value fills the last ``_t_end``.
            start_column: User-facing name for the start column. Defaults to ``"start"``.
            end_column: User-facing name for the end column. Defaults to ``"end"``.
            destination: ``None`` → ephemeral result; path → new persistent store.
            overwrite: Replace *destination* if it already exists.

        Returns:
            A new :class:`StateSequencePool`.
        """
        if isinstance(end_value, str):
            self.settings.validate_features([end_value], is_static=True)
            if self.metadata.time_index.is_datetime:
                if not self.metadata.is_datetime_feature(end_value, is_static=True):
                    got = self.metadata.feature_info(end_value, is_static=True).dtype
                    raise TypeError(
                        f"end_value column {end_value!r} must be a Datetime-compatible type "
                        "when the time index is Datetime. "
                        f"Got: {got}."
                    )
            else:
                if not self.metadata.is_numeric_feature(end_value, is_static=True):
                    got = self.metadata.feature_info(end_value, is_static=True).dtype
                    raise TypeError(
                        f"end_value column {end_value!r} must be numeric "
                        "when the time index is a timestep (non-Datetime). "
                        f"Got: {got}."
                    )
        return self._as_state_impl(
            end_value, start_column, end_column, destination, overwrite
        )

    # ------------------------------------------------------------------
    # Type conversion helpers (private)
    # ------------------------------------------------------------------

    def _as_interval_impl(
        self,
        duration: Duration,
        start_column: str,
        end_column: str,
        destination: str | Path | None,
        overwrite: bool = False,
    ) -> IntervalSequencePool:
        """Fork the virtual context with ``(_t_start, _t_end)`` computed from *duration*, then branch on *destination*."""
        new_uuid = self._store._fork_event_to_interval(
            self._virtual_id,
            duration,
            feature_caster=(
                self._casts.entity_caster(duration)
                if isinstance(duration, str)
                else None
            ),
            time_index_caster=self._casts.time_index_caster(),
        )
        new_settings = {
            "id_column": self.settings.id_column,
            "start_column": start_column,
            "end_column": end_column,
            "entity_features": list(self.settings.entity_features),
            "static_features": list(self.settings.static_features),
        }
        interval_cls = SequencePool.get_registered("interval")
        if destination is None:
            return self._reinterpret_as(interval_cls, new_settings, new_uuid)
        return self._persist_as(
            interval_cls, new_settings, new_uuid, destination, overwrite
        )

    def _as_state_impl(
        self,
        end_value: datetime | int | float | str | None,
        start_column: str,
        end_column: str,
        destination: str | Path | None,
        overwrite: bool = False,
    ) -> StateSequencePool:
        """Fork the virtual context with shift-based ``(_t_start, _t_end)``, then branch on *destination*."""
        new_uuid = self._store._fork_event_to_state(
            self._virtual_id,
            end_value,
            time_index_caster=self._casts.time_index_caster(),
            static_caster=(
                self._casts.static_caster(end_value)
                if isinstance(end_value, str)
                else None
            ),
        )
        new_settings = {
            "id_column": self.settings.id_column,
            "start_column": start_column,
            "end_column": end_column,
            "entity_features": list(self.settings.entity_features),
            "static_features": list(self.settings.static_features),
        }
        state_cls = SequencePool.get_registered("state")
        if destination is None:
            return self._reinterpret_as(state_cls, new_settings, new_uuid)
        return self._persist_as(
            state_cls, new_settings, new_uuid, destination, overwrite
        )
