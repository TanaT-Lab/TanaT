#!/usr/bin/env python3
"""
Sequence Store Base Class.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import logging
import shutil
from pathlib import Path
from typing import Callable
import pandas as pd
import polars as pl

from ...metadata.sequence import SequenceMetadata
from ..base.store import BaseStore
from ..base.utils import (
    apply_cast_exprs,
    check_no_reserved_names,
    drop_columns_from_file,
    hconcat_physical_virtual,
    normalise_to_lazyframe,
    probe_cast_recipe,
)
from .schema import StoreSchema as SCH

LOGGER = logging.getLogger(__name__)


class SequenceStore(BaseStore):
    """
    Sequence store.

    Delegates virtual (temporary) feature storage to a ``VirtualStore``
    and inherits shared I/O helpers from ``BaseStore``
    (which itself inherits ``StaticStoreMixin``).
    """

    _MAIN_INDEX_PROPERTY: str = "sequence_index"
    _MAIN_ID_PROPERTY: str = "seq_id_col"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, root_path: str | Path) -> None:
        """
        Initialise the store (lazy loading).

        Args:
            root_path: Root directory of the store.
        """
        super().__init__(root_path)

        # Name cache (physical columns only; invalidated after drop/snapshot)
        self._phys_entity_names: list[str] | None = None

    def _required_files(self) -> list[str]:
        """List of file names that must exist in the root directory."""
        return [
            SCH.Files.CORE,
            SCH.Files.SEQUENCE_INDEX,
            SCH.Files.TIME_INDEX,
            SCH.Files.ENTITY_FEATURES,
        ]

    # ------------------------------------------------------------------
    # Settings / Manifest
    # ------------------------------------------------------------------

    def get_sequence_type(self) -> str:
        """Returns the sequence type declared in ``core.json``."""
        seq_type = self.core.get("type")
        if seq_type is None:
            raise ValueError(
                f"Invalid Store: sequence type not found in core.json in {self._root_path}."
            )
        return seq_type

    @staticmethod
    def write_core_json(
        path: Path,
        sequence_type: str,
        n_sequences: int,
        n_entities: int,
    ) -> None:
        """Writes ``core.json`` - static store facts set once at build time."""
        core = {
            "__NOTICE__": "auto-generated. DO NOT EDIT BY HAND",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "container": "sequence",
            "type": sequence_type,
            "n_sequences": n_sequences,
            "n_entities": n_entities,
        }
        with open(path / SCH.Files.CORE, "w", encoding="utf-8") as fh:
            json.dump(core, fh, indent=4)

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    @property
    def sequence_index(self) -> pl.LazyFrame:
        """Navigation index (seq_id, offset, length) - physical, no cast overlay."""
        return pl.scan_ipc(self._root_path / SCH.Files.SEQUENCE_INDEX)

    def time_index(self, virtual_id: str | None = None) -> pl.LazyFrame:
        """Time-index rows (``_t_event`` or ``_t_start`` / ``_t_end``), with optional virtual override.

        When *virtual_id* is given and ``tmp/<virtual_id>/time_index.arrow``
        exists, the virtual time index is returned **instead of** the
        physical one (full replacement - the whole temporal structure changes
        during type conversions).  Falls back to the physical file when the
        virtual override is absent.

        When *virtual_id* is ``None``, always returns the physical time index.

        Args:
            virtual_id: Optional virtual context identifier.

        Returns:
            A :class:`polars.LazyFrame` of the time-index rows.
        """
        if virtual_id is None:
            return pl.scan_ipc(self._root_path / SCH.Files.TIME_INDEX)
        virtual_ti = self._virtual.time_index(virtual_id)
        if virtual_ti is not None:
            return virtual_ti
        return pl.scan_ipc(self._root_path / SCH.Files.TIME_INDEX)

    def write_virtual_time_index(
        self,
        virtual_id: str,
        time_index_lf: pl.LazyFrame,
    ) -> None:
        """Write a virtual time-index override for *virtual_id*.

        Creates ``tmp/<virtual_id>/`` if it does not exist, then writes
        *time_index_lf* as ``time_index.arrow`` there.  A subsequent call
        to :meth:`time_index` with the same *virtual_id* will return this
        override instead of the physical time index.

        Args:
            virtual_id: Virtual context identifier (a UUID string).
            time_index_lf: LazyFrame containing the new time columns
                (``_t_start`` + ``_t_end`` for period types, or ``_t_event``
                for event types).
        """
        self._virtual.write_time_index(virtual_id, time_index_lf)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get_id_lf(
        self,
        id_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        *,
        explode: bool = False,
        **_kw,
    ) -> pl.LazyFrame:
        """Sequence IDs as a single-column lazy frame, with optional cast recipe applied.

        Args:
            id_caster: If set, applied to the ID column as a column-agnostic
                ``Callable[[pl.Expr], pl.Expr]`` (e.g. from
                :meth:`SequenceCastRecipe.id_caster`).
            explode: When ``False`` (default), returns one row per sequence
                (unique IDs).  When ``True``, expands each ID by its entity
                count so the result is row-aligned with :meth:`entity` and
                :meth:`get_time_index`.

        Returns:
            A :class:`polars.LazyFrame` with a single column of sequence IDs.
        """
        if explode:
            lf = (
                self.sequence_index.select([SCH.SEQ_ID, SCH.LENGTH])
                .filter(pl.col(SCH.LENGTH) > 0)
                .select(pl.col(SCH.SEQ_ID).repeat_by(pl.col(SCH.LENGTH)).explode())
            )
        else:
            lf = self.sequence_index.select(SCH.SEQ_ID)
        if id_caster is not None:
            lf = lf.with_columns(id_caster(pl.col(SCH.SEQ_ID)))
        return lf

    def get_slice(
        self,
        id_value,
        id_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> tuple[int, int]:
        """Returns ``(offset, length)`` for slicing a pool-level row mask.

        *id_caster* is the user-facing cast recipe for *id_value* - when set,
        the store applies it to its own column before filtering.
        """
        idx = self.sequence_index
        if id_caster is not None:
            idx = idx.with_columns(id_caster(pl.col(SCH.SEQ_ID)))
        row = (
            idx.filter(pl.col(SCH.SEQ_ID) == id_value)
            .select(SCH.OFFSET, SCH.LENGTH)
            .collect()
        )
        return int(row[SCH.OFFSET][0]), int(row[SCH.LENGTH][0])

    def get_sequence_length(
        self,
        id_value,
        id_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> int:
        """Returns the number of entity rows for a given sequence ID.

        *id_caster* is the user-facing cast recipe for *id_value*; the store
        applies it to its own column before filtering.
        """
        idx = self.sequence_index
        if id_caster is not None:
            idx = idx.with_columns(id_caster(pl.col(SCH.SEQ_ID)))
        return int(
            idx.filter(pl.col(SCH.SEQ_ID) == id_value)
            .select(SCH.LENGTH)
            .collect()
            .item()
        )

    @property
    def seq_id_col(self) -> str:
        """Internal name of the sequence ID column."""
        return SCH.SEQ_ID

    @property
    def seq_id_dtype(self) -> pl.DataType:
        """Physical dtype of the sequence ID column (IPC header read, no data scan)."""
        return self.sequence_index.collect_schema()[SCH.SEQ_ID]

    # ------------------------------------------------------------------
    # Cast probes (fast validation on a small sample before accepting a cast)
    # ------------------------------------------------------------------

    def probe_time_cast_recipe(
        self, recipe: list[pl.DataType], n_rows: int = 10
    ) -> None:
        """Validate the time-index cast recipe on a small sample."""
        t_lf = self.time_index()
        t_names = SCH.time_index_columns()
        present = [c for c in t_lf.collect_schema().names() if c in t_names]
        if present:
            probe_cast_recipe(t_lf, {c: recipe for c in present}, n_rows)

    def probe_entity_cast_recipe(
        self, entity: dict[str, list[pl.DataType]], n_rows: int = 10
    ) -> None:
        """Validate entity-feature cast recipes on a small sample."""
        probe_cast_recipe(self.entity(), entity, n_rows)

    def structural_columns(
        self, is_static: bool = False, virtual_id: str | None = None
    ) -> list[str]:
        """
        Returns the structural column names for a data access call.

        Always includes the sequence ID column.  For entity data
        (non-static) also includes the time columns actually
        present in this store's time index (physical or virtual).
        """
        cols = [SCH.SEQ_ID]
        if not is_static:
            schema_names = set(self.time_index(virtual_id).collect_schema().names())
            cols += [c for c in SCH.time_index_columns() if c in schema_names]
        return cols

    # ------------------------------------------------------------------
    # Entity features
    # ------------------------------------------------------------------

    def entity_features(self, virtual_id: str | None = None) -> list[str]:
        """List of entity feature column names (physical + virtual when *virtual_id* is set)."""
        if virtual_id is not None:
            return self.entity(virtual_id).collect_schema().names()
        if self._phys_entity_names is None:
            self._phys_entity_names = (
                pl.scan_ipc(self._root_path / SCH.Files.ENTITY_FEATURES)
                .collect_schema()
                .names()
            )
        return self._phys_entity_names

    def entity(self, virtual_id: str | None = None) -> pl.LazyFrame:
        """Entity feature rows (physical + virtual), without seq_id.

        Virtual features take precedence: any physical column whose name is
        also present in the virtual context is silently shadowed, so the
        virtual value is always returned.
        """
        return hconcat_physical_virtual(
            pl.scan_ipc(self._root_path / SCH.Files.ENTITY_FEATURES),
            self._virtual.features(virtual_id, is_static=False) if virtual_id else None,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _infer_metadata(
        self,
        virtual_id: str | None = None,
    ) -> SequenceMetadata:
        """
        Infers **all** metadata (base + virtual), unfiltered.

        Used internally by :meth:`copy_to`.
        The view layer (Pool / Sequence) builds its own metadata
        directly from its cast-and-masked LazyFrames.

        Cached when ``virtual_id is None``.

        Args:
            virtual_id: If set, includes virtual features.
        """
        if virtual_id is None and self._metadata_cache is not None:
            LOGGER.debug("Metadata cache hit: %s", self._root_path)
            return self._metadata_cache

        entity_lf = self.entity(virtual_id)
        static_lf = self.static(virtual_id)
        result = SequenceMetadata(
            seq_id=self.sequence_index.collect_schema()[SCH.SEQ_ID],
            time_index=SequenceMetadata.infer_time_index(self.time_index()),
            entity_features=SequenceMetadata.infer_entity_features(entity_lf),
            static_features=SequenceMetadata.infer_static_features(static_lf),
        )

        if virtual_id is None:
            self._metadata_cache = result
        return result

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def get_temporal_data(
        self,
        virtual_id: str | None = None,
        *,
        id_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        time_index_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        feature_exprs: list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Returns the full temporal data: seq_id + time index + entity features.

        Cast recipes are applied after assembly - physical store is never touched.
        """
        lf = pl.concat(
            [
                self.get_id_lf(id_caster=id_caster, explode=True),
                self.get_time_index(virtual_id, time_index_caster=time_index_caster),
                self.entity(virtual_id),
            ],
            how="horizontal",
        )
        if feature_exprs:
            lf = apply_cast_exprs(lf, feature_exprs)
        return lf

    def get_time_index(
        self,
        virtual_id: str | None = None,
        *,
        time_index_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Returns time-index columns only, with optional cast recipe applied.

        Cheaper than :meth:`get_id_time_index` when the sequence ID
        column is not needed.

        Args:
            virtual_id: Optional virtual context identifier.
            time_index_caster: If set, applied to each time column as a
                column-agnostic ``Callable[[pl.Expr], pl.Expr]``.

        Returns:
            A :class:`polars.LazyFrame` of the time-index columns.
        """
        lf = self.time_index(virtual_id)
        if time_index_caster is not None:
            cols = lf.collect_schema().names()
            lf = lf.with_columns([time_index_caster(pl.col(c)) for c in cols])
        return lf

    def get_id_time_index(
        self,
        virtual_id: str | None = None,
        *,
        id_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        time_index_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Returns seq_id + time-index columns only (no entity features).

        Cheaper than :meth:`get_temporal_data` when entity features are not needed.
        Cast recipes are applied after assembly.
        """
        return pl.concat(
            [
                self.get_id_lf(id_caster=id_caster, explode=True),
                self.get_time_index(virtual_id, time_index_caster=time_index_caster),
            ],
            how="horizontal",
        )

    def get_entity_row(
        self,
        id_value,
        rank: int,
        virtual_id: str | None = None,
        *,
        feature_exprs: list[pl.Expr] | None = None,
        id_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> dict:
        """
        Returns the feature values for the entity at *rank* within *id_value*.

        Only feature columns are returned - ``_seq_id``, time and
        transient columns are stripped internally.  Column selection is
        the responsibility of the caller (Entity).

        Args:
            id_value: The sequence identifier (user-facing type).
            rank: 0-based row index within that sequence.
            virtual_id: Optional virtual feature context.
            feature_exprs: Pre-built cast expressions applied before collecting.
            id_caster: User-facing cast recipe for *id_value*; the store
                applies it to its own column before the slice lookup.

        Returns:
            ``dict`` mapping feature name → scalar value.
        """
        lf = self.entity(virtual_id)
        offset, _ = self.get_slice(id_value, id_caster=id_caster)
        physical_rank = offset + rank
        row = (
            lf.with_row_index(SCH.ROW_IDX)
            .filter(pl.col(SCH.ROW_IDX) == physical_rank)
            .drop(SCH.ROW_IDX)
        )
        if not feature_exprs:
            return row.collect().row(0, named=True)
        return apply_cast_exprs(row, feature_exprs).collect().row(0, named=True)

    def get_time_at(
        self,
        id_value,
        rank: int,
        *,
        time_index_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        id_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ):
        """
        Returns the time-index value(s) for the entity at *rank* within *id_value*.

        Returns a single scalar for event sequences, or a two-element
        list ``[start, end]`` for interval sequences.

        *id_caster* is the user-facing cast recipe for *id_value*.
        """
        offset, _ = self.get_slice(id_value, id_caster=id_caster)
        physical_rank = offset + rank
        lf = (
            self.time_index()
            .with_row_index(SCH.ROW_IDX)
            .filter(pl.col(SCH.ROW_IDX) == physical_rank)
            .drop(SCH.ROW_IDX)
        )
        if time_index_caster is not None:
            cols = lf.collect_schema().names()
            lf = lf.with_columns([time_index_caster(pl.col(c)) for c in cols])
        row = lf.collect().row(0, named=True)
        values = list(row.values())
        return values[0] if len(values) == 1 else values

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def drop_features(
        self,
        features: list[str],
        is_static: bool = False,
        virtual_id: str | None = None,
    ) -> None:
        """
        Removes feature columns from disk.

        The full list is sent to both physical and virtual stores;
        each one silently ignores columns it doesn't own.

        Args:
            features: Column names to remove.
            is_static: Static or entity features.
            virtual_id: Optional virtual context.
        """
        if not features:
            return

        path = self._resolve_feature_path(self._root_path, is_static)
        if drop_columns_from_file(path, features):
            self._clear_feature_cache(is_static)

        if virtual_id:
            self._virtual.drop_features(virtual_id, features, is_static=is_static)

    def add_entity_features(
        self,
        virtual_id: str,
        df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    ) -> list[str]:
        """Add positional entity features to a virtual store context.

        Computes the expected row count from the time index and delegates
        height validation to :meth:`VirtualStore.add_entity_features`.

        Args:
            virtual_id: Must already exist.
            df: Feature-only DataFrame positionally aligned with entity rows.

        Returns:
            The list of column names written.
        """
        lf = normalise_to_lazyframe(df)
        check_no_reserved_names(
            lf.collect_schema().names(),
            SCH.internal_columns(),
            context="internal sequence store columns",
        )
        expected_height = self.time_index().select(pl.len()).collect().item()
        return self._virtual.add_entity_features(
            virtual_id=virtual_id,
            df=lf,
            expected_height=expected_height,
        )

    # ------------------------------------------------------------------
    # Temporal conversion helpers
    # ------------------------------------------------------------------

    def _fork_event_to_interval(
        self,
        virtual_id: str | None,
        duration: timedelta | int | float | str,
        feature_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        time_index_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> str:
        """Fork a virtual context with ``(_t_start, _t_end)`` computed from an event index.

        ``_t_start = _t_event``.  ``_t_end`` is determined by *duration*:

        - ``str``: column name of a per-row entity feature.
        - :class:`~datetime.timedelta` or numeric scalar: applied uniformly.

        Args:
            virtual_id: Active virtual context to inherit from (``None`` → physical only).
            duration: Scalar offset or entity feature column name.
            feature_caster: Callable applying the full cast recipe to the duration
                column (``str`` case only), or ``None``.
            time_index_caster: Callable applying the full cast recipe to the time
                column, or ``None``.

        Returns:
            UUID of the new forked context.
        """
        LOGGER.debug(
            "Fork event -> interval (virtual_id=%r, duration=%r)", virtual_id, duration
        )
        active_ti = self.time_index(virtual_id)
        if time_index_caster is not None:
            active_ti = apply_cast_exprs(
                active_ti, [time_index_caster(pl.col(SCH.T_EVENT))]
            )
        if isinstance(duration, str):
            entity_col = self.entity(virtual_id).select(pl.col(duration))
            if feature_caster is not None:
                entity_col = apply_cast_exprs(
                    entity_col, [feature_caster(pl.col(duration))]
                )
            combined = pl.concat(
                [
                    self.get_id_lf(explode=True),
                    active_ti,
                    entity_col,
                ],
                how="horizontal",
            )
            new_ti = combined.select(
                pl.col(SCH.T_EVENT).alias(SCH.T_START),
                (pl.col(SCH.T_EVENT) + pl.col(duration)).alias(SCH.T_END),
            )
        else:
            new_ti = active_ti.select(
                pl.col(SCH.T_EVENT).alias(SCH.T_START),
                (pl.col(SCH.T_EVENT) + pl.lit(duration)).alias(SCH.T_END),
            )
        new_uuid = self.fork_virtual_context(virtual_id) or self._virtual.new_context()
        self.write_virtual_time_index(new_uuid, new_ti)
        return new_uuid

    def _fork_event_to_state(
        self,
        virtual_id: str | None,
        end_value: datetime | int | float | str | None,
        time_index_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        static_caster: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> str:
        """Fork a virtual context with ``(_t_start, _t_end)`` where ``_t_end`` is the next event start.

        ``_t_start = _t_event``.  ``_t_end = _t_event.shift(-1).over(seq_id)``.
        The last row per sequence gets *end_value* as ``_t_end`` (or stays
        ``null`` when *end_value* is ``None``).

        Args:
            virtual_id: Active virtual context to inherit from (``None`` → physical only).
            end_value: Fill value for the last row's ``_t_end``, or ``None``.
                A ``str`` names a static feature column whose per-sequence value is used.
            time_index_caster: Callable applying the full cast recipe to the time
                column, or ``None``.
            static_caster: Callable applying the full cast recipe to the static
                end_value column (``str`` case only), or ``None``.

        Returns:
            UUID of the new forked context.
        """
        LOGGER.debug(
            "Fork event -> state (virtual_id=%r, end_value=%r)", virtual_id, end_value
        )
        active_ti = self.time_index(virtual_id)
        if time_index_caster is not None:
            active_ti = apply_cast_exprs(
                active_ti, [time_index_caster(pl.col(SCH.T_EVENT))]
            )
        combined = pl.concat(
            [self.get_id_lf(explode=True), active_ti], how="horizontal"
        )
        # Sequences are contiguous in the store: the last row of each sequence is the
        # one where the next row belongs to a different sequence (or doesn't exist).
        is_last = (pl.col(SCH.SEQ_ID).shift(-1) != pl.col(SCH.SEQ_ID)).fill_null(True)
        if isinstance(end_value, str):
            static_col = self.get_static_data(virtual_id).select(
                [SCH.SEQ_ID, end_value]
            )
            if static_caster is not None:
                static_col = apply_cast_exprs(
                    static_col, [static_caster(pl.col(end_value))]
                )
            new_ti = (
                combined.join(static_col, on=SCH.SEQ_ID, how="left")
                .select(
                    pl.col(SCH.SEQ_ID),
                    pl.col(SCH.T_EVENT).alias(SCH.T_START),
                    pl.col(SCH.T_EVENT).shift(-1).over(SCH.SEQ_ID).alias(SCH.T_END),
                    pl.col(end_value),
                )
                .with_columns(
                    pl.when(is_last)
                    .then(pl.col(end_value))
                    .otherwise(pl.col(SCH.T_END))
                    .alias(SCH.T_END)
                )
                .drop([SCH.SEQ_ID, end_value])
            )
        else:
            new_ti = combined.select(
                pl.col(SCH.SEQ_ID),
                pl.col(SCH.T_EVENT).alias(SCH.T_START),
                pl.col(SCH.T_EVENT).shift(-1).over(SCH.SEQ_ID).alias(SCH.T_END),
            )
            if end_value is not None:
                new_ti = new_ti.with_columns(
                    pl.when(is_last)
                    .then(pl.lit(end_value))
                    .otherwise(pl.col(SCH.T_END))
                    .alias(SCH.T_END)
                )
            new_ti = new_ti.drop(SCH.SEQ_ID)
        new_uuid = self.fork_virtual_context(virtual_id) or self._virtual.new_context()
        self.write_virtual_time_index(new_uuid, new_ti)
        return new_uuid

    def _fork_period_to_event(
        self,
        virtual_id: str | None,
        anchor: str,
        time_index_caster: Callable[[pl.Expr], pl.Expr] | None = None,
        time_index_dtype: pl.DataType | None = None,
    ) -> str:
        """Fork a virtual context with ``_t_event`` projected from a period time index.

        - ``'start'``:  ``_t_event = _t_start``
        - ``'end'``:    ``_t_event = _t_end``
        - ``'middle'``: ``_t_event = (_t_start + _t_end) / 2``
          (works for both Datetime and numeric timestep axes).

        Args:
            virtual_id: Active virtual context to inherit from (``None`` → physical only).
            anchor: One of ``'start'``, ``'end'``, ``'middle'``.
            time_index_caster: Callable applying the full cast recipe to the time
                columns, or ``None``.
            time_index_dtype: Final dtype after the cast recipe.  When provided,
                avoids a ``collect_schema`` call on the middle-anchor path.

        Returns:
            UUID of the new forked context.
        """
        LOGGER.debug(
            "Fork period -> event (virtual_id=%r, anchor=%r)", virtual_id, anchor
        )
        active_ti = self.time_index(virtual_id)
        if time_index_caster is not None:
            active_ti = apply_cast_exprs(
                active_ti,
                [
                    time_index_caster(pl.col(SCH.T_START)),
                    time_index_caster(pl.col(SCH.T_END)),
                ],
            )
        if anchor == "start":
            new_ti = active_ti.select(pl.col(SCH.T_START).alias(SCH.T_EVENT))
        elif anchor == "end":
            new_ti = active_ti.select(pl.col(SCH.T_END).alias(SCH.T_EVENT))
        else:  # middle
            # Resolve the effective dtype without a lazy-plan collect when possible.
            col_type = (
                time_index_dtype or self.time_index().collect_schema()[SCH.T_START]
            )
            if isinstance(col_type, (pl.Datetime, pl.Date)):
                # Polars forbids adding two absolute timestamps (`start + end`),
                # so the midpoint must be expressed as `start + (end - start) / 2`
                # where `(end - start)` produces a Duration that can be added back.
                new_ti = active_ti.select(
                    (
                        pl.col(SCH.T_START)
                        + (pl.col(SCH.T_END) - pl.col(SCH.T_START)) / 2
                    ).alias(SCH.T_EVENT)
                )
            else:
                # Numeric path: `(start + end) / 2` is the natural form.
                # Polars promotes integer `/` to Float64; cast back to preserve dtype.
                midpoint = (pl.col(SCH.T_START) + pl.col(SCH.T_END)) / 2
                if not isinstance(col_type, (pl.Float32, pl.Float64)):
                    midpoint = midpoint.cast(col_type)
                new_ti = active_ti.select(midpoint.alias(SCH.T_EVENT))
        new_uuid = self.fork_virtual_context(virtual_id) or self._virtual.new_context()
        self.write_virtual_time_index(new_uuid, new_ti)
        return new_uuid

    # ------------------------------------------------------------------
    # Copy / snapshot helpers
    # ------------------------------------------------------------------

    def copy_to(self, target: Path, type_name: str, *, exist_ok: bool = False) -> None:
        """Copy all store files to *target*, writing *type_name* into ``core.json``.

        Fast-path used by :meth:`~tanat.sequence.base.pool.SequencePool.save`
        when no transformation is needed (no virtual features, no masks, no casts).
        *type_name* is always provided by the pool via
        ``get_registration_name()`` - never copied verbatim from the physical
        ``core.json`` (avoids the silent-corruption bug where a converted pool
        would persist the wrong type).

        Args:
            target: Destination directory (must already exist or be creatable).
            type_name: Sequence type to write into ``core.json``.
            exist_ok: If True, allows the target directory to already exist.
        """
        target.mkdir(parents=True, exist_ok=exist_ok)
        for fname in (SCH.Files.SEQUENCE_INDEX, SCH.Files.TIME_INDEX):
            src = self._root_path / fname
            if src.exists():
                shutil.copy2(src, target / fname)
        for is_static in (False, True):
            src = self._resolve_feature_path(self._root_path, is_static)
            if src.exists():
                shutil.copy2(src, self._resolve_feature_path(target, is_static))
        self._write_core_snapshot(target, type_name)
        src = self._root_path / SCH.Files.METADATA
        if src.exists():
            shutil.copy2(src, target / SCH.Files.METADATA)

    def _write_core_snapshot(self, target: Path, type_name: str) -> None:
        """Writes ``core.json`` to *target* with *type_name* as the sequence type.

        Reads the physical ``core.json``, overrides ``"type"`` and refreshes
        ``"created_at"`` to now (UTC).  All other fields are preserved.
        """
        core = self.core
        core["type"] = type_name
        core["created_at"] = datetime.now(timezone.utc).isoformat()
        with open(target / SCH.Files.CORE, "w", encoding="utf-8") as fh:
            json.dump(core, fh, indent=4)

    def _invalidate_all_caches(self) -> None:
        """Clears every name/metadata cache after an in-place rewrite."""
        self._clear_feature_cache(is_static=False)
        self._clear_feature_cache(is_static=True)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _clear_feature_cache(self, is_static: bool) -> None:
        """Invalidates feature-related caches."""
        if is_static:
            self._clear_static_cache()
        else:
            self._phys_entity_names = None
        self._metadata_cache = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_feature_path(self, root: Path, is_static: bool) -> Path:
        """Returns the physical path to the entity or static feature file."""
        fname = SCH.Files.STATIC_FEATURES if is_static else SCH.Files.ENTITY_FEATURES
        return root / fname
