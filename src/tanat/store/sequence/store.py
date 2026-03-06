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
import pandas as pd
import polars as pl

from ...metadata.sequence import SequenceMetadata
from ..common.virtual import VirtualStore
from ..common.static import StaticStoreMixin
from ..common.utils import (
    apply_casts,
    check_no_reserved_names,
    drop_columns_from_file,
    hconcat_physical_virtual,
    normalise_to_lazyframe,
    probe_cast,
)
from .schema import StoreSchema as SCH

LOGGER = logging.getLogger(__name__)


class SequenceStore(StaticStoreMixin):
    """
    Sequence store.

    Delegates virtual (temporary) feature storage to a ``VirtualStore``
    and inherits shared I/O helpers from ``StaticStoreMixin``
    (which itself inherits ``StoreMixin``).
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
        self._root_path = Path(root_path)
        self._check_structure()

        # Virtual store delegate
        self._virtual = VirtualStore(self._root_path)

        # Name cache (physical columns only; invalidated after drop/snapshot)
        self._phys_entity_names: list[str] | None = None

        # Metadata cache
        self._metadata_cache: SequenceMetadata | None = None

    @property
    def root_path(self) -> Path:
        """Root directory of this store."""
        return self._root_path

    def _check_structure(self) -> None:
        """Validates that the store directory contains the required files."""
        if not self._root_path.exists():
            raise FileNotFoundError(f"Store path not found: {self._root_path}")

        required = [
            SCH.Files.CORE,
            SCH.Files.SEQUENCE_INDEX,
            SCH.Files.TEMPORAL_INDEX,
            SCH.Files.ENTITY_FEATURES,
        ]
        for fname in required:
            if not (self._root_path / fname).exists():
                raise FileNotFoundError(
                    f"Invalid Store: Missing required file '{fname}' in {self._root_path}"
                )

    # ------------------------------------------------------------------
    # Settings / Manifest
    # ------------------------------------------------------------------

    @property
    def core(self) -> dict:
        """Returns the static store facts written once at build time."""
        with open(self._root_path / SCH.Files.CORE, "r", encoding="utf-8") as f:
            return json.load(f)

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

    @staticmethod
    def write_metadata_json(metadata: SequenceMetadata, path: Path) -> None:
        """Writes *metadata* to *path* as ``metadata.json``."""
        data = {
            "__NOTICE__": "auto-generated. DO NOT EDIT BY HAND",
            **metadata.to_json_dict(),
        }
        with open(path / SCH.Files.METADATA, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    @property
    def sequence_index(self) -> pl.LazyFrame:
        """Navigation index (seq_id, offset, length) - physical, no cast overlay."""
        return pl.scan_ipc(self._root_path / SCH.Files.SEQUENCE_INDEX)

    def temporal(self, virtual_id: str | None = None) -> pl.LazyFrame:
        """Temporal rows, with optional virtual override.

        When *virtual_id* is given and ``tmp/<virtual_id>/temporal_index.arrow``
        exists, the virtual temporal index is returned **instead of** the
        physical one (full replacement - the whole temporal structure changes
        during type conversions).  Falls back to the physical file when the
        virtual override is absent.

        When *virtual_id* is ``None``, always returns the physical temporal.

        Args:
            virtual_id: Optional virtual context identifier.

        Returns:
            A :class:`polars.LazyFrame` of the temporal rows.
        """
        if virtual_id is None:
            return pl.scan_ipc(self._root_path / SCH.Files.TEMPORAL_INDEX)
        virtual_temporal = self._virtual.temporal(virtual_id)
        if virtual_temporal is not None:
            return virtual_temporal
        return pl.scan_ipc(self._root_path / SCH.Files.TEMPORAL_INDEX)

    def write_virtual_temporal(
        self,
        virtual_id: str,
        temporal_lf: pl.LazyFrame,
    ) -> None:
        """Write a virtual temporal override for *virtual_id*.

        Creates ``tmp/<virtual_id>/`` if it does not exist, then writes
        *temporal_lf* as ``temporal_index.arrow`` there.  A subsequent call
        to :meth:`temporal` with the same *virtual_id* will return this
        override instead of the physical temporal.

        Args:
            virtual_id: Virtual context identifier (a UUID string).
            temporal_lf: LazyFrame containing the new temporal columns
                (``_t_start`` + ``_t_end`` for period types, or ``_t_event``
                for event types).
        """
        self._virtual.write_temporal(virtual_id, temporal_lf)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get_sorted_ids(self, id_cast: pl.DataType | None = None) -> list:
        """All sequence IDs in their stored order, cast to *id_cast* when provided.

        Use this at view boundaries to expose IDs in the user-facing type while
        keeping all internal navigation on the physical (stored) type.
        """
        ids = self.sequence_index.select(SCH.SEQ_ID).collect().to_series().to_list()
        if id_cast is None:
            return ids
        return pl.Series("id", ids).cast(id_cast).to_list()

    def get_slice(
        self, id_value, id_cast: pl.DataType | None = None
    ) -> tuple[int, int]:
        """Returns ``(offset, length)`` for slicing a pool-level row mask.

        *id_cast* is the user-facing cast type of *id_value* - when set, the
        store casts its own column before filtering (no reverse cast needed).
        """
        idx = self.sequence_index
        if id_cast is not None:
            idx = idx.with_columns(pl.col(SCH.SEQ_ID).cast(id_cast))
        row = (
            idx.filter(pl.col(SCH.SEQ_ID) == id_value)
            .select(SCH.OFFSET, SCH.LENGTH)
            .collect()
        )
        return int(row[SCH.OFFSET][0]), int(row[SCH.LENGTH][0])

    def get_sequence_length(self, id_value, id_cast: pl.DataType | None = None) -> int:
        """Returns the number of entity rows for a given sequence ID.

        *id_cast* is the user-facing cast type of *id_value* - when set, the
        store casts its own column before filtering.
        """
        idx = self.sequence_index
        if id_cast is not None:
            idx = idx.with_columns(pl.col(SCH.SEQ_ID).cast(id_cast))
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

    def probe_entity_cast(
        self, schema: dict[str, pl.DataType], n_rows: int = 10
    ) -> None:
        """
        Validates *schema* against a sample of entity-feature rows.

        Reads at most *n_rows* rows from ``entity_features.arrow``
        (IPC footer + a tiny slice - no full scan) and attempts every
        cast.  Raises :exc:`TypeError` if any cast is incompatible.

        Args:
            schema: Mapping of feature name → target dtype.
            n_rows: Sample size (default: 10).
        """
        probe_cast(self.entity(), schema, n_rows)

    def probe_id_cast(self, dtype: pl.DataType, n_rows: int = 10) -> None:
        """
        Validates casting the sequence-ID column to *dtype*.

        Args:
            dtype: Target Polars DataType.
            n_rows: Sample size (default: 10).
        """
        probe_cast(self.sequence_index.select(SCH.SEQ_ID), {SCH.SEQ_ID: dtype}, n_rows)

    def probe_temporal_cast(self, dtype: pl.DataType, n_rows: int = 10) -> None:
        """
        Validates casting all temporal columns to *dtype*.

        Only columns actually present in ``temporal_index.arrow`` are
        checked (the schema is read from the IPC footer - no full scan).

        Args:
            dtype: Target Polars DataType.
            n_rows: Sample size (default: 10).
        """
        t_lf = self.temporal()
        t_names = SCH.temporal_columns()
        present = [c for c in t_lf.collect_schema().names() if c in t_names]
        if present:
            probe_cast(t_lf, {c: dtype for c in present}, n_rows)

    def structural_columns(
        self, is_static: bool = False, virtual_id: str | None = None
    ) -> list[str]:
        """
        Returns the structural column names for a data access call.

        Always includes the sequence ID column.  For entity data
        (non-static) also includes the temporal columns actually
        present in this store's temporal index (physical or virtual).
        """
        cols = [SCH.SEQ_ID]
        if not is_static:
            schema_names = set(self.temporal(virtual_id).collect_schema().names())
            cols += [c for c in SCH.temporal_columns() if c in schema_names]
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
            temporal=SequenceMetadata.infer_temporal(self.temporal()),
            entity_features=SequenceMetadata.infer_entity_features(entity_lf),
            static_features=SequenceMetadata.infer_static_features(static_lf),
        )

        if virtual_id is None:
            self._metadata_cache = result
        return result

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _ids_col(self) -> pl.LazyFrame:
        """Expands seq_id into one row per entity, aligned with entity rows."""
        return (
            self.sequence_index.select([SCH.SEQ_ID, SCH.LENGTH])
            .filter(pl.col(SCH.LENGTH) > 0)
            .select(pl.col(SCH.SEQ_ID).repeat_by(pl.col(SCH.LENGTH)).explode())
        )

    def get_sequence_data(
        self,
        virtual_id: str | None = None,
        *,
        id_cast: pl.DataType | None = None,
        temporal_cast: pl.DataType | None = None,
        feature_casts: dict[str, pl.DataType] | None = None,
    ) -> pl.LazyFrame:
        """Returns the full sequence data: seq_id + temporal + entity features.

        Cast overlays are applied after assembly - physical store is never touched.
        """
        lf = pl.concat(
            [self._ids_col(), self.temporal(virtual_id), self.entity(virtual_id)],
            how="horizontal",
        )
        cast_schema: dict[str, pl.DataType] = {}
        if id_cast is not None:
            cast_schema[SCH.SEQ_ID] = id_cast
        if temporal_cast is not None:
            for col in self.temporal(virtual_id).collect_schema().names():
                cast_schema[col] = temporal_cast
        if feature_casts:
            cast_schema.update(feature_casts)
        return apply_casts(lf, cast_schema) if cast_schema else lf

    def get_entity_row(
        self,
        id_value,
        rank: int,
        virtual_id: str | None = None,
        *,
        feature_casts: dict[str, pl.DataType] | None = None,
        id_cast: pl.DataType | None = None,
    ) -> dict:
        """
        Returns the feature values for the entity at *rank* within *id_value*.

        Only feature columns are returned - ``_seq_id``, temporal and
        transient columns are stripped internally.  Column selection is
        the responsibility of the caller (Entity).

        Args:
            id_value: The sequence identifier (user-facing type).
            rank: 0-based row index within that sequence.
            virtual_id: Optional virtual feature context.
            feature_casts: Optional cast schema applied before collecting.
            id_cast: User-facing cast type of *id_value*; the store casts
                its own column before the slice lookup.

        Returns:
            ``dict`` mapping feature name → scalar value.
        """
        lf = self.entity(virtual_id)
        offset, _ = self.get_slice(id_value, id_cast=id_cast)
        physical_rank = offset + rank
        row = (
            lf.with_row_index(SCH.ROW_IDX)
            .filter(pl.col(SCH.ROW_IDX) == physical_rank)
            .drop(SCH.ROW_IDX)
        )
        if feature_casts is None:
            return row.collect().row(0, named=True)
        return apply_casts(row, feature_casts).collect().row(0, named=True)

    def get_temporal_at(
        self,
        id_value,
        rank: int,
        *,
        temporal_cast: pl.DataType | None = None,
        id_cast: pl.DataType | None = None,
    ):
        """
        Returns the temporal value(s) for the entity at *rank* within *id_value*.

        Returns a single scalar for event sequences, or a two-element
        list ``[start, end]`` for interval sequences.

        *id_cast* is the user-facing cast type of *id_value*.
        """
        offset, _ = self.get_slice(id_value, id_cast=id_cast)
        physical_rank = offset + rank
        lf = (
            self.temporal()
            .with_row_index(SCH.ROW_IDX)
            .filter(pl.col(SCH.ROW_IDX) == physical_rank)
            .drop(SCH.ROW_IDX)
        )
        if temporal_cast is not None:
            cols = lf.collect_schema().names()
            lf = apply_casts(lf, dict.fromkeys(cols, temporal_cast))
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

        Computes the expected row count from the temporal index and delegates
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
        expected_height = self.temporal().select(pl.len()).collect().item()
        return self._virtual.add_entity_features(
            virtual_id=virtual_id,
            df=lf,
            expected_height=expected_height,
        )

    def clear_virtual_context(self, virtual_id: str) -> None:
        """Removes a virtual context (features on disk + cast registries)."""
        self._virtual.clear_context(virtual_id)

    def fork_virtual_context(self, source_virtual_id: str | None) -> str | None:
        """Fork *source_virtual_id* into a new context, or ``None`` if nothing to inherit.

        Returns ``None`` immediately when *source_virtual_id* is ``None``;
        otherwise delegates to :meth:`~tanat.store.common.virtual.VirtualStore.fork_context`.
        """
        if source_virtual_id is None:
            return None
        return self._virtual.fork_context(source_virtual_id)

    # ------------------------------------------------------------------
    # Temporal conversion helpers
    # ------------------------------------------------------------------

    def _fork_event_to_interval(
        self,
        virtual_id: str | None,
        duration: timedelta | int | float | str,
    ) -> str:
        """Fork a virtual context with ``(_t_start, _t_end)`` computed from an event index.

        ``_t_start = _t_event``.  ``_t_end`` is determined by *duration*:

        - ``str``: column name of a per-row entity feature.
        - :class:`~datetime.timedelta` or numeric scalar: applied uniformly.

        Args:
            virtual_id: Active virtual context to inherit from (``None`` → physical only).
            duration: Scalar offset or entity feature column name.

        Returns:
            UUID of the new forked context.
        """
        LOGGER.debug(
            "Fork event -> interval (virtual_id=%r, duration=%r)", virtual_id, duration
        )
        active_temporal = self.temporal(virtual_id)
        if isinstance(duration, str):
            combined = pl.concat(
                [
                    self._ids_col(),
                    active_temporal,
                    self.entity(virtual_id).select(pl.col(duration)),
                ],
                how="horizontal",
            )
            temporal_lf = combined.select(
                pl.col(SCH.T_EVENT).alias(SCH.T_START),
                (pl.col(SCH.T_EVENT) + pl.col(duration)).alias(SCH.T_END),
            )
        else:
            temporal_lf = active_temporal.select(
                pl.col(SCH.T_EVENT).alias(SCH.T_START),
                (pl.col(SCH.T_EVENT) + pl.lit(duration)).alias(SCH.T_END),
            )
        new_uuid = self.fork_virtual_context(virtual_id) or self._virtual.new_context()
        self.write_virtual_temporal(new_uuid, temporal_lf)
        return new_uuid

    def _fork_event_to_state(
        self,
        virtual_id: str | None,
        end_value: datetime | int | float | None,
    ) -> str:
        """Fork a virtual context with ``(_t_start, _t_end)`` where ``_t_end`` is the next event start.

        ``_t_start = _t_event``.  ``_t_end = _t_event.shift(-1).over(seq_id)``.
        The last row per sequence gets *end_value* as ``_t_end`` (or stays
        ``null`` when *end_value* is ``None``).

        Args:
            virtual_id: Active virtual context to inherit from (``None`` → physical only).
            end_value: Fill value for the last row's ``_t_end``, or ``None``.

        Returns:
            UUID of the new forked context.
        """
        LOGGER.debug(
            "Fork event -> state (virtual_id=%r, end_value=%r)", virtual_id, end_value
        )
        combined = pl.concat(
            [self._ids_col(), self.temporal(virtual_id)], how="horizontal"
        )
        temporal_lf = combined.select(
            pl.col(SCH.T_EVENT).alias(SCH.T_START),
            pl.col(SCH.T_EVENT).shift(-1).over(SCH.SEQ_ID).alias(SCH.T_END),
        )
        if end_value is not None:
            temporal_lf = temporal_lf.with_columns(
                pl.col(SCH.T_END).fill_null(pl.lit(end_value))
            )
        new_uuid = self.fork_virtual_context(virtual_id) or self._virtual.new_context()
        self.write_virtual_temporal(new_uuid, temporal_lf)
        return new_uuid

    def _fork_period_to_event(
        self,
        virtual_id: str | None,
        anchor: str,
    ) -> str:
        """Fork a virtual context with ``_t_event`` projected from a period temporal index.

        - ``'start'``:  ``_t_event = _t_start``
        - ``'end'``:    ``_t_event = _t_end``
        - ``'middle'``: ``_t_event = (_t_start + _t_end) / 2``
          (works for both Datetime and numeric timestep axes).

        Args:
            virtual_id: Active virtual context to inherit from (``None`` → physical only).
            anchor: One of ``'start'``, ``'end'``, ``'middle'``.

        Returns:
            UUID of the new forked context.
        """
        LOGGER.debug(
            "Fork period -> event (virtual_id=%r, anchor=%r)", virtual_id, anchor
        )
        active_temporal = self.temporal(virtual_id)
        if anchor == "start":
            temporal_lf = active_temporal.select(pl.col(SCH.T_START).alias(SCH.T_EVENT))
        elif anchor == "end":
            temporal_lf = active_temporal.select(pl.col(SCH.T_END).alias(SCH.T_EVENT))
        else:  # middle
            col_type = active_temporal.collect_schema()[SCH.T_START]
            if isinstance(col_type, (pl.Datetime, pl.Date)):
                temporal_lf = active_temporal.select(
                    (
                        pl.col(SCH.T_START)
                        + (pl.col(SCH.T_END) - pl.col(SCH.T_START)) / 2
                    ).alias(SCH.T_EVENT)
                )
            else:
                temporal_lf = active_temporal.select(
                    ((pl.col(SCH.T_START) + pl.col(SCH.T_END)) / 2).alias(SCH.T_EVENT)
                )
        new_uuid = self.fork_virtual_context(virtual_id) or self._virtual.new_context()
        self.write_virtual_temporal(new_uuid, temporal_lf)
        return new_uuid

    # ------------------------------------------------------------------
    # Copy / snapshot helpers
    # ------------------------------------------------------------------

    def _effective_row_mask(
        self,
        id_mask: set | None,
        row_mask: pl.Series | None,
    ) -> pl.Series:
        """
        Combines ``id_mask`` and ``row_mask`` into a single row-level
        boolean ``pl.Series`` aligned with entity rows.
        """
        if id_mask is not None:
            id_row_filter = (
                self._ids_col()
                .select(pl.col(SCH.SEQ_ID).is_in(list(id_mask)))
                .collect()
                .to_series()
            )
            if row_mask is not None:
                return row_mask & id_row_filter
            return id_row_filter
        # id_mask is None → row_mask alone
        return row_mask

    def _static_id_mask(self, id_mask: set) -> pl.Series:
        """
        Boolean ``pl.Series`` aligned with the sequence index,
        keeping only IDs present in *id_mask*.
        """
        sids = self.sequence_index.select(SCH.SEQ_ID).collect().to_series()
        return sids.is_in(list(id_mask))

    def _rebuild_sequence_index(
        self,
        id_mask: set | None,
        row_filter: pl.Series | None,
    ) -> pl.LazyFrame:
        """
        Returns a filtered sequence index with recalculated offsets/lengths.
        """
        # __ord preserves original physical sequence order across the join
        lf = self.sequence_index.with_row_index("__ord")

        # Filter by ID
        if id_mask is not None:
            lf = lf.filter(pl.col(SCH.SEQ_ID).is_in(list(id_mask)))

        # Recalculate lengths from the row filter
        if row_filter is not None:
            # Zip seq_id (one per entity row) with the boolean keep-mask
            per_row = pl.concat(
                [
                    self._ids_col(),
                    pl.DataFrame({"_keep": row_filter}).lazy(),
                ],
                how="horizontal",
            )
            new_lengths_lf = per_row.group_by(SCH.SEQ_ID).agg(
                pl.col("_keep").sum().alias(SCH.LENGTH)
            )
            # Replace old lengths; sort restores physical order after the hash-join
            lf = (
                lf.drop(SCH.LENGTH)
                .join(new_lengths_lf, on=SCH.SEQ_ID, how="left")
                .sort("__ord")
            )

        # Drop the helper column, empty sequences, rebuild contiguous offsets
        return (
            lf.drop("__ord")
            .filter(pl.col(SCH.LENGTH) > 0)
            .drop(SCH.OFFSET)
            .with_columns(
                pl.col(SCH.LENGTH).cum_sum().shift(1, fill_value=0).alias(SCH.OFFSET)
            )
        )

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
        for fname in (SCH.Files.SEQUENCE_INDEX, SCH.Files.TEMPORAL_INDEX):
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
