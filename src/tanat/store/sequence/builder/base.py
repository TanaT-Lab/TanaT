#!/usr/bin/env python3
"""
SequenceStoreBuilder: fluent base class for constructing a SequenceStore.

Each ``add_*`` call declares the source-local column names; the builder
renames them directly to ``SCH.*`` internal names before writing.
No intermediate canonical layer exists - column naming for display is
a View-layer concern handled by Pool/Sequence settings.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import shutil

import polars as pl
from tanat_utils import DisplayMixin, Registrable

from ..schema import StoreSchema as SCH
from ..store import SequenceStore
from ....core.path import resolve_path
from ....metadata.sequence import SequenceMetadata

LOGGER = logging.getLogger(__name__)


class SequenceStoreBuilder(ABC, Registrable, DisplayMixin):
    """
    Fluent builder that accumulates data sources then writes a :class:`SequenceStore`.

    Each ``add_*`` call declares:

    * ``id_column``      - which source column is the sequence ID
    * time index kwargs  - which column(s) are the time index dimension
    * ``features``    - feature columns to extract

    The builder renames every source directly to ``SCH.*`` internal names,
    concatenates/joins all sources, then runs the write pipeline inline.

    Obtain an instance via :meth:`SequencePool.builder` (recommended).
    """

    _REGISTER = {}
    _TYPE_SUBMODULE = "type"

    # Subclasses declare the mapping from time index kwarg name to SCH.* constant,
    # e.g. {"time_column": SCH.T_EVENT} or {"start_column": SCH.T_START, ...}.
    _TIME_INDEX_SCHEMA_MAP: dict[str, str] = {}

    def __init__(
        self,
    ) -> None:
        self._entity_entries: list[dict] = []
        self._static_entries: list[dict] = []
        # Time index dtype reference set by the first entity source; compared against
        # subsequent ones to detect silent coercions by diagonal_relaxed concat.
        self._time_index_dtypes: dict[str, pl.DataType] | None = None

    # ------------------------------------------------------------------
    # Source registration - implemented by typed subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def add_dataframe(
        self,
        data,
        *,
        id_column: str,
        features: list[str],
        is_static: bool = False,
        **time_index_kwargs,
    ) -> SequenceStoreBuilder:
        """Register an in-memory Polars / Pandas DataFrame."""

    @abstractmethod
    def add_csv(
        self,
        path,
        *,
        id_column: str,
        features: list[str],
        is_static: bool = False,
        **kw,
    ) -> SequenceStoreBuilder:
        """Register a CSV file as a source."""

    @abstractmethod
    def add_parquet(
        self,
        path,
        *,
        id_column: str,
        features: list[str],
        is_static: bool = False,
        **kw,
    ) -> SequenceStoreBuilder:
        """Register a Parquet file (glob patterns supported)."""

    @abstractmethod
    def add_sql(
        self,
        connection: str,
        query: str,
        *,
        id_column: str,
        features: list[str],
        is_static: bool = False,
        **kw,
    ) -> SequenceStoreBuilder:
        """Register a SQL query (requires ``connectorx``)."""

    # ------------------------------------------------------------------
    # Sort hook - override to add intra-sequence temporal ordering
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare_entity(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Prepare the entity ``LazyFrame`` before writing.

        Implementors must at minimum sort by ``SEQ_ID`` and the relevant
        time column(s).  Subclasses may also enrich the frame here
        (e.g. deriving ``T_END`` for state sequences) or run validation
        checks before the data reaches the write pipeline.
        """

    # ------------------------------------------------------------------
    # Build - public entry point
    # ------------------------------------------------------------------

    def build(self, store_path: str | Path, *, exist_ok: bool = False) -> Path:
        """
        Run the write pipeline and persist the store to *store_path*.

        Args:
            store_path: Destination directory for the store.
            exist_ok:   Overwrite an existing store if ``True``.

        Returns:
            The resolved :class:`Path` to the written store directory.

        Raises:
            FileExistsError: If the store exists and ``exist_ok=False``.
            ValueError:      If no entity source is registered.
        """
        store_path = (
            resolve_path(store_path)
            if isinstance(store_path, str)
            else Path(store_path)
        )
        if not self._entity_entries:
            raise ValueError(
                "No entity source registered. "
                "Call .add_csv(), .add_parquet(), .add_dataframe(), or .add_sql() first."
            )
        if not exist_ok and store_path.exists():
            raise FileExistsError(
                f"Store already exists at '{store_path}'. "
                "Use exist_ok=True to overwrite."
            )
        if exist_ok and store_path.exists():
            shutil.rmtree(store_path)

        entity_lf = self._merge_entity()
        static_lf = self._merge_static() if self._static_entries else None

        self._run(store_path, entity_lf, static_lf)
        return store_path

    def build_from_frames(
        self,
        store_path: str | Path,
        entity_lf: pl.LazyFrame,
        static_lf: pl.LazyFrame | None = None,
        *,
        presorted: bool = False,
        exist_ok: bool = False,
    ) -> Path:
        """Write a store directly from pre-prepared LazyFrames.

        Unlike :meth:`build`, this method bypasses source registration and
        :meth:`_to_internal` renaming.  The caller provides frames already in
        ``SCH.*`` internal names:

        * ``entity_lf``: ``SEQ_ID | time_cols | feature_cols``
        * ``static_lf``: ``SEQ_ID | static_cols``  (optional)

        Intended for :meth:`~tanat.sequence.base.pool.SequencePool.save` so
        that the pool can prepare its frames (filter, cast, virtual merge) and
        delegate all I/O to the builder - keeping stores as read-only objects.

        Args:
            store_path: Destination directory.
            entity_lf:  Entity LazyFrame in ``SCH.*`` names.
            static_lf:  Optional static LazyFrame in ``SCH.*`` names
                (with ``SEQ_ID`` column included).
            presorted:  Skip the :meth:`_prepare_entity` step when frames are already
                ordered by ``SEQ_ID`` then by time column within each
                sequence (always the case for frames read from an existing store).
            exist_ok:   Overwrite an existing store if ``True``.

        Returns:
            The resolved :class:`Path` to the written store directory.

        Raises:
            FileExistsError: If the store exists and ``exist_ok=False``.
        """
        store_path = (
            resolve_path(store_path)
            if isinstance(store_path, str)
            else Path(store_path)
        )
        if not exist_ok and store_path.exists():
            raise FileExistsError(
                f"Store already exists at '{store_path}'. "
                "Use exist_ok=True to overwrite."
            )
        if exist_ok and store_path.exists():
            shutil.rmtree(store_path)

        self._run(store_path, entity_lf, static_lf, presorted=presorted)
        return store_path

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run(
        self,
        store_path: Path,
        entity_lf: pl.LazyFrame,
        static_lf: pl.LazyFrame | None,
        presorted: bool = False,
    ) -> dict[str, int]:
        """
        Execute the write pipeline and persist all store files.

        Files written:

        * ``sequence_index.arrow``  -- ``_seq_id | offset | length``
        * ``time_index.arrow``      -- whichever ``SCH.T_*`` cols are present
        * ``entity_features.arrow`` -- every non-internal column
        * ``static_features.arrow`` -- (optional) static feature columns
        * ``core.json``             -- container type + counts (immutable)
        * ``metadata.json``         -- computed feature/temporal metadata

        Returns:
            ``dict`` with keys ``n_sequences`` and ``n_entities``.
        """
        store_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("ETL starting -> %s", store_path)

        step3_desc = (
            "Writing entity, time index & static features"
            if static_lf is not None
            else "Writing entity & time index features"
        )
        seq_type = self.__class__.get_registration_name().capitalize()
        self._display_header(f"{seq_type} SequenceStore")

        self._display_step(1, 4, "Sorting & preparing data")
        entity_lf, time_cols, entity_cols, master_ids = self._etl_prepare(
            entity_lf, static_lf, presorted
        )

        self._display_step(2, 4, "Building sequence index")
        sequence_index_df, n_sequences, n_entities = self._etl_build_index(
            store_path, entity_lf, master_ids
        )

        self._display_step(3, 4, step3_desc)
        time_index_lf, entity_feature_lf, static_lf = self._etl_write_features(
            store_path, entity_lf, static_lf, time_cols, entity_cols, master_ids
        )

        self._display_step(4, 4, "Computing & writing metadata")
        self._etl_write_metadata(
            store_path,
            sequence_index_df,
            time_index_lf,
            entity_feature_lf,
            static_lf,
            n_sequences,
            n_entities,
        )

        LOGGER.info(
            "ETL complete -> %s  [%d sequences, %d entities]",
            store_path,
            n_sequences,
            n_entities,
        )
        self._display_footer(f"{n_sequences:,} sequences · {n_entities:,} entities")
        return {"n_sequences": n_sequences, "n_entities": n_entities}

    # ------------------------------------------------------------------
    # ETL steps
    # ------------------------------------------------------------------

    def _etl_prepare(
        self,
        entity_lf: pl.LazyFrame,
        static_lf: pl.LazyFrame | None,
        presorted: bool,
    ) -> tuple[pl.LazyFrame, list[str], list[str], pl.LazyFrame]:
        """Sort/enrich the entity frame, derive column lists, and build the master ID set."""
        # Sort by SEQ_ID + time columns; subclasses may also enrich or validate
        # the frame (e.g. deriving T_END for states).
        # Skipped when presorted=True (frames come from an existing store).
        if not presorted:
            entity_lf = self._prepare_entity(entity_lf)
        schema_names = entity_lf.collect_schema().names()
        time_cols = [c for c in SCH.time_index_columns() if c in schema_names]
        entity_cols = [c for c in schema_names if c not in SCH.internal_columns()]
        # Master seq_id list: sorted union of every ID present across all sources.
        # This is the authority — sequence_index and static are both aligned to it.
        master_ids = self._master_ids(entity_lf, static_lf)
        return entity_lf, time_cols, entity_cols, master_ids

    def _etl_build_index(
        self,
        store_path: Path,
        entity_lf: pl.LazyFrame,
        master_ids: pl.LazyFrame,
    ) -> tuple[pl.DataFrame, int, int]:
        """Build and persist ``sequence_index.arrow``; return ``(df, n_sequences, n_entities)``."""
        # Collect is fine here — one row per sequence, always small.
        # Reuse the DF to extract counts at zero extra cost.
        sequence_index_df = self._build_sequence_index(master_ids, entity_lf).collect()
        sequence_index_df.write_ipc(store_path / SCH.Files.SEQUENCE_INDEX)
        n_sequences: int = len(sequence_index_df)
        n_entities: int = sequence_index_df[SCH.LENGTH].sum()
        return sequence_index_df, n_sequences, n_entities

    def _etl_write_features(
        self,
        store_path: Path,
        entity_lf: pl.LazyFrame,
        static_lf: pl.LazyFrame | None,
        time_cols: list[str],
        entity_cols: list[str],
        master_ids: pl.LazyFrame,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame | None]:
        """Sink time index, entity (and optional static) feature files to *store_path*."""
        # entity_lf comes from get_temporal_data(): a horizontal concat of three
        # lazy branches (SEQ_ID repeat_by/explode, time index scan, entity scan +
        # optional virtual hconcat).  The Polars ≥1.38.1 streaming engine panics
        # with SchemaMismatch when sink_ipc encounters that nested plan structure.
        # SEQ_ID is not written to any output file, so we materialise only the
        # columns that will actually land on disk (time_cols + entity_cols), then
        # derive both sinks from that single in-memory frame.
        entity_lf = entity_lf.select(time_cols + entity_cols).collect().lazy()
        entity_feature_lf = entity_lf.select(entity_cols)
        time_index = entity_lf.select(time_cols)
        time_index.sink_ipc(store_path / SCH.Files.TIME_INDEX)
        entity_feature_lf.sink_ipc(store_path / SCH.Files.ENTITY_FEATURES)
        if static_lf is not None:
            static_lf = self._write_static(store_path, master_ids, static_lf)
        return time_index, entity_feature_lf, static_lf

    def _etl_write_metadata(
        self,
        store_path: Path,
        sequence_index_df: pl.DataFrame,
        time_index_lf: pl.LazyFrame,
        entity_feature_lf: pl.LazyFrame,
        static_lf: pl.LazyFrame | None,
        n_sequences: int,
        n_entities: int,
    ) -> None:
        """Write ``core.json`` and ``metadata.json``."""
        SequenceStore.write_core_json(
            store_path,
            sequence_type=self.__class__.get_registration_name(),
            n_sequences=n_sequences,
            n_entities=n_entities,
        )
        metadata = self._get_metadata(
            sequence_index=sequence_index_df,
            time_index=time_index_lf,
            entity_features=entity_feature_lf,
            static_features=static_lf,
        )
        SequenceStore.write_metadata_json(metadata, store_path)

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _master_ids(
        self, entity_lf: pl.LazyFrame, static_lf: pl.LazyFrame | None
    ) -> pl.LazyFrame:
        """
        Return a sorted ``LazyFrame`` of every ``SEQ_ID`` present across
        all sources (entity ∪ static), with no duplicates.

        This frame drives both the sequence index and the static alignment,
        ensuring every source has exactly the same set of IDs in the same order.
        """
        ids = entity_lf.select(SCH.SEQ_ID).unique()
        if static_lf is not None:
            ids = pl.concat([ids, static_lf.select(SCH.SEQ_ID).unique()]).unique()
        return ids.sort(SCH.SEQ_ID)

    def _build_sequence_index(
        self, master_ids: pl.LazyFrame, entity_lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Build the sequence index aligned to *master_ids*.

        Left-join master_ids ← entity row counts so that seq_ids present
        only in static get ``length=0`` and a valid offset.

        The result is explicitly sorted by ``SEQ_ID`` to guarantee:
        - a deterministic, stable order in the written IPC file, and
        - correct offset computation via ``cum_sum`` (offsets must align
          with the physical row order of the entity files, which are also
          sorted by ``SEQ_ID``).
        """
        row_counts = entity_lf.group_by(SCH.SEQ_ID).agg(pl.len().alias(SCH.LENGTH))
        return (
            master_ids.join(row_counts, on=SCH.SEQ_ID, how="left")
            .with_columns(pl.col(SCH.LENGTH).fill_null(0))
            .sort(
                SCH.SEQ_ID
            )  # explicit: LazyFrame joins don't guarantee left-side order
            .with_columns(
                pl.col(SCH.LENGTH).cum_sum().shift(1, fill_value=0).alias(SCH.OFFSET)
            )
            .select([SCH.SEQ_ID, SCH.OFFSET, SCH.LENGTH])
        )

    def _write_static(
        self,
        store_path: Path,
        master_ids: pl.LazyFrame,
        static_lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """
        Write ``static_features.arrow`` aligned to *master_ids*.

        Left-join master_ids ← static so that the output has exactly one row
        per master seq_id in the canonical order (nulls for missing IDs).

        Returns the aligned static features frame without the SEQ_ID column.
        """
        static_cols = [c for c in static_lf.collect_schema().names() if c != SCH.SEQ_ID]
        static_features_lf = (
            master_ids.collect()  # materialise sorted IDs: streaming engine does
            .lazy()  # not preserve JOIN row order from lazy left sides
            .join(
                static_lf.select([SCH.SEQ_ID] + static_cols),
                on=SCH.SEQ_ID,
                how="left",
            )
            .select(static_cols)
            .collect()  # materialise before sink_ipc to guarantee row order
            .lazy()
        )

        static_features_lf.sink_ipc(store_path / SCH.Files.STATIC_FEATURES)

        return static_features_lf

    def _get_metadata(
        self,
        sequence_index: pl.LazyFrame,
        time_index: pl.LazyFrame,
        entity_features: pl.LazyFrame,
        static_features: pl.LazyFrame | None,
    ) -> SequenceMetadata:
        """Build :class:`SequenceMetadata` from the live lazy frames."""
        return SequenceMetadata(
            seq_id=sequence_index.collect_schema()[SCH.SEQ_ID],
            time_index=SequenceMetadata.infer_time_index(time_index),
            entity_features=SequenceMetadata.infer_entity_features(entity_features),
            static_features=SequenceMetadata.infer_static_features(static_features),
        )

    # ------------------------------------------------------------------
    # Source helpers
    # ------------------------------------------------------------------

    def _validate_source(
        self,
        source,
        *,
        id_column: str,
        features: list[str],
        is_static: bool,
        time_index_kwargs: dict,
    ) -> None:
        """Validate that all declared columns exist in *source*."""
        required = [id_column] + list(features)
        if not is_static:
            for key, val in time_index_kwargs.items():
                if val is None:
                    raise ValueError(
                        f"'{key}' is required for entity sources but was not provided."
                    )
            required += list(time_index_kwargs.values())
        available = set(source.schema().names())
        missing = [c for c in required if c not in available]
        if missing:
            raise ValueError(
                f"{source!r}: missing column(s) {missing}. "
                f"Available: {sorted(available)}"
            )

    def _stage(
        self,
        source,
        *,
        id_column: str,
        features: list[str],
        is_static: bool,
        time_index_kwargs: dict,
    ) -> SequenceStoreBuilder:
        """Append the source entry to the build queue."""
        entry = {
            "source": source,
            "id_column": id_column,
            "features": list(features),
            "time_index_kwargs": {
                k: v for k, v in time_index_kwargs.items() if v is not None
            },
        }
        if not is_static:
            self._warn_time_index_dtype_mismatch(source, entry)
        (self._static_entries if is_static else self._entity_entries).append(entry)
        return self

    def _warn_time_index_dtype_mismatch(self, source, entry: dict) -> None:
        """Warn when the incoming source's time index dtypes differ from the first registered source.

        The first entity source sets the reference dtypes.  Every subsequent
        ``add_*`` call compares its time index columns against that reference.
        Called at registration time so the user gets immediate feedback.
        """
        schema = source.schema()
        incoming = {
            internal_col: schema[src_col]
            for kwarg, internal_col in self._TIME_INDEX_SCHEMA_MAP.items()
            if (src_col := entry["time_index_kwargs"].get(kwarg)) and src_col in schema
        }
        if self._time_index_dtypes is None:
            self._time_index_dtypes = incoming
            return
        for col, dtype in incoming.items():
            ref = self._time_index_dtypes.get(col)
            if ref is not None and type(ref) is not type(dtype):
                LOGGER.warning(
                    "Time index dtype mismatch detected on %r: %s (reference) vs %s (new source). "
                    "At build time Polars will silently coerce both to a common supertype - "
                    "cast to a consistent dtype before registering if this is unintended.",
                    col,
                    ref,
                    dtype,
                )

    def _to_internal(self, lf: pl.LazyFrame, entry: dict) -> pl.LazyFrame:
        """
        Rename source-local column names to ``SCH.*`` internal names and
        select only the columns needed for the pipeline.

        This is the single place where source-local names are translated to
        internal store names.
        """
        rename: dict[str, str] = {}

        # Rename the ID column if needed
        if entry["id_column"] != SCH.SEQ_ID:
            rename[entry["id_column"]] = SCH.SEQ_ID

        # Rename each time column if it was provided and not already named correctly
        for kwarg_name, internal_name in self._TIME_INDEX_SCHEMA_MAP.items():
            src_col = entry["time_index_kwargs"].get(kwarg_name)
            if src_col is not None and src_col != internal_name:
                rename[src_col] = internal_name

        # Only select time columns that were actually provided by the caller.
        time_cols = [
            internal_name
            for kwarg_name, internal_name in self._TIME_INDEX_SCHEMA_MAP.items()
            if kwarg_name in entry["time_index_kwargs"]
        ]
        select_cols = [SCH.SEQ_ID] + time_cols + entry["features"]
        if rename:
            lf = lf.rename(rename)
        return lf.select(select_cols)

    def _merge_entity(self) -> pl.LazyFrame:
        """Normalise each entity source to ``SCH.*``, then concatenate."""
        frames = [
            self._to_internal(e["source"].read(), e) for e in self._entity_entries
        ]
        return pl.concat(frames, how="diagonal_relaxed")

    def _merge_static(self) -> pl.LazyFrame:
        """Rename id_column → ``SCH.SEQ_ID`` for each static source, then left-join."""
        frames = []
        for entry in self._static_entries:
            lf = entry["source"].read()
            if entry["id_column"] != SCH.SEQ_ID:
                lf = lf.rename({entry["id_column"]: SCH.SEQ_ID})
            frames.append(lf.select([SCH.SEQ_ID] + entry["features"]))
        base = frames[0]
        for other in frames[1:]:
            base = base.join(other, on=SCH.SEQ_ID, how="left")
        return base
