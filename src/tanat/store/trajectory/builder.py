#!/usr/bin/env python3
"""TrajectoryStoreBuilder: fluent builder for creating a TrajectoryPool."""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

import polars as pl

from tanat_utils import DisplayIndentManager, DisplayMixin

from ...core.path import resolve_path
from ...metadata.trajectory import TrajectoryMetadata
from ...sequence.base.pool import SequencePool
from ..source.base import AbstractSource
from ..sequence.store import SequenceStore
from ..sequence.schema import StoreSchema as SCH
from .schema import TrajectorySchema as TSCH
from .store import TrajectoryStore

LOGGER = logging.getLogger(__name__)


class TrajectoryStoreBuilder(DisplayMixin):
    """
    Fluent builder for constructing a :class:`~tanat.trajectory.pool.TrajectoryPool`.

    Each :meth:`add` call registers a :class:`SequencePool` under an alias.
    Call :meth:`build` to write the trajectory store to disk and obtain
    the resolved path - symmetric with :class:`SequenceStoreBuilder`.

    Usage::

        store_path = (
            TrajectoryPool.builder()
            .add("admissions", admissions_pool)
            .add("pharmacy", pharmacy_pool)
            .build("my_trajectories")
        )
        pool = TrajectoryPool(store=store_path)
    """

    def __init__(self) -> None:
        self._pools: dict[str, SequencePool] = {}
        self._static_entries: list[dict] = []

    # ------------------------------------------------------------------
    # Pool registration
    # ------------------------------------------------------------------

    def add(
        self,
        alias: str,
        pool: SequencePool,
        *,
        overwrite: bool = False,
    ) -> TrajectoryStoreBuilder:
        """
        Register a :class:`SequencePool` under *alias*.

        Args:
            alias: Short name (e.g. ``"admissions"``).
            pool: A :class:`SequencePool` instance.
            overwrite: If ``True``, replaces an existing alias silently.

        Returns:
            ``self`` for method chaining.

        Raises:
            TypeError: If *pool* is not a SequencePool.
            ValueError: If the alias is already registered and *overwrite*
                is ``False``.
            TypeError: If the pool's schema (ID dtype or time index type)
                is incompatible with already-registered pools.
        """
        if not isinstance(pool, SequencePool):
            raise TypeError(f"Expected a SequencePool, got {type(pool).__name__}")
        if alias in self._pools and not overwrite:
            raise ValueError(
                f"Alias '{alias}' already added. Use overwrite=True to replace it."
            )
        # Compatibility check against the first other registered pool
        for other_alias, other_pool in self._pools.items():
            if other_alias != alias:
                pool.metadata.assert_id_compatible_with(
                    other_pool.metadata,
                    alias,
                    context="All sequence pools in a TrajectoryPool must share the same ID type.",
                )
                pool.metadata.assert_time_index_compatible_with(
                    other_pool.metadata,
                    alias,
                    context="All sequence pools in a TrajectoryPool must share the same time index schema.",
                )
                break

        self._pools[alias] = pool
        return self

    # ------------------------------------------------------------------
    # Static source registration
    # ------------------------------------------------------------------

    def add_dataframe(
        self,
        data,
        *,
        id_column: str,
        features: list[str],
    ) -> TrajectoryStoreBuilder:
        """Register an in-memory Polars / Pandas DataFrame as static trajectory features."""
        source = AbstractSource.get_registered("dataframe")(data)
        self._validate_static_source(source, id_column=id_column, features=features)
        return self._stage_static(source, id_column=id_column, features=features)

    def add_csv(
        self,
        path,
        *,
        id_column: str,
        features: list[str],
        **reader_kwargs,
    ) -> TrajectoryStoreBuilder:
        """Register a CSV file as static trajectory features."""
        source = AbstractSource.get_registered("csv")(path, **reader_kwargs)
        self._validate_static_source(source, id_column=id_column, features=features)
        return self._stage_static(source, id_column=id_column, features=features)

    def add_parquet(
        self,
        path,
        *,
        id_column: str,
        features: list[str],
        **reader_kwargs,
    ) -> TrajectoryStoreBuilder:
        """Register a Parquet file (glob patterns supported) as static trajectory features."""
        source = AbstractSource.get_registered("parquet")(path, **reader_kwargs)
        self._validate_static_source(source, id_column=id_column, features=features)
        return self._stage_static(source, id_column=id_column, features=features)

    def add_sql(
        self,
        connection: str,
        query: str,
        *,
        id_column: str,
        features: list[str],
        **sql_kwargs,
    ) -> TrajectoryStoreBuilder:
        """Register a SQL query as static trajectory features (requires ``connectorx``)."""
        source = AbstractSource.get_registered("sql")(connection, query, **sql_kwargs)
        self._validate_static_source(source, id_column=id_column, features=features)
        return self._stage_static(source, id_column=id_column, features=features)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        store_path: str | Path,
        *,
        exist_ok: bool = False,
    ) -> Path:
        """
        Persist the trajectory store to *store_path*.

        Resolves *store_path* via the workspace (bare name → workspace
        directory), then writes ``core.json``, ``trajectory_index.arrow``,
        and ``metadata.json``.

        Args:
            store_path: Destination directory.  Can be a workspace store
                name (no ``/``), a relative path, or an absolute path.
            exist_ok: If ``True``, overwrites an existing store on disk.

        Returns:
            The resolved :class:`Path` to the written store directory.

        Raises:
            RuntimeError:    If no pools have been registered.
            FileExistsError: If the store exists and ``exist_ok=False``.
        """
        if not self._pools:
            raise RuntimeError(
                "Nothing to build. Call .add(alias, pool) at least once first."
            )

        resolved = (
            resolve_path(store_path)
            if isinstance(store_path, str)
            else Path(store_path)
        )
        resolved.mkdir(parents=True, exist_ok=exist_ok)

        self._display_header("TrajectoryStore")

        n_pools = len(self._pools)
        aliases = ", ".join(self._pools)
        self._display_step(1, 2, f"Linking pools: {aliases}")
        links = self._resolve_pool_links(resolved, self._pools, overwrite=exist_ok)

        self._display_step(2, 2, "Building trajectory index & metadata")
        stats = self._run(resolved, links)

        self._display_footer(
            f"{stats['n_trajectories']:,} trajectories · {n_pools} pool(s)"
        )
        return resolved

    def build_from_frames(
        self,
        store_path: str | Path,
        traj_idx: pl.LazyFrame,
        static_lf: pl.LazyFrame | None,
        links: dict[str, str],
        *,
        exist_ok: bool = False,
    ) -> Path:
        """
        Write trajectory store files from pre-prepared LazyFrames and links.

        Args:
            store_path: Destination directory.
            traj_idx: Trajectory index frame (``TRAJ_ID`` + bool presence
                columns), already filtered and cast.
            static_lf: Optional static frame **including the** ``TRAJ_ID``
                column, already filtered and cast.  The frame is aligned to
                *traj_idx* via a left-join before writing - guaranteeing
                row order regardless of the frame's input order.
                ``None`` if no static features.
            links: ``{alias: path_string}`` mapping written verbatim to
                ``core.json``.
            exist_ok: If ``True``, the destination may already exist.

        Returns:
            The resolved :class:`Path` to the written store directory.
        """
        resolved = (
            resolve_path(store_path)
            if isinstance(store_path, str)
            else Path(store_path)
        )
        resolved.mkdir(parents=True, exist_ok=exist_ok)

        self._display_header("TrajectoryStore")

        self._display_step(1, 2, "Writing trajectory index")
        index_df, n_trajectories, written_static_lf = self._traj_write_index(
            resolved, traj_idx, static_lf
        )

        self._display_step(2, 2, "Computing & writing metadata")
        self._traj_write_metadata(resolved, index_df, links, written_static_lf)

        self._display_footer(f"{n_trajectories:,} trajectories · {len(links)} pool(s)")
        return resolved

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_pool_links(
        cls,
        resolved: Path,
        pools: dict[str, SequencePool],
        *,
        deep: bool = False,
        overwrite: bool = False,
    ) -> dict[str, str]:
        """Core link-resolution logic shared by :meth:`build` and
        :meth:`~tanat.trajectory.pool.TrajectoryPool.save`.

        For each pool:

        * **Dirty** (any pending view state) or *deep* → materialise into
          ``stores/<alias>/`` via :meth:`~tanat.sequence.base.pool.SequencePool.save`;
          link is relative.
        * **Clean** and not *deep* → link is the absolute path to the original
          store; nothing is written.

        Args:
            resolved: Root destination directory.
            pools: ``{alias: SequencePool}`` to resolve.
            deep: Force materialisation even for clean pools.
            overwrite: Passed through to :meth:`~tanat.sequence.base.pool.SequencePool.save`.
                When ``False`` (default) the pool will raise if
                ``stores/<alias>/`` already exists on disk.
        """
        stores_dir = resolved / TSCH.Files.DIR_STORES
        links: dict[str, str] = {}
        for alias, pool in pools.items():
            if pool.is_dirty or deep:
                if pool.is_dirty:
                    reasons = [
                        r
                        for r, cond in [
                            ("id_mask", pool._id_mask is not None),
                            ("entity_row_mask", pool._entity_row_mask is not None),
                            ("virtual features", pool._virtual_id is not None),
                            ("cast overrides", not pool._casts.is_empty()),
                            ("soft drops", pool._has_soft_drops),
                        ]
                        if cond
                    ]
                    warnings.warn(
                        f"Pending changes detected in pool '{alias}' ({', '.join(reasons)}). "
                        f"Materialising a copy into stores/{alias}/. "
                        "Call pool.save() before build() to avoid data duplication on disk.",
                        UserWarning,
                        stacklevel=2,
                    )

                dest = stores_dir / alias
                with DisplayIndentManager.nested():
                    pool.save(destination=dest, overwrite=overwrite)
                links[alias] = cls._to_relative(resolved, dest)
            else:
                links[alias] = cls._to_relative(resolved, pool._store.root_path)
        return links

    def _run(self, resolved: Path, links: dict[str, str]) -> dict[str, int]:
        """
        Execute the write pipeline and persist all trajectory store files.

        Files written:

        * ``trajectory_index.arrow``  -- traj_id presence-map across linked stores
        * ``core.json``               -- container type + counts + store links
        * ``metadata.json``           -- computed time index / feature metadata
        """
        LOGGER.info("Building trajectory store -> %s", resolved)

        seq_stores = {
            alias: SequenceStore(root_path=(resolved / rel_path).resolve())
            for alias, rel_path in links.items()
        }

        # Merge static first so _build_trajectory_index can extend the master ID set
        # with IDs that exist only in static sources.
        raw_static_lf = self._merge_static() if self._static_entries else None

        index_df = self._build_trajectory_index(seq_stores, raw_static_lf)
        index_df.write_ipc(resolved / TSCH.Files.TRAJECTORY_INDEX)
        n_trajectories = len(index_df)

        written_static_lf = None
        if raw_static_lf is not None:
            written_static_lf = self._write_static(resolved, index_df, raw_static_lf)

        TrajectoryStore.write_core_json(resolved, links, n_trajectories=n_trajectories)
        metadata = self._get_metadata(index_df, seq_stores, written_static_lf)
        TrajectoryStore.write_metadata_json(metadata, resolved)

        LOGGER.info(
            "Trajectory store built -> %s  [%d trajectories, %d pools]",
            resolved,
            n_trajectories,
            len(links),
        )

        return {"n_trajectories": n_trajectories}

    # ------------------------------------------------------------------
    # Build steps
    # ------------------------------------------------------------------

    def _traj_write_index(
        self,
        resolved: Path,
        traj_idx: pl.LazyFrame,
        static_lf: pl.LazyFrame | None,
    ) -> tuple[pl.DataFrame, int, pl.LazyFrame | None]:
        """Collect and persist ``trajectory_index.arrow``; align and write static features if present."""
        index_df = traj_idx.collect()
        index_df.write_ipc(resolved / TSCH.Files.TRAJECTORY_INDEX)
        n_trajectories = len(index_df)
        written_static_lf: pl.LazyFrame | None = None
        if static_lf is not None:
            # _write_static does a LEFT JOIN index → static to guarantee alignment.
            written_static_lf = self._write_static(resolved, index_df, static_lf)
        return index_df, n_trajectories, written_static_lf

    def _traj_write_metadata(
        self,
        resolved: Path,
        index_df: pl.DataFrame,
        links: dict[str, str],
        written_static_lf: pl.LazyFrame | None,
    ) -> None:
        """Write ``core.json`` and ``metadata.json`` for the trajectory store."""
        seq_stores = {
            alias: SequenceStore(root_path=(resolved / rel).resolve())
            for alias, rel in links.items()
        }
        TrajectoryStore.write_core_json(resolved, links, n_trajectories=len(index_df))
        metadata = self._get_metadata(index_df, seq_stores, written_static_lf)
        TrajectoryStore.write_metadata_json(metadata, resolved)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_relative(root_path: Path, path: Path) -> str:
        """Makes *path* relative to *root_path* if possible, absolute otherwise."""
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(root_path.resolve()))
        except ValueError:
            return str(resolved)

    @staticmethod
    def _build_trajectory_index(
        stores: dict[str, SequenceStore],
        static_lf: pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        """
        Build the presence-map from *stores*, extended with any IDs that
        appear only in *static_lf*.

        Trajectory IDs present only in static sources receive ``False`` for
        every pool presence column - symmetric with the sequence builder's
        ``length=0`` treatment for static-only sequences.
        """
        if not stores:
            if static_lf is not None:
                return (
                    static_lf.select(TSCH.TRAJ_ID).unique().sort(TSCH.TRAJ_ID).collect()
                )
            return pl.DataFrame({TSCH.TRAJ_ID: []})

        items = list(stores.items())
        alias0, store0 = items[0]
        result = (
            store0.sequence_index.select(pl.col(SCH.SEQ_ID).alias(TSCH.TRAJ_ID))
            .unique()
            .collect()
            .with_columns(pl.lit(True).alias(alias0))
        )

        for alias, store in items[1:]:
            ids = (
                store.sequence_index.select(pl.col(SCH.SEQ_ID).alias(TSCH.TRAJ_ID))
                .unique()
                .collect()
                .with_columns(pl.lit(True).alias(alias))
            )
            result = result.join(ids, on=TSCH.TRAJ_ID, how="full", coalesce=True)

        for col in result.columns:
            if col != TSCH.TRAJ_ID:
                result = result.with_columns(pl.col(col).fill_null(False))

        # Extend with IDs present only in static (full outer join, fill False)
        if static_lf is not None:
            static_ids = static_lf.select(TSCH.TRAJ_ID).unique().collect()
            result = result.join(static_ids, on=TSCH.TRAJ_ID, how="full", coalesce=True)
            for col in result.columns:
                if col != TSCH.TRAJ_ID:
                    result = result.with_columns(pl.col(col).fill_null(False))

        return result.sort(TSCH.TRAJ_ID)

    def _write_static(
        self,
        resolved: Path,
        index_df: pl.DataFrame,
        static_lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """
        Write ``static_features.arrow`` aligned to the trajectory index.

        Left-joins the master traj IDs from *index_df* against *static_lf*
        so the output has exactly one row per trajectory in canonical order.

        Returns the aligned features frame (without the ID column).
        """
        static_cols = [
            c for c in static_lf.collect_schema().names() if c != TSCH.TRAJ_ID
        ]
        static_features_lf = (
            index_df.lazy()
            .select(TSCH.TRAJ_ID)
            .join(
                static_lf.select([TSCH.TRAJ_ID] + static_cols),
                on=TSCH.TRAJ_ID,
                how="left",
            )
            .select(static_cols)
            .collect()  # materialise before sink_ipc: streaming engine does not
            .lazy()  # preserve JOIN row order without collect().lazy()
        )
        static_features_lf.sink_ipc(resolved / TSCH.Files.STATIC_FEATURES)
        return static_features_lf

    def _merge_static(self) -> pl.LazyFrame:
        """Rename id_column → ``TSCH.TRAJ_ID`` for each static source, then left-join."""
        frames = []
        for entry in self._static_entries:
            lf = entry["source"].read()
            if entry["id_column"] != TSCH.TRAJ_ID:
                lf = lf.rename({entry["id_column"]: TSCH.TRAJ_ID})
            frames.append(lf.select([TSCH.TRAJ_ID] + entry["features"]))
        base = frames[0]
        for other in frames[1:]:
            base = base.join(other, on=TSCH.TRAJ_ID, how="left")
        return base

    def _get_metadata(
        self,
        index_df,
        seq_stores: dict[str, SequenceStore],
        static_lf: pl.LazyFrame | None = None,
    ) -> TrajectoryMetadata:
        """Build :class:`TrajectoryMetadata` from *index_df*, *seq_stores*, and *static_lf*."""
        return TrajectoryMetadata(
            traj_id=index_df.schema[TSCH.TRAJ_ID],
            time_index=TrajectoryMetadata.infer_time_index(seq_stores),
            static_features=TrajectoryMetadata.infer_static(static_lf),
        )

    # ------------------------------------------------------------------
    # Source helpers
    # ------------------------------------------------------------------

    def _validate_static_source(
        self, source, *, id_column: str, features: list[str]
    ) -> None:
        """Validate that all declared columns exist in *source*."""
        required = [id_column] + list(features)
        available = set(source.schema().names())
        missing = [c for c in required if c not in available]
        if missing:
            raise ValueError(
                f"{source!r}: missing column(s) {missing}. "
                f"Available: {sorted(available)}"
            )

    def _stage_static(
        self, source, *, id_column: str, features: list[str]
    ) -> TrajectoryStoreBuilder:
        """Append the static source entry to the build queue."""
        self._static_entries.append(
            {"source": source, "id_column": id_column, "features": list(features)}
        )
        return self
