#!/usr/bin/env python3
"""
Trajectory Store: persistent storage for trajectory data.

Layout::

    <root>/
    ├── core.json         ← source of truth: {alias: relative_path}
    ├── metadata.json         ← trajectory-level metadata (ID dtype + static features)
    ├── trajectory_index.arrow   ← _traj_id + bool presence columns
    ├── static_features.arrow    ← (optional) trajectory-level features
    ├── stores/                  ← materialised stores (from masked pools)
    └── tmp/                     ← virtual feature contexts
        └── <virtual_id>/
            └── static_features.arrow
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
import polars as pl

from ..common.virtual import VirtualStore
from ..common.static import StaticStoreMixin
from ..common.utils import (
    apply_casts,
    probe_cast,
)
from ...metadata.trajectory import TrajectoryMetadata
from ..sequence.store import SequenceStore
from .schema import TrajectorySchema as TSCH

if TYPE_CHECKING:
    from ...trajectory.cast import TrajectoryCastRecipe

LOGGER = logging.getLogger(__name__)


class TrajectoryStore(StaticStoreMixin):
    """
    Persistent storage for a :class:`TrajectoryPool`.

    Manages on disk:

    * ``store_links.json``: ``{alias: relative_path}`` (user-editable)
    * ``trajectory_index.arrow``: presence-map
    * ``static_features.arrow``: trajectory-level features
    """

    # All file-name constants live in TSCH.Files
    # FILE_STATIC_FEATURES inherited from StaticStoreMixin

    _MAIN_INDEX_PROPERTY: str = "trajectory_index"
    _MAIN_ID_PROPERTY: str = "traj_id_col"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, root_path: str | Path) -> None:
        self._root_path = Path(root_path)
        self._check_structure()
        self._virtual = VirtualStore(self._root_path)
        # Caches
        self._store_links: dict[str, str] | None = None
        self._seq_stores: dict[str, SequenceStore] | None = None
        self._metadata_cache: TrajectoryMetadata | None = None

    @property
    def root_path(self) -> Path:
        """Root directory of this store."""
        return self._root_path

    def _check_structure(self) -> None:
        """Validates that the store directory contains the required files."""
        if not self._root_path.exists():
            raise FileNotFoundError(f"Trajectory store not found: {self._root_path}")
        required = [TSCH.Files.CORE, TSCH.Files.TRAJECTORY_INDEX]
        for fname in required:
            if not (self._root_path / fname).exists():
                raise FileNotFoundError(
                    f"Invalid trajectory store: missing '{fname}' in {self._root_path}. "
                    "Use TrajectoryPool.builder().add(...).build(path) to create it first."
                )

    # ------------------------------------------------------------------
    # Store links (JSON)
    # ------------------------------------------------------------------

    @property
    def core(self) -> dict:
        """Contents of ``core.json`` (written once at build time)."""
        path = self._root_path / TSCH.Files.CORE
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @property
    def store_links(self) -> dict[str, str]:
        """``{alias: relative_path}`` from ``core.json``."""
        if self._store_links is None:
            self._store_links = self.core.get("store_links", {})
        return self._store_links

    @property
    def store_aliases(self) -> list[str]:
        """List of registered store aliases."""
        return list(self.store_links.keys())

    @property
    def sequence_stores(self) -> dict[str, SequenceStore]:
        """Linked :class:`SequenceStore` instances, keyed by alias."""
        if self._seq_stores is None:
            self._seq_stores = {}
            for alias, rel_path in self.store_links.items():
                abs_path = (self._root_path / rel_path).resolve()
                self._seq_stores[alias] = SequenceStore(root_path=abs_path)
        return self._seq_stores

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    @property
    def trajectory_index(self) -> pl.LazyFrame:
        """Navigation index (_traj_id + bool presence columns) - physical, no cast overlay."""
        path = self._root_path / TSCH.Files.TRAJECTORY_INDEX
        if path.exists():
            return pl.scan_ipc(path)
        return pl.DataFrame({TSCH.TRAJ_ID: []}).lazy()

    def get_id_lf(self, id_cast: pl.DataType | None = None, **_kw) -> pl.LazyFrame:
        """All trajectory IDs as a single-column lazy frame, optionally cast.

        Preserves the physical dtype — stays lazy until collected.
        """
        lf = self.trajectory_index.select(TSCH.TRAJ_ID)
        if id_cast is not None:
            lf = lf.with_columns(pl.col(TSCH.TRAJ_ID).cast(id_cast))
        return lf

    @property
    def traj_id_col(self) -> str:
        """Internal name of the trajectory ID column."""
        return TSCH.TRAJ_ID

    @property
    def traj_id_dtype(self) -> pl.DataType:
        """Physical dtype of the trajectory ID column (IPC header read, no data scan)."""
        return self.trajectory_index.collect_schema()[TSCH.TRAJ_ID]

    # ------------------------------------------------------------------
    # Cast probes (fast validation on a small sample before accepting a cast)
    # ------------------------------------------------------------------

    def probe_id_cast(self, dtype: pl.DataType, n_rows: int = 10) -> None:
        """
        Validates casting the trajectory-ID column to *dtype*.

        Args:
            dtype: Target Polars DataType.
            n_rows: Sample size (default: 10).
        """
        probe_cast(
            self.trajectory_index.select(TSCH.TRAJ_ID),
            {TSCH.TRAJ_ID: dtype},
            n_rows,
        )

    def probe_time_cast(self, dtype: pl.DataType, n_rows: int = 10) -> None:
        """
        Validates casting time index columns to *dtype* against the first
        linked sequence store.

        All sequence stores are guaranteed to share the same time index
        schema by the build-time compatibility check
        (:meth:`SequenceMetadata.assert_id_compatible_with` and :meth:`~SequenceMetadata.assert_time_index_compatible_with`), so probing
        one is sufficient.

        Args:
            dtype: Target Polars DataType.
            n_rows: Sample size (default: 10).

        Raises:
            RuntimeError: If no sequence stores are linked.
            TypeError: If the cast is incompatible with the time index data.
        """
        stores = self.sequence_stores
        if not stores:
            raise RuntimeError(
                "No sequence stores linked - cannot probe time index cast."
            )
        first_store = next(iter(stores.values()))
        first_store.probe_time_cast(dtype, n_rows)

    def clear_virtual_context(self, virtual_id: str) -> None:
        """Removes a virtual context directory and all its feature files."""
        self._virtual.clear_context(virtual_id)

    def fork_virtual_context(self, source_virtual_id: str | None) -> str | None:
        """Fork *source_virtual_id* into a new context, or ``None`` if nothing to inherit.

        Returns ``None`` immediately when *source_virtual_id* is ``None``;
        otherwise delegates to :meth:`~tanat.store.common.virtual.VirtualStore.fork_context`.
        """
        if source_virtual_id is None:
            return None
        return self._virtual.fork_context(source_virtual_id)

    def _filter_by_id(self, lf: pl.LazyFrame, id_value) -> pl.LazyFrame:
        """Filters a LazyFrame to rows belonging to *id_value* (physical type)."""
        return lf.filter(pl.col(TSCH.TRAJ_ID) == id_value)

    def _filter_by_ids(self, lf: pl.LazyFrame, ids: list | set | None) -> pl.LazyFrame:
        """Filters a LazyFrame to rows whose ID is in *ids* (physical type)."""
        if ids is None:
            return lf
        return lf.filter(pl.col(TSCH.TRAJ_ID).is_in(list(ids)))

    def get_frames_for_save(
        self,
        id_mask: set | None,
        cast_recipe: TrajectoryCastRecipe | None,
        virtual_id: str | None,
        features: list[str] | None = None,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
        """Prepare trajectory-level frames ready to be handed to the builder.

        Centralises all filtering and casting logic that was previously
        scattered inside :meth:`~tanat.trajectory.pool.TrajectoryPool.save`.

        Args:
            id_mask: Optional set of trajectory IDs to keep (``None`` = all).
            cast_recipe: :class:`~tanat.trajectory.cast.TrajectoryCastRecipe`
                whose ``id`` and ``static`` fields are applied.
            virtual_id: Virtual context UUID for merged static features
                (``None`` if no virtual features).
            features: Static feature names to materialise.  ``None`` keeps all
                available columns.  Pass ``settings.static_features`` to
                materialise soft drops (columns absent from the list are
                excluded from the written frame).

        Returns:
            ``(traj_idx, static_lf)`` where:

            * *traj_idx* is the trajectory-index LazyFrame with filters and
              ID cast applied.
            * *static_lf* includes the ``TRAJ_ID`` column (required by
              :meth:`~tanat.store.trajectory.builder.TrajectoryStoreBuilder.build_from_frames`
              for left-join alignment), or ``None`` when no feature columns
              exist after filtering.
        """
        has_casts = cast_recipe is not None and not cast_recipe.is_empty()

        # --- trajectory index ---
        traj_idx = self.trajectory_index
        if id_mask is not None:
            traj_idx = self._filter_by_ids(traj_idx, id_mask)
        if has_casts and cast_recipe.id is not None:
            traj_idx = apply_casts(traj_idx, {TSCH.TRAJ_ID: cast_recipe.id})

        # --- static features (physical + virtual, with TRAJ_ID) ---
        static_lf = self.get_static_data(virtual_id=virtual_id)
        if static_lf is not None:
            if id_mask is not None:
                static_lf = self._filter_by_ids(static_lf, id_mask)
            if has_casts and cast_recipe.id is not None:
                static_lf = apply_casts(static_lf, {TSCH.TRAJ_ID: cast_recipe.id})
            if has_casts and cast_recipe.static:
                static_lf = apply_casts(static_lf, cast_recipe.static)
            # Materialise soft drops: keep only requested feature columns.
            if features is not None:
                keep = [TSCH.TRAJ_ID] + [f for f in features if f != TSCH.TRAJ_ID]
                available = set(static_lf.collect_schema().names())
                static_lf = static_lf.select([c for c in keep if c in available])
            # Discard frame entirely when there are no feature columns
            feat_cols = [
                c for c in static_lf.collect_schema().names() if c != TSCH.TRAJ_ID
            ]
            if not feat_cols:
                static_lf = None

        return traj_idx, static_lf

    @staticmethod
    def write_core_json(
        path: Path, links: dict[str, str], *, n_trajectories: int | None = None
    ) -> None:
        """Writes ``core.json`` to *path* with the given *links*."""
        core = {
            "__NOTICE__": "auto-generated. DO NOT EDIT BY HAND",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "container": "trajectory",
            "n_trajectories": n_trajectories,
            "store_links": links,
        }
        with open(path / TSCH.Files.CORE, "w", encoding="utf-8") as fh:
            json.dump(core, fh, indent=4)

    @staticmethod
    def write_metadata_json(metadata: TrajectoryMetadata, path: Path) -> None:
        """Writes *metadata* to *path* as ``metadata.json``."""
        data = {
            "__NOTICE__": "auto-generated. DO NOT EDIT BY HAND",
            **metadata.to_json_dict(),
        }
        with open(path / TSCH.Files.METADATA, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

    def _write_core_snapshot(self, target: Path) -> None:
        """Writes ``core.json`` to *target*, refreshing ``created_at`` to now (UTC).

        Reads the physical ``core.json``, preserves all fields (store links,
        counts, etc.) and only updates the timestamp.  Used by
        :meth:`copy_to` so that copied stores carry a fresh creation date
        instead of the original's.
        """
        core = self.core
        core["created_at"] = datetime.now(timezone.utc).isoformat()
        with open(target / TSCH.Files.CORE, "w", encoding="utf-8") as fh:
            json.dump(core, fh, indent=4)

    def copy_to(self, target: Path, *, exist_ok: bool = False) -> None:
        """Copy all trajectory store files to *target*.

        Fast-path used by :meth:`~tanat.trajectory.pool.TrajectoryPool.save`
        when no transformation is needed (no virtual features, no masks, no casts).

        Copies only the physical store files - ``tmp/`` (virtual feature contexts)
        is intentionally skipped so the destination starts with a clean slate.
        ``stores/`` (materialised sub-stores) is copied when present so that
        relative store links in ``core.json`` remain valid at the new location.

        Args:
            target: Destination directory (must not exist, or ``exist_ok=True``).
            exist_ok: If ``True``, allows the target directory to already exist.
        """
        target.mkdir(parents=True, exist_ok=exist_ok)
        # Navigation index
        src = self._root_path / TSCH.Files.TRAJECTORY_INDEX
        if src.exists():
            shutil.copy2(src, target / TSCH.Files.TRAJECTORY_INDEX)
        # Optional trajectory-level static features
        src = self._root_path / TSCH.Files.STATIC_FEATURES
        if src.exists():
            shutil.copy2(src, target / TSCH.Files.STATIC_FEATURES)
        # Materialised sub-stores (relative links in core.json point here)
        stores_dir = self._root_path / TSCH.Files.DIR_STORES
        if stores_dir.exists():
            shutil.copytree(stores_dir, target / TSCH.Files.DIR_STORES)
        # Manifest - refreshed timestamp, relative links remain valid
        self._write_core_snapshot(target)
        # Metadata
        src = self._root_path / TSCH.Files.METADATA
        if src.exists():
            shutil.copy2(src, target / TSCH.Files.METADATA)
