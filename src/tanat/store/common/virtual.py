#!/usr/bin/env python3
"""
VirtualStore: manages temporary feature engineering in ./tmp/<virtual_id>/.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import pandas as pd
import polars as pl

from ..sequence.schema import StoreSchema as SCH
from .utils import (
    atomic_write,
    normalise_to_lazyframe,
    scan_if_exists,
    drop_columns_from_file,
)

LOGGER = logging.getLogger(__name__)


class VirtualStore:
    """
    Manages temporary (virtual) feature storage under ``<root>/tmp/<virtual_id>/``.

    The VirtualStore is used by stores (SequenceStore, TrajectoryStore)
    to handle ephemeral feature engineering contexts:
    - Creating / clearing virtual contexts
    - Temporal overrides (sequence store only, e.g for temporal conversions)
    - Reading / writing virtual feature files
    - Listing virtual feature names
    """

    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def _tmp_root(self) -> Path:
        """Absolute path to the ``./tmp/`` directory."""
        return self._root_path / "tmp"

    def clear(self) -> None:
        """Removes the entire ``./tmp/`` directory tree."""
        if self._tmp_root.exists():
            shutil.rmtree(self._tmp_root)

    def create(self, virtual_id: str) -> None:
        """
        Initialise a virtual context directory for *virtual_id*.

        Raises nothing if the directory already exists.
        """
        vdir = self._tmp_root / virtual_id
        vdir.mkdir(parents=True, exist_ok=True)

    def exists(self, virtual_id: str) -> bool:
        """Returns ``True`` if the virtual context directory exists."""
        return (self._tmp_root / virtual_id).is_dir()

    def clear_context(self, virtual_id: str) -> None:
        """Removes a single virtual context directory."""
        vdir = self._tmp_root / virtual_id
        if vdir.exists():
            shutil.rmtree(vdir)

    def fork_context(self, source_virtual_id: str) -> str | None:
        """Copy *source_virtual_id* into a fresh context; return ``None`` if nothing to inherit.

        Returns ``None`` when the source directory is absent or empty,
        preserving the invariant that ``_virtual_id is None`` means
        "no virtual content".  The caller is responsible for the
        ``source_virtual_id is None`` guard (see
        :meth:`~tanat.store.sequence.store.SequenceStore.fork_virtual_context`).
        Callers that unconditionally need a real UUID (e.g. temporal-conversion
        helpers that are about to write) should chain with :meth:`new_context`::

            uuid = self.fork_virtual_context(virtual_id) or self._virtual.new_context()

        Args:
            source_virtual_id: Existing non-null context UUID to inherit from.

        Returns:
            The new virtual context identifier (a UUID string), or ``None``.
        """
        src = self._tmp_root / source_virtual_id
        if not src.is_dir() or not any(src.iterdir()):
            return None
        new_uuid = str(uuid.uuid4())
        shutil.copytree(src, self._tmp_root / new_uuid)
        return new_uuid

    def new_context(self) -> str:
        """Create a fresh empty virtual context directory and return its UUID.

        Use this when you need a real UUID to write into and there is no
        source context to inherit from (or :meth:`fork_context` returned
        ``None``).

        Returns:
            A new UUID string; the corresponding ``tmp/<uuid>/`` directory
            is guaranteed to exist.
        """
        new_uuid = str(uuid.uuid4())
        self.create(new_uuid)
        return new_uuid

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def _virtual_path(self, virtual_id: str, is_static: bool) -> Path:
        """Arrow file path for a virtual context."""
        fname = SCH.Files.STATIC_FEATURES if is_static else SCH.Files.ENTITY_FEATURES
        return self._tmp_root / virtual_id / fname

    def features(self, virtual_id: str, is_static: bool = False) -> pl.LazyFrame | None:
        """Scans the virtual feature file; returns ``None`` when absent."""
        return scan_if_exists(self._virtual_path(virtual_id, is_static))

    def temporal(self, virtual_id: str) -> pl.LazyFrame | None:
        """Scan the virtual temporal override for *virtual_id*.

        Returns ``None`` when no ``temporal_index.arrow`` has been written
        for this context.

        Args:
            virtual_id: Virtual context identifier.

        Returns:
            A :class:`polars.LazyFrame` of the override temporal rows, or
            ``None`` if absent.
        """
        path = self._tmp_root / virtual_id / SCH.Files.TEMPORAL_INDEX
        return scan_if_exists(path)

    def write_temporal(self, virtual_id: str, temporal_lf: pl.LazyFrame) -> None:
        """Write a temporal override into the virtual context.

        Creates the context directory if it does not already exist, then
        writes *temporal_lf* as ``temporal_index.arrow`` inside
        ``tmp/<virtual_id>/``.

        Args:
            virtual_id: Virtual context identifier.
            temporal_lf: LazyFrame containing the new temporal columns.
        """
        self.create(virtual_id)
        path = self._tmp_root / virtual_id / SCH.Files.TEMPORAL_INDEX
        atomic_write(temporal_lf, path)

    # ------------------------------------------------------------------
    # Mutations (add/drop)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_input(
        df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    ) -> tuple[int, pl.LazyFrame]:
        """Normalise *df* to a LazyFrame and return ``(input_height, lf)``.

        The height is materialised before conversion so that pandas and Polars
        eager frames do not trigger an extra ``collect()`` on the resulting LazyFrame.
        """
        if isinstance(df, pd.DataFrame):
            return len(df), normalise_to_lazyframe(df)
        if isinstance(df, pl.DataFrame):
            return df.height, normalise_to_lazyframe(df)
        lf = df  # already a LazyFrame
        height = lf.select(pl.len()).collect().item()
        return height, lf

    def _merge_and_write(
        self,
        virtual_id: str,
        lf: pl.LazyFrame,
        is_static: bool,
    ) -> list[str]:
        """Merge *lf* with any existing virtual file, then write atomically.

        Existing columns with the same name are always replaced by incoming columns.

        Uses ``select(cols_to_keep)`` on the existing frame rather than
        ``drop()`` so that Polars projection-pushdown handles the IPC scan
        cleanly without risk of duplicate columns in the resolved plan.

        Returns the list of column names present in *lf* (all written names).
        """
        full_path = self._virtual_path(virtual_id, is_static)
        existing_lf = scan_if_exists(full_path)

        new_col_names = lf.collect_schema().names()

        if existing_lf is not None:
            new_col_set = set(new_col_names)
            cols_to_keep = [
                c for c in existing_lf.collect_schema().names() if c not in new_col_set
            ]
            if cols_to_keep:
                final_lf = pl.concat(
                    [existing_lf.select(cols_to_keep), lf], how="horizontal"
                )
            else:
                final_lf = lf
        else:
            final_lf = lf

        atomic_write(final_lf, full_path)
        return new_col_names

    def add_entity_features(
        self,
        virtual_id: str,
        df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        *,
        expected_height: int,
    ) -> list[str]:
        """Add positional entity features into the virtual context.

        Validates that the input row count matches *expected_height* before
        writing.  The virtual context directory is created if absent.
        Existing columns with the same name are always replaced.

        Args:
            virtual_id: Virtual context identifier.
            df: Feature-only DataFrame (no ID column), positionally aligned
                with the entity rows in the store.
            expected_height: Expected row count (validated against the
                temporal index length).

        Returns:
            The list of column names written.

        Raises:
            ValueError: If ``df`` height does not match *expected_height*.
        """
        input_height, lf = self._normalise_input(df)
        self.create(virtual_id)

        if input_height != expected_height:
            raise ValueError(
                f"Dimension Mismatch: Input has {input_height} rows, "
                f"but temporal index has {expected_height}."
            )

        return self._merge_and_write(virtual_id, lf, is_static=False)

    def add_static_features(
        self,
        virtual_id: str,
        df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    ) -> list[str]:
        """Add pre-aligned static features into the virtual context.

        No height validation is performed: the caller (``StaticStoreMixin``)
        is responsible for producing a correctly shaped frame via a LEFT JOIN
        against the main index.  The virtual context directory is created if
        absent.  Existing columns with the same name are always replaced.

        Args:
            virtual_id: Virtual context identifier.
            df: Feature-only DataFrame already aligned to the main index
                (one row per ID, nulls for absent IDs).

        Returns:
            The list of column names written.
        """
        lf = normalise_to_lazyframe(df)
        self.create(virtual_id)
        return self._merge_and_write(virtual_id, lf, is_static=True)

    def drop_features(
        self,
        virtual_id: str,
        features: list[str],
        is_static: bool = False,
    ) -> None:
        """
        Physically removes feature columns from a virtual context.

        Columns not present in the virtual file are silently ignored.

        Args:
            virtual_id: The virtual context identifier.
            features: Column names to remove.
            is_static: Static or entity features.
        """
        drop_columns_from_file(
            self._virtual_path(virtual_id, is_static),
            features,
        )
