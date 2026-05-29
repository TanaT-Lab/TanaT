#!/usr/bin/env python3
"""
BaseStore: abstract base class shared by SequenceStore and TrajectoryStore.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from .virtual import VirtualStore
from .static import StaticStoreMixin

LOGGER = logging.getLogger(__name__)


class BaseStore(ABC, StaticStoreMixin):
    """
    Abstract base class for all stores.

    Subclasses **must** define two class-level attributes:

    * ``_MAIN_INDEX_PROPERTY: str`` - name of the property that returns the
      main navigation index (e.g. ``"sequence_index"`` or ``"trajectory_index"``).
    * ``_MAIN_ID_PROPERTY: str`` - name of the property that returns the
      ID column name (e.g. ``"seq_id_col"`` or ``"traj_id_col"``).
    """

    _MAIN_INDEX_PROPERTY: str
    _MAIN_ID_PROPERTY: str

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, root_path: str | Path) -> None:
        self._root_path = Path(root_path)
        self._check_structure()
        self._virtual = VirtualStore(self._root_path)
        self._metadata_cache = None

    @property
    def root_path(self) -> Path:
        """Root directory of this store."""
        return self._root_path

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------

    @abstractmethod
    def _required_files(self) -> list[str]:
        """List of file names that must exist in the root directory."""

    def _check_structure(self) -> None:
        """Validates that the store directory contains the required files."""
        if not self._root_path.exists():
            raise FileNotFoundError(f"Store path not found: {self._root_path}")
        for fname in self._required_files():
            if not (self._root_path / fname).exists():
                raise FileNotFoundError(
                    f"Invalid Store: Missing required file '{fname}' in {self._root_path}"
                )

    # ------------------------------------------------------------------
    # Core / Metadata JSON I/O
    # ------------------------------------------------------------------

    @property
    def core(self) -> dict:
        """Returns the static store facts written once at build time."""
        with open(self._root_path / "core.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def write_metadata_json(
        metadata, path: Path, filename: str = "metadata.json"
    ) -> None:
        """Writes *metadata* to *path* / *filename*.

        Works with any metadata object that implements ``to_json_dict()``.
        """
        data = {
            "__NOTICE__": "auto-generated. DO NOT EDIT BY HAND",
            **metadata.to_json_dict(),
        }
        with open(path / filename, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

    # ------------------------------------------------------------------
    # Main index / ID helpers (dispatched via _MAIN_INDEX_PROPERTY)
    # ------------------------------------------------------------------

    @property
    def main_index(self) -> pl.LazyFrame:
        """Primary ID index (e.g. ``sequence_index`` or ``trajectory_index``)."""
        return getattr(self, self._MAIN_INDEX_PROPERTY)

    @property
    def main_id_col(self) -> str:
        """Name of the ID column in the main index."""
        return getattr(self, self._MAIN_ID_PROPERTY)

    # ------------------------------------------------------------------
    # Virtual-context lifecycle
    # ------------------------------------------------------------------

    def clear_virtual_context(self, virtual_id: str) -> None:
        """Removes a virtual context directory and all its feature files."""
        self._virtual.clear_context(virtual_id)

    def fork_virtual_context(self, source_virtual_id: str | None) -> str | None:
        """Fork *source_virtual_id* into a new context, or ``None`` if nothing to inherit.

        Returns ``None`` immediately when *source_virtual_id* is ``None``;
        otherwise delegates to :meth:`~tanat.store.base.virtual.VirtualStore.fork_context`.
        """
        if source_virtual_id is None:
            return None
        return self._virtual.fork_context(source_virtual_id)
