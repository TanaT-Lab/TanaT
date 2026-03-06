#!/usr/bin/env python3
"""Store factory class."""

from __future__ import annotations

import json
from pathlib import Path

from ..store.sequence.store import SequenceStore
from ..store.trajectory.store import TrajectoryStore


class StoreFactory:
    """Factory for creating stores based on manifest definitions."""

    CORE_FILENAME = "core.json"

    @staticmethod
    def get_core_path(store_path: Path) -> Path:
        """Returns the full path to the core file for a given store."""
        return store_path / StoreFactory.CORE_FILENAME

    @staticmethod
    def load_core(core_path: Path) -> dict:
        """Loads the core JSON file from the given path."""
        if not core_path.exists():
            raise FileNotFoundError(f"Core file not found at {core_path}")

        with open(core_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def from_path(cls, store_path: Path):
        """
        Load a store instance based on the core file.

        Args:
            store_path: Path to the store directory containing the core file.

        Returns:
            An instance of the specified store.
        """
        core_path = cls.get_core_path(store_path)
        core = cls.load_core(core_path)
        container_type = core.get("container")

        if container_type == "sequence":
            return SequenceStore(root_path=store_path)

        if container_type == "trajectory":
            return TrajectoryStore(root_path=store_path)

        raise ValueError(f"Unsupported container type: {container_type}")
