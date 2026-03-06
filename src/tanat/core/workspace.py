#!/usr/bin/env python3
"""
Workspace class.
"""

from pathlib import Path
import shutil

from . import registry as _registry
from ..store.factory import StoreFactory


class Workspace:
    """Workspace class to manage directory for data storage."""

    def __init__(self, root_path: str):
        self.root = Path(root_path).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def get_store_path(self, store_name: str) -> Path:
        """Returns the full path for a given store name."""
        return self.root / store_name

    def list_stores(self) -> list[str]:
        """Lists all store names in the workspace."""
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def clear(self):
        """Clears the workspace by removing all files and directories."""
        for path in self.root.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def clear_tmp(self):
        """
        Remove all virtual (tmp) contexts from every store.

        Use this to reclaim disk space after crashes or interrupted
        sessions that left orphan virtual contexts behind.

        .. warning::
            Only call when **no** Pool is actively using virtual features.
        """
        for store_name in self.list_stores():
            tmp_dir = self.get_store_path(store_name) / "tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def __getitem__(self, store_name: str):
        """Allows access to stores via workspace[store_name]."""
        store_path = self.get_store_path(store_name)
        manifest = StoreFactory.load_core(StoreFactory.get_core_path(store_path))
        container_type = manifest.get("container", "sequence")
        return _registry.build_pool(container_type, store_path)

    def __repr__(self):
        n_stores = len(self.list_stores())
        return (
            f"TanaT Workspace\n"
            f"∟ Root: {self.root}\n"
            f"∟ Content: {n_stores} stores detected"
        )
