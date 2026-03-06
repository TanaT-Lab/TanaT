#!/usr/bin/env python3
"""
Mimic4 zenodo accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..zenodo import ZenodoAccessor

if TYPE_CHECKING:
    from pathlib import Path


class Mimic4ZenodoAccessor(ZenodoAccessor, register_name="mimic4"):
    """
    Mimic4 zenodo accessor.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        ZenodoAccessor.__init__(
            self,
            record_id=16368465,
            filename="mimic4.db",
            cache_dir=cache_dir,
        )

    def _access_impl(self) -> Path:
        """
        Mimic4 access implementation.

        Returns:
            Path: Local path to the cached SQLite database file.
        """
        return self.local_path
