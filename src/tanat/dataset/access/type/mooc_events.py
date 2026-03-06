#!/usr/bin/env python3
"""
MOOC-events zenodo accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..zenodo import ZenodoAccessor

if TYPE_CHECKING:
    from pathlib import Path


class MOOCEventsZenodoAccessor(ZenodoAccessor, register_name="mooc_events"):
    """
    MOOCEvents zenodo accessor.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        ZenodoAccessor.__init__(
            self,
            record_id=16421055,
            filename="mooc_events.csv",
            cache_dir=cache_dir,
        )

    def _access_impl(self) -> pd.DataFrame:
        """
        MOOCEvent access implementation.

        Returns:
            pd.DataFrame: MOOC events dataset with ``timecreated`` parsed as datetime.
        """
        return pd.read_csv(self.local_path, parse_dates=["timecreated"])
