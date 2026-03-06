#!/usr/bin/env python3
"""
Mvad zenodo accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..zenodo import ZenodoAccessor

if TYPE_CHECKING:
    from pathlib import Path


class MvadZenodoAccessor(ZenodoAccessor, register_name="mvad"):
    """
    Mvad zenodo accessor.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        ZenodoAccessor.__init__(
            self,
            record_id=16367106,
            filename="data_mvad.csv",
            cache_dir=cache_dir,
        )

    def _access_impl(self) -> pd.DataFrame:
        """
        Mvad access implementation.

        Returns:
            pd.DataFrame: MVAD dataset with ``start`` and ``end`` parsed as datetimes.
        """
        return pd.read_csv(self.local_path, parse_dates=["start", "end"])
