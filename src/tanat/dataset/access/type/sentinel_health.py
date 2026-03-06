#!/usr/bin/env python3
"""
Sentinel Health Dataset zenodo accessor.

Synthetic dataset for pathogen emergence detection POC.
See: https://gitlab.inria.fr/tanat/datasets/sentinel-health-dataset
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..zenodo import ZenodoAccessor

if TYPE_CHECKING:
    from pathlib import Path


class SentinelHealthZenodoAccessor(ZenodoAccessor, register_name="sentinel_health"):
    """
    Sentinel Health Dataset accessor from Zenodo.

    A synthetic dataset of ~2000 medical reports spanning 12 months with 3 phases:
    - Baseline (Jan-Apr): Normal hospital activity, diverse pathologies
    - Pre-emergence (May-Aug): Subtle appearance of a new pathogen
    - Outbreak (Sep-Dec): Massive semantic convergence toward critical symptoms
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        ZenodoAccessor.__init__(
            self,
            record_id=18161141,
            filename="sentinel_health_dataset.csv",
            cache_dir=cache_dir,
        )

    def _access_impl(self) -> pd.DataFrame:
        """
        Sentinel Health Dataset access implementation.

        Returns:
            pd.DataFrame: Dataset with columns ``date``, ``patient_id``,
            ``report_text``, ``label``. ``date`` is parsed as datetime.
        """
        return pd.read_csv(self.local_path, parse_dates=["date"])
