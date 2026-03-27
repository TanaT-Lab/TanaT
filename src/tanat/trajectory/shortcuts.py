#!/usr/bin/env python3
"""
Quick-build helper for trajectory pools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd
import polars as pl

from ..store.base.utils import infer_features, validate_required_columns
from .pool import TrajectoryPool

if TYPE_CHECKING:
    from ..sequence.base.pool import SequencePool


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def build_trajectories(
    pools: dict[str, SequencePool],
    *,
    static_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame | None = None,
    id_column: str | None = None,
    store_name: str | None = None,
) -> TrajectoryPool:
    """Build a :class:`TrajectoryPool` from a dict of pre-built sequence pools.

    Args:
        pools: Mapping of ``{alias: SequencePool}``.  Each alias becomes the
            key used to access the sub-sequence inside a trajectory
            (e.g. ``traj["admissions"]``).
        static_data: Optional DataFrame or LazyFrame with per-trajectory
            static features.  When provided, ``id_column`` must also be set.
        id_column: Name of the id column in ``static_data``.  Required when
            ``static_data`` is not ``None``.  Ignored otherwise.
        store_name: Name for the on-disk store.  When ``None`` a unique name
            is generated automatically (``_quick_trajectory_<hex8>``).

    Returns:
        A ready-to-use :class:`TrajectoryPool`.

    Raises:
        ValueError: If ``static_data`` is provided without ``id_column``, or
            if ``id_column`` is absent from ``static_data``.

    Examples::

        tpool = build_trajectories(
            pools={"admissions": adm_pool, "procedures": proc_pool},
        )
        tpool[tpool.unique_ids[0]]["admissions"].temporal_data(output_format="polars")
    """
    if static_data is not None and id_column is None:
        raise ValueError(
            "``id_column`` must be provided when ``static_data`` is not None."
        )

    store_name = store_name or f"_quick_trajectory_{uuid4().hex[:8]}"

    builder = TrajectoryPool.builder()
    for alias, pool in pools.items():
        builder = builder.add(alias, pool)

    if static_data is not None:
        validate_required_columns(static_data, required={id_column})
        builder = builder.add_dataframe(
            static_data,
            id_column=id_column,
            features=infer_features(static_data, exclude={id_column}),
        )

    store_path = builder.build(store_name, exist_ok=True)
    if id_column is None:
        id_column = "id"  # Fallback default
    return TrajectoryPool(store=store_path, id_column=id_column)
