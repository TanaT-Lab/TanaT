#!/usr/bin/env python3
"""simulate_intervals: generate synthetic interval sequence data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ._generate import (
    _prepare_sequence,
    _sample_interval_times,
)


def simulate_intervals(
    *,
    n_ids: int = 100,
    seq_length_range: tuple[int, int] = (3, 10),
    duration_range: tuple[int, int] = (1, 30),
    allow_overlaps: bool = True,
    features: int | list[str] = 2,
    time_range: tuple[datetime, datetime] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic interval sequence data.

    Returns a ``pd.DataFrame`` with one row per interval (entity).

    The DataFrame contains columns: ``id`` (int64),
    ``start`` (datetime64[us]), ``end`` (datetime64[us]), plus one
    column per entity feature.
    The ``features`` argument controls the entity-level columns, i.e.
    the per-interval measurements attached to each sequence row.

    Feature types are assigned by cycling through numeric, categorical
    and boolean in that order.

    Use :func:`~tanat.dataset.simulate_static` to generate a separate
    per-sequence static DataFrame when needed.

    Args:
        n_ids: Number of distinct sequence IDs to generate.
        seq_length_range: (min, max) number of intervals per ID.
        duration_range: (min, max) interval duration in days.
        allow_overlaps: When True intervals within an ID may overlap.
            When False each interval starts after the previous ends.
        features: Number of entity feature columns to generate
            (auto-named ``f_0``, ``f_1``, ...) or explicit list of
            column names. These become the per-interval (entity-level)
            measurements in the pool.
        time_range: (start, end) datetime bounds. Defaults to
            2000-01-01 to 2025-01-01.
        seed: Random seed for reproducibility.

    Returns:
        A ``pd.DataFrame`` with columns ``[id, start, end, <features>]``.

    Examples::

        df = simulate_intervals(n_ids=200, allow_overlaps=False, seed=0)
        df = simulate_intervals(n_ids=50, features=["duration_days", "label"], seed=0)
    """
    rng, tr, id_col, lengths, feat_data = _prepare_sequence(
        n_ids,
        seq_length_range,
        features,
        time_range,
        seed,
    )
    starts, ends = _sample_interval_times(
        lengths, duration_range, allow_overlaps, tr, rng
    )
    return pd.DataFrame({"id": id_col, "start": starts, "end": ends, **feat_data})
