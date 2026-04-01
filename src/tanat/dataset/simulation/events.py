#!/usr/bin/env python3
"""simulate_events: generate synthetic event sequence data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ._generate import (
    _prepare_sequence,
    _sample_event_times,
)


def simulate_events(
    *,
    n_ids: int = 100,
    seq_length_range: tuple[int, int] = (3, 10),
    features: int | list[str] = 2,
    time_range: tuple[datetime, datetime] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic event sequence data.

    Returns a ``pd.DataFrame`` with one row per event (entity).

    The DataFrame contains columns: ``id`` (int64),
    ``time`` (datetime64[us]), plus one column per entity feature.
    The ``features`` argument controls the entity-level columns, i.e.
    the per-event measurements attached to each sequence row.

    Feature types are assigned by cycling through numeric, categorical
    and boolean in that order.

    Use :func:`~tanat.dataset.simulate_static` to generate a separate
    per-sequence static DataFrame when needed.

    Args:
        n_ids: Number of distinct sequence IDs to generate.
        seq_length_range: (min, max) number of events per ID (inclusive).
        features: Number of entity feature columns to generate
            (auto-named ``f_0``, ``f_1``, ...) or explicit list of
            column names. These become the per-event (entity-level)
            measurements in the pool.
        time_range: (start, end) datetime bounds for generated
            timestamps. Defaults to 2000-01-01 to 2025-01-01.
        seed: Random seed for reproducibility.

    Returns:
        A ``pd.DataFrame`` with columns ``[id, time, <features>]``.

    Examples::

        df = simulate_events(n_ids=50, seed=42)
        df = simulate_events(n_ids=50, features=["value", "category"], seed=42)
    """
    rng, tr, id_col, lengths, feat_data = _prepare_sequence(
        n_ids,
        seq_length_range,
        features,
        time_range,
        seed,
    )
    times = _sample_event_times(lengths, tr, rng)
    return pd.DataFrame({"id": id_col, "time": times, **feat_data})
