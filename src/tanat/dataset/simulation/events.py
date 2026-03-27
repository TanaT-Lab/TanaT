#!/usr/bin/env python3
"""simulate_events: generate synthetic event sequence data."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ._generate import (
    _DEFAULT_TIME_RANGE,
    _assign_feature_types,
    _build_static,
    _generate_features,
    _maybe_with_static,
    _resolve_feature_names,
    _sample_event_times,
    _sample_lengths,
)


def simulate_events(
    *,
    n_ids: int = 100,
    seq_length_range: tuple[int, int] = (3, 10),
    entity_features: int | list[str] = 2,
    static_features: int | list[str] | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    seed: int | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic event sequence data (temporal + static).

    Returns a single temporal DataFrame when ``static_features`` is None,
    or a ``(temporal_df, static_df)`` tuple when static features are
    requested.

    The temporal DataFrame contains columns: ``id`` (int64),
    ``time`` (datetime64[us]), plus one column per entity feature.

    Args:
        n_ids: Number of distinct sequence IDs to generate.
        seq_length_range: (min, max) number of events per ID (inclusive).
        entity_features: Number of entity feature columns to generate
            (auto-named ``f_0``, ``f_1``, ...) or explicit list of
            column names. Types are assigned automatically by cycling
            through numeric, categorical, boolean.
        static_features: Number of static feature columns (auto-named
            ``s_0``, ``s_1``, ...) or explicit list of column names.
            When None no static data is generated and the function
            returns a single DataFrame instead of a tuple.
        time_range: (start, end) datetime bounds for generated
            timestamps. Defaults to 2000-01-01 to 2025-01-01.
        seed: Random seed for reproducibility.

    Returns:
        A ``pd.DataFrame``, or a ``(temporal, static)`` tuple.

    Examples::

        df = simulate_events(n_ids=50, seed=42)
        temporal, static = simulate_events(
            n_ids=50,
            static_features=2,
            seed=42,
        )
    """
    rng = np.random.default_rng(seed)
    tr = time_range if time_range is not None else _DEFAULT_TIME_RANGE
    ids = np.arange(1, n_ids + 1, dtype=np.int64)
    lengths = _sample_lengths(n_ids, seq_length_range, rng)
    id_col = np.repeat(ids, lengths)
    times = _sample_event_times(lengths, tr, rng)
    feat_names = _resolve_feature_names(entity_features, prefix="f")
    typed = _assign_feature_types(feat_names)
    total = int(lengths.sum())
    feat_data = _generate_features(total, typed, rng)
    temporal = pd.DataFrame({"id": id_col, "time": times, **feat_data})
    static = (
        _build_static(ids, static_features, rng)
        if static_features is not None
        else None
    )
    return _maybe_with_static(temporal, static)
