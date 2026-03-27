#!/usr/bin/env python3
"""simulate_states: generate synthetic contiguous state sequence data."""

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
    _sample_lengths,
    _sample_state_times,
)


def simulate_states(
    *,
    n_ids: int = 100,
    seq_length_range: tuple[int, int] = (3, 10),
    duration_range: tuple[int, int] = (5, 45),
    entity_features: int | list[str] = 2,
    static_features: int | list[str] | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    seed: int | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic contiguous state sequence data (temporal + static).

    States are strictly contiguous: ``end[i] == start[i+1]`` within
    each ID. The ``end`` column of the last state per ID is set to
    the end of ``time_range``.

    The temporal DataFrame contains columns: ``id`` (int64),
    ``start`` (datetime64[us]), ``end`` (datetime64[us]), plus one
    column per entity feature.

    Args:
        n_ids: Number of distinct sequence IDs to generate.
        seq_length_range: (min, max) number of states per ID.
        duration_range: (min, max) state duration in days.
        entity_features: Number of entity feature columns (auto-named
            ``f_0``, ``f_1``, ...) or explicit list of column names.
            Types cycle through numeric, categorical, boolean.
        static_features: Number of static feature columns or list of
            names. When None returns a single DataFrame.
        time_range: (start, end) datetime bounds. Defaults to
            2000-01-01 to 2025-01-01.
        seed: Random seed for reproducibility.

    Returns:
        A ``pd.DataFrame``, or a ``(temporal, static)`` tuple.

    Examples::

        temporal, static = simulate_states(
            n_ids=100,
            static_features=2,
            seed=42,
        )
    """
    rng = np.random.default_rng(seed)
    tr = time_range if time_range is not None else _DEFAULT_TIME_RANGE
    ids = np.arange(1, n_ids + 1, dtype=np.int64)
    lengths = _sample_lengths(n_ids, seq_length_range, rng)
    id_col = np.repeat(ids, lengths)
    starts, ends = _sample_state_times(lengths, duration_range, tr, rng)
    feat_names = _resolve_feature_names(entity_features, prefix="f")
    typed = _assign_feature_types(feat_names)
    total = int(lengths.sum())
    feat_data = _generate_features(total, typed, rng)
    temporal = pd.DataFrame({"id": id_col, "start": starts, "end": ends, **feat_data})
    static = (
        _build_static(ids, static_features, rng)
        if static_features is not None
        else None
    )
    return _maybe_with_static(temporal, static)
