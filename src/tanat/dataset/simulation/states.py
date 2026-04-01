#!/usr/bin/env python3
"""simulate_states: generate synthetic contiguous state sequence data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ._generate import (
    _prepare_sequence,
    _sample_state_times,
)


def simulate_states(
    *,
    n_ids: int = 100,
    seq_length_range: tuple[int, int] = (3, 10),
    duration_range: tuple[int, int] = (5, 45),
    features: int | list[str] = 2,
    time_range: tuple[datetime, datetime] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic contiguous state sequence data.

    Returns a ``pd.DataFrame`` with one row per state (entity).

    States are strictly contiguous: ``end[i] == start[i+1]`` within
    each ID. The ``end`` column of the last state per ID is set to
    the end of ``time_range``.

    The DataFrame contains columns: ``id`` (int64),
    ``start`` (datetime64[us]), ``end`` (datetime64[us]), plus one
    column per entity feature.
    The ``features`` argument controls the entity-level columns, i.e.
    the per-state measurements attached to each sequence row.

    Feature types are assigned by cycling through numeric, categorical
    and boolean in that order.

    Use :func:`~tanat.dataset.simulate_static` to generate a separate
    per-sequence static DataFrame when needed.

    Args:
        n_ids: Number of distinct sequence IDs to generate.
        seq_length_range: (min, max) number of states per ID.
        duration_range: (min, max) state duration in days.
        features: Number of entity feature columns to generate
            (auto-named ``f_0``, ``f_1``, ...) or explicit list of
            column names. These become the per-state (entity-level)
            measurements in the pool.
        time_range: (start, end) datetime bounds. Defaults to
            2000-01-01 to 2025-01-01.
        seed: Random seed for reproducibility.

    Returns:
        A ``pd.DataFrame`` with columns ``[id, start, end, <features>]``.

    Examples::

        df = simulate_states(n_ids=50, seed=42)
        df = simulate_states(n_ids=50, features=["score", "status"], seed=42)
    """
    rng, tr, id_col, lengths, feat_data = _prepare_sequence(
        n_ids,
        seq_length_range,
        features,
        time_range,
        seed,
    )
    starts, ends = _sample_state_times(lengths, duration_range, tr, rng)
    return pd.DataFrame({"id": id_col, "start": starts, "end": ends, **feat_data})
