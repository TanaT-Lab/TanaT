#!/usr/bin/env python3
"""simulate_static: generate synthetic per-sequence static data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._generate import _assign_feature_types, _generate_features, _resolve_feature_names


def simulate_static(
    *,
    n_ids: int = 100,
    features: int | list[str] = 2,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic static DataFrame with one row per sequence ID.

    This function is independent of any temporal simulation: it can be
    used alongside any ``simulate_events``, ``simulate_intervals`` or
    ``simulate_states`` call that shares the same ``n_ids``.  IDs are
    generated as consecutive integers ``1 … n_ids``.

    Feature types are assigned by cycling through numeric, categorical
    and boolean in that order.

    Args:
        n_ids: Number of distinct sequence IDs (and rows) to generate.
        features: Number of feature columns to generate (auto-named
            ``s_0``, ``s_1``, ...) or an explicit list of column names.
            Types cycle through numeric, categorical, boolean.
        seed: Random seed for reproducibility.

    Returns:
        A ``pd.DataFrame`` with columns ``id`` plus one column per
        feature.

    Examples::

        static_df = simulate_static(n_ids=50, features=["age", "group"], seed=0)
        static_df.head()
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n_ids + 1, dtype=np.int64)
    names = _resolve_feature_names(features, prefix="s")
    typed = _assign_feature_types(names)
    data: dict[str, np.ndarray] = {"id": ids}
    data.update(_generate_features(n_ids, typed, rng))
    return pd.DataFrame(data)
