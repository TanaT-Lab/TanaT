#!/usr/bin/env python3
"""simulate_trajectories: generate synthetic multi-sequence trajectory data."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .events import simulate_events
from .intervals import simulate_intervals
from .states import simulate_states

_SIMULATE_FUNC: dict[str, Callable] = {
    "event": simulate_events,
    "interval": simulate_intervals,
    "state": simulate_states,
}


def simulate_trajectories(
    sequences: dict[str, dict],
    *,
    shared_ids: bool = True,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic data for multiple sequence types at once.

    Convenience wrapper that calls ``simulate_events``,
    ``simulate_intervals``, or ``simulate_states`` for each entry
    in ``sequences``.

    Args:
        sequences: Mapping of ``{alias: config_dict}``. Each
            config_dict must contain a ``"type"`` key (``"event"``,
            ``"interval"``, or ``"state"``) and may contain any
            keyword accepted by the corresponding ``simulate_*``
            function, including ``features`` to name the entity-level
            columns.
        shared_ids: When True all generated sequences use the same
            ID space (1..n_ids). When False each sequence gets its
            own independent ID range.
        seed: Master seed. Per-sequence seeds are derived
            deterministically when individual configs omit ``seed``.

    Returns:
        Dict of ``{alias: DataFrame}`` matching the input keys,
        ready to be piped into ``build_*`` and then
        ``build_trajectories``.

        Use :func:`~tanat.dataset.simulate_static` separately to
        generate per-trajectory static data.

    Raises:
        ValueError: When ``shared_ids=True`` and ``n_ids`` values
            differ across sequence configs.
        ValueError: When an unknown sequence type is provided.

    Examples::

        data = simulate_trajectories(
            sequences={
                "admissions": {"type": "interval", "n_ids": 500},
                "procedures": {"type": "event", "n_ids": 500},
            },
            seed=42,
        )
    """
    if shared_ids:
        n_ids_values = {cfg.get("n_ids", 100) for cfg in sequences.values()}
        if len(n_ids_values) > 1:
            raise ValueError(
                f"shared_ids=True requires all sequences to have the same n_ids, "
                f"got: {n_ids_values}"
            )

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(sequences))

    result: dict[str, pd.DataFrame] = {}
    for (alias, config), child_seed in zip(sequences.items(), child_seeds):
        config = dict(config)
        seq_type = config.pop("type")
        if seq_type not in _SIMULATE_FUNC:
            raise ValueError(
                f"Unknown sequence type {seq_type!r}. "
                f"Must be one of: {list(_SIMULATE_FUNC)}"
            )
        if "seed" not in config:
            config["seed"] = int(child_seed.generate_state(1)[0])
        result[alias] = _SIMULATE_FUNC[seq_type](**config)

    return result
