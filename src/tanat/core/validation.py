#!/usr/bin/env python3
"""Lightweight type guards used across the library.

These helpers exist to break import cycles.

The check relies on a duck-typing marker (e.g. :attr:`Criterion.__criterion__`)
so this module has **zero** dependency on the validated types.
"""


def ensure_criterion(obj) -> None:
    """Raise :class:`TypeError` if *obj* is not a :class:`Criterion` instance.

    Uses the :attr:`Criterion.__criterion__` marker for an import-cycle-safe
    duck-typing check, so internal callers (pools, sequences, trajectories)
    do not need to import :class:`Criterion` at module load time.
    """
    if not getattr(obj, "__criterion__", False):
        raise TypeError(
            f"'criterion' must be a Criterion object, got {type(obj).__name__}."
        )
