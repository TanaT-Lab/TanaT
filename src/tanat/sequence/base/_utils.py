#!/usr/bin/env python3
"""
Internal helpers.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ...core.path import resolve_path
from ...store.sequence.store import SequenceStore


def resolve_store(store: str | Path | SequenceStore) -> SequenceStore:
    """Resolve a store argument to a :class:`SequenceStore` instance.

    Accepts a resolved :class:`SequenceStore`, a path-like, or a store name
    string.  Used by both :class:`SequenceViewMixin` and :class:`Entity`.
    """
    if isinstance(store, SequenceStore):
        return store
    if isinstance(store, (str, Path)):
        return SequenceStore(root_path=resolve_path(store))
    raise TypeError(
        f"'store' must be a store name, Path, or SequenceStore instance, "
        f"got {type(store).__name__}"
    )


def resolve_ids_to_add(
    self_ids: set,
    other_ids_list: list,
    on_duplicate: str,
    *,
    entity_label: str = "IDs",
) -> list:
    """Return the subset of *other_ids_list* not already in *self_ids*.

    Raises ``ValueError`` when ``on_duplicate="raise"`` and overlapping IDs
    are found.  Returns an empty list when all IDs are already present.
    """
    duplicates = [uid for uid in other_ids_list if uid in self_ids]
    if duplicates and on_duplicate == "raise":
        raise ValueError(
            f"Duplicate {entity_label} found in 'other': {duplicates}. "
            "Use on_duplicate='skip' to ignore them."
        )
    return (
        [uid for uid in other_ids_list if uid not in self_ids]
        if duplicates
        else other_ids_list
    )


def merge_optional_frames(
    lf_a: pl.LazyFrame | None,
    lf_b: pl.LazyFrame | None,
) -> pl.LazyFrame | None:
    """Concatenate two optional LazyFrames with ``diagonal_relaxed``.

    Returns ``None`` when both are ``None``.  Returns the non-``None`` frame
    directly when only one is present.
    """
    if lf_a is None and lf_b is None:
        return None
    frames = [f for f in [lf_a, lf_b] if f is not None]
    return pl.concat(frames, how="diagonal_relaxed") if len(frames) == 2 else frames[0]
