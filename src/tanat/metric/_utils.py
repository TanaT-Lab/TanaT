#!/usr/bin/env python3
"""
Shared utilities for Sequence and Trajectory metric ABCs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .matrix import DistanceMatrix
from ._storage import StorageOptions

if TYPE_CHECKING:
    from typing import Callable


def resolve_storage(
    class_name: str,
    memmap_support: bool,
    storage: StorageOptions | dict | None = None,
    *,
    store_path: str | Path | None = None,
    chunk_size: int = 500,
    resume: bool = True,
    dtype: str = "float32",
) -> StorageOptions | None:
    """Normalise *storage* and enforce memmap support guard.

    Args:
        class_name:    Name of the metric class (for the warning message).
        memmap_support: Whether the metric class supports disk-backed computation.
        storage:       Existing StorageOptions, plain dict, or None.
        store_path:    When provided, overrides *storage* entirely.
        chunk_size:    Forwarded to StorageOptions.
        resume:        Forwarded to StorageOptions.
        dtype:         Forwarded to StorageOptions.

    Returns:
        Resolved StorageOptions, or None (in-memory fallback).
    """
    if store_path is not None:
        storage = StorageOptions(
            store_path=store_path,
            chunk_size=chunk_size,
            resume=resume,
            dtype=dtype,
        )
    elif isinstance(storage, dict):
        storage = StorageOptions(**storage)

    if storage is None:
        return None

    if not memmap_support:
        warnings.warn(
            f"{class_name} does not support disk-backed computation "
            f"(MEMMAP_SUPPORT=False). Falling back to in-memory computation.",
            UserWarning,
            stacklevel=4,
        )
        return None

    return storage


def default_pairwise_matrix(
    items: dict,
    ids: list,
    compute_fn: Callable,
    progress_bar_fn: Callable,
) -> DistanceMatrix:
    """Default O(n^2) double-loop pairwise distance matrix.

    Args:
        items:           ``{id: item}`` mapping (sequences or trajectories).
        ids:             Ordered list of identifiers.
        compute_fn:      ``(item_a, item_b) -> float`` distance function.
        progress_bar_fn: Callable returning a context-managed progress bar
                         (e.g. ``self._create_progress_bar``).

    Returns:
        In-memory DistanceMatrix.
    """
    n = len(ids)
    result = np.zeros((n, n), dtype=np.float32)

    with progress_bar_fn(total=n * (n - 1), desc="Pairs") as pbar:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                result[i, j] = float(compute_fn(items[ids[i]], items[ids[j]]))
                pbar.update(1)

    return DistanceMatrix(result, ids)


def validate_type(value: object, expected_type: type, param_name: str) -> None:
    """Raise TypeError if *value* is not an instance of *expected_type*."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{param_name} must be a {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def validate_pair(
    a: object,
    b: object,
    expected_type: type,
    name_a: str,
    name_b: str,
) -> None:
    """Type-check both arguments of a pair."""
    validate_type(a, expected_type, name_a)
    validate_type(b, expected_type, name_b)
