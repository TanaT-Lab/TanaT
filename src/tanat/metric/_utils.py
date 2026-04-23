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
from ._storage import StorageOptions, save_progress

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
    *,
    storage: StorageOptions | None = None,
    result: np.ndarray | None = None,
    is_resuming: bool = False,
    completed: int = 0,
) -> DistanceMatrix:
    """O(n²) double-loop pairwise distance matrix with optional storage.

    Computes the full ``(n, n)`` square matrix.  When *storage* is provided,
    flushes to disk in chunks and supports resuming from a partial run.

    Args:
        items:           ``{id: item}`` mapping (sequences or trajectories).
        ids:             Ordered list of identifiers.
        compute_fn:      ``(item_a, item_b) -> float`` distance function.
        progress_bar_fn: Callable returning a context-managed progress bar
                         (e.g. ``self._create_progress_bar``).
        storage:         Optional :class:`StorageOptions` for disk-backed runs.
        result:          Pre-opened memmap array or ``None`` for in-memory.
        is_resuming:     Whether partial chunks are already on disk.
        completed:       Number of chunks already flushed.

    Returns:
        :class:`~tanat.metric.DistanceMatrix` of shape ``(n, n)``.
    """
    n = len(ids)

    if result is None:
        result = np.full((n, n), np.nan, dtype=np.float32)

    chunk_size = storage.chunk_size if storage is not None else n
    chunks = list(range(0, n, chunk_size))

    with progress_bar_fn(total=n * n, desc="Pairs") as pbar:
        for chunk_idx, chunk_start in enumerate(chunks):
            chunk_end = min(chunk_start + chunk_size, n)
            if is_resuming and chunk_idx < completed:
                pbar.update((chunk_end - chunk_start) * n)
                continue
            for i in range(chunk_start, chunk_end):
                for j in range(n):
                    result[i, j] = float(compute_fn(items[ids[i]], items[ids[j]]))
                    pbar.update(1)
            if storage is not None:
                result.flush()
                completed += 1
                save_progress(storage, completed, status="computing")

    if storage is not None:
        result.flush()
        save_progress(storage, completed, status="complete")

    return DistanceMatrix(result, ids)


def default_cross_matrix(
    items_rows: dict,
    ids_rows: list,
    items_cols: dict,
    ids_cols: list,
    compute_fn: Callable,
) -> np.ndarray:
    """Default O(n×k) double-loop cross-pool distance matrix.

    Args:
        items_rows: ``{id: item}`` mapping for row items.
        ids_rows: Ordered list of row identifiers.
        items_cols: ``{id: item}`` mapping for column items.
        ids_cols: Ordered list of column identifiers.
        compute_fn: ``(item_a, item_b) -> float`` distance function.

    Returns:
        ``float32`` numpy array of shape ``(n, k)``.
    """
    n, k = len(ids_rows), len(ids_cols)
    result = np.empty((n, k), dtype=np.float32)

    for i, id_row in enumerate(ids_rows):
        for j, id_col in enumerate(ids_cols):
            result[i, j] = float(compute_fn(items_rows[id_row], items_cols[id_col]))

    return result


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
