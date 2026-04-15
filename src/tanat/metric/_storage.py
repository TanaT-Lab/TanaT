#!/usr/bin/env python3
"""
Disk storage helpers for DistanceMatrix computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tanat_utils import settings_dataclass as dataclass

from ..core.path import resolve_path

# ---------------------------------------------------------------------------
# StorageOptions
# ---------------------------------------------------------------------------


@dataclass
class StorageOptions:
    """Disk-backed storage options for distance matrix computation.

    Args:
        store_path: Directory where the matrix file and metadata are stored
            (required). Accepts the same formats as ``resolve_path``:

            - a plain name (e.g. ``"distances"``) - resolved via workspace store,
            - a relative or absolute path (e.g. ``"./distances"``, ``Path(...)``).

        chunk_size: Number of matrix rows computed per chunk before flushing
            to disk. Larger = fewer I/O ops, smaller = finer resume granularity.
            Default: 500.
        resume: If ``True`` (default), skip chunks already computed.
            If ``False``, delete and recompute from scratch.
        dtype: Numpy dtype string for the matrix. Default: ``"float32"``.
    """

    store_path: str | Path
    chunk_size: int = 500
    resume: bool = True
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_metric_config(metric) -> dict:
    """Build the config dict used for cache invalidation.

    Args:
        metric: SettingsMixin instance.

    Returns:
        JSON-serialisable dict from ``metric.to_config()``.
    """
    return metric.to_config()


def save_matrix_metadata(
    storage: StorageOptions,
    ids: list,
    metric_config: dict,
) -> None:
    """Write metadata.json to the storage directory.

    Args:
        storage: Storage configuration.
        ids: Identifiers (order matters).
        metric_config: ``to_config()`` dict (storage excluded).
    """
    path = resolve_path(storage.store_path)
    n = len(ids)
    metadata = {
        "shape": [n, n],
        "dtype": storage.dtype,
        "ids": list(ids),
        "metric_config": metric_config,
        "chunk_size": storage.chunk_size,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def save_progress(
    storage: StorageOptions,
    completed_chunks: int,
    status: str = "complete",
) -> None:
    """Write progress.json to the storage directory.

    Args:
        storage: Storage configuration.
        completed_chunks: Number of chunks already flushed.
        status: ``"computing"`` or ``"complete"``.
    """
    path = resolve_path(storage.store_path)
    progress = {
        "completed_chunks": completed_chunks,
        "status": status,
    }
    (path / "progress.json").write_text(json.dumps(progress))


def _wipe_matrix(path: Path) -> None:
    """Remove all matrix files from the storage directory (leaves dir intact)."""
    for filename in ("matrix.dat", "metadata.json", "progress.json"):
        f = path / filename
        if f.exists():
            f.unlink()


def _parse_and_validate_metadata(
    path: Path,
    n: int,
    ids: list,
    metric_config: dict,
) -> tuple[dict, dict] | tuple[None, None]:
    """Read and cross-validate the three storage files.

    Checks that all three files exist, that ``metadata.json`` matches the
    expected ``n``, ``ids`` and ``metric_config``, and that both JSON files
    are parseable.

    Args:
        path: Resolved storage directory.
        n: Expected matrix dimension.
        ids: Expected identifiers (order-sensitive).
        metric_config: Current metric ``to_config()`` dict.

    Returns:
        ``(meta, progress)`` dicts when everything is consistent,
        ``(None, None)`` on any mismatch or I/O error.
    """
    metadata_path = path / "metadata.json"
    progress_path = path / "progress.json"
    matrix_path = path / "matrix.dat"

    if not (metadata_path.exists() and progress_path.exists() and matrix_path.exists()):
        return None, None

    try:
        meta = json.loads(metadata_path.read_text())
        progress = json.loads(progress_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None

    if meta.get("ids") != list(ids):
        return None, None
    if meta.get("shape") != [n, n]:
        return None, None
    if meta.get("metric_config") != metric_config:
        return None, None

    return meta, progress


def open_or_create_matrix(
    storage: StorageOptions,
    n: int,
    ids: list,
    metric_config: dict,
) -> tuple[np.memmap, bool, int, bool]:
    """Open an existing memmap or create a new one.

    Single entry-point for all disk-backed matrix operations.  Returns
    everything callers need to either short-circuit (already complete) or
    continue the chunk loop (fresh or resuming).

    If a compatible matrix already exists on disk and ``resume=True``,
    it is reopened. Otherwise, a fresh NaN-filled memmap is created.

    Args:
        storage: Storage configuration.
        n: Matrix dimension (n x n).
        ids: Sequence/trajectory identifiers.
        metric_config: ``to_config()`` dict of the metric. Used to detect
            parameter changes between runs.

    Returns:
        ``(memmap, is_resuming, completed_chunks, is_complete)``:

        - *memmap*: float32 (n x n) memory-mapped array.
        - *is_resuming*: ``True`` if an existing matrix was reopened.
        - *completed_chunks*: number of chunks already flushed (0 if fresh).
        - *is_complete*: ``True`` when the matrix is fully computed and
          callers should return it immediately without further work.
    """
    path = resolve_path(storage.store_path)
    path.mkdir(parents=True, exist_ok=True)

    matrix_path = path / "matrix.dat"

    _, progress = _parse_and_validate_metadata(path, n, ids, metric_config)
    # Wipe when: validation failed (progress is None), or resume explicitly disabled
    should_create_fresh = progress is None or not storage.resume

    if should_create_fresh:
        _wipe_matrix(path)
        mm = np.memmap(matrix_path, dtype=storage.dtype, mode="w+", shape=(n, n))
        mm[:] = np.nan
        mm.flush()
        save_matrix_metadata(storage, ids, metric_config)
        save_progress(storage, completed_chunks=0, status="computing")
        return mm, False, 0, False

    # --- Existing matrix: resume or already complete ---
    mm = np.memmap(matrix_path, dtype=storage.dtype, mode="r+", shape=(n, n))
    completed_chunks = int(progress.get("completed_chunks", 0))
    is_complete = progress.get("status") == "complete"
    return mm, True, completed_chunks, is_complete
