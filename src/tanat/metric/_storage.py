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


def open_or_create_matrix(
    storage: StorageOptions,
    n: int,
    ids: list,
    metric_config: dict,
) -> tuple[np.memmap, bool, int]:
    """Open an existing memmap or create a new one.

    If a compatible matrix already exists on disk and ``resume=True``,
    it is reopened. Otherwise, a fresh NaN-filled memmap is created.

    Args:
        storage: Storage configuration.
        n: Matrix dimension (n x n).
        ids: Sequence/trajectory identifiers.
        metric_config: ``to_config()`` dict of the metric. Used to detect
            parameter changes between runs.

    Returns:
        ``(memmap, is_resuming, completed_chunks)``:

        - *memmap*: float32 (n x n) memory-mapped array.
        - *is_resuming*: ``True`` if resuming from an existing matrix,
          ``False`` if freshly created.
        - *completed_chunks*: number of chunks already computed (0 if not
          resuming).

    Raises:
        ValueError: If existing matrix has incompatible shape or ids.
    """
    path = resolve_path(storage.store_path)
    path.mkdir(parents=True, exist_ok=True)

    metadata_path = path / "metadata.json"
    matrix_path = path / "matrix.dat"
    progress_path = path / "progress.json"

    should_create_fresh = True
    completed_chunks = 0

    if metadata_path.exists():
        try:
            existing_meta = json.loads(metadata_path.read_text())
            existing_ids = existing_meta.get("ids", [])
            existing_shape = existing_meta.get("shape", [])
            existing_config = existing_meta.get("metric_config", {})

            ids_match = existing_ids == list(ids)
            shape_match = existing_shape == [n, n]
            config_match = existing_config == metric_config

            if not storage.resume:
                # resume=False → always wipe and recompute
                should_create_fresh = True
            elif not ids_match or not shape_match:
                # Pool changed (different ids or different size) → wipe
                should_create_fresh = True
            elif not config_match:
                # Metric configuration changed → wipe
                should_create_fresh = True
            else:
                # All conditions satisfied → resume
                should_create_fresh = False

        except (json.JSONDecodeError, KeyError, OSError):
            should_create_fresh = True

    if should_create_fresh:
        _wipe_matrix(path)
        mm = np.memmap(matrix_path, dtype=storage.dtype, mode="w+", shape=(n, n))
        mm[:] = np.nan
        mm.flush()
        save_matrix_metadata(storage, ids, metric_config)
        save_progress(storage, completed_chunks=0, status="computing")
        return mm, False, 0

    # --- Resume path: open existing memmap ---
    mm = np.memmap(matrix_path, dtype=storage.dtype, mode="r+", shape=(n, n))
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
            completed_chunks = int(progress.get("completed_chunks", 0))
        except (json.JSONDecodeError, KeyError, OSError):
            completed_chunks = 0
    return mm, True, completed_chunks
