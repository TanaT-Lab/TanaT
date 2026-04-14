#!/usr/bin/env python3
"""
DistanceMatrix: thin numpy wrapper with associated IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from ..core.path import resolve_path


class DistanceMatrix:
    """Pairwise distance matrix with associated IDs.

    Wrapper around a square ``numpy.ndarray`` associating each row/column
    with a sequence (or trajectory) identifier.

    Example::

        dm = DistanceMatrix(np.zeros((3, 3), dtype="float32"), ids=[1, 2, 3])
        dm.to_frame()           # pandas DataFrame (default)
        dm.to_frame("polars")   # polars DataFrame
        dm.to_numpy()           # raw array
    """

    def __init__(self, data: np.ndarray, ids: list) -> None:
        """Create a DistanceMatrix.

        Args:
            data: Square numpy array of shape ``(n, n)``.
            ids:  List of n identifiers matching the matrix rows/columns.
                  Stored as-is (order preserved from ``pool.unique_ids``).
        """
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"data must be a square 2-D array, got shape {data.shape}")
        if len(ids) != data.shape[0]:
            raise ValueError(
                f"len(ids)={len(ids)} must match data.shape[0]={data.shape[0]}"
            )
        self._data = data
        self._ids = list(ids)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Raw numpy array (shape ``(n, n)``)."""
        return self._data

    @property
    def ids(self) -> list:
        """Ordered list of identifiers (same order as rows/columns)."""
        return self._ids

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying array."""
        return self._data.shape  # type: ignore[return-value]

    @property
    def is_memmap(self) -> bool:
        """``True`` if the underlying data is a memory-mapped file."""
        return isinstance(self._data, np.memmap)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_frame(
        self, output_format: Literal["pandas", "polars"] = "pandas"
    ) -> pd.DataFrame | pl.DataFrame:
        """Return a labelled dataframe with IDs as index/columns.

        Args:
            output_format: ``"pandas"`` (default) returns a
                :class:`pandas.DataFrame`; ``"polars"`` returns a
                :class:`polars.DataFrame` with an extra ``"id"`` column
                (Polars has no row index).

        Returns:
            Square dataframe of shape ``(n, n)``.

        Raises:
            ValueError: If *output_format* is not ``"pandas"`` or ``"polars"``.
        """
        if output_format == "pandas":
            return pd.DataFrame(self._data, index=self._ids, columns=self._ids)
        if output_format == "polars":
            return pl.DataFrame(
                self._data, schema=[str(i) for i in self._ids]
            ).insert_column(0, pl.Series("id", self._ids))
        raise ValueError(
            f"Unknown output_format '{output_format}'. Expected 'pandas' or 'polars'."
        )

    def to_numpy(self) -> np.ndarray:
        """Return the underlying numpy array.

        Returns:
            Square array of shape ``(n, n)``.
        """
        return self._data

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, ids: list, dtype: str = "float32") -> DistanceMatrix:
        """Create a zero-initialised square matrix.

        Args:
            ids:   List of identifiers.
            dtype: Numpy dtype string (default ``"float32"``).

        Returns:
            A :class:`DistanceMatrix` of shape ``(n, n)`` filled with zeros.
        """
        n = len(ids)
        return cls(np.zeros((n, n), dtype=dtype), ids)

    @classmethod
    def from_path(cls, path: str | Path) -> DistanceMatrix:
        """Load a previously computed distance matrix from disk.

        Uses ``resolve_path`` to resolve the storage directory (workspace
        name or filesystem path), then opens the memmap in read-only mode.

        Args:
            path: Storage directory. Same formats as ``StorageOptions.store_path``:
                plain name (``"distances"``), relative path (``"./distances"``),
                or absolute path.

        Returns:
            A :class:`DistanceMatrix` backed by a read-only memmap.

        Raises:
            FileNotFoundError: If the directory or required files don't exist.
            ValueError: If progress.json status is not ``"complete"``
                (incomplete computation).
        """
        resolved = resolve_path(path)
        metadata_path = resolved / "metadata.json"
        progress_path = resolved / "progress.json"
        matrix_path = resolved / "matrix.dat"

        if not resolved.exists():
            raise FileNotFoundError(f"Storage directory not found: {resolved}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in: {resolved}")
        if not progress_path.exists():
            raise FileNotFoundError(f"progress.json not found in: {resolved}")

        progress = json.loads(progress_path.read_text())
        if progress.get("status") != "complete":
            raise ValueError(
                f"Matrix at {resolved} is incomplete "
                f"(status={progress.get('status')!r}). "
                "Run compute_matrix() to finish computation first."
            )

        metadata = json.loads(metadata_path.read_text())
        shape = tuple(metadata["shape"])
        dtype = metadata["dtype"]
        ids = metadata["ids"]

        mm = np.memmap(matrix_path, dtype=dtype, mode="r", shape=shape)
        return cls(mm, ids)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DistanceMatrix(shape={self.shape}, n={len(self._ids)})"

    def __len__(self) -> int:
        return len(self._ids)
