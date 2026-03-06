#!/usr/bin/env python3
"""File-based sources: CSV and Parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..base import AbstractSource


class CsvSource(AbstractSource, register_name="csv"):
    """
    Reads a CSV file via :func:`polars.scan_csv`.

    Args:
        path:   Path to the CSV file.
        kwargs: Forwarded verbatim to :func:`polars.scan_csv`
                (e.g. ``separator``, ``schema_overrides``, ``null_values``).
    """

    def __init__(self, path: str | Path, **kwargs) -> None:
        self._path = Path(path)
        self._kwargs = kwargs

    def read(self) -> pl.LazyFrame:
        kwargs = {"try_parse_dates": True, **self._kwargs}
        return pl.scan_csv(self._path, **kwargs)

    def schema(self) -> pl.Schema:
        kwargs = {"try_parse_dates": True, **self._kwargs}
        return pl.scan_csv(self._path, **kwargs).collect_schema()

    def __repr__(self) -> str:
        return f"CsvSource(path={self._path})"


class ParquetSource(AbstractSource, register_name="parquet"):
    """
    Reads a Parquet file via :func:`polars.scan_parquet`.

    Args:
        path:   Path to the Parquet file (glob patterns supported).
        kwargs: Forwarded verbatim to :func:`polars.scan_parquet`.
    """

    def __init__(self, path: str | Path, **kwargs) -> None:
        self._path = Path(path)
        self._kwargs = kwargs

    def read(self) -> pl.LazyFrame:
        return pl.scan_parquet(self._path, **self._kwargs)

    def schema(self) -> pl.Schema:
        return pl.scan_parquet(self._path, **self._kwargs).collect_schema()

    def __repr__(self) -> str:
        return f"ParquetSource(path={self._path})"
