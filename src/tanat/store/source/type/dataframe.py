#!/usr/bin/env python3
"""In-memory DataFrame source (Polars or Pandas)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ..base import AbstractSource

if TYPE_CHECKING:
    import pandas as pd


class DataFrameSource(AbstractSource, register_name="dataframe"):
    """
    Wraps an in-memory Polars or Pandas DataFrame as a source.

    Args:
        data: A :class:`polars.DataFrame`, :class:`polars.LazyFrame`,
              or :class:`pandas.DataFrame`.
    """

    def __init__(self, data: pl.DataFrame | pl.LazyFrame | pd.DataFrame) -> None:
        self._data = data

    def read(self) -> pl.LazyFrame:
        if isinstance(self._data, pl.LazyFrame):
            return self._data
        if isinstance(self._data, pl.DataFrame):
            return self._data.lazy()
        # Pandas fallback
        return pl.from_pandas(self._data).lazy()

    def schema(self) -> pl.Schema:
        if isinstance(self._data, pl.LazyFrame):
            return self._data.collect_schema()
        if isinstance(self._data, pl.DataFrame):
            return self._data.schema
        return pl.from_pandas(self._data).schema

    def __repr__(self) -> str:
        return f"DataFrameSource(type={type(self._data).__name__})"
