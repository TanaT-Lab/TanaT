#!/usr/bin/env python3
"""SQL source via Polars + connectorx (optional dependency)."""

from __future__ import annotations

import polars as pl

from ..base import AbstractSource

# Formats tried in order when auto-parsing string columns.
_DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d",
]


def _detect_datetime_format(col: pl.Series) -> str | None:
    """Return the first format that parses the sample without producing nulls."""
    sample_size = 20
    sample = col.drop_nulls().filter(col != "").head(sample_size)
    if sample.len() == 0:
        return None

    for fmt in _DATETIME_FORMATS:
        # Strict=False allows parsing failures but doesn't raise an error
        parsed = sample.str.to_datetime(format=fmt, strict=False)

        # No Nulls means the format successfully parsed all values in the sample
        if parsed.null_count() == 0:
            return fmt

    return None


class SqlSource(AbstractSource, register_name="sql"):
    """
    Executes a SQL query and returns the result as a :class:`polars.LazyFrame`.

    String columns that look like dates/datetimes are automatically cast to
    :class:`polars.Datetime`.

    Requires ``connectorx``::

        pip install tanat[sql]

    Args:
        connection: Connection string (e.g. ``"postgresql://user:pwd@host/db"``).
        query:      SQL SELECT query to execute.
        kwargs:     Forwarded to :func:`polars.read_database_uri`.
    """

    def __init__(self, connection: str, query: str, **kwargs) -> None:
        self._connection = connection
        self._query = query
        self._kwargs = kwargs

    def schema(self) -> pl.Schema:
        """
        Probe the SQL schema with a zero-row query.

        Wraps the user query as ``SELECT * FROM (...) AS _q LIMIT 0``
        so the database resolves column names and types without transferring
        any data rows.
        """
        probe_query = f"SELECT * FROM ({self._query}) AS _q LIMIT 0"
        try:
            probe = pl.read_database_uri(
                query=probe_query,
                uri=self._connection,
                **self._kwargs,
            )
        except ImportError as exc:
            raise ImportError(
                "SQL sources require 'connectorx'. "
                "Install it with: pip install tanat[sql]"
            ) from exc
        return probe.schema

    def read(self) -> pl.LazyFrame:
        try:
            df = pl.read_database_uri(
                query=self._query,
                uri=self._connection,
                **self._kwargs,
            )
        except ImportError as exc:
            raise ImportError(
                "SQL sources require 'connectorx'. "
                "Install it with: pip install tanat[sql]"
            ) from exc

        # SQLite (and other DBs) return date/datetime columns as strings.
        # Detect the format on a small sample then apply it to the full column.
        casts = []
        for col_name, dtype in zip(df.columns, df.dtypes):
            if dtype in (pl.String, pl.Utf8):
                fmt = _detect_datetime_format(df[col_name])
                if fmt is not None:
                    casts.append(
                        pl.col(col_name).str.to_datetime(format=fmt, strict=False)
                    )
        if casts:
            df = df.with_columns(casts)

        return df.lazy()

    def __repr__(self) -> str:
        return f"SqlSource(connection={self._connection!r})"
