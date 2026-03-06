#!/usr/bin/env python3
"""Interval sequence store builder."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..base import SequenceStoreBuilder
from ....source.base import AbstractSource
from ....sequence.schema import StoreSchema as SCH


class IntervalSequenceStoreBuilder(SequenceStoreBuilder, register_name="interval"):
    """
    Fluent builder for **Interval** sequence stores.

    Exposes ``start_column`` and ``end_column`` explicitly on every ``add_*`` call.
    Pass ``sort_anchor`` at construction time to control the intra-sequence sort
    column: ``"start"`` (default), ``"end"``, or ``"middle"`` (midpoint of each
    interval).
    """

    _TEMPORAL_SCHEMA_MAP = {"start_column": SCH.T_START, "end_column": SCH.T_END}

    def __init__(self, *, sort_anchor: str = "start") -> None:
        if sort_anchor not in ("start", "end", "middle"):
            raise ValueError(
                f"sort_anchor must be 'start', 'end', or 'middle', got {sort_anchor!r}."
            )
        super().__init__()
        self._sort_anchor = sort_anchor

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------

    def add_dataframe(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        id_column: str,
        features: str | list[str],
        start_column: str | None = None,
        end_column: str | None = None,
        is_static: bool = False,
        **_kw,
    ) -> IntervalSequenceStoreBuilder:
        """Register an in-memory Polars / Pandas DataFrame."""
        features = [features] if isinstance(features, str) else list(features)
        temporal = (
            {}
            if is_static
            else {"start_column": start_column, "end_column": end_column}
        )
        source = AbstractSource.get_registered("dataframe")(data)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )

    def add_csv(
        self,
        path: str | Path,
        *,
        id_column: str,
        features: str | list[str],
        start_column: str | None = None,
        end_column: str | None = None,
        is_static: bool = False,
        **reader_kwargs,
    ) -> IntervalSequenceStoreBuilder:
        """Register a CSV file."""
        features = [features] if isinstance(features, str) else list(features)
        temporal = (
            {}
            if is_static
            else {"start_column": start_column, "end_column": end_column}
        )
        source = AbstractSource.get_registered("csv")(path, **reader_kwargs)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )

    def add_parquet(
        self,
        path: str | Path,
        *,
        id_column: str,
        features: str | list[str],
        start_column: str | None = None,
        end_column: str | None = None,
        is_static: bool = False,
        **reader_kwargs,
    ) -> IntervalSequenceStoreBuilder:
        """Register a Parquet file (glob patterns supported)."""
        features = [features] if isinstance(features, str) else list(features)
        temporal = (
            {}
            if is_static
            else {"start_column": start_column, "end_column": end_column}
        )
        source = AbstractSource.get_registered("parquet")(path, **reader_kwargs)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )

    def add_sql(
        self,
        connection: str,
        query: str,
        *,
        id_column: str,
        features: str | list[str],
        start_column: str | None = None,
        end_column: str | None = None,
        is_static: bool = False,
        **sql_kwargs,
    ) -> IntervalSequenceStoreBuilder:
        """Register a SQL query (requires ``connectorx``)."""
        features = [features] if isinstance(features, str) else list(features)
        temporal = (
            {}
            if is_static
            else {"start_column": start_column, "end_column": end_column}
        )
        source = AbstractSource.get_registered("sql")(connection, query, **sql_kwargs)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            temporal_kwargs=temporal,
        )

    # ------------------------------------------------------------------
    # Sort hook
    # ------------------------------------------------------------------

    def _prepare_entity(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Sort by sequence ID then by the anchor within each sequence.

        ``"middle"`` sorts by the interval midpoint
        ``(T_START + T_END) / 2``, cast to ``Int64`` to support both
        numeric and datetime temporal indexes.
        """
        if self._sort_anchor == "start":
            return lf.sort([SCH.SEQ_ID, SCH.T_START])
        if self._sort_anchor == "end":
            return lf.sort([SCH.SEQ_ID, SCH.T_END])
        # middle: sort by midpoint - cast to Int64 to handle datetime and numeric
        mid_expr = (
            (pl.col(SCH.T_START).cast(pl.Int64) + pl.col(SCH.T_END).cast(pl.Int64)) / 2
        ).alias("__mid__")
        return lf.with_columns(mid_expr).sort([SCH.SEQ_ID, "__mid__"]).drop("__mid__")
