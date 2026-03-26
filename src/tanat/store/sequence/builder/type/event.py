#!/usr/bin/env python3
"""Event sequence store builder."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..base import SequenceStoreBuilder
from ....source.base import AbstractSource
from ....sequence.schema import StoreSchema as SCH


class EventSequenceStoreBuilder(SequenceStoreBuilder, register_name="event"):
    """
    Fluent builder for **Event** sequence stores.

    Exposes ``time_column`` explicitly on every ``add_*`` call.
    """

    _TIME_INDEX_SCHEMA_MAP = {"time_column": SCH.T_EVENT}

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------

    def add_dataframe(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        id_column: str,
        features: str | list[str],
        time_column: str | None = None,
        is_static: bool = False,
    ) -> EventSequenceStoreBuilder:
        """Register an in-memory Polars / Pandas DataFrame."""
        features = [features] if isinstance(features, str) else list(features)
        ti_kwargs = {} if is_static else {"time_column": time_column}
        source = AbstractSource.get_registered("dataframe")(data)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )

    def add_csv(
        self,
        path: str | Path,
        *,
        id_column: str,
        features: str | list[str],
        time_column: str | None = None,
        is_static: bool = False,
        **reader_kwargs,
    ) -> EventSequenceStoreBuilder:
        """Register a CSV file."""
        features = [features] if isinstance(features, str) else list(features)
        ti_kwargs = {} if is_static else {"time_column": time_column}
        source = AbstractSource.get_registered("csv")(path, **reader_kwargs)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )

    def add_parquet(
        self,
        path: str | Path,
        *,
        id_column: str,
        features: str | list[str],
        time_column: str | None = None,
        is_static: bool = False,
        **reader_kwargs,
    ) -> EventSequenceStoreBuilder:
        """Register a Parquet file (glob patterns supported)."""
        features = [features] if isinstance(features, str) else list(features)
        ti_kwargs = {} if is_static else {"time_column": time_column}
        source = AbstractSource.get_registered("parquet")(path, **reader_kwargs)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )

    def add_sql(
        self,
        connection: str,
        query: str,
        *,
        id_column: str,
        features: str | list[str],
        time_column: str | None = None,
        is_static: bool = False,
        **sql_kwargs,
    ) -> EventSequenceStoreBuilder:
        """Register a SQL query (requires ``connectorx``)."""
        features = [features] if isinstance(features, str) else list(features)
        ti_kwargs = {} if is_static else {"time_column": time_column}
        source = AbstractSource.get_registered("sql")(connection, query, **sql_kwargs)
        self._validate_source(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )
        return self._stage(
            source,
            id_column=id_column,
            features=features,
            is_static=is_static,
            time_index_kwargs=ti_kwargs,
        )

    # ------------------------------------------------------------------
    # Sort hook
    # ------------------------------------------------------------------

    def _prepare_entity(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Sort by sequence ID then by event time."""
        return lf.sort(SCH.SEQ_ID, SCH.T_EVENT)
