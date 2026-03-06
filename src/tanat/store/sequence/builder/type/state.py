#!/usr/bin/env python3
"""State sequence store builder."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..base import SequenceStoreBuilder
from ....source.base import AbstractSource
from ....sequence.schema import StoreSchema as SCH

if TYPE_CHECKING:
    from datetime import datetime


class StateSequenceStoreBuilder(SequenceStoreBuilder, register_name="state"):
    """
    Fluent builder for **State** sequence stores.

    States are contiguous and non-overlapping intervals defined by
    ``start_column`` and an optional ``end_column``.

    When ``end_column`` is omitted at registration time, ``T_END`` is
    auto-computed as the next ``T_START`` within each sequence.  Pass
    ``end_value`` at construction time to set the sentinel for the last
    state of every sequence (``None`` → leaves the last ``T_END`` as
    ``null``).

    When ``end_column`` *is* provided by the user, ``validate_continuity``
    (default: ``True``) checks that states are truly contiguous
    (``T_END[i] == T_START[i+1]`` within each sequence) before writing.
    Set it to ``False`` to skip this check on large datasets where the
    cost of a full ``collect()`` is unacceptable.
    """

    _TEMPORAL_SCHEMA_MAP = {"start_column": SCH.T_START, "end_column": SCH.T_END}

    def __init__(
        self,
        *,
        end_value: datetime | int | float | None = None,
        validate_continuity: bool = True,
    ) -> None:
        super().__init__()
        self._end_value = end_value
        self._validate_continuity = validate_continuity

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
    ) -> StateSequenceStoreBuilder:
        """Register an in-memory Polars / Pandas DataFrame."""
        features = [features] if isinstance(features, str) else list(features)
        if is_static:
            temporal = {}
        else:
            temporal = {"start_column": start_column}
            if end_column is not None:
                temporal["end_column"] = end_column
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
    ) -> StateSequenceStoreBuilder:
        """Register a CSV file."""
        features = [features] if isinstance(features, str) else list(features)
        if is_static:
            temporal = {}
        else:
            temporal = {"start_column": start_column}
            if end_column is not None:
                temporal["end_column"] = end_column
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
    ) -> StateSequenceStoreBuilder:
        """Register a Parquet file (glob patterns supported)."""
        features = [features] if isinstance(features, str) else list(features)
        if is_static:
            temporal = {}
        else:
            temporal = {"start_column": start_column}
            if end_column is not None:
                temporal["end_column"] = end_column
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
    ) -> StateSequenceStoreBuilder:
        """Register a SQL query (requires ``connectorx``)."""
        features = [features] if isinstance(features, str) else list(features)
        if is_static:
            temporal = {}
        else:
            temporal = {"start_column": start_column}
            if end_column is not None:
                temporal["end_column"] = end_column
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
        """Sort by sequence ID and start time; derive or validate ``T_END``.

        When ``end_column`` was not provided at registration time, ``T_END``
        is derived here as the next ``T_START`` within each sequence, using
        ``self._end_value`` as the sentinel for the last state.

        When ``end_column`` *was* provided and ``validate_continuity=True``,
        continuity is verified after sorting (``T_END[i] == T_START[i+1]``
        for every sequence).
        """
        lf = lf.sort([SCH.SEQ_ID, SCH.T_START])
        if SCH.T_END not in lf.collect_schema().names():
            lf = lf.with_columns(
                pl.col(SCH.T_START).shift(-1).over(SCH.SEQ_ID).alias(SCH.T_END)
            )
            if self._end_value is not None:
                lf = lf.with_columns(pl.col(SCH.T_END).fill_null(self._end_value))
        elif self._validate_continuity:
            self._check_continuity(lf)
        return lf

    def _check_continuity(self, lf: pl.LazyFrame) -> None:
        """Verify that states are contiguous within each sequence.

        For each sequence, checks that ``T_END[i] == T_START[i+1]`` holds
        for every consecutive pair of states.  Raises :exc:`ValueError` if
        any gap or overlap is detected.

        This method performs a full ``collect()`` and should only be called
        when the dataset fits comfortably in memory.  Set
        ``validate_continuity=False`` on the builder to skip it.

        Args:
            lf: Sorted entity ``LazyFrame`` containing both ``T_START`` and
                ``T_END`` in ``SCH.*`` internal names.

        Raises:
            ValueError: If any sequence contains a gap or overlap between
                        consecutive states.
        """
        violations = (
            lf.with_columns(
                pl.col(SCH.T_START).shift(-1).over(SCH.SEQ_ID).alias("__next_start__")
            )
            .filter(
                pl.col("__next_start__").is_not_null()
                & (pl.col(SCH.T_END) != pl.col("__next_start__"))
            )
            .select(SCH.SEQ_ID)
            .unique()
            .collect()
        )
        if len(violations) > 0:
            bad_ids = violations[SCH.SEQ_ID].to_list()
            sample = bad_ids[:5]
            suffix = f" … (+{len(bad_ids) - 5} more)" if len(bad_ids) > 5 else ""
            raise ValueError(
                f"State continuity violation: {len(bad_ids)} sequence(s) contain gaps "
                f"or overlaps between consecutive states: {sample}{suffix}.\n"
                "State sequences require strictly contiguous, non-overlapping periods "
                "(T_END[i] == T_START[i+1]). If your data contains gaps or overlaps, "
                "use IntervalSequencePool instead - it is designed for that case.\n"
            )
