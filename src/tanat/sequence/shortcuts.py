#!/usr/bin/env python3
"""
Quick-build helpers for sequence pools.
"""

from __future__ import annotations

from uuid import uuid4

import pandas as pd
import polars as pl

from ..store.base.utils import (
    get_column_names,
    infer_features,
    validate_required_columns,
)
from .type.event.pool import EventSequencePool
from .type.interval.pool import IntervalSequencePool
from .type.state.pool import StateSequencePool

# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _add_static(
    builder, static_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame, id_column: str
):
    """Validate *static_data* and append it as static features to *builder*."""
    validate_required_columns(static_data, required={id_column})
    return builder.add_dataframe(
        static_data,
        id_column=id_column,
        is_static=True,
        features=infer_features(static_data, exclude={id_column}),
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_events(
    temporal_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    *,
    id_column: str,
    time_column: str,
    static_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame | None = None,
    store_name: str | None = None,
) -> EventSequencePool:
    """Build an :class:`EventSequencePool` from a single DataFrame.

    All columns in ``temporal_data`` except ``id_column`` and ``time_column``
    are treated as entity features.  All columns in ``static_data`` except
    ``id_column`` are treated as static features.

    Args:
        temporal_data: DataFrame or LazyFrame with at least *id*, *time*, and
            one feature column.
        id_column: Name of the sequence identifier column (present in both
            ``temporal_data`` and ``static_data`` if provided).
        time_column: Name of the timestamp column.
        static_data: Optional DataFrame or LazyFrame with per-id static
            features.  Must contain a column named ``id_column`` for joining.
        store_name: Name for the on-disk store.  When ``None`` a unique name
            is generated automatically (``_quick_event_<hex8>``).

    Returns:
        A ready-to-use :class:`EventSequencePool`.

    Raises:
        ValueError: If ``id_column`` or ``time_column`` are missing, if no
            feature columns remain, or if ``id_column`` is absent from
            ``static_data``.

    Examples::

        pool = build_events(df, id_column="patient", time_column="date")
        pool.temporal_data(fmt="polars").head()
    """
    structural = {id_column, time_column}
    validate_required_columns(temporal_data, required=structural)
    builder = EventSequencePool.builder().add_dataframe(
        temporal_data,
        id_column=id_column,
        time_column=time_column,
        features=infer_features(temporal_data, exclude=structural),
    )
    if static_data is not None:
        builder = _add_static(builder, static_data, id_column)
    store_path = builder.build(
        store_name or f"_quick_event_{uuid4().hex[:8]}", exist_ok=True
    )
    return EventSequencePool(
        store=store_path, id_column=id_column, time_column=time_column
    )


def build_intervals(
    temporal_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    *,
    id_column: str,
    start_column: str,
    end_column: str,
    static_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame | None = None,
    store_name: str | None = None,
) -> IntervalSequencePool:
    """Build an :class:`IntervalSequencePool` from a single DataFrame.

    All columns in ``temporal_data`` except ``id_column``, ``start_column``,
    and ``end_column`` are treated as entity features.

    Args:
        temporal_data: DataFrame or LazyFrame with at least *id*, *start*,
            *end*, and one feature column.
        id_column: Name of the sequence identifier column.
        start_column: Name of the interval start column.
        end_column: Name of the interval end column.
        static_data: Optional DataFrame or LazyFrame with per-id static
            features.
        store_name: Name for the on-disk store.  When ``None`` a unique name
            is generated automatically (``_quick_interval_<hex8>``).

    Returns:
        A ready-to-use :class:`IntervalSequencePool`.

    Raises:
        ValueError: If required columns are missing, if no feature columns
            remain, or if ``id_column`` is absent from ``static_data``.

    Examples::

        pool = build_intervals(
            df, id_column="id", start_column="start", end_column="end",
        )
        pool.temporal_data(fmt="polars").head()
    """
    structural = {id_column, start_column, end_column}
    validate_required_columns(temporal_data, required=structural)
    builder = IntervalSequencePool.builder().add_dataframe(
        temporal_data,
        id_column=id_column,
        start_column=start_column,
        end_column=end_column,
        features=infer_features(temporal_data, exclude=structural),
    )
    if static_data is not None:
        builder = _add_static(builder, static_data, id_column)
    store_path = builder.build(
        store_name or f"_quick_interval_{uuid4().hex[:8]}", exist_ok=True
    )
    return IntervalSequencePool(
        store=store_path,
        id_column=id_column,
        start_column=start_column,
        end_column=end_column,
    )


def build_states(
    temporal_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    *,
    id_column: str,
    start_column: str,
    end_column: str | None = None,
    static_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame | None = None,
    store_name: str | None = None,
) -> StateSequencePool:
    """Build a :class:`StateSequencePool` from a single DataFrame.

    When ``end_column`` is ``None`` the end of each state is derived from
    the start of the next state (last state stays open-ended with ``null``).

    All columns in ``temporal_data`` except the structural columns
    (id, start, and optionally end) are treated as entity features.

    Args:
        temporal_data: DataFrame or LazyFrame with at least *id*, *start*, and
            one feature column.
        id_column: Name of the sequence identifier column.
        start_column: Name of the state start column.
        end_column: Name of the state end column.  When ``None`` the builder
            derives end values automatically.
        static_data: Optional DataFrame or LazyFrame with per-id static
            features.
        store_name: Name for the on-disk store.  When ``None`` a unique name
            is generated automatically (``_quick_state_<hex8>``).

    Returns:
        A ready-to-use :class:`StateSequencePool`.

    Raises:
        ValueError: If required columns are missing, if no feature columns
            remain, or if ``id_column`` is absent from ``static_data``.

    Examples::

        pool = build_states(df, id_column="id", start_column="start")
        pool.temporal_data(fmt="polars").head()
    """
    required = {id_column, start_column}
    if end_column is not None:
        required.add(end_column)
    validate_required_columns(temporal_data, required=required)

    # When end_column is omitted the pool outputs a column named "end".
    # If the data already has an "end" column it would silently become a
    # feature and cause a DuplicateError at read time — catch it early.
    if end_column is None:
        candidate_cols = [
            c for c in get_column_names(temporal_data) if c not in required
        ]
        if "end" in candidate_cols:
            raise ValueError(
                "temporal_data has an 'end' column but end_column was not provided. "
                "Pass end_column='end' to use it as the explicit state boundary, "
                "or drop/rename it before calling build_states()."
            )

    builder = StateSequencePool.builder().add_dataframe(
        temporal_data,
        id_column=id_column,
        start_column=start_column,
        features=infer_features(temporal_data, exclude=required),
        **({"end_column": end_column} if end_column is not None else {}),
    )
    if static_data is not None:
        builder = _add_static(builder, static_data, id_column)
    store_path = builder.build(
        store_name or f"_quick_state_{uuid4().hex[:8]}", exist_ok=True
    )
    return StateSequencePool(
        store=store_path,
        id_column=id_column,
        start_column=start_column,
        end_column=end_column or "end",
    )
