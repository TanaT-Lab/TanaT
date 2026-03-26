#!/usr/bin/env python3
"""
Pure Polars data-preparation functions for the spanplot builder.
"""

from __future__ import annotations

import polars as pl

from ...base.literals import DisplayUnit
from ...base.utils import MS_PER_DISPLAY_UNIT


def rename_time_index_columns(
    lf: pl.LazyFrame,
    time_cols: list[str],
) -> pl.LazyFrame:
    """Rename the two time index columns to ``__START__`` and ``__END__``.

    Args:
        lf: Input LazyFrame.
        time_cols: List of exactly two time index column names (start, end).

    Returns:
        LazyFrame with time index columns renamed to ``__START__`` and ``__END__``.

    Raises:
        ValueError: If *time_cols* does not contain exactly 2 elements.
    """
    if len(time_cols) != 2:
        raise ValueError(
            f"spanplot requires exactly 2 time index columns (start + end), "
            f"got {len(time_cols)}: {time_cols}. "
            "Only state and interval sequences expose a duration."
        )
    return lf.rename({time_cols[0]: "__START__", time_cols[1]: "__END__"})


def extract_durations(
    lf: pl.LazyFrame,
    display_unit: DisplayUnit | None,
    *,
    extra_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Compute per-row ``__DURATION__`` from ``__END__ - __START__``.

    - *display_unit=None*: raw numeric difference (timestep sequences).
    - *display_unit=<unit>*: datetime difference converted to the requested
      unit via total milliseconds.

    The caller is responsible for ensuring consistency between the time
    column dtype and *display_unit* (see ``IncompatibleDisplayUnitError``).

    Args:
        lf: LazyFrame with ``__ID__``, ``__LABEL__``, ``__START__``, ``__END__``.
        display_unit: Target duration unit, or ``None`` for raw numeric values.
        extra_cols: Additional columns to keep in the output (e.g. ``["__FACET__"]``
            when faceting is active).  Columns that are absent from *lf* are
            silently ignored.

    Returns:
        LazyFrame with columns ``[__ID__, __LABEL__, __DURATION__]`` plus any
        *extra_cols* that were present.

    Raises:
        ValueError: If *display_unit* is not in :data:`MS_PER_DISPLAY_UNIT`.
    """
    if display_unit is not None and display_unit not in MS_PER_DISPLAY_UNIT:
        raise ValueError(
            f"Unknown display_unit={display_unit!r}. "
            f"Accepted: {', '.join(MS_PER_DISPLAY_UNIT)}"
        )

    diff = pl.col("__END__") - pl.col("__START__")

    if display_unit is None:
        dur_expr = diff.alias("__DURATION__")
    else:
        dur_expr = (
            diff.dt.total_milliseconds() / MS_PER_DISPLAY_UNIT[display_unit]
        ).alias("__DURATION__")

    base_cols = ["__ID__", "__LABEL__", "__DURATION__"]
    keep_cols = base_cols + [c for c in (extra_cols or []) if c not in base_cols]
    return lf.with_columns(dur_expr).select(keep_cols)


def sort_labels(df: pl.DataFrame, sort: str) -> list[str]:
    """Return ``__LABEL__`` values in the requested sort order.

    Args:
        df: Collected DataFrame with ``__LABEL__`` and ``__DURATION__`` columns.
        sort: One of ``"ascending"`` (ascending median), ``"descending"``
            (descending median), or ``"alphabetic"`` (string sort).

    Returns:
        Ordered list of unique label strings.
    """
    if sort == "alphabetic":
        return sorted(df["__LABEL__"].unique().to_list())
    order = (
        df.group_by("__LABEL__")
        .agg(pl.col("__DURATION__").median().alias("_med"))
        .sort("_med", descending=(sort == "descending"))
    )
    return order["__LABEL__"].to_list()


def sort_ids(df: pl.DataFrame, sort: str) -> list[str]:
    """Return ``__ID__`` values (as strings) in the requested sort order.

    Args:
        df: Collected DataFrame with ``__ID__`` and ``__DURATION__`` columns.
        sort: One of ``"ascending"`` (ascending median), ``"descending"``
            (descending median), or ``"alphabetic"`` (string sort).

    Returns:
        Ordered list of unique ID strings.
    """
    if sort == "alphabetic":
        return sorted(df["__ID__"].cast(pl.Utf8).unique().to_list())
    order = (
        df.with_columns(pl.col("__ID__").cast(pl.Utf8).alias("__ID_STR__"))
        .group_by("__ID_STR__")
        .agg(pl.col("__DURATION__").median().alias("_med"))
        .sort("_med", descending=(sort == "descending"))
    )
    return order["__ID_STR__"].to_list()
