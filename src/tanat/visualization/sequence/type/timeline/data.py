#!/usr/bin/env python3
"""
Pure Polars data-preparation functions for the timeline builder.
"""

from __future__ import annotations

import polars as pl

from ...base.literals import GroupBy


def rename_time_index_columns(
    lf: pl.LazyFrame,
    time_cols: list[str],
) -> pl.LazyFrame:
    """Rename pool time index columns to internal standard names.

    - 1 column  -> renamed to ``__TIME__`` (event pool).
    - 2 columns -> first renamed to ``__TIME__``, second to ``__END__`` (interval/state).

    Args:
        lf: Input LazyFrame.
        time_cols: List of time index column names (1 or 2 elements).

    Returns:
        LazyFrame with time index columns renamed to ``__TIME__`` (and ``__END__``).
    """
    rename_map: dict[str, str] = {time_cols[0]: "__TIME__"}
    if len(time_cols) == 2:
        rename_map[time_cols[1]] = "__END__"
    return lf.rename(rename_map)


def assign_y_positions(
    lf: pl.LazyFrame,
    mode: GroupBy,
) -> pl.LazyFrame:
    """Add ``__Y_POSITION__`` (0-based integer) based on *mode*.

    - ``"id"``       -> dense rank on ``__ID__``.
    - ``"category"`` -> dense rank on ``__LABEL__``.

    Args:
        lf: Input LazyFrame (must contain ``__ID__`` and ``__LABEL__``).
        mode: Grouping mode.

    Returns:
        LazyFrame with an additional ``__Y_POSITION__`` integer column.
    """
    rank_col = "__ID__" if mode == "id" else "__LABEL__"
    return lf.with_columns(
        pl.col(rank_col).rank("dense").sub(1).cast(pl.Int32).alias("__Y_POSITION__")
    )


def build_y_tick_map(
    df: pl.DataFrame,
    mode: GroupBy,
) -> dict[int, str]:
    """Return ``{y_position: tick_label}`` for y-axis formatting.

    - ``"id"``       -> ``{pos: str(id_value)}``.
    - ``"category"`` -> ``{pos: label_string}``.

    Args:
        df: Collected DataFrame (must contain ``__Y_POSITION__``,
            ``__ID__``, and ``__LABEL__``).
        mode: Grouping mode used for y-position assignment.

    Returns:
        Mapping from integer y position to its tick label string.
    """
    label_col = "__ID__" if mode == "id" else "__LABEL__"
    mapping = df.select(["__Y_POSITION__", label_col]).unique().sort("__Y_POSITION__")
    return {
        int(row["__Y_POSITION__"]): str(row[label_col])
        for row in mapping.iter_rows(named=True)
    }
