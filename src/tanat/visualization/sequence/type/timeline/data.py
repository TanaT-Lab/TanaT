#!/usr/bin/env python3
"""
Timeline data transformations: pure Polars functions, no matplotlib dependency.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

StackingMode = Literal["flat", "by_category"]
TimeMode = Literal["absolute", "relative"]


def rename_id_column(
    lf: pl.LazyFrame,
    id_col: str,
) -> pl.LazyFrame:
    """Rename the store ID column to ``__ID__``.

    Args:
        lf: Input LazyFrame.
        id_col: Name of the column that holds the sequence identifier.

    Returns:
        LazyFrame with *id_col* renamed to ``__ID__``.
    """
    return lf.rename({id_col: "__ID__"})


def rename_temporal_columns(
    lf: pl.LazyFrame,
    temporal_cols: list[str],
) -> pl.LazyFrame:
    """Rename pool temporal columns to internal standard names.

    - 1 column  -> renamed to ``__TIME__`` (event pool).
    - 2 columns -> first renamed to ``__TIME__``, second to ``__END__`` (interval/state).

    Args:
        lf: Input LazyFrame.
        temporal_cols: List of temporal column names (1 or 2 elements).

    Returns:
        LazyFrame with temporal columns renamed to ``__TIME__`` (and ``__END__``).
    """
    rename_map: dict[str, str] = {temporal_cols[0]: "__TIME__"}
    if len(temporal_cols) == 2:
        rename_map[temporal_cols[1]] = "__END__"
    return lf.rename(rename_map)


def resolve_label(
    lf: pl.LazyFrame,
    feature: str,
) -> pl.LazyFrame:
    """Rename *feature* to ``__LABEL__``.

    Args:
        lf: Input LazyFrame.
        feature: Name of the entity feature column to use as label.

    Returns:
        LazyFrame with *feature* renamed to ``__LABEL__``.
    """
    return lf.rename({feature: "__LABEL__"})


def drop_null_labels(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop rows where ``__LABEL__`` is null.

    Called before y-position assignment so null rows do not consume a y slot.

    Args:
        lf: Input LazyFrame (must contain ``__LABEL__``).

    Returns:
        LazyFrame with null-label rows removed.
    """
    return lf.filter(pl.col("__LABEL__").is_not_null())


def assign_y_positions(
    lf: pl.LazyFrame,
    mode: StackingMode,
) -> pl.LazyFrame:
    """Add ``__Y_POSITION__`` (0-based integer) based on *mode*.

    - ``"flat"``        -> dense rank on ``__ID__``.
    - ``"by_category"`` -> dense rank on ``__LABEL__``.

    Args:
        lf: Input LazyFrame (must contain ``__ID__`` and ``__LABEL__``).
        mode: Stacking mode.

    Returns:
        LazyFrame with an additional ``__Y_POSITION__`` integer column.
    """
    rank_col = "__ID__" if mode == "flat" else "__LABEL__"
    return lf.with_columns(
        pl.col(rank_col).rank("dense").sub(1).cast(pl.Int32).alias("__Y_POSITION__")
    )


def build_y_tick_map(
    df: pl.DataFrame,
    mode: StackingMode,
) -> dict[int, str]:
    """Return ``{y_position: tick_label}`` for y-axis formatting.

    - ``"flat"``        -> ``{pos: str(id_value)}``.
    - ``"by_category"`` -> ``{pos: label_string}``.

    Args:
        df: Collected DataFrame (must contain ``__Y_POSITION__``,
            ``__ID__``, and ``__LABEL__``).
        mode: Stacking mode used for y-position assignment.

    Returns:
        Mapping from integer y position to its tick label string.
    """
    label_col = "__ID__" if mode == "flat" else "__LABEL__"
    mapping = df.select(["__Y_POSITION__", label_col]).unique().sort("__Y_POSITION__")
    return {
        int(row["__Y_POSITION__"]): str(row[label_col])
        for row in mapping.iter_rows(named=True)
    }
