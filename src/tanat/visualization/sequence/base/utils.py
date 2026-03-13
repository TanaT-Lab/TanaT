#!/usr/bin/env python3
"""
Shared pure-Polars utility functions for sequence visualization data preparation.
"""

from __future__ import annotations

import polars as pl

# Milliseconds per display unit, shared across barplot and spanplot.
MS_PER_DISPLAY_UNIT: dict[str, int] = {
    "days": 86_400_000,
    "hours": 3_600_000,
    "minutes": 60_000,
    "seconds": 1_000,
}


def rename_id_column(
    lf: pl.LazyFrame,
    id_col: str,
) -> pl.LazyFrame:
    """Rename the sequence ID column to ``__ID__``.

    Args:
        lf: Input LazyFrame.
        id_col: Name of the column that holds the sequence identifier.

    Returns:
        LazyFrame with *id_col* renamed to ``__ID__``.
    """
    return lf.rename({id_col: "__ID__"})


def resolve_label(
    lf: pl.LazyFrame,
    feature: str,
) -> pl.LazyFrame:
    """Rename *feature* to the internal ``__LABEL__`` column.

    Args:
        lf: Input LazyFrame.
        feature: Name of the entity feature column to use as label.

    Returns:
        LazyFrame with *feature* renamed to ``__LABEL__``.
    """
    return lf.rename({feature: "__LABEL__"})


def drop_null_labels(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop rows where ``__LABEL__`` is null.

    Args:
        lf: Input LazyFrame (must contain ``__LABEL__``).

    Returns:
        LazyFrame with null-label rows removed.
    """
    return lf.filter(pl.col("__LABEL__").is_not_null())
