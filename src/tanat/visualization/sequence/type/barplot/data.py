#!/usr/bin/env python3
"""
Pure Polars data-preparation functions for the barplot builder.
"""

from __future__ import annotations

import polars as pl

from ...base.literals import DisplayUnit, SortOrder
from ...base.utils import MS_PER_DISPLAY_UNIT


def aggregate_count(
    lf: pl.LazyFrame,
    label_col: str,
    *,
    facet_col: str | None = None,
) -> pl.LazyFrame:
    """Count rows per category (and optionally per facet).

    Returns a LazyFrame with columns ``[label_col, __VALUE__]`` or
    ``[facet_col, label_col, __VALUE__]`` when *facet_col* is set.
    """
    group_cols = [label_col] if facet_col is None else [facet_col, label_col]
    return lf.group_by(group_cols).agg(pl.len().alias("__VALUE__"))


def aggregate_rate(
    lf: pl.LazyFrame,
    label_col: str,
    *,
    facet_col: str | None = None,
) -> pl.LazyFrame:
    """Relative frequency per category (and optionally per facet).

    Rates sum to 1 within each facet (or globally when *facet_col* is
    ``None``).  Single-pass: one ``group_by`` + window division.

    Returns a LazyFrame with columns ``[label_col, __VALUE__]`` or
    ``[facet_col, label_col, __VALUE__]``.
    """
    group_cols = [label_col] if facet_col is None else [facet_col, label_col]
    over_cols = [label_col] if facet_col is None else [facet_col]
    return (
        lf.group_by(group_cols)
        .agg(pl.len().alias("__COUNT__"))
        .with_columns(
            (pl.col("__COUNT__") / pl.col("__COUNT__").sum().over(over_cols)).alias(
                "__VALUE__"
            )
        )
        .drop("__COUNT__")
    )


def aggregate_duration(
    lf: pl.LazyFrame,
    label_col: str,
    start_col: str,
    end_col: str,
    display_unit: DisplayUnit | None = None,
    *,
    facet_col: str | None = None,
) -> pl.LazyFrame:
    """Summed (end - start) duration per category (and optionally per facet).

    - *display_unit=None*: raw numeric difference (timestep sequences).
    - *display_unit=<unit>*: datetime difference converted through
      milliseconds into the requested unit.

    The caller is responsible for ensuring consistency between the
    temporal type and *display_unit* (see ``IncompatibleDisplayUnitError``).

    Raises:
        ValueError: If *display_unit* is not in ``MS_PER_DISPLAY_UNIT``.
    """
    if display_unit is not None and display_unit not in MS_PER_DISPLAY_UNIT:
        raise ValueError(
            f"Unknown display_unit={display_unit!r}. "
            f"Accepted: {', '.join(MS_PER_DISPLAY_UNIT)}"
        )

    diff = pl.col(end_col) - pl.col(start_col)
    group_cols = [label_col] if facet_col is None else [facet_col, label_col]

    if display_unit is None:
        return lf.group_by(group_cols).agg(diff.sum().alias("__VALUE__"))

    value_expr = (
        diff.sum().dt.total_milliseconds() / MS_PER_DISPLAY_UNIT[display_unit]
    ).alias("__VALUE__")
    return lf.group_by(group_cols).agg(value_expr)


def apply_sort(lf: pl.LazyFrame, order: SortOrder) -> pl.LazyFrame:
    """Sort *lf* according to *order*.

    Keeps the sort inside the lazy plan so Polars can optimise it
    (e.g. push-down on columnar sources).

    - ``"alphabetic"``  -> ascending __LABEL__ string sort.
    - ``"ascending"``   -> ascending __VALUE__.
    - ``"descending"``  -> descending __VALUE__.
    """
    if order == "alphabetic":
        return lf.sort("__LABEL__")
    if order == "ascending":
        return lf.sort("__VALUE__")
    # "descending"
    return lf.sort("__VALUE__", descending=True)
