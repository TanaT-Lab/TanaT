#!/usr/bin/env python3
"""
Shared pure-Polars utility functions for sequence visualization data preparation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import polars as pl

from ....zeroing import _T0
from .literals import DisplayUnit, NaLabel, NaTimeIndex

if TYPE_CHECKING:
    from ....sequence.base.pool import SequencePool
    from ....sequence.base.sequence import Sequence

# Milliseconds per display unit, shared across barplot and spanplot.
MS_PER_DISPLAY_UNIT: dict[str, int] = {
    "days": 86_400_000,
    "hours": 3_600_000,
    "minutes": 60_000,
    "seconds": 1_000,
}

# Human-readable unit labels for relative time x-axis.
UNIT_LABELS: dict[str, str] = {
    "days": "Days",
    "hours": "Hours",
    "minutes": "Minutes",
    "seconds": "Seconds",
}


def resolve_display_unit(
    display_unit: DisplayUnit | None,
    *,
    is_datetime: bool,
) -> DisplayUnit | None:
    """Resolve *display_unit* for relative time mode.

    - datetime + ``None``   → ``"days"`` (silent default)
    - datetime + explicit   → pass through unchanged
    - timestep + ``None``   → ``None`` (numeric offsets, no conversion)
    - timestep + explicit   → emits :class:`UserWarning`, returns ``None``

    Args:
        display_unit: Requested unit, or ``None`` for the default.
        is_datetime: ``True`` when the pool uses ``Datetime`` time columns.

    Returns:
        Resolved ``DisplayUnit`` value, or ``None`` for timestep pools.
    """
    if not is_datetime and display_unit is not None:
        warnings.warn(
            "display_unit is ignored for timestep (numeric) pools. "
            "The x-axis will show raw numeric offsets from T0.",
            UserWarning,
            stacklevel=4,
        )
        return None
    if is_datetime and display_unit is None:
        return "days"

    return display_unit


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


def rename_time_index_columns(
    lf: pl.LazyFrame,
    time_cols: list[str],
) -> pl.LazyFrame:
    """Rename the two time index boundary columns to ``__START__`` / ``__END__``.

    Used by builders that operate on interval or state pools
    (barplot-duration, distribution, spanplot).  The timeline builder
    uses its own variant because events have a single ``__TIME__`` column.

    Args:
        lf: Input LazyFrame.
        time_cols: Exactly two time index column names ``[start_col, end_col]``.

    Returns:
        LazyFrame with *start_col* → ``__START__`` and *end_col* → ``__END__``.

    Raises:
        ValueError: If *time_cols* does not contain exactly two entries.
    """
    if len(time_cols) != 2:
        raise ValueError(
            f"Expected exactly 2 time columns, got {len(time_cols)}: {time_cols!r}."
        )
    return lf.rename({time_cols[0]: "__START__", time_cols[1]: "__END__"})


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


def handle_null_time_index(
    lf: pl.LazyFrame,
    strategy: NaTimeIndex,
) -> pl.LazyFrame:
    """Handle null values in time index columns (``__TIME__``, ``__START__``, ``__END__``).

    Inspects the schema to determine which internal time columns are present
    and applies the chosen *strategy* uniformly.

    Args:
        lf: Input LazyFrame (columns already renamed to internal names).
        strategy: ``"drop"`` removes null rows with a :class:`UserWarning`;
            ``"raise"`` raises :exc:`ValueError` immediately.

    Returns:
        LazyFrame with null time index rows handled.

    Raises:
        ValueError: If *strategy* is ``"raise"`` and nulls are found, **or**
            if ``"drop"`` would remove every row.
    """
    time_cols = [
        c
        for c in ("__TIME__", "__START__", "__END__")
        if c in lf.collect_schema().names()
    ]
    if not time_cols:
        return lf

    # Build a filter: row is null if ANY time column is null.
    null_mask = pl.lit(False)
    for col in time_cols:
        null_mask = null_mask | pl.col(col).is_null()

    # Materialise counts (two scalars only).
    counts = lf.select(
        null_mask.sum().alias("n_null"),
        pl.len().alias("n_total"),
    ).collect()
    n_null = counts["n_null"][0]
    n_total = counts["n_total"][0]

    if n_null == 0:
        return lf

    # Identify which columns actually have nulls (for the message).
    null_col_counts = lf.select(
        [pl.col(c).is_null().sum().alias(c) for c in time_cols]
    ).collect()
    affected = [c for c in time_cols if null_col_counts[c][0] > 0]
    col_list = ", ".join(affected)

    if strategy == "raise":
        raise ValueError(
            f"{n_null} row(s) have a null time index ({col_list}). "
            "Clean your data or use na_time_index='drop' to exclude them."
        )

    # strategy == "drop"
    if n_null >= n_total:
        raise ValueError(
            f"All {n_total} row(s) have a null time index ({col_list}). "
            "Nothing to render."
        )

    warnings.warn(
        f"{n_null} row(s) have a null time index ({col_list}) and will be "
        "excluded from the visualisation.",
        UserWarning,
        stacklevel=4,
    )
    return lf.filter(~null_mask)


def handle_null_labels(
    lf: pl.LazyFrame,
    strategy: NaLabel,
) -> pl.LazyFrame:
    """Handle null values in the ``__LABEL__`` column.

    Args:
        lf: Input LazyFrame (must contain ``__LABEL__``).
        strategy: ``"drop"`` removes null-label rows with a :class:`UserWarning`;
            ``"raise"`` raises :exc:`ValueError` immediately;
            ``"category"`` replaces nulls with the string ``"N/A"``.

    Returns:
        LazyFrame with null labels handled.

    Raises:
        ValueError: If *strategy* is ``"raise"`` and nulls are found, **or**
            if ``"drop"`` would remove every row.
    """
    counts = lf.select(
        pl.col("__LABEL__").is_null().sum().alias("n_null"),
        pl.len().alias("n_total"),
    ).collect()
    n_null = counts["n_null"][0]
    n_total = counts["n_total"][0]

    if n_null == 0:
        return lf

    if strategy == "raise":
        raise ValueError(
            f"{n_null} row(s) have a null label. "
            "Clean your data or use na_label='drop' to exclude them."
        )

    if strategy == "category":
        return lf.with_columns(pl.col("__LABEL__").fill_null(pl.lit("N/A")))

    # strategy == "drop"
    if n_null >= n_total:
        raise ValueError(f"All {n_total} row(s) have a null label. Nothing to render.")

    warnings.warn(
        f"{n_null} row(s) have a null label and will be excluded "
        "from the visualisation.",
        UserWarning,
        stacklevel=4,
    )
    return lf.filter(pl.col("__LABEL__").is_not_null())


def shift_time_to_relative(
    lf: pl.LazyFrame,
    sequence_or_pool: SequencePool | Sequence,
    time_col: str,
    end_col: str | None = None,
    *,
    display_unit: DisplayUnit = "days",
) -> pl.LazyFrame:
    """Subtract per-ID T0 from temporal columns, converting to relative time.

    Joins the pool's T0 table on ``__ID__`` and replaces *time_col* (and
    optionally *end_col*) with ``col - _T0_``.  Rows whose ``_T0_`` is null
    are dropped with a :exc:`UserWarning` stating the count of dropped IDs.
    Raises :exc:`ValueError` when **all** T0 values are null.

    For datetime pools the shifted columns are further converted to
    ``Float64`` in the requested *display_unit* so that the x-axis shows a
    human-readable numeric scale.  For timestep pools the numeric offset is
    left unchanged and *display_unit* is ignored.

    Args:
        lf: LazyFrame with ``__ID__`` and the named time columns already
            renamed to their internal names (e.g. ``__TIME__``, ``__START__``).
        sequence_or_pool: Source pool or sequence (provides ``_get_t0_df()``
            and settings).
        time_col: Internal time column name (``"__TIME__"`` or ``"__START__"``).
        end_col: Optional end column name (``"__END__"`` for interval/state pools).
        display_unit: Target unit for datetime pools (``"days"`` by default).
            Ignored for timestep pools.

    Returns:
        LazyFrame with time columns shifted to offsets from T0.  Datetime pools
        yield ``Float64`` in *display_unit*; timestep pools yield a numeric offset.

    Raises:
        ValueError: If every sequence has a null T0 value.
    """
    id_col = sequence_or_pool.settings.id_column
    is_datetime = sequence_or_pool.metadata.time_index.is_datetime

    # Build a small collected [id_col, _T0_] lookup table.
    # _get_t0_df() is cached and triggers the lazy default when no set_t0() was called.
    t0_df = sequence_or_pool._get_t0_df().select([id_col, _T0])

    # Null-T0 handling: raise when all are null, warn when some are.
    n_null = t0_df.filter(pl.col(_T0).is_null()).height
    if n_null > 0:
        if n_null == t0_df.height:
            raise ValueError(
                "Cannot use time_mode='relative': all sequences have a null T0 value. "
                "Call set_t0() with a valid strategy before drawing in relative time mode."
            )
        warnings.warn(
            f"{n_null} sequence(s) have no valid T0 value and will be excluded "
            "from the relative-time visualisation.",
            UserWarning,
            stacklevel=4,
        )

    # Rename id_col → __ID__ to align with the already-renamed lf.
    t0_lf = t0_df.rename({id_col: "__ID__"}).lazy()

    # Left-join, then drop rows whose T0 is null (unknown reference date).
    lf = lf.join(t0_lf, on="__ID__", how="left")
    lf = lf.filter(pl.col(_T0).is_not_null())

    # Subtract T0 from each temporal column in-place.
    exprs: list[pl.Expr] = [(pl.col(time_col) - pl.col(_T0)).alias(time_col)]
    if end_col is not None:
        exprs.append((pl.col(end_col) - pl.col(_T0)).alias(end_col))
    lf = lf.with_columns(exprs)

    # For datetime pools: convert Duration → Float64 in the target display_unit
    # so that matplotlib renders a clean numeric x-axis instead of raw timedelta objects.
    if is_datetime:
        ms_divisor = MS_PER_DISPLAY_UNIT[display_unit]
        unit_exprs: list[pl.Expr] = [
            (
                pl.col(time_col).dt.total_milliseconds().cast(pl.Float64) / ms_divisor
            ).alias(time_col)
        ]
        if end_col is not None:
            unit_exprs.append(
                (
                    pl.col(end_col).dt.total_milliseconds().cast(pl.Float64)
                    / ms_divisor
                ).alias(end_col)
            )
        lf = lf.with_columns(unit_exprs)

    # Drop the helper _T0_ column added by the join.
    lf = lf.drop(_T0)
    return lf
