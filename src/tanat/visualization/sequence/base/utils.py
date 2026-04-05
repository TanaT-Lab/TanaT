#!/usr/bin/env python3
"""
Shared pure-Polars utility functions for sequence visualization data preparation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import polars as pl

from ....zeroing import _T0
from .literals import DisplayUnit

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
