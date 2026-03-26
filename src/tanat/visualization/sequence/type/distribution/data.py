#!/usr/bin/env python3
"""
Pure-Polars data preparation for DistributionVizBuilder.

Functions
---------
rename_time_index_columns
    Rename raw start/end columns to ``__START__`` / ``__END__``.
assign_time_bins
    Cross-join each segment against a generated bin axis; keep only bins the
    segment occupies (occupancy-based binning).
aggregate_distribution
    Count label occurrences per bin, then derive proportion or percentage.
"""

from __future__ import annotations

import polars as pl


def rename_time_index_columns(
    lf: pl.LazyFrame,
    time_cols: list[str],
) -> pl.LazyFrame:
    """Rename the two time index boundary columns to internal names.

    Args:
        lf: Input LazyFrame.
        time_cols: Exactly two time index column names ``[start_col, end_col]``.

    Returns:
        LazyFrame with the boundary columns renamed to
        ``__START__`` and ``__END__``.

    Raises:
        ValueError: If *time_cols* does not contain exactly two entries.
    """
    if len(time_cols) != 2:
        raise ValueError(
            f"Expected exactly 2 time columns, got {len(time_cols)}: " f"{time_cols!r}."
        )
    start_col, end_col = time_cols
    return lf.rename({start_col: "__START__", end_col: "__END__"})


def assign_time_bins(
    lf: pl.LazyFrame,
    bin_size: str | int | float,
    *,
    is_datetime: bool,
) -> pl.LazyFrame:
    """Add ``__TIME_BIN__`` via occupancy-based binning.

    A segment contributes to every bin it overlaps, i.e., every bin *b* where
    ``__START__ <= b < __END__``. This avoids the start-bin artefact where a
    long segment spanning many bins would be counted in only the first one.

    Implementation note: a single ``.collect()`` is performed to read the
    global temporal bounds (two scalars). The cross-join and filter remain lazy.

    Args:
        lf: LazyFrame with ``__START__``, ``__END__``, and ``__LABEL__`` columns.
        bin_size: Polars duration string (e.g. ``"1d"`` or ``"12h"``) for
            datetime pools, or a numeric step for timestep pools.
        is_datetime: ``True`` when the pool uses ``Datetime`` time columns.

    Returns:
        LazyFrame with ``__TIME_BIN__`` added and one row per
        *(original row, bin)* pair.

    Raises:
        TypeError: If *bin_size* type does not match the pool temporal type.
    """
    if is_datetime and not isinstance(bin_size, str):
        raise TypeError(
            f"bin_size must be a Polars duration string for datetime pools "
            f"(e.g. '1d', '12h', '1w'), got {type(bin_size).__name__!r}."
        )
    if not is_datetime and isinstance(bin_size, str):
        raise TypeError(
            f"bin_size must be int or float for numeric timestep pools, "
            f"got a string {bin_size!r}. Use a numeric value instead."
        )

    # Single collect to read the two global bounds
    bounds = lf.select(
        pl.col("__START__").min().alias("t_min"),
        pl.col("__END__").max().alias("t_max"),
    ).collect()
    t_min = bounds["t_min"][0]
    t_max = bounds["t_max"][0]

    if is_datetime:
        bins_series = pl.datetime_range(t_min, t_max, bin_size, eager=True)
    else:
        step = int(bin_size) if isinstance(bin_size, int) else float(bin_size)
        bins_series = pl.arange(t_min, t_max, step, eager=True)

    bins_lf = bins_series.to_frame("__TIME_BIN__").lazy()

    return lf.join(bins_lf, how="cross").filter(
        (pl.col("__TIME_BIN__") >= pl.col("__START__"))
        & (pl.col("__TIME_BIN__") < pl.col("__END__"))
    )


def aggregate_distribution(
    lf: pl.LazyFrame,
    mode: str,
    *,
    facet_col: str | None = None,
) -> pl.LazyFrame:
    """Count label occurrences per time bin and compute the requested metric.

    After grouping by ``[__TIME_BIN__, __LABEL__]`` (or
    ``[facet_col, __TIME_BIN__, __LABEL__]`` when *facet_col* is set), the raw
    count is either kept as-is (``"count"``) or normalised within each bin
    (``"proportion"`` / ``"percentage"``) using a window expression so no
    additional collect is required.

    Args:
        lf: LazyFrame with ``__TIME_BIN__`` and ``__LABEL__`` columns.
        mode: One of ``"count"``, ``"proportion"``, or ``"percentage"``.
        facet_col: When set, include this column as the leading group-by
            dimension so that counts and normalisations are computed per
            facet × bin rather than globally.

    Returns:
        LazyFrame with columns ``__TIME_BIN__``, ``__LABEL__``, ``__VALUE__``
        (plus *facet_col* when set).
    """
    group_cols = (
        ["__TIME_BIN__", "__LABEL__"]
        if facet_col is None
        else [facet_col, "__TIME_BIN__", "__LABEL__"]
    )
    over_cols = ["__TIME_BIN__"] if facet_col is None else [facet_col, "__TIME_BIN__"]
    out_cols = (
        ["__TIME_BIN__", "__LABEL__", "__VALUE__"]
        if facet_col is None
        else [facet_col, "__TIME_BIN__", "__LABEL__", "__VALUE__"]
    )

    counted = lf.group_by(group_cols).agg(pl.len().alias("__COUNT__"))

    if mode == "count":
        return counted.rename({"__COUNT__": "__VALUE__"})

    bin_total_expr = pl.col("__COUNT__").sum().over(over_cols).alias("__BIN_TOTAL__")

    if mode == "proportion":
        return (
            counted.with_columns(bin_total_expr)
            .with_columns(
                (pl.col("__COUNT__") / pl.col("__BIN_TOTAL__")).alias("__VALUE__")
            )
            .select(out_cols)
        )

    # percentage
    return (
        counted.with_columns(bin_total_expr)
        .with_columns(
            (pl.col("__COUNT__") / pl.col("__BIN_TOTAL__") * 100.0).alias("__VALUE__")
        )
        .select(out_cols)
    )
