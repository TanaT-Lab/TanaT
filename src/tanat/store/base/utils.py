#!/usr/bin/env python3
"""
Shared Arrow I/O utilities for the store layer.

All functions are stateless and operate directly on paths / LazyFrames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl


def get_column_names(
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> list[str]:
    """Return column names without triggering a PerformanceWarning on LazyFrames.

    Args:
        df: Any supported tabular format.

    Returns:
        Ordered list of column names.
    """
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    return list(df.columns)


def validate_required_columns(
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    required: set[str],
) -> None:
    """Raise ``ValueError`` if any column in *required* is missing from *df*.

    Args:
        df: Input DataFrame or LazyFrame.
        required: Set of column names that must be present.

    Raises:
        ValueError: If any column in *required* is absent from *df*.
    """
    available = set(get_column_names(df))
    missing = required - available
    if missing:
        raise ValueError(
            f"The following required columns are missing: "
            f"{sorted(missing)}. Available: {sorted(available)}."
        )


def infer_features(
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    *,
    exclude: set[str],
) -> list[str]:
    """Return all column names from *df* that are not in *exclude*.

    Args:
        df: Input DataFrame or LazyFrame.
        exclude: Structural column names to exclude.

    Returns:
        Ordered list of feature column names.

    Raises:
        ValueError: If no feature columns remain after exclusion.
    """
    features = [c for c in get_column_names(df) if c not in exclude]
    if not features:
        raise ValueError(
            f"No feature columns found after excluding {sorted(exclude)}. "
            f"Available columns: {get_column_names(df)}."
        )
    return features


def check_no_reserved_names(
    cols: list[str],
    reserved: frozenset[str],
    *,
    context: str = "reserved",
) -> None:
    """Raise ``ValueError`` if any column in *cols* collides with *reserved*.

    Used at both the store layer (internal schema names) and the pool layer
    (user-facing structural names such as the ID or time columns).

    Args:
        cols: Incoming column names to validate.
        reserved: Set of forbidden names.
        context: Human-readable description of the reserved set, shown in
            the error message (e.g. ``"internal store columns"`` or
            ``"time columns"``).

    Raises:
        ValueError: If any name in *cols* is in *reserved*.
    """
    bad = [c for c in cols if c in reserved]
    if bad:
        raise ValueError(
            f"Column name(s) {bad} conflict with {context} "
            "and cannot be used as feature names."
        )


def normalise_to_lazyframe(
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> pl.LazyFrame:
    """Convert a DataFrame (pandas, Polars eager, or Polars lazy) to a LazyFrame.

    Args:
        df: Input data in any supported tabular format.

    Returns:
        A :class:`polars.LazyFrame` wrapping *df*.
    """
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df).lazy()
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return df  # already a LazyFrame


def atomic_write(lf: pl.LazyFrame, path: Path) -> None:
    """Atomic Arrow file writing (write to tmp then rename)."""
    temp_path = path.with_suffix(".tmp.arrow")
    try:
        lf.sink_ipc(temp_path, compression="lz4")
        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def scan_if_exists(path: Path) -> pl.LazyFrame | None:
    """Scans an IPC file if it exists, otherwise returns None."""
    if path.exists():
        return pl.scan_ipc(path)
    return None


def hconcat_datasets(
    *datasets: pl.LazyFrame | None,
) -> pl.LazyFrame | None:
    """
    Horizontally concatenates non-``None``, non-empty LazyFrames.

    Returns ``None`` when no data is available.
    """
    parts = [ds for ds in datasets if ds is not None and len(ds.collect_schema()) > 0]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return pl.concat(parts, how="horizontal")


def hconcat_physical_virtual(
    physical_lf: pl.LazyFrame | None,
    virtual_lf: pl.LazyFrame | None,
) -> pl.LazyFrame | None:
    """Horizontal concat of physical and virtual feature frames.

    Virtual columns take precedence: any physical column whose name is
    also present in *virtual_lf* is silently shadowed via ``select``,
    so the virtual value is always returned.  This avoids ``DuplicateError``
    when a feature has been overridden in the virtual context after a
    ``save()`` baked it into the physical file.

    Args:
        physical_lf: Physical feature frame (may be ``None``).
        virtual_lf:  Virtual feature frame (may be ``None``).

    Returns:
        A merged :class:`~polars.LazyFrame`, or ``None`` when both inputs
        are absent / empty.
    """
    if physical_lf is not None and virtual_lf is not None:
        virtual_cols = set(virtual_lf.collect_schema().names())
        phys_keep = [
            c for c in physical_lf.collect_schema().names() if c not in virtual_cols
        ]
        physical_lf = physical_lf.select(phys_keep) if phys_keep else None
    return hconcat_datasets(physical_lf, virtual_lf)


def drop_columns_from_file(path: Path, columns: list[str]) -> bool:
    """
    Removes *columns* from an IPC file on disk.

    Columns not present in the file are silently ignored.
    If no columns remain after the drop, the file is deleted.

    Returns:
        ``True`` if the file was modified, ``False`` otherwise.
    """
    if not path.exists():
        return False

    lf = pl.scan_ipc(path)
    existing = lf.collect_schema().names()
    to_drop = [c for c in columns if c in existing]

    if not to_drop:
        return False

    remaining = lf.drop(to_drop)
    if len(remaining.collect_schema()) == 0:
        path.unlink()
    else:
        atomic_write(remaining, path)
    return True


def apply_casts(
    lf: pl.LazyFrame,
    schema: dict[str, pl.DataType],
) -> pl.LazyFrame:
    """
    Applies a cast schema to *lf*, silently skipping absent columns.

    Args:
        lf: The LazyFrame to apply casts to.
        schema: Mapping of column name → target dtype.

    Returns:
        A new LazyFrame with the cast expressions applied (or *lf* unchanged
        when *schema* is empty or no matching column is found).
    """
    if not schema:
        return lf
    existing = set(lf.collect_schema().names())
    exprs = [pl.col(c).cast(schema[c]) for c in schema if c in existing]
    return lf.with_columns(exprs) if exprs else lf


def filter_and_cast(
    lf: pl.LazyFrame,
    row_filter: pl.Series | None,
    cast_schema: dict[str, pl.DataType] | None,
) -> pl.LazyFrame:
    """
    Applies an optional row filter and an optional cast schema to *lf*.

    Both arguments are safe to pass as ``None`` - the LazyFrame is returned
    unchanged when neither operation is needed.

    Args:
        lf: The LazyFrame to transform.
        row_filter: Boolean ``pl.Series`` aligned with *lf* rows, or ``None``.
        cast_schema: Column-name → dtype mapping, or ``None`` / empty dict.

    Returns:
        The transformed LazyFrame.
    """
    if row_filter is not None:
        lf = lf.filter(row_filter)
    if cast_schema:
        lf = apply_casts(lf, cast_schema)
    return lf


def probe_cast(
    lf: pl.LazyFrame,
    schema: dict[str, pl.DataType],
    n_rows: int = 10,
) -> None:
    """
    Validates cast *schema* against at most *n_rows* rows sampled from *lf*.

    Columns absent from *lf* are silently skipped (same contract as
    :func:`apply_casts`).  Only the minimal IPC footer is read to resolve
    the column list; actual data is fetched for at most *n_rows* rows.

    Args:
        lf: LazyFrame to probe (physical store data, no full scan needed).
        schema: Mapping of column name → target dtype to validate.
        n_rows: Number of rows to sample (default: 10).

    Raises:
        TypeError: Wrapping the Polars error when a cast fails, with a
            description of which columns / target dtypes were involved.
    """
    existing = set(lf.collect_schema().names())
    to_check = {c: dt for c, dt in schema.items() if c in existing}
    if not to_check:
        return
    # Build a sample that excludes rows where ALL probed columns are null,
    # so we always test on real values (a fully-null slice would trivially succeed).
    non_null_filter = pl.any_horizontal(pl.col(c).is_not_null() for c in to_check)
    sample = lf.filter(non_null_filter).limit(n_rows)
    exprs = [pl.col(c).cast(dt) for c, dt in to_check.items()]
    cols_desc = ", ".join(f"'{c}' → {dt}" for c, dt in to_check.items())
    try:
        sample.with_columns(exprs).collect()
    except Exception as exc:
        raise TypeError(
            f"Cast validation failed on {n_rows}-row sample " f"({cols_desc}): {exc}"
        ) from exc
