#!/usr/bin/env python3
"""Helpers for output format validation and DataFrame conversions."""

from __future__ import annotations

import pandas as pd
import polars as pl


def resolve_fmt(fmt: str, allowed: tuple[str, ...], default: str) -> str:
    """Validate and normalize the output format.

    Args:
        fmt: Requested output format.
        allowed: Allowed format values.
        default: Default format used when *fmt* is ``None``.

    Returns:
        The validated output format.

    Raises:
        ValueError: If *fmt* is not one of *allowed*.
    """
    resolved = default if fmt is None else fmt
    if resolved not in allowed:
        allowed_txt = ", ".join(repr(v) for v in allowed)
        raise ValueError(f"Invalid fmt {resolved!r}. Expected one of: {allowed_txt}.")
    return resolved


def to_pandas(df: pl.DataFrame, use_arrow: bool = True) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas.

    Args:
        df: Input Polars DataFrame.
        use_arrow: Whether to preserve Arrow extension arrays in pandas.

    Returns:
        The converted pandas DataFrame.
    """
    return df.to_pandas(use_pyarrow_extension_array=use_arrow)
