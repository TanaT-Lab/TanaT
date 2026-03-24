#!/usr/bin/env python3
"""Polars expression builders for sequence describe() metrics.

All functions return ``pl.Expr`` objects.
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Shared metrics (all sequence types)
# ---------------------------------------------------------------------------


def _n_unique_entities_expr(entity_features: list[str]) -> pl.Expr:
    """Number of distinct entities in the sequence.

    For single-feature sequences, counts unique values of that feature.
    For multi-feature sequences, counts unique ``(feat1, feat2, …)`` tuples.
    """
    if len(entity_features) == 1:
        return pl.col(entity_features[0]).n_unique().alias("n_unique_entities")
    return pl.struct(entity_features).n_unique().alias("n_unique_entities")


def _temporal_span_expr(temporal_cols: list[str]) -> pl.Expr:
    """Time elapsed between the earliest start and the latest end.

    - 1 column  (event):           ``max(time) - min(time)``
    - 2 columns (state/interval):  ``max(end)  - min(start)``
    """
    if len(temporal_cols) == 1:
        col = temporal_cols[0]
        return (pl.col(col).max() - pl.col(col).min()).alias("temporal_span")
    start_col, end_col = temporal_cols
    return (pl.col(end_col).max() - pl.col(start_col).min()).alias("temporal_span")


# ---------------------------------------------------------------------------
# State / Interval metrics
# ---------------------------------------------------------------------------


def _duration_stats_exprs(start_col: str, end_col: str) -> list[pl.Expr]:
    """Mean, median, and std of individual entity durations (``end - start``).

    Null values are skipped natively by Polars aggregations.
    """
    dur = pl.col(end_col) - pl.col(start_col)
    return [
        dur.mean().alias("mean_duration"),
        dur.median().alias("median_duration"),
        dur.std().alias("duration_std"),
    ]


# ---------------------------------------------------------------------------
# Event metrics
# ---------------------------------------------------------------------------


def _median_gap_expr(time_col: str) -> pl.Expr:
    """Median inter-event gap (``diff().median()`` on the time column)."""
    return pl.col(time_col).diff().median().alias("median_gap")


def _gap_std_expr(time_col: str) -> pl.Expr:
    """Standard deviation of inter-event gaps (``diff().std()`` on the time column)."""
    return pl.col(time_col).diff().std().alias("gap_std")


# ---------------------------------------------------------------------------
# State metrics
# ---------------------------------------------------------------------------


def _n_transitions_expr(entity_features: list[str]) -> pl.Expr:
    """Number of state transitions (consecutive rows where the state changed).

    For single-feature states: ``sum(state[i] != state[i-1]) - 1``.
    For multi-feature states: a transition occurs when **any** feature changes.
    The ``- 1`` removes the first row comparison which always yields a "change"
    relative to the virtual ``shift(1)`` null.
    """
    if len(entity_features) == 1:
        feat = entity_features[0]
        return ((pl.col(feat) != pl.col(feat).shift(1)).sum() - 1).alias(
            "n_transitions"
        )

    changed = pl.lit(False)
    for feat in entity_features:
        changed = changed | (pl.col(feat) != pl.col(feat).shift(1))
    return (changed.sum() - 1).alias("n_transitions")
