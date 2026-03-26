#!/usr/bin/env python3
"""Query-based T0 strategy: T0 = timestamp of first/last row matching a Polars expression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from tanat_utils import settings_dataclass as dataclass

from ..base import T0Setter, _T0

if TYPE_CHECKING:
    from ...sequence.base.pool import SequencePool
    from ...sequence.base.sequence import Sequence


@dataclass(config={"arbitrary_types_allowed": True})
class QueryT0Settings:
    """Settings for the query-based T0 strategy.

    The *query* expression is evaluated against the full sequence row set
    (time columns + entity features), so it can reference any of those
    columns.
    """

    query: pl.Expr
    anchor: Literal["start", "end", "middle"] | None = (
        None  # None = auto-resolve from pool type
    )
    use_first: bool = True


class QueryT0Setter(T0Setter, register_name="query"):
    """T0 = timestamp of the first (or last) entity row matching *query*.

    * ``query``: any ``pl.Expr`` that evaluates to a boolean Series on any
      column of the sequence data (time columns **or** entity features),
      e.g. ``pl.col("event") == "admission"`` or ``pl.col("t_start") > threshold``.
    * ``use_first=True`` (default): earliest matching row per sequence.
    * ``use_first=False``: latest matching row.
    * ``anchor``: which time column to read (``"start"`` / ``"end"``).
      For event sequences the anchor is ignored.

    Sequences with no matching row receive ``_t0 = null``.
    """

    def __init__(
        self,
        *,
        query: pl.Expr,
        anchor: Literal["start", "end", "middle"] | None = None,
        use_first: bool = True,
    ):
        super().__init__(
            QueryT0Settings(query=query, anchor=anchor, use_first=use_first)
        )

    @property
    def strategy_summary(self) -> str:
        """e.g. ``'query, anchor=start'``."""
        return f"query, anchor={self.settings.anchor}"

    def _compute_t0(self, target: SequencePool | Sequence, id_col: str) -> pl.LazyFrame:
        cols = target.settings.get_time_columns()
        t_expr = self._t0_temporal_expr(
            self.settings.anchor, cols, target.metadata.is_datetime
        )
        # Full temporal data (time cols + entity features), masks applied.
        # Row numbers are stable within each sequence (physical order preserved).
        # pylint: disable=protected-access
        lf = target._temporal_data_lf().with_columns(
            pl.int_range(pl.len()).over(id_col).alias("__rn__"),
        )
        matched = lf.filter(self.settings.query)
        pick = t_expr.sort_by("__rn__")
        pick = pick.first() if self.settings.use_first else pick.last()
        return matched.group_by(id_col).agg(pick.alias(_T0))
