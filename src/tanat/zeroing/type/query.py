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
    (temporal columns + entity features), so it can reference any of those
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
      column of the sequence data (temporal columns **or** entity features),
      e.g. ``pl.col("event") == "admission"`` or ``pl.col("t_start") > threshold``.
    * ``use_first=True`` (default): earliest matching row per sequence.
    * ``use_first=False``: latest matching row.
    * ``anchor``: which temporal column to read (``"start"`` / ``"end"``).
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

    def compute(self, target: SequencePool | Sequence) -> pl.DataFrame:
        # 1. Normalise anchor (frozen-safe writeback).
        self._guard_anchor(target)

        id_col = target.settings.id_column

        # 2. Resolve temporal expression (start / end, dtype-aware for middle).
        cols = target.settings.get_temporal_columns()
        t_expr = self._t0_temporal_expr(
            self.settings.anchor, cols, target.metadata.is_datetime
        )

        # 3. Full sequence data (temporal + entity features), masks and rename applied.
        #    Row numbers are stable within each sequence (physical order preserved).
        lf = target._sequence_data_lf().with_columns(
            pl.int_range(pl.len()).over(id_col).alias("__rn__"),
        )

        # 4. Keep only rows matching the user query.
        matched = lf.filter(self.settings.query)

        # 5. Per sequence: pick the first or last matching row by row number.
        pick = t_expr.sort_by("__rn__")
        pick = pick.first() if self.settings.use_first else pick.last()
        result_lf = matched.group_by(id_col).agg(pick.alias(_T0))

        # 6. Left-join so sequences with no matching row receive _t0 = null.
        ids: list = (
            target.unique_ids if hasattr(target, "unique_ids") else [target.id_value]
        )
        t0_df = (
            pl.LazyFrame({id_col: ids}).join(result_lf, on=id_col, how="left").collect()
        )

        self._warn_nulls(t0_df.filter(pl.col(_T0).is_null())[id_col].to_list())
        self._df = t0_df
        return self._df
