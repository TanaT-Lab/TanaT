#!/usr/bin/env python3
"""Position-based T0 strategy: T0 = row at a given position (0-based, negative ok)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from tanat_utils import settings_dataclass as dataclass

from ..base import T0Setter, _T0

if TYPE_CHECKING:
    from ...sequence.base.pool import SequencePool
    from ...sequence.base.sequence import Sequence


@dataclass
class PositionT0Settings:
    """Settings for the position-based T0 strategy."""

    position: int = 0
    anchor: Literal["start", "end", "middle"] | None = (
        None  # None = auto-resolve from pool type
    )


class PositionT0Setter(T0Setter, register_name="position"):
    """T0 = temporal value of the row at *position* for each sequence.

    Supports 0-based positive indexing (``position=0`` → first row) and
    negative indexing (``position=-1`` → last row).  Sequences shorter
    than ``abs(position) + 1`` rows receive ``_t0 = null``.
    """

    def __init__(
        self,
        *,
        position: int = 0,
        anchor: Literal["start", "end", "middle"] | None = None,
    ):
        super().__init__(PositionT0Settings(position=position, anchor=anchor))

    def _compute_t0(
        self, target: SequencePool | Sequence, ids: list, id_col: str
    ) -> pl.LazyFrame:
        pos = self.settings.position
        cols = target.settings.get_temporal_columns()
        t_expr = self._t0_temporal_expr(
            self.settings.anchor, cols, target.metadata.is_datetime
        )
        # pylint: disable=protected-access
        lf = target._temporal_data_lf().select(id_col, t_expr.alias(_T0))
        # Negative indexing: -1 → last row.
        target_rn = pl.len().over(id_col) + pos if pos < 0 else pos
        return lf.filter(pl.int_range(pl.len()).over(id_col) == target_rn)
