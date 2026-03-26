#!/usr/bin/env python3
"""Direct T0 strategy: T0 = a user-provided scalar or per-sequence dict."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from tanat_utils import settings_dataclass as dataclass

from ..base import T0Setter, T0Value, _T0

if TYPE_CHECKING:
    from ...sequence.base.pool import SequencePool
    from ...sequence.base.sequence import Sequence

_VALID_SCALAR_TYPES = (datetime, date, int, float, type(None))


@dataclass
class DirectT0Settings:
    """Settings for the direct T0 strategy."""

    direct: Any  # T0Value | dict[Any, T0Value]
    anchor: Literal["start", "end", "middle"] | None = (
        None  # None = auto-resolve from pool type
    )


class DirectT0Setter(T0Setter, register_name="direct"):
    """T0 = a user-supplied scalar or per-sequence mapping.

    * Scalar: the same value is assigned to every sequence.
    * Dict (``{seq_id: value}``): per-sequence assignment.
      Keys not present in the pool emit a :class:`UserWarning` and are
      ignored.  Sequences with no matching key receive ``_t0 = null``.

    ``_t0_nearest_rank`` is resolved as the last row where ``t_col ≤ T0``
    (floor semantics).  ``null`` when ``_t0 = null`` or when no row
    satisfies the constraint.
    """

    def __init__(
        self,
        *,
        direct: T0Value | dict[Any, T0Value],
        anchor: Literal["start", "end", "middle"] | None = None,
    ):
        if not isinstance(direct, (dict, *_VALID_SCALAR_TYPES)):
            raise TypeError(
                f"'direct' must be a scalar T0Value (datetime, date, int, float, None) "
                f"or a per-sequence dict, got {type(direct).__name__}."
            )
        super().__init__(DirectT0Settings(direct=direct, anchor=anchor))

    @property
    def strategy_summary(self) -> str:
        """e.g. ``'direct, anchor=start'``."""
        return f"direct, anchor={self.settings.anchor}"

    def _compute_t0(self, target: SequencePool | Sequence, id_col: str) -> pl.LazyFrame:
        direct = self.settings.direct
        # pylint: disable=protected-access
        id_lf = target._id_lf
        if isinstance(direct, dict):
            return self._compute_dict(id_lf, direct, id_col)
        return self._compute_scalar(id_lf, direct, id_col)

    @staticmethod
    def _compute_scalar(
        id_lf: pl.LazyFrame, value: T0Value, id_col: str
    ) -> pl.LazyFrame:
        """Add a constant ``_T0_`` column to the typed ID frame."""
        return id_lf.with_columns(pl.lit(value).alias(_T0))

    @staticmethod
    def _compute_dict(
        id_lf: pl.LazyFrame,
        mapping: dict[Any, T0Value],
        id_col: str,
    ) -> pl.LazyFrame:
        """Join the typed ID frame with a mapping frame built from *mapping*."""
        ids = id_lf.collect().to_series().to_list()
        unknown_keys = [k for k in mapping if k not in set(ids)]
        if unknown_keys:
            warnings.warn(
                f"DirectT0Setter: {len(unknown_keys)} key(s) in 'direct' dict are not "
                f"present in the pool and will be ignored: {unknown_keys}",
                UserWarning,
                stacklevel=5,  # user → set_t0 → compute → _compute_t0 → _compute_dict
            )

        # Build a lookup frame with the same ID dtype as the store.
        id_dtype = id_lf.collect_schema()[id_col]
        lookup = pl.DataFrame(
            {id_col: list(mapping.keys()), _T0: list(mapping.values())}
        ).with_columns(pl.col(id_col).cast(id_dtype))

        return id_lf.join(lookup.lazy(), on=id_col, how="left")
