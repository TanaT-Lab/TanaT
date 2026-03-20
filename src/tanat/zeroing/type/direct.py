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

    def compute(self, target: SequencePool | Sequence) -> pl.DataFrame:
        """Build ``[id_col, _T0]``, assign to ``self._df``, and return it.

        ``_T0_NEAREST_RANK_`` is NOT computed here; that is ViewMixin's job.
        """
        # Normalise anchor against pool type and write back to frozen settings.
        self._guard_anchor(target)
        direct = self.settings.direct
        id_col = target.settings.id_column
        ids: list = (
            target.unique_ids if hasattr(target, "unique_ids") else [target.id_value]
        )

        if isinstance(direct, dict):
            t0_df = self._compute_dict(ids, direct, id_col)
        else:
            t0_df = self._compute_scalar(ids, direct, id_col)

        null_ids = t0_df.filter(pl.col(_T0).is_null())[id_col].to_list()
        self._warn_nulls(null_ids)

        self._df = t0_df
        return self._df

    def _compute_scalar(self, ids: list, value: T0Value, id_col: str) -> pl.DataFrame:
        """Build ``[id_col, _T0_]`` with the same value for all sequences."""
        return pl.DataFrame({id_col: ids, _T0: [value] * len(ids)})

    def _compute_dict(
        self, ids: list, mapping: dict[Any, T0Value], id_col: str
    ) -> pl.DataFrame:
        """Build ``[id_col, _T0_]`` from a per-sequence mapping."""
        unknown_keys = [k for k in mapping if k not in set(ids)]
        if unknown_keys:
            warnings.warn(
                f"DirectT0Setter: {len(unknown_keys)} key(s) in 'direct' dict are not "
                f"present in the pool and will be ignored: {unknown_keys}",
                UserWarning,
                stacklevel=3,
            )

        return pl.DataFrame({id_col: ids, _T0: [mapping.get(i) for i in ids]})
