#!/usr/bin/env python3
"""
RankCriterion: select entities by their positional rank within a sequence.

Compatibility: ENTITY only.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import polars as pl
from pydantic import Field, field_validator, model_validator
from tanat_utils import settings_dataclass as dataclass

from ..base import Criterion, CriterionLevel
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from ...trajectory.pool import TrajectoryPool
from ...trajectory.trajectory import Trajectory
from ...zeroing import _T0_NEAREST_RANK, _T0

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class RankCriterionSettings:
    """Settings for :class:`RankCriterion`.

    Select entities by their 0-based position within each sequence.
    Exactly **one** parameter group must be specified:
    ``first`` **or** ``last`` **or** ``start``/``end``/``step`` **or** ``ranks``.

    Args:
        first: Keep the first N entities.
            - ``N > 0``: keep first N (like ``[:N]``).
            - ``N < 0``: keep all except last ``|N|`` (like ``[:-|N|]``).
            - Cannot be ``0``. cd /home/aduvermy/Documents/Dev/TanaT-Lab/TanaT && source ../.venv/bin/activate && python -m pytest tests/criterion/
        last: Keep the last N entities.
            - ``N > 0``: keep last N (like ``[-N:]``).
            - ``N < 0``: keep all except first ``|N|`` (like ``[|N|:]``).
            - Cannot be ``0``.
        start: Start rank (inclusive, 0-based). Python-style negative indexing
            supported (e.g. ``-5`` = 5th from the end).
        end: End rank (exclusive, 0-based). Python-style negative indexing
            supported (e.g. ``-2`` = 2nd from the end).
        step: Sub-sample every N-th entity (must be ``>= 1``).
            Only compatible with ``start``/``end``.
        ranks: Specific ranks to keep (0-based, negative = from end).
            A single ``int`` is accepted and coerced to a one-element list.
        relative: If ``True``, interpret ranks relative to T0 (nearest entity to time 0).
    """

    first: int | None = None
    last: int | None = None
    start: int | None = None
    end: int | None = None
    step: int | None = Field(default=None, ge=1)
    ranks: list[int] | None = None
    relative: bool = False

    @field_validator("first")
    @classmethod
    def _validate_first_non_zero(cls, v: int | None) -> int | None:
        if v == 0:
            raise ValueError(
                "first cannot be 0. "
                "Use first > 0 for first N, first < 0 for all except last |N|."
            )
        return v

    @field_validator("last")
    @classmethod
    def _validate_last_non_zero(cls, v: int | None) -> int | None:
        if v == 0:
            raise ValueError(
                "last cannot be 0. "
                "Use last > 0 for last N, last < 0 for all except first |N|."
            )
        return v

    @field_validator("ranks", mode="before")
    @classmethod
    def _coerce_ranks(cls, v):
        """Accept a single int as a one-element list."""
        if isinstance(v, int):
            return [v]
        return v

    @model_validator(mode="after")
    def _validate_at_least_one(self) -> RankCriterionSettings:
        if all(
            v is None
            for v in [
                self.first,
                self.last,
                self.start,
                self.end,
                self.step,
                self.ranks,
            ]
        ):
            raise ValueError(
                "At least one parameter must be specified: "
                "(first) OR (last) OR (start/end/step) OR (ranks)."
            )
        return self

    @model_validator(mode="after")
    def _validate_exclusivity(self) -> RankCriterionSettings:
        """Ensure only one parameter group is active at a time."""
        groups = [
            self.first is not None,
            self.last is not None,
            (self.start is not None)
            or (self.end is not None)
            or (self.step is not None),
            self.ranks is not None,
        ]
        if sum(groups) > 1:
            raise ValueError(
                "Only one parameter group is allowed: "
                "(first) OR (last) OR (start/end/step) OR (ranks)."
            )
        return self

    @model_validator(mode="after")
    def _validate_step_compatibility(self) -> RankCriterionSettings:
        """Ensure step is only used with start/end."""
        if self.step is not None:
            if self.first is not None or self.last is not None:
                raise ValueError("step is not compatible with first/last.")
            if self.ranks is not None:
                raise ValueError("step is not compatible with ranks.")
        return self

    @model_validator(mode="after")
    def _validate_relative_compatibility(self) -> RankCriterionSettings:
        """first/last have no meaning relative to T0; require start/end/step/ranks instead."""
        if self.relative and (self.first is not None or self.last is not None):
            raise ValueError(
                "first/last are not compatible with relative=True. "
                "Use start, end, step, or ranks instead."
            )
        return self


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class RankCriterion(Criterion):
    """Select entities by their positional rank within a sequence.

    Supported levels: **ENTITY** only.

    Entities are numbered 0-based within each sequence in their natural
    store order.  Negative indices use Python-style semantics (from the end).

    Example::

        # keep the first 3 entities
        pool2 = pool.filter_entities(RankCriterion(first=3))

        # keep all except the last 2 entities
        pool2 = pool.filter_entities(RankCriterion(first=-2))

        # keep the last 2 entities
        pool2 = pool.filter_entities(RankCriterion(last=2))

        # keep all except the first 3 entities
        pool2 = pool.filter_entities(RankCriterion(last=-3))

        # keep entities at ranks 2, 3, 4 (Python slice semantics)
        pool2 = pool.filter_entities(RankCriterion(start=2, end=5))

        # keep from rank 5 to 2nd-from-end
        pool2 = pool.filter_entities(RankCriterion(start=5, end=-2))

        # keep every other entity
        pool2 = pool.filter_entities(RankCriterion(step=2))

        # keep specific ranks (first and last)
        pool2 = pool.filter_entities(RankCriterion(ranks=[0, -1]))

        # keep the 3 entities centered on T0 (requires set_t0())
        pool2 = pool.filter_entities(RankCriterion(start=-1, end=2, relative=True))
    """

    SETTINGS_CLASS = RankCriterionSettings
    LEVELS: ClassVar[frozenset[CriterionLevel]] = frozenset({CriterionLevel.ENTITY})

    def __init__(
        self,
        *,
        first: int | None = None,
        last: int | None = None,
        start: int | None = None,
        end: int | None = None,
        step: int | None = None,
        ranks: list[int] | int | None = None,
        relative: bool = False,
    ) -> None:
        super().__init__(
            settings=RankCriterionSettings(
                first=first,
                last=last,
                start=start,
                end=end,
                step=step,
                ranks=ranks,
                relative=relative,
            )
        )

    # ------------------------------------------------------------------
    # Impl hooks
    # ------------------------------------------------------------------

    def _compute_entity_mask(
        self,
        target: Sequence | SequencePool,
    ) -> pl.Series:
        # Stamp __store_idx__ so we can reconstruct a store-space mask regardless
        # of whether *target* has an active _id_mask (subset() view).
        lf = target._id_time_index_lf(
            with_store_index=True
        )  # pylint: disable=protected-access

        if self._settings.relative:
            kept_idx = self._kept_idx_relative(target, lf)
        else:
            kept_idx = self._kept_idx_absolute(target, lf)

        return self._build_store_space_mask(
            target._store.n_entities, kept_idx
        )  # pylint: disable=protected-access

    def _kept_idx_absolute(
        self,
        target: Sequence | SequencePool,
        lf: pl.LazyFrame,
    ) -> pl.Series:
        """Return the surviving ``__store_idx__`` rows in absolute-rank mode."""
        return (
            lf.filter(self._to_expr(target))
            .select("__store_idx__")
            .collect()["__store_idx__"]
        )

    def _kept_idx_relative(
        self,
        target: Sequence | SequencePool,
        lf: pl.LazyFrame,
    ) -> pl.Series:
        """Return the surviving ``__store_idx__`` rows in T0-relative mode.

        IDs without a T0 are kept unfiltered; a :class:`UserWarning` is emitted
        on pools when at least one ID is missing T0.
        """
        id_col = target.settings.id_column
        t0_df = target._get_t0_df()  # pylint: disable=protected-access

        if isinstance(target, SequencePool):
            missing = set(target.unique_ids) - set(t0_df[id_col].to_list())
            if missing:
                warnings.warn(
                    f"{type(self).__name__}(relative=True): "
                    f"{len(missing):,} ID(s) have no T0, their rows are kept unfiltered.",
                    UserWarning,
                    stacklevel=2,
                )

        t0_lf = t0_df.lazy().select([id_col, _T0, _T0_NEAREST_RANK])
        over_col = id_col if isinstance(target, SequencePool) else None

        lf_with_nr = lf.join(t0_lf, on=id_col, how="left")

        abs_rank = self._over(pl.int_range(pl.len()), over_col)
        n = self._over(pl.len(), over_col)
        rel_rank = abs_rank - pl.col(_T0_NEAREST_RANK).cast(pl.Int64)

        expr = self._build_rank_expr_on(rel_rank, n, resolve_negative=False)
        no_t0 = pl.col(_T0_NEAREST_RANK).is_null()

        return (
            lf_with_nr.filter(no_t0 | expr)
            .select("__store_idx__")
            .collect()["__store_idx__"]
        )

    def _to_expr(self, target: Sequence | SequencePool) -> pl.Expr:
        """Return a boolean expression adapted to pool or sequence context.

        Pool path uses ``.over(id_col)`` for per-sequence ranking;
        sequence path uses bare ``pl.int_range``.
        """
        if isinstance(target, SequencePool):
            return self._build_rank_expr(over_col=target.settings.id_column)
        return self._build_rank_expr()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _over(expr: pl.Expr, over_col: str | None) -> pl.Expr:
        """Apply ``.over(over_col)`` when *over_col* is set, otherwise return *expr* as-is."""
        return expr.over(over_col) if over_col is not None else expr

    def _build_rank_expr(self, over_col: str | None = None) -> pl.Expr:
        """Build boolean filter on 0-based absolute rank."""
        rank = self._over(pl.int_range(pl.len()), over_col)
        n = self._over(pl.len(), over_col)
        return self._build_rank_expr_on(rank, n)

    def _build_rank_expr_on(
        self,
        rank: pl.Expr,
        n: pl.Expr,
        *,
        resolve_negative: bool = True,
    ) -> pl.Expr:
        """Build the boolean filter expression given pre-built *rank* and *n* expressions.

        Used by both the absolute path (:meth:`_build_rank_expr`) and the
        T0-relative path (:meth:`filter_lf` with ``relative=True``).

        Args:
            rank: Per-entity rank expression (absolute or relative to T0).
            n: Sequence length expression (used to resolve negative indices).
            resolve_negative: When ``True`` (default, absolute mode), negative
                indices for ``start``/``end``/``ranks`` are resolved as
                ``n + value`` (Python-style from-end semantics).  When
                ``False`` (relative mode), negative values are used as-is
                because they already express a position relative to T0
                (e.g. ``start=-1`` means one entity before T0).
        """
        s = self._settings
        parts: list[pl.Expr] = []

        def _resolve(v: int) -> pl.Expr:
            """Return the Polars expression for value *v*, resolving negatives when needed."""
            if v < 0 and resolve_negative:
                return n + v
            return pl.lit(v)

        if s.first is not None:
            # first > 0: keep first N  →  rank < N
            # first < 0: keep all except last |N|  →  rank < n + first
            parts.append(rank < s.first if s.first > 0 else rank < n + s.first)

        elif s.last is not None:
            # last > 0: keep last N  →  rank >= n - N
            # last < 0: keep all except first |N|  →  rank >= |N|
            parts.append(rank >= n - s.last if s.last > 0 else rank >= -s.last)

        elif s.start is not None or s.end is not None:
            if s.start is not None:
                parts.append(rank >= _resolve(s.start))
            if s.end is not None:
                parts.append(rank < _resolve(s.end))
            if s.step is not None:
                parts.append((rank % s.step) == 0)

        elif s.step is not None:
            # step alone: sub-sample every N-th entity from the start
            parts.append((rank % s.step) == 0)

        elif s.ranks is not None:
            rank_parts: list[pl.Expr] = []
            for r in s.ranks:
                if r >= 0 or not resolve_negative:
                    rank_parts.append(rank == r)
                else:
                    rank_parts.append(rank == n + r)
            if rank_parts:
                combined = rank_parts[0]
                for p in rank_parts[1:]:
                    combined = combined | p
                parts.append(combined)

        if not parts:
            return pl.lit(True)
        result = parts[0]
        for p in parts[1:]:
            result = result & p
        return result
