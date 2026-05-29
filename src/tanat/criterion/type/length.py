#!/usr/bin/env python3
"""
LengthCriterion: filter sequences by number of entities.

Compatibility: SEQUENCE only.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
from pydantic import model_validator
from tanat_utils import settings_dataclass as dataclass

from ..base import Criterion, CriterionLevel
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from ...trajectory.pool import TrajectoryPool
from ...trajectory.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class LengthCriterionSettings:
    """Validation settings for :class:`LengthCriterion`.

    At least one bound must be provided. Contradictory bounds are rejected
    at construction time.

    Args:
        gt: Strictly greater than.
        ge: Greater than or equal to.
        lt: Strictly less than.
        le: Less than or equal to.
    """

    gt: int | None = None
    ge: int | None = None
    lt: int | None = None
    le: int | None = None

    @model_validator(mode="after")
    def _check_at_least_one_and_consistent(self) -> LengthCriterionSettings:
        bounds = [self.gt, self.ge, self.lt, self.le]
        if all(b is None for b in bounds):
            raise ValueError("At least one bound (gt, ge, lt, le) must be provided.")
        lo = self.ge if self.gt is None else (self.gt + 1)
        hi = self.le if self.lt is None else (self.lt - 1)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"Contradictory bounds: effective range [{lo}, {hi}] is empty."
            )
        return self


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class LengthCriterion(Criterion):
    """Select sequences by their number of entities (rows).

    Supported levels: **SEQUENCE**.

    Example::

        # sequences with more than 5 entities
        ids = pool.which(LengthCriterion(gt=5))
        pool2 = pool.subset(ids)

        # a single sequence
        ok = seq.match(LengthCriterion(ge=3, lt=20))
    """

    SETTINGS_CLASS = LengthCriterionSettings
    LEVELS: ClassVar[frozenset[CriterionLevel]] = frozenset({CriterionLevel.SEQUENCE})

    def __init__(
        self,
        *,
        gt: int | None = None,
        ge: int | None = None,
        lt: int | None = None,
        le: int | None = None,
    ) -> None:
        super().__init__(settings=LengthCriterionSettings(gt=gt, ge=ge, lt=lt, le=le))

    # ------------------------------------------------------------------
    # Impl hooks
    # ------------------------------------------------------------------

    def _which_ids_impl(
        self,
        pool: SequencePool | TrajectoryPool,
    ) -> set:
        """Vectorized: group-count via the time-index lazy frame."""
        id_col = pool.settings.id_column
        counts = (
            pool._frames.id_time_index()  # pylint: disable=protected-access
            .select(id_col)
            .group_by(id_col)
            .agg(pl.len().alias("__len__"))
            .filter(self._length_predicate("__len__"))
            .collect()
        )
        return set(counts[id_col].to_list())

    def _match_impl(self, target: Sequence | Trajectory) -> bool:
        """Count entities in *target* sequence and check bounds."""
        return self._satisfies(len(target))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _length_predicate(self, length_col: str) -> pl.Expr:
        """Build a Polars expression that returns True when *length_col* satisfies the bounds."""
        s = self._settings
        parts: list[pl.Expr] = []
        if s.gt is not None:
            parts.append(pl.col(length_col) > s.gt)
        if s.ge is not None:
            parts.append(pl.col(length_col) >= s.ge)
        if s.lt is not None:
            parts.append(pl.col(length_col) < s.lt)
        if s.le is not None:
            parts.append(pl.col(length_col) <= s.le)

        expr = parts[0]
        for p in parts[1:]:
            expr = expr & p
        return expr

    def _satisfies(self, length: int) -> bool:
        s = self._settings
        if s.gt is not None and not (length > s.gt):
            return False
        if s.ge is not None and not (length >= s.ge):
            return False
        if s.lt is not None and not (length < s.lt):
            return False
        if s.le is not None and not (length <= s.le):
            return False
        return True
