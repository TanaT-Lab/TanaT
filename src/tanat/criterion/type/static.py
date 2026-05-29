#!/usr/bin/env python3
"""
StaticCriterion: filter sequences/trajectories by a static-data expression.

Compatibility: SEQUENCE, TRAJECTORY.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
from tanat_utils import settings_dataclass as dataclass

from ..base import Criterion, CriterionLevel
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from ...trajectory.pool import TrajectoryPool
from ...trajectory.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(config={"arbitrary_types_allowed": True})
class StaticCriterionSettings:
    """Settings for :class:`StaticCriterion`.

    Args:
        query: A ``polars.Expr`` evaluated against the static data frame.
    """

    query: pl.Expr


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class StaticCriterion(Criterion):
    """Select sequences or trajectories using a static-feature expression.

    Supported levels: **SEQUENCE**, **TRAJECTORY**.

    Example::

        # sequence pool: keep IDs where age > 50
        ids = seq_pool.which(StaticCriterion(query=pl.col("age") > 50))
        seq_pool2 = seq_pool.subset(ids)

        # trajectory pool
        ids = traj_pool.which(StaticCriterion(query=pl.col("group") == "A"))
        traj_pool2 = traj_pool.subset(ids)

        # single match
        ok = seq.match(StaticCriterion(query=pl.col("age") > 50))
        ok = traj.match(StaticCriterion(query=pl.col("group") == "A"))
    """

    SETTINGS_CLASS = StaticCriterionSettings
    LEVELS: ClassVar[frozenset[CriterionLevel]] = frozenset(
        {CriterionLevel.SEQUENCE, CriterionLevel.TRAJECTORY}
    )

    def __init__(self, query: pl.Expr) -> None:
        super().__init__(settings=StaticCriterionSettings(query=query))

    # ------------------------------------------------------------------
    # Impl hooks
    # ------------------------------------------------------------------

    def _which_ids_impl(
        self,
        pool: SequencePool | TrajectoryPool,
    ) -> set:
        """IDs whose static row satisfies the expression."""
        id_col = pool.settings.id_column
        lf = self._require_static_lf(pool)
        self._probe_boolean(lf, self._settings.query, kind="Static")
        result = lf.filter(self._settings.query).select(id_col).collect()
        return set(result[id_col].to_list())

    def _match_impl(self, target: Sequence | Trajectory) -> bool:
        """True if this item's static row satisfies the expression."""
        lf = self._require_static_lf(target)
        self._probe_boolean(lf, self._settings.query, kind="Static")
        return lf.filter(self._settings.query).limit(1).collect().height > 0

    @staticmethod
    def _require_static_lf(obj) -> pl.LazyFrame:
        """Return the static-data LazyFrame or raise if unavailable."""
        lf = obj._frames.static()  # pylint: disable=protected-access
        if lf is None:
            raise ValueError(
                "StaticCriterion requires static features, but none are available."
            )
        return lf
