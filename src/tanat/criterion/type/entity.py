#!/usr/bin/env python3
"""EntityCriterion: filter entities/sequences by a Polars expression.

Compatibility: ENTITY, SEQUENCE.
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
class EntityCriterionSettings:
    """Settings for :class:`EntityCriterion`.

    Args:
        query: A ``polars.Expr`` used for lazy filtering.
    """

    query: pl.Expr


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class EntityCriterion(Criterion):
    """Filter entities or select sequences using a Polars expression.

    Supported levels: **ENTITY**, **SEQUENCE**.

    Example::

        # entity-level pruning (keep only rows where diag_type == "DP")
        pool2 = pool.filter_entities(EntityCriterion(query=pl.col("diag_type") == "DP"))

        # sequence selection: IDs that have at least one such row
        ids = pool.which(EntityCriterion(query=pl.col("diag_type") == "DP"))

        # single sequence match
        ok = seq.match(EntityCriterion(query=pl.col("diag_type") == "DP"))
    """

    SETTINGS_CLASS = EntityCriterionSettings
    LEVELS: ClassVar[frozenset[CriterionLevel]] = frozenset(
        {CriterionLevel.ENTITY, CriterionLevel.SEQUENCE}
    )

    def __init__(self, query: pl.Expr) -> None:
        super().__init__(settings=EntityCriterionSettings(query=query))

    # ------------------------------------------------------------------
    # Impl hooks
    # ------------------------------------------------------------------

    def _which_ids_impl(
        self,
        pool: SequencePool | TrajectoryPool,
    ) -> set:
        """IDs that have at least one entity row matching the query."""
        s = pool.settings
        id_col = s.id_column
        lf = pool._temporal_data_lf(
            features=s.entity_features
        )  # pylint: disable=protected-access
        result = lf.filter(self._settings.query).select(id_col).unique().collect()
        return set(result[id_col].to_list())

    def _compute_entity_mask(self, target: Sequence | SequencePool) -> pl.Series:
        s = target.settings
        lf = target._temporal_data_lf(  # pylint: disable=protected-access
            features=s.entity_features, with_store_index=True
        )

        self._probe_boolean(lf, self._settings.query, kind="Entity")

        kept_idx = (
            lf.filter(self._settings.query)
            .select("__store_idx__")
            .collect()["__store_idx__"]
        )
        return self._build_store_space_mask(
            # pylint: disable=protected-access
            target._store.n_entities,
            kept_idx,
        )

    def _match_impl(self, target: Sequence | Trajectory) -> bool:
        """True if at least one entity row satisfies the query."""
        s = target.settings
        lf = target._temporal_data_lf(
            features=s.entity_features
        )  # pylint: disable=protected-access
        return lf.filter(self._settings.query).limit(1).collect().height > 0
