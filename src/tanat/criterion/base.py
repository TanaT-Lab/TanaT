#!/usr/bin/env python3
"""
Criterion base: abstract class, level enum, and core exceptions.
"""

from __future__ import annotations

import enum
import warnings
from abc import ABC
from typing import ClassVar, Literal

import numpy as np
import polars as pl
from tqdm import tqdm
from tanat_utils import Registrable, SettingsMixin

from ..sequence.base.pool import SequencePool
from ..sequence.base.sequence import Sequence
from ..trajectory.pool import TrajectoryPool
from ..trajectory.trajectory import Trajectory
from ..exceptions import TanaTException

# ---------------------------------------------------------------------------
# Enum & Exceptions
# ---------------------------------------------------------------------------


class CriterionLevel(enum.Enum):
    """Compatibility level for a criterion."""

    ENTITY = enum.auto()
    SEQUENCE = enum.auto()
    TRAJECTORY = enum.auto()


class CriterionLevelError(TanaTException):
    """Raised when a criterion is applied at an incompatible level."""


class CriterionError(TanaTException):
    """Raised when a criterion expression is invalid or fails schema probing."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Criterion(SettingsMixin, Registrable, ABC):
    """Abstract base for all filtering criteria.

    Subclasses must declare :attr:`LEVELS` and implement the three
    ``_*_impl`` hooks.  All public methods enforce compatibility and
    delegate to those hooks.
    """

    #: Marker used by pools/sequences for import-cycle-safe type checks.
    __criterion__ = True

    #: Declare which levels this criterion supports.
    LEVELS: ClassVar[frozenset[CriterionLevel]]

    _REGISTER: dict = {}

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def ensure_compatible(self, level: CriterionLevel) -> None:
        """Raise :exc:`CriterionLevelError` if *level* not in :attr:`LEVELS`."""
        if level not in self.LEVELS:
            supported = ", ".join(
                lv.name for lv in sorted(self.LEVELS, key=lambda x: x.name)
            )
            raise CriterionLevelError(
                f"{type(self).__name__} does not support level {level.name!r}. "
                f"Supported: {supported}."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check(
        self,
        obj: object,
        caller: Literal["which", "entities", "match"],
    ) -> CriterionLevel:
        """Validate *obj* for *caller* and return the corresponding :class:`CriterionLevel`.

        +-------------+-------------------------------+------------+
        | caller      | accepted types                | level      |
        +=============+===============================+============+
        | ``which``   | SequencePool, TrajectoryPool  | inferred   |
        +-------------+-------------------------------+------------+
        | ``match``   | Sequence, Trajectory          | inferred   |
        +-------------+-------------------------------+------------+
        | ``entities``| Sequence, SequencePool        | ENTITY     |
        +-------------+-------------------------------+------------+

        Raises :exc:`TypeError` when *obj* is not the expected type.
        """
        if caller == "which":
            if isinstance(obj, TrajectoryPool):
                return CriterionLevel.TRAJECTORY
            if isinstance(obj, SequencePool):
                return CriterionLevel.SEQUENCE
            raise TypeError(
                f"which_ids() expects a SequencePool or TrajectoryPool, "
                f"got {type(obj).__name__}."
            )
        if caller == "match":
            if isinstance(obj, Trajectory):
                return CriterionLevel.TRAJECTORY
            if isinstance(obj, Sequence):
                return CriterionLevel.SEQUENCE
            raise TypeError(
                f"match() expects a Sequence or Trajectory, "
                f"got {type(obj).__name__}."
            )
        # caller == "entities"
        if not isinstance(obj, (Sequence, SequencePool)):
            raise TypeError(
                f"filter_entities() expects a Sequence or SequencePool, "
                f"got {type(obj).__name__}."
            )
        return CriterionLevel.ENTITY

    # ------------------------------------------------------------------
    # Verbose reporting helpers
    # ------------------------------------------------------------------

    def _report_which(self, n_before: int, n_after: int) -> None:
        """Emit a ``[which]`` report line."""
        pct = (n_after / n_before * 100) if n_before else 0.0
        tqdm.write(
            f"[which]           {type(self).__name__} "
            f"→ {n_after:,} / {n_before:,} IDs ({pct:.1f}%)"
        )

    @staticmethod
    def _entity_row_mask_attr(target: Sequence | SequencePool) -> str:
        """Return the attribute name of the mutable entity row mask for *target*."""
        if isinstance(target, SequencePool):
            return "_entity_row_mask"
        return "_own_entity_row_mask"

    # ------------------------------------------------------------------
    # Public API (template method)
    # ------------------------------------------------------------------

    def which_ids(
        self,
        pool: SequencePool | TrajectoryPool,
        *,
        verbose: bool = False,
    ) -> set:
        """Return the set of IDs in *pool* that satisfy this criterion.

        Args:
            pool: A :class:`~tanat.sequence.base.pool.SequencePool` or
                :class:`~tanat.trajectory.pool.TrajectoryPool`.
            verbose: If ``True``, emit a one-line report.

        Returns:
            Set of matching IDs.

        Raises:
            TypeError: If *pool* is not a supported pool type.
            CriterionLevelError: If the criterion is incompatible with the
                pool's level.
        """
        level = self._check(pool, "which")
        self.ensure_compatible(level)
        n_before = len(pool.unique_ids) if verbose else 0
        result = self._which_ids_impl(pool)
        if verbose:
            self._report_which(n_before, len(result))
        return result

    def filter_entities(
        self,
        target: Sequence | SequencePool,
        *,
        inplace: bool = False,
        verbose: bool = True,
    ):
        """Return a filtered view at entity level.

        Args:
            target: A :class:`~tanat.sequence.base.sequence.Sequence` or
                :class:`~tanat.sequence.base.pool.SequencePool`.
            inplace: If ``True``, modify *target* in place.
            verbose: If ``True``, emit a one-line report.

        Returns:
            Filtered *target* (or *target* itself when *inplace=True*).

        Raises:
            CriterionLevelError: If :attr:`ENTITY` is not in :attr:`LEVELS`.
            TypeError: If *target* is not a Sequence or SequencePool.
            CriterionError: If the criterion fails probing.
        """
        self.ensure_compatible(self._check(target, "entities"))

        # 1. Get current mask
        attr_name = self._entity_row_mask_attr(target)
        current_mask = getattr(target, attr_name, None)

        # 2. Compute the new mask (without modifying *target* yet)
        mask = self._compute_entity_mask(target)

        # 3. Verbose report
        if verbose:
            rows_before = current_mask.sum() if current_mask is not None else len(mask)
            rows_after = mask.sum()
            pct = (rows_after / rows_before * 100) if rows_before else 0.0

            if isinstance(target, SequencePool):
                id_col = target.settings.id_column
                # mask is store-space; align to view-space rows via __store_idx__
                df = (
                    target._id_time_index_lf(
                        with_store_index=True
                    )  # pylint: disable=protected-access
                    .with_columns(
                        pl.lit(mask).gather(pl.col("__store_idx__")).alias("mask")
                    )
                    .collect()
                )
                ids_before = df.select(pl.col(id_col).n_unique()).item()
                ids_after = (
                    df.filter(pl.col("mask")).select(pl.col(id_col).n_unique()).item()
                )
                ids_affected = ids_before - ids_after
                suffix = f" · {ids_affected:,} IDs affected"
            else:
                suffix = ""

            tqdm.write(
                f"[filter_entities] {type(self).__name__} "
                f"→ {rows_after:,} / {rows_before:,} entities ({pct:.1f}%){suffix}"
            )

        # 4. Copy target if needed
        if not inplace or (
            isinstance(target, Sequence)
            and target._parent_pool is not None  # pylint: disable=protected-access
        ):
            if inplace:
                warnings.warn(
                    "filter_entities(inplace=True) on a managed Sequence returns a detached copy. "
                    "Use pool.filter_entities(criterion) to filter in place.",
                    stacklevel=2,
                )
            target = target.copy()

        # 5. Apply the mask
        new_mask = mask if current_mask is None else (current_mask & mask)
        setattr(target, attr_name, new_mask)

        target.clear_cache()
        return target

    def match(self, target: Sequence | Trajectory) -> bool:
        """Return ``True`` if *target* satisfies this criterion.

        Args:
            target: A :class:`~tanat.sequence.base.sequence.Sequence` or
                :class:`~tanat.trajectory.trajectory.Trajectory`.

        Returns:
            ``True`` when *target* matches.

        Raises:
            TypeError: If *target* is not a Sequence or Trajectory.
            CriterionLevelError: If the criterion is incompatible with *target*'s level.
        """
        level = self._check(target, "match")
        self.ensure_compatible(level)
        return self._match_impl(target)

    # ------------------------------------------------------------------
    # Subtype hooks
    # ------------------------------------------------------------------
    #
    # Defaults raise NotImplementedError. Public methods call ``ensure_compatible``
    # first, so a hook is only reached when the level is in :attr:`LEVELS`. A
    # subclass therefore only needs to override the hooks for its supported levels.

    def _which_ids_impl(self, pool: SequencePool | TrajectoryPool) -> set:
        """Level-specific ID extraction. Called after compatibility check."""
        raise NotImplementedError

    def _match_impl(self, target: Sequence | Trajectory) -> bool:
        """Level-specific single-item match. Called after compatibility check."""
        raise NotImplementedError

    def _compute_entity_mask(self, target: Sequence | SequencePool) -> pl.Series:
        """Level-specific entity mask computation (returns a Polars Series)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_boolean(lf: pl.LazyFrame, expr: pl.Expr, *, kind: str) -> None:
        """Raise :class:`TypeError` if *expr* on *lf* does not produce a Boolean column.

        Args:
            lf: LazyFrame the expression will be evaluated on.
            expr: Polars expression to probe.
            kind: Human-readable label inserted in the error message (e.g. ``"Static"``).
        """
        schema = lf.select(expr.alias("__probe__")).collect_schema()
        if schema["__probe__"] != pl.Boolean:
            raise TypeError(
                f"{kind} expression must return a Boolean column, "
                f"got {schema['__probe__']} instead."
            )

    @staticmethod
    def _build_store_space_mask(n_store: int, kept_store_idx: pl.Series) -> pl.Series:
        """Reconstruct a full store-space boolean mask from surviving row indices.

        Args:
            n_store: Total number of entity rows in the physical store.
            kept_store_idx: ``__store_idx__`` values of rows that should be ``True``.

        Returns:
            A boolean :class:`~polars.Series` of length *n_store*.
        """
        arr = np.zeros(n_store, dtype=bool)
        if len(kept_store_idx) > 0:
            arr[kept_store_idx.to_numpy()] = True
        return pl.Series("", arr, dtype=pl.Boolean)
