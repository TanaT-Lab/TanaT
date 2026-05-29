#!/usr/bin/env python3
"""
Criterion base: abstract class, level enum, and core exceptions.
"""

from __future__ import annotations

import enum
import warnings
from abc import ABC
from collections.abc import Callable
from typing import ClassVar, Literal

import polars as pl
from tqdm import tqdm
from tanat_utils import Registrable, SettingsMixin

from ..sequence.base.pool import SequencePool
from ..sequence.base.sequence import Sequence
from ..trajectory.pool import TrajectoryPool
from ..trajectory.trajectory import Trajectory
from ..exceptions import TanaTException
from ..store.sequence.schema import StoreSchema as SCH

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
    def _entity_filter_expr_attr(target: Sequence | SequencePool) -> str:
        """Return the attribute name of the mutable entity filter expression."""
        if isinstance(target, SequencePool):
            return "_entity_filter_expr"
        return "_own_entity_filter_expr"

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

        attr_name = self._entity_filter_expr_attr(target)
        current_expr = getattr(target, attr_name, None)
        expr = self._entity_filter_expr_impl(target)
        new_expr = expr if current_expr is None else (current_expr & expr)

        if verbose:
            pre_lf = target._frames.temporal(  # pylint: disable=protected-access
                with_store_index=True
            )
            post_lf = pre_lf.filter(expr)
            rows_before = pre_lf.select(pl.len()).collect().item()
            rows_after = post_lf.select(pl.len()).collect().item()
            pct = (rows_after / rows_before * 100) if rows_before else 0.0

            suffix = ""
            if isinstance(target, SequencePool):
                id_col = target.settings.id_column
                ids_before = pre_lf.select(pl.col(id_col).n_unique()).collect().item()
                ids_after = post_lf.select(pl.col(id_col).n_unique()).collect().item()
                suffix = f" · {ids_before - ids_after:,} IDs affected"

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

        setattr(target, attr_name, new_expr)

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

    def _entity_filter_expr_impl(self, target: Sequence | SequencePool) -> pl.Expr:
        """Return the entity-level predicate for *target*."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers for subclasses
    # ------------------------------------------------------------------

    @property
    def _store_index_col(self) -> str:
        """Column name for the absolute physical row index."""
        return SCH.STORE_INDEX

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

    def _materialise_as_store_idx_expr(
        self,
        target: Sequence | SequencePool,
        kept_predicate: Callable[[pl.LazyFrame], pl.LazyFrame],
        *,
        base_lf: pl.LazyFrame | None = None,
    ) -> pl.Expr:
        """Materialise once and return a stable store-index predicate.

        Convention used by criteria whose logic cannot be expressed as a
        cardinality-stable :class:`polars.Expr`. The criterion materialises
        the rows it wants to keep, then returns the equivalent stable
        expression ``pl.col(SCH.STORE_INDEX).is_in(kept)`` so the rest of
        the view pipeline can apply it like any other filter.

        Args:
            target: The view used to build the base LazyFrame when
                *base_lf* is not provided.
            kept_predicate: Callable that consumes the base LazyFrame and
                returns a filtered LazyFrame retaining the store index column.
            base_lf: Optional custom base LazyFrame. Must contain
                :attr:`_store_index_col`. Defaults to
                ``target._frames.id_time_index(with_store_index=True)``, which is
                enough for criteria that only need id/time columns.
        """
        idx_col = self._store_index_col
        if base_lf is None:
            base_lf = target._frames.id_time_index(  # pylint: disable=protected-access
                with_store_index=True
            )
        kept = kept_predicate(base_lf).select(idx_col).collect()[idx_col]
        return pl.col(idx_col).is_in(kept)
