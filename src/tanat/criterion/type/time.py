#!/usr/bin/env python3
"""
TimeCriterion: filter entities/sequences by temporal bounds.

Compatibility: ENTITY, SEQUENCE.
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar, Union

import polars as pl
from pydantic import model_validator
from tanat_utils import settings_dataclass as dataclass

from ..base import Criterion, CriterionLevel
from ...metadata.sequence import TimeIndexInfo
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from ...trajectory.pool import TrajectoryPool
from ...trajectory.trajectory import Trajectory

# Acceptable Python types for temporal bounds.
TimeBound = Union[dt.datetime, dt.date, int, float]

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(config={"arbitrary_types_allowed": True})
class TimeCriterionSettings:
    """Settings for :class:`TimeCriterion`.

    Temporal bounds are inclusive on both sides.  At least one of *start_ge*
    / *start_le* / *end_ge* / *end_le* must be provided.

    Args:
        start_ge: Minimum value for the start time column (inclusive).
        start_le: Maximum value for the start time column (inclusive).
        end_ge: Minimum value for the end time column (inclusive).
            Only applicable to interval/state sequences (two time columns).
        end_le: Maximum value for the end time column (inclusive).
            Only applicable to interval/state sequences (two time columns).
        duration_within: Entity-level (two time columns only).  When ``True``,
            the entity interval must be **fully contained** within the bounds
            (``start >= start_ge`` AND ``end <= end_le``).  When ``False``
            (default), any overlap is sufficient.  Ignored for single-column
            (event) sequences.
        all_entities: Sequence-level.  When ``True``, **every** entity row of
            the sequence must satisfy the time filter (sequence is fully
            within the window).  When ``False`` (default), at least one
            entity row must match.
    """

    start_ge: TimeBound | None = None
    start_le: TimeBound | None = None
    end_ge: TimeBound | None = None
    end_le: TimeBound | None = None
    duration_within: bool = False
    all_entities: bool = False

    @model_validator(mode="after")
    def _check_at_least_one_bound(self) -> TimeCriterionSettings:
        if all(
            v is None for v in [self.start_ge, self.start_le, self.end_ge, self.end_le]
        ):
            raise ValueError(
                "At least one bound (start_ge, start_le, end_ge, end_le) must be provided."
            )
        return self

    @model_validator(mode="after")
    def _check_bounds_homogeneous(self) -> TimeCriterionSettings:
        """All provided bounds must share the same Python type."""
        bounds = {
            name: v
            for name, v in [
                ("start_ge", self.start_ge),
                ("start_le", self.start_le),
                ("end_ge", self.end_ge),
                ("end_le", self.end_le),
            ]
            if v is not None
        }
        types = {type(v) for v in bounds.values()}
        # Allow datetime to coexist with date (datetime is a subclass of date).
        if dt.datetime in types and dt.date in types:
            types.discard(dt.date)
        if len(types) > 1:
            detail = ", ".join(f"{k}={type(v).__name__}" for k, v in bounds.items())
            raise TypeError(f"All bounds must have the same type, got: {detail}.")
        return self

    # ------------------------------------------------------------------
    # Sequence-metadata coherence helper
    # ------------------------------------------------------------------

    def validate_against(self, time_index: TimeIndexInfo) -> None:
        """Check that the bounds are compatible with *time_index*.

        Raises:
            TypeError: If the bound Python type is inconsistent with the
                sequence time index (e.g. numeric bound for a datetime index
                or vice-versa).
        """
        bounds = [
            v
            for v in (self.start_ge, self.start_le, self.end_ge, self.end_le)
            if v is not None
        ]
        if not bounds:
            return  # already caught by _check_at_least_one_bound

        # Determine what the bound type is (all same thanks to _check_bounds_homogeneous).
        sample = bounds[0]
        bound_is_datetime = isinstance(sample, (dt.datetime, dt.date))

        if time_index.is_datetime and not bound_is_datetime:
            raise TypeError(
                f"Sequence has a datetime/date time index ({time_index.dtype}) "
                f"but bounds are numeric ({type(sample).__name__}). "
                "Pass datetime or date objects as bounds."
            )
        if not time_index.is_datetime and bound_is_datetime:
            raise TypeError(
                f"Sequence has a numeric timestep time index ({time_index.dtype}) "
                f"but bounds are datetime-like ({type(sample).__name__}). "
                "Pass numeric values as bounds."
            )


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class TimeCriterion(Criterion):
    """Filter entities or select sequences by temporal position.

    Supported levels: **ENTITY**, **SEQUENCE**.

    Example::

        import datetime as dt
        from tanat.criterion import TimeCriterion

        t0 = dt.datetime(2020, 1, 1)
        t1 = dt.datetime(2021, 1, 1)

        # entity pruning: keep rows whose start time is in [t0, t1]
        pool2 = pool.filter_entities(TimeCriterion(start_ge=t0, start_le=t1))

        # entity pruning: interval must be fully contained in [t0, t1]
        pool3 = pool.filter_entities(
            TimeCriterion(start_ge=t0, end_le=t1, duration_within=True)
        )

        # sequence selection: IDs with at least one row in the window (default)
        ids = pool.which(TimeCriterion(start_ge=t0))

        # sequence selection: IDs where ALL rows are in the window
        ids = pool.which(TimeCriterion(start_ge=t0, end_le=t1, all_entities=True))

        # match
        ok = seq.match(TimeCriterion(start_le=t1))
    """

    SETTINGS_CLASS = TimeCriterionSettings
    LEVELS: ClassVar[frozenset[CriterionLevel]] = frozenset(
        {CriterionLevel.ENTITY, CriterionLevel.SEQUENCE}
    )

    def __init__(
        self,
        *,
        start_ge: TimeBound | None = None,
        start_le: TimeBound | None = None,
        end_ge: TimeBound | None = None,
        end_le: TimeBound | None = None,
        duration_within: bool = False,
        all_entities: bool = False,
    ) -> None:
        super().__init__(
            settings=TimeCriterionSettings(
                start_ge=start_ge,
                start_le=start_le,
                end_ge=end_ge,
                end_le=end_le,
                duration_within=duration_within,
                all_entities=all_entities,
            )
        )

    # ------------------------------------------------------------------
    # Impl hooks
    # ------------------------------------------------------------------

    def _entity_filter_expr_impl(self, target: Sequence | SequencePool) -> pl.Expr:
        self._settings.validate_against(target.metadata.time_index)
        start_col, end_col = self._resolve_time_cols(target)
        return self._build_entity_filter_expr(start_col, end_col)

    def _which_ids_impl(
        self,
        pool: SequencePool | TrajectoryPool,
    ) -> set:
        self._settings.validate_against(pool.metadata.time_index)
        id_col = pool.settings.id_column
        start_col, end_col = self._resolve_time_cols(pool)
        lf = pool._frames.id_time_index()  # pylint: disable=protected-access
        filter_expr = self._build_entity_filter_expr(start_col, end_col)

        if not self._settings.all_entities:
            # At least one row matches → keep any ID that has a surviving row.
            result = lf.filter(filter_expr).select(id_col).unique().collect()
            return set(result[id_col].to_list())

        # All rows must match → keep IDs where every row passes.
        result = (
            lf.with_columns(filter_expr.alias("__ok__"))
            .group_by(id_col)
            .agg(pl.col("__ok__").all())
            .filter(pl.col("__ok__"))
            .select(id_col)
            .collect()
        )
        return set(result[id_col].to_list())

    def _match_impl(self, target: Sequence | Trajectory) -> bool:
        """True if the sequence satisfies the time criterion.

        - ``all_entities=False`` (default): at least one entity row matches.
        - ``all_entities=True``: every entity row must match.
        """
        self._settings.validate_against(target.metadata.time_index)
        start_col, end_col = self._resolve_time_cols(target)
        lf = target._frames.id_time_index()  # pylint: disable=protected-access
        filter_expr = self._build_entity_filter_expr(start_col, end_col)

        if not self._settings.all_entities:
            return lf.filter(filter_expr).limit(1).collect().height > 0

        # All rows must pass: no row should fail.
        return lf.filter(~filter_expr).limit(1).collect().height == 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_time_cols(self, obj) -> tuple[str, str | None]:
        """Infer start / end column names from object settings."""
        time_cols = obj.settings.get_time_columns()
        return time_cols[0], (time_cols[1] if len(time_cols) > 1 else None)

    def _build_entity_filter_expr(self, start_col: str, end_col: str | None) -> pl.Expr:
        """Build a boolean row expression for entity-level filtering.

        Single time column (event)
        --------------------------
        Always uses direct bound comparisons: ``duration_within`` has no effect.

        Two time columns (interval/state), ``duration_within=False`` (default)
        -----------------------------------------------------------------------
        **Overlap** mode.  An entity ``[s, e]`` overlaps a window ``[lo, hi]`` iff
        ``s <= hi AND e >= lo``.  Provide ``start_ge=lo, end_le=hi`` to express the
        full window: this generates ``start <= hi AND end >= lo``.

        * ``start_ge`` → overlap lower bound → ``end >= start_ge``
        * ``end_le``   → overlap upper bound → ``start <= end_le``
        * ``start_le`` / ``end_ge`` → direct additional bounds (narrowing).

        Two time columns (interval/state), ``duration_within=True``
        ------------------------------------------------------------
        **Containment** mode.  The entity interval must be fully inside the window.
        Provide ``start_ge=lo, end_le=hi``: generates ``start >= lo AND end <= hi``.

        * ``start_ge`` → containment lower bound → ``start >= start_ge``
        * ``end_le``   → containment upper bound → ``end <= end_le``
        * ``start_le`` / ``end_ge`` → direct additional bounds (narrowing).
        """
        s = self._settings
        parts: list[pl.Expr] = []

        if end_col is None:
            # Single-column (event): direct bounds, duration_within irrelevant.
            if s.start_ge is not None:
                parts.append(pl.col(start_col) >= s.start_ge)
            if s.start_le is not None:
                parts.append(pl.col(start_col) <= s.start_le)
            if s.end_ge is not None or s.end_le is not None:
                raise ValueError(
                    "end_ge / end_le bounds require a two-column time index "
                    "(interval/state). This sequence has a single-column time index."
                )
        elif not s.duration_within:
            # Overlap: entity [s,e] overlaps window [lo, hi] iff s <= hi AND e >= lo.
            # start_ge=lo → end >= lo  |  end_le=hi → start <= hi
            # Null end means "state still ongoing" → treat as +∞ → always satisfies
            # end >= lo, so fill_null(True).
            if s.start_ge is not None:
                parts.append((pl.col(end_col) >= s.start_ge).fill_null(True))
            if s.end_le is not None:
                parts.append(pl.col(start_col) <= s.end_le)
            # Additional direct narrowing bounds.
            if s.start_le is not None:
                parts.append(pl.col(start_col) <= s.start_le)
            if s.end_ge is not None:
                parts.append((pl.col(end_col) >= s.end_ge).fill_null(True))
        else:
            # Containment: entity [s,e] fully inside window → s >= lo AND e <= hi.
            if s.start_ge is not None:
                parts.append(pl.col(start_col) >= s.start_ge)
            if s.end_le is not None:
                parts.append(pl.col(end_col) <= s.end_le)
            # Additional direct narrowing bounds.
            if s.start_le is not None:
                parts.append(pl.col(start_col) <= s.start_le)
            if s.end_ge is not None:
                parts.append(pl.col(end_col) >= s.end_ge)

        expr = parts[0]
        for p in parts[1:]:
            expr = expr & p
        return expr
