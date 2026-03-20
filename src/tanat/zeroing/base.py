#!/usr/bin/env python3
"""
T0Setter ABC + column constants shared across all zeroing strategies.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import replace as _dc_replace
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

import polars as pl
from tanat_utils import Registrable

if TYPE_CHECKING:
    from ..sequence.base.pool import SequencePool
    from ..sequence.base.sequence import Sequence

# ---------------------------------------------------------------------------
# Column name constants (used throughout the zeroing layer)
# ---------------------------------------------------------------------------

_T0: Final[str] = "_T0_"
_T0_NEAREST_RANK: Final[str] = "_T0_NEAREST_RANK_"

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

T0Value = datetime | date | int | float | None


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class T0Setter(ABC, Registrable):
    """Base class for all T0 strategies.

    Subclasses register themselves via ``register_name`` and live in the
    ``type/`` subdirectory so that :meth:`get_registered` can auto-discover
    them on first lookup.

    The setter is the single T0 attribute on the pool.  After ``set_t0()``
    runs, ``_df`` holds the pre-computed ``[id_col, _T0]`` DataFrame and
    ``anchor`` exposes the hint from settings.  The setter is set-once and
    never mutated again.
    """

    _REGISTER: dict = {}
    _TYPE_SUBMODULE = "type"

    def __init__(self, settings: Any) -> None:
        self._df: pl.DataFrame | None = None
        self.settings: Any = settings

    @property
    def df(self) -> pl.DataFrame | None:
        """Pre-computed ``[id_col, _T0]`` DataFrame.  ``None`` until ``set_t0()`` assigns it."""
        return self._df

    @property
    def anchor(self) -> Literal["start", "end", "middle"] | None:
        """Anchor hint from settings, or ``None`` if the strategy has no anchor field."""
        return getattr(self.settings, "anchor", None)

    @classmethod
    def default(cls, *, is_event: bool = False) -> T0Setter:
        """Return the default T0 strategy: :class:`PositionT0Setter` at position 0.

        Uses the registry so no direct import of the subclass is needed.
        The ``position`` type is auto-discovered on first call.

        Args:
            is_event: ``True`` when the target has a single temporal column.
                Used to pre-set ``anchor`` so that :meth:`_guard_anchor` does
                not emit a spurious warning on the implicit default.
        """
        anchor = None if is_event else "start"
        return cls.get_registered("position")(anchor=anchor)

    @abstractmethod
    def compute(self, target: SequencePool | Sequence) -> pl.DataFrame:
        """Compute T0 for a pool (all sequences) or a standalone sequence.

        Implementations **must** assign ``self._df`` before returning.
        Always returns a DataFrame with two columns:

        * ``id_col``: sequence identifier (same dtype as in the store).
        * ``_T0_``: T0 value (``null`` when no valid row found).

        ``_T0_NEAREST_RANK_`` is **not** produced here; it is a view-layer
        concern computed by the pool's ``_resolve_nearest_rank()``.

        When *target* is a :class:`~tanat.sequence.base.sequence.Sequence`
        the result has exactly one row.
        """

    # ── Anchor helpers ────────────────────────────────────────────────────

    @staticmethod
    def _t0_temporal_expr(
        anchor: Literal["start", "end", "middle"] | None,
        cols: list[str],
        is_datetime: bool = True,
    ) -> pl.Expr:
        """Return a Polars expression for the T0 reference timestamp.

        Args:
            anchor: ``"start"`` / ``None`` → start column; ``"end"`` → end
                    column; ``"middle"`` → midpoint (dtype-aware arithmetic).
            cols:   Temporal column names from
                    ``target.settings.get_temporal_columns()``.
                    Single-element list (event) → anchor is ignored.
            is_datetime: Flag from ``target.metadata.is_datetime``. Controls
                    the midpoint formula when ``anchor == "middle"``:

                    * ``True`` (default): ``start + (end - start) / 2``.
                      Required because Polars forbids adding two absolute
                      timestamps directly.
                    * ``False`` (numeric timestep): ``(start + end) / 2``.
        """
        if len(cols) == 1:
            return pl.col(cols[0])
        if anchor == "end":
            return pl.col(cols[1])
        if anchor == "middle":
            if is_datetime:
                # Datetime-safe: start + duration/2  (also safe default when dtype unknown)
                return pl.col(cols[0]) + (pl.col(cols[1]) - pl.col(cols[0])) / 2
            # Numeric path
            return (pl.col(cols[0]) + pl.col(cols[1])) / 2
        return pl.col(cols[0])  # "start" or None

    @staticmethod
    def normalize_anchor(
        anchor: Literal["start", "end", "middle"] | None,
        is_event: bool,
        pool_type: str = "",
        stacklevel: int = 3,
    ) -> Literal["start", "end", "middle"] | None:
        """Validate and normalise an anchor value against the pool type.

        Args:
            anchor: Raw anchor value to normalise.
            is_event: ``True`` when the pool has a single temporal column.
            pool_type: Registration name of the pool (used in the warning
                message).  Pass ``""`` when unknown.
            stacklevel: Passed to :func:`warnings.warn` so the warning points
                to the user's call site.  Caller must count frames:
                ``3`` for a direct ``t0_data`` call,
                ``5`` when called from ``_guard_anchor`` → ``compute`` →
                ``set_t0`` → user.
        """
        if is_event and anchor is not None:
            warnings.warn(
                f"anchor={anchor!r} has no effect on event pools "
                "(single temporal column). The argument will be ignored.",
                UserWarning,
                stacklevel=stacklevel,
            )
            return None
        if not is_event and anchor is None:
            type_hint = f" {pool_type}" if pool_type else ""
            warnings.warn(
                f"anchor= not specified for a{type_hint} pool: defaulting to 'start'. "
                "Pass anchor='start', 'end', or 'middle' explicitly to silence this warning.",
                UserWarning,
                stacklevel=stacklevel,
            )
            return "start"
        return anchor

    def _guard_anchor(self, target: SequencePool | Sequence) -> None:
        """Validate and normalise ``self.settings.anchor`` against *target*'s pool type.

        Delegates to :meth:`normalize_anchor` for the warning/defaulting
        logic, then writes the result back via :func:`dataclasses.replace`
        (safe for frozen pydantic dataclasses).

        Must be called at the top of :meth:`compute` by setters that expose an
        ``anchor`` field (``position``, ``direct``, ``query``).  Setters without
        an anchor field (e.g. ``feature``) skip this call.
        """
        pool_type = target.get_registration_name()
        is_event = pool_type == "event"
        anchor = T0Setter.normalize_anchor(
            self.settings.anchor,
            is_event=is_event,
            pool_type=pool_type,
            stacklevel=5,  # user → set_t0 → compute → _guard_anchor → normalize_anchor
        )
        self.settings = _dc_replace(self.settings, anchor=anchor)

    # ── Warning helper ─────────────────────────────────────────────────────

    def _warn_nulls(self, null_ids: list) -> None:
        """Emit a :exc:`UserWarning` for sequences that received ``_t0 = null``."""
        if null_ids:
            warnings.warn(
                f"{len(null_ids)} sequence(s) received _t0 = null (no valid row found): "
                f"{null_ids}",
                UserWarning,
                stacklevel=4,  # user → set_t0 → compute → _warn_nulls
            )
