#!/usr/bin/env python3
"""
PatternCriterion: filter sequences by an ordered pattern of string values.

Compatibility: ENTITY, SEQUENCE.
"""

from __future__ import annotations

import re
from typing import ClassVar

import polars as pl
from pydantic import field_validator
from tanat_utils import settings_dataclass as dataclass

from ..base import Criterion, CriterionLevel
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from ...trajectory.pool import TrajectoryPool
from ...trajectory.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

#: Matches **exactly one** non-null value at that position (any value).
WILDCARD = "*"

#: Matches **zero or more** elements — free gap between adjacent segments.
ANY = "..."

# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

# A compiled segment: WILDCARD kept as str, others as re.Pattern or literal str.
_Segment = list[str | re.Pattern[str]]


def _build_segments(
    pattern: list[str], *, regex: bool, case_sensitive: bool
) -> list[_Segment]:
    """Split *pattern* on :data:`ANY` and compile each element once.

    Called once at construction time; results are cached on the instance.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    segments: list[_Segment] = []
    current: _Segment = []
    for p in pattern:
        if p == ANY:
            if current:
                segments.append(current)
            current = []
        elif p == WILDCARD:
            current.append(WILDCARD)
        elif regex:
            current.append(re.compile(p, flags))
        else:
            # literal: lowercase once here if case-insensitive
            current.append(p if case_sensitive else p.lower())
    if current:
        segments.append(current)
    return segments


def _matches_element(
    val: str, pattern: str | re.Pattern[str], *, case_sensitive: bool
) -> bool:
    """Return ``True`` if *val* satisfies *pattern* (sentinel, compiled regex, or literal)."""
    if pattern == WILDCARD:
        return True
    if isinstance(pattern, re.Pattern):
        return bool(pattern.search(val))
    # literal substring (already lowercased at build time if case-insensitive)
    return (pattern in val) if case_sensitive else (pattern in val.lower())


def _match_segment_at(
    values: list[str | None],
    pos: int,
    segment: _Segment,
    *,
    case_sensitive: bool,
) -> bool:
    """Return ``True`` if *values[pos:pos+len(segment)]* matches *segment* adjacently."""
    if pos + len(segment) > len(values):
        return False
    return all(
        values[pos + i] is not None
        and _matches_element(
            values[pos + i],  # type: ignore[arg-type]
            segment[i],
            case_sensitive=case_sensitive,
        )
        for i in range(len(segment))
    )


def _find_ranks(
    values: list[str | None],
    segments: list[_Segment],
    *,
    case_sensitive: bool,
) -> list[int]:
    """Return 0-based witness indices for pre-compiled *segments*.

    Segments are matched adjacently; gaps between them are free.
    Returns ``[]`` if the full pattern is not found.
    """
    pos = 0
    matched: list[int] = []
    for segment in segments:
        found = False
        for start in range(pos, len(values) - len(segment) + 1):
            if _match_segment_at(values, start, segment, case_sensitive=case_sensitive):
                matched.extend(range(start, start + len(segment)))
                pos = start + len(segment)
                found = True
                break
        if not found:
            return []
    return matched


def _has_match(
    values: list[str | None], segments: list[_Segment], *, case_sensitive: bool
) -> bool:
    """Return ``True`` if *values* matches pre-compiled *segments*."""
    return bool(_find_ranks(values, segments, case_sensitive=case_sensitive))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class PatternCriterionSettings:
    """Settings for :class:`PatternCriterion`.

    Args:
        feature: Name of the string feature to match against.
        pattern: Ordered pattern to match against the feature column.

            Elements are matched **adjacently** (consecutively) within each
            segment.  Use :data:`ANY` (``"..."``) to introduce a free gap of
            any length between segments:

            * ``["A", "B"]`` — A **directly** followed by B.
            * ``["A", WILDCARD, "B"]`` — A, then exactly one element, then B.
            * ``["A", ANY, "B"]`` — A before B with any number of rows in between.
            * ``["A", ANY, "B", "C"]`` — A anywhere, then B directly followed by C.

            A plain ``str`` is shorthand for a single-element list.

        present: When ``True`` (default), the pattern must be present:

            * **Sequence level**: sequences that contain the pattern are selected.
            * **Entity level**: the "witness" rows of the greedy first match are kept.

            When ``False``, the pattern must be absent:

            * **Sequence level**: sequences that do **not** contain the pattern are selected.
            * **Entity level**: all rows that are **not** witnesses are kept
              (rows that don't participate in the pattern match).

        regex: If ``True`` (default), each non-sentinel element of *pattern*
            is treated as a regular expression.  Set to ``False`` for literal
            substring matching.  :data:`WILDCARD` and :data:`ANY` always
            behave as sentinels regardless of this flag.
        case_sensitive: Case-sensitive matching (default: ``True``).
    """

    feature: str
    pattern: str | list[str]
    present: bool = True
    regex: bool = True
    case_sensitive: bool = True

    @field_validator("pattern", mode="before")
    @classmethod
    def _normalise_pattern(cls, v: str | list[str]) -> list[str]:
        """Coerce a bare string into a single-element list."""
        return [v] if isinstance(v, str) else list(v)


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class PatternCriterion(Criterion):
    """Filter entities or sequences by an ordered pattern of string values.

    A sequence matches when its entities (in temporal order) contain the
    given pattern as an **ordered sub-sequence**: pattern element ``k`` must
    appear *after* element ``k-1`` in the sequence.

    Supported levels: **ENTITY**, **SEQUENCE**.

    **Entity level** (``filter_entities``):
        * ``present=True``: keeps only the rows that are "witnesses" of the
          greedy first match. Sequences without a complete match → 0 rows.
        * ``present=False``: keeps all rows that are **not** witnesses
          (rows that don't participate in the pattern). Sequences without a
          complete match → all their rows are kept.

    **Sequence level** (``which``, ``match``):
        Keeps (or excludes, with ``present=False``) whole sequences based on
        whether the ordered pattern is found.

    Example::

        # IDs where "A" appears directly before "B" (adjacent)
        ids = pool.which(PatternCriterion(feature="code", pattern=["A", "B"]))

        # Entity pruning: keep only the matched witness rows
        pool2 = pool.filter_entities(
            PatternCriterion(feature="code", pattern=["A", "B"])
        )

        # Free gap: A before B with any rows in between
        ids = pool.which(
            PatternCriterion(feature="code", pattern=["A", ANY, "B"])
        )

        # Exactly one element between A and B
        ids = pool.which(
            PatternCriterion(feature="code", pattern=["A", WILDCARD, "B"])
        )

        # Single-element pattern: at least one row matching "ICU"
        ids = pool.which(PatternCriterion(feature="label", pattern="ICU"))

        # Exclusion: sequences that never contain adjacent A→B
        ids = pool.which(
            PatternCriterion(feature="code", pattern=["A", "B"], present=False)
        )

        # Literal, case-insensitive
        ids = pool.which(
            PatternCriterion(feature="code", pattern="icu", regex=False, case_sensitive=False)
        )

        # Single-sequence match
        ok = seq.match(PatternCriterion(feature="code", pattern=["A", "B"]))
    """

    SETTINGS_CLASS = PatternCriterionSettings
    LEVELS: ClassVar[frozenset[CriterionLevel]] = frozenset(
        {CriterionLevel.ENTITY, CriterionLevel.SEQUENCE}
    )

    def __init__(
        self,
        feature: str,
        pattern: str | list[str],
        present: bool = True,
        regex: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        super().__init__(
            settings=PatternCriterionSettings(
                feature=feature,
                pattern=pattern,
                present=present,
                regex=regex,
                case_sensitive=case_sensitive,
            )
        )
        s = self._settings
        # Pre-compile once: split on ANY + compile regex patterns.
        self._segments: list[_Segment] = _build_segments(
            s.pattern, regex=s.regex, case_sensitive=s.case_sensitive
        )

    # ------------------------------------------------------------------
    # Impl hooks
    # ------------------------------------------------------------------

    def _which_ids_impl(
        self,
        pool: SequencePool | TrajectoryPool,
    ) -> set:
        s = self._settings
        id_col = pool.settings.id_column
        lf = pool._frames.temporal(  # pylint: disable=protected-access
            features=[s.feature]
        )
        result = (
            lf.group_by(id_col)
            .agg(pl.col(s.feature).alias("__vals__"))
            .filter(
                pl.col("__vals__").map_elements(
                    lambda vals: _has_match(
                        vals.to_list(),
                        self._segments,
                        case_sensitive=s.case_sensitive,
                    )
                    == s.present,
                    return_dtype=pl.Boolean,
                )
            )
            .select(id_col)
            .collect()
        )
        return set(result[id_col].to_list())

    def _kept_rows_lf(
        self,
        target: Sequence | SequencePool,
        lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Filter *lf* down to the rows kept by the pattern witness logic.

        *lf* must contain the feature column and ``__store_idx__`` — typically
        ``target._frames.temporal(features=[feature], with_store_index=True)``.
        """
        s = self._settings
        id_col = target.settings.id_column
        df = lf.with_columns(
            pl.int_range(pl.len()).over(id_col).alias("__rank__")
        ).collect()

        matched_lists = (
            df.group_by(id_col, maintain_order=True)
            .agg(pl.col(s.feature))
            .with_columns(
                pl.col(s.feature)
                .map_elements(
                    lambda feats: _find_ranks(
                        feats,
                        self._segments,
                        case_sensitive=s.case_sensitive,
                    ),
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("__matched_ranks__")
            )
        )

        df = df.join(
            matched_lists.select(id_col, "__matched_ranks__"), on=id_col, how="left"
        )
        row_matches = df.select(
            pl.col("__matched_ranks__")
            .list.contains(pl.col("__rank__"))
            .fill_null(False)
            .alias("__in__")
        )["__in__"]
        if not s.present:
            row_matches = ~row_matches

        return df.filter(row_matches).lazy()

    def _entity_filter_expr_impl(self, target: Sequence | SequencePool) -> pl.Expr:
        base_lf = target._frames.temporal(  # pylint: disable=protected-access
            features=[self._settings.feature], with_store_index=True
        )
        return self._materialise_as_store_idx_expr(
            target,
            lambda lf: self._kept_rows_lf(target, lf),
            base_lf=base_lf,
        )

    def _match_impl(self, target: Sequence | Trajectory) -> bool:
        """True if the sequence contains the ordered pattern (respecting ``present``)."""
        s = self._settings
        vals = (
            target._frames.temporal(  # pylint: disable=protected-access
                features=[s.feature]
            )
            .collect()[s.feature]
            .to_list()
        )
        found = _has_match(vals, self._segments, case_sensitive=s.case_sensitive)
        return found == s.present
