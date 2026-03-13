#!/usr/bin/env python3
"""
SequenceVisualizer: fluent entry point for sequence visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .type.barplot.builder import BarplotVizBuilder
from .type.barplot.settings import BarplotSettings
from .type.spanplot.builder import SpanplotVizBuilder
from .type.spanplot.settings import SpanplotSettings
from .type.timeline.builder import TimelineVizBuilder
from .type.timeline.settings import TimelineSettings

if TYPE_CHECKING:
    from .base.literals import DisplayUnit, GroupBy, Orientation, SortOrder
    from .type.barplot.settings import ShowAs
    from .type.spanplot.settings import SpanKind
    from .type.timeline.settings import TimeMode


class SequenceVisualizer:
    """Factory for sequence visualization builders.

    All class methods return a configured builder ready for chaining.

    Example::

        SequenceVisualizer.barplot(show_as="rate") \\
            .title("Relative frequencies") \\
            .draw(pool) \\
            .show()
    """

    @classmethod
    def barplot(
        cls,
        *,
        show_as: ShowAs = "count",
        sort: SortOrder = "alphabetic",
        orientation: Orientation = "vertical",
        display_unit: DisplayUnit | None = None,
        allow_large: bool = False,
    ) -> BarplotVizBuilder:
        """Create a barplot builder.

        Args:
            show_as: What each bar represents: "count", "rate", or "duration".
            sort: Bar sort order: "alphabetic", "ascending", or "descending".
            orientation: "vertical" (default) or "horizontal".
            display_unit: Output unit for DURATION mode ("days", "hours",
                          "minutes", "seconds"). None keeps raw ms / raw timestep.
                          Only meaningful when show_as="duration" and the pool is datetime-based.
            allow_large: Bypass the :attr:`~BaseSequenceVizBuilder.MAX_CATEGORY` safety guard.

        Returns:
            A configured :class:`~tanat.visualization.sequence.type.barplot.builder.BarplotVizBuilder`.
        """
        settings = BarplotSettings(
            aesthetics={
                "show_as": show_as,
                "sort": sort,
                "orientation": orientation,
                "display_unit": display_unit,
            }
        )
        return BarplotVizBuilder(settings=settings, allow_large=allow_large)

    @classmethod
    def timeline(
        cls,
        *,
        group_by: GroupBy = "id",
        time_mode: TimeMode = "absolute",
        allow_large: bool = False,
    ) -> TimelineVizBuilder:
        """Create a timeline builder.

        Args:
            group_by: Row organisation:

                * ``"id"``: one row per sequence ID (default).
                * ``"category"``: one row per unique label value.

            time_mode: ``"absolute"`` (real timestamps as-is) or ``"relative"``
                (all sequences aligned to t=0). Note: ``"relative"`` is accepted
                by the settings layer but raises :exc:`NotImplementedError` at
                ``prepare_data`` time until the alignment logic is implemented.
            allow_large: Bypass the :attr:`~TimelineVizBuilder.MAX_MARKERS` and
                :attr:`~TimelineVizBuilder.MAX_IDS` safety guards.

        Returns:
            A configured :class:`~tanat.visualization.sequence.type.timeline.builder.TimelineVizBuilder`.
        """
        settings = TimelineSettings(
            aesthetics={
                "group_by": group_by,
                "time_mode": time_mode,
            }
        )
        return TimelineVizBuilder(settings=settings, allow_large=allow_large)

    @classmethod
    def spanplot(
        cls,
        *,
        group_by: GroupBy = "category",
        kind: SpanKind = "box",
        display_unit: DisplayUnit | None = None,
        sort: SortOrder = "ascending",
        orientation: Orientation = "vertical",
        allow_large: bool = False,
    ) -> SpanplotVizBuilder:
        """Create a spanplot (duration distribution) builder.

        Only compatible with **state** and **interval** sequence types. Duration
        is undefined for event sequences; passing one raises
        :exc:`~tanat.visualization.sequence.type.spanplot.exception.UnsupportedSequenceTypeError`.

        Args:
            group_by: Grouping dimension:

                * ``"category"``: one column per unique label value (default).
                * ``"id"``: one column per sequence ID.

            kind: Chart variety:

                * ``"box"``: standard box-and-whisker (default).
                * ``"violin"``: kernel-density violin.
                * ``"strip"``: individual points with horizontal jitter.

            display_unit: Output unit for datetime-based pools: ``"days"``,
                ``"hours"``, ``"minutes"``, or ``"seconds"``.  Required when
                the pool is datetime-based; must be ``None`` for numeric timestep
                pools.
            sort: Sort order for x-axis groups:

                * ``"ascending"``: ascending median duration (default).
                * ``"descending"``: descending median duration.
                * ``"alphabetic"``: alphabetical order of labels or IDs.

            orientation: Chart orientation:

                * ``"vertical"``: groups on the x-axis, durations on y (default).
                * ``"horizontal"``: groups on the y-axis, durations on x.

            allow_large: Bypass the
                :attr:`~SpanplotVizBuilder.MAX_CATEGORY` /
                :attr:`~SpanplotVizBuilder.MAX_IDS` safety guards.

        Returns:
            A configured :class:`~tanat.visualization.sequence.type.spanplot.builder.SpanplotVizBuilder`.
        """
        settings = SpanplotSettings(
            aesthetics={
                "group_by": group_by,
                "kind": kind,
                "display_unit": display_unit,
                "sort": sort,
                "orientation": orientation,
            }
        )
        return SpanplotVizBuilder(settings=settings, allow_large=allow_large)
