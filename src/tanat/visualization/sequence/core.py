#!/usr/bin/env python3
"""
SequenceVisualizer: fluent entry point for sequence visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .type.barplot.builder import BarplotVizBuilder
from .type.barplot.settings import BarplotSettings
from .type.timeline.builder import TimelineVizBuilder
from .type.timeline.settings import TimelineSettings

if TYPE_CHECKING:
    from .base.literals import DisplayUnit, GroupBy, Orientation, SortOrder
    from .type.barplot.settings import ShowAs
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
