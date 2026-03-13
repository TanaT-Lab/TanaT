#!/usr/bin/env python3
"""
SequenceVisualizer: fluent entry point for sequence visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .type.barplot.builder import BarplotVizBuilder
from .type.barplot.settings import BarplotSettings

if TYPE_CHECKING:
    from .type.barplot.settings import DisplayUnit, Orientation, ShowAs, SortOrder


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
    ) -> BarplotVizBuilder:
        """Create a barplot builder.

        Args:
            show_as: What each bar represents: "count", "rate", or "duration".
            sort: Bar sort order: "alphabetic", "ascending", or "descending".
            orientation: "vertical" (default) or "horizontal".
            display_unit: Output unit for DURATION mode ("days", "hours",
                          "minutes", "seconds"). None keeps raw ms / raw timestep.
                          Only meaningful when show_as="duration" and the pool is datetime-based.

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
        return BarplotVizBuilder(settings=settings)
