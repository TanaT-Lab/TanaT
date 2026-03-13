#!/usr/bin/env python3
"""
Spanplot settings.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from tanat_utils import settings_dataclass as dataclass

from ...base.literals import DisplayUnit, GroupBy, Orientation, SortOrder
from ....style.base import BaseVizSettings
from ....style.legend import LegendSettings

SpanKind = Literal["box", "violin", "strip"]


@dataclass
class SpanAesthetics:
    """Visual aesthetics for the spanplot builder.

    Attributes:
        group_by: Dimension to group distributions along.
            ``"category"``: one distribution per label value (default).
            ``"id"``: one distribution per sequence ID.
        kind: Chart style. ``"box"`` (default), ``"violin"``, or ``"strip"``.
        display_unit: Duration output unit for datetime-based pools.
            ``None`` keeps raw numeric values (correct for timestep sequences).
        sort: Sort order for groups.
            ``"ascending"``: ascending median duration (default).
            ``"descending"``: descending median duration.
            ``"alphabetic"``: alphabetical order of label / ID strings.
        orientation: Chart orientation.
            ``"vertical"``: groups on the x-axis, durations on y (default).
            ``"horizontal"``: groups on the y-axis, durations on x.
    """

    group_by: GroupBy = "category"
    kind: SpanKind = "box"
    display_unit: DisplayUnit | None = None
    sort: SortOrder = "ascending"
    orientation: Orientation = "vertical"


@dataclass
class SpanMarkerSettings:
    """Marker visual properties for the spanplot builder.

    Attributes:
        alpha: Global opacity (0–1).
        edge_color: Marker/box border color. ``None`` means no explicit border.
        point_size: Scatter point diameter in points (strip kind only).
        line_width: Box/violin outline thickness.
    """

    alpha: float = 0.75
    edge_color: str | None = None
    point_size: float = 4.0  # scatter point diameter (strip kind)
    line_width: float = 1.5  # box/violin outline thickness


@dataclass
class SpanplotSettings(BaseVizSettings):
    """Full settings for the spanplot builder."""

    # pylint: disable=invalid-field-call
    aesthetics: SpanAesthetics = field(default_factory=SpanAesthetics)
    marker: SpanMarkerSettings = field(default_factory=SpanMarkerSettings)
    # Override BaseVizSettings default: spanplots hide the legend by default.
    legend: LegendSettings = field(default_factory=lambda: LegendSettings(show=False))
    # Inherited from BaseVizSettings:
    #   title     : str | None
    #   colors    : str | dict | list | None
    #   figsize   : tuple[float, float]
    #   grid      : bool
    #   x_axis    : XAxisSettings
    #   y_axis    : YAxisSettings
