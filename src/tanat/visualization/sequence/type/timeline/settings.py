#!/usr/bin/env python3
"""
Timeline settings.
"""

from __future__ import annotations

from dataclasses import field

from tanat_utils import settings_dataclass as dataclass

from ...base.literals import GroupBy, TimeMode
from ....style.axis import XAxisSettings, YAxisSettings
from ....style.base import BaseVizSettings


@dataclass
class TimelineAesthetics:
    """Visual aesthetics for the timeline.

    Attributes:
        group_by: Row organisation. ``"id"`` (default) for one row per sequence ID,
            or ``"category"`` for one row per unique label value.
        time_mode: ``"absolute"`` (default) displays real timestamps.
            ``"relative"`` would align all sequences to t=0 (not yet implemented).
    """

    group_by: GroupBy = "id"
    time_mode: TimeMode = "absolute"


@dataclass
class TimelineMarkerSettings:
    """Marker visual properties for the timeline.

    Attributes:
        size: Scatter point size (event pools).
        bar_height: Height fraction of each row slot (interval/state pools).
        alpha: Opacity of markers.
        edge_color: Marker edge color. ``None`` means no edge.
        shape: Any matplotlib marker string (e.g. "o", "s", "^").
            Validated by matplotlib at render time.
    """

    size: float = 4.0
    bar_height: float = 0.6
    alpha: float = 0.75
    edge_color: str | None = None
    shape: str = "o"


@dataclass
class TimelineSettings(BaseVizSettings):
    """Full settings for the timeline builder."""

    # pylint: disable=invalid-field-call
    aesthetics: TimelineAesthetics = field(default_factory=TimelineAesthetics)
    marker: TimelineMarkerSettings = field(default_factory=TimelineMarkerSettings)
    # Override BaseVizSettings defaults for timeline
    y_axis: YAxisSettings = field(
        default_factory=lambda: YAxisSettings(label=None, tick_rotation=0)
    )
    x_axis: XAxisSettings = field(
        default_factory=lambda: XAxisSettings(autofmt_xdate=True)
    )
