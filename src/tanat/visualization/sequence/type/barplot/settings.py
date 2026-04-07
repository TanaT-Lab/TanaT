#!/usr/bin/env python3
"""
Barplot settings.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from tanat_utils import settings_dataclass as dataclass

from ...base.literals import DisplayUnit, Orientation, SortOrder
from ...base.settings import NullHandling
from ....style.base import BaseVizSettings
from ....style.legend import LegendSettings

ShowAs = Literal["count", "rate", "duration"]


@dataclass
class BarAesthetics:
    """Visual aesthetics for the barplot.

    Attributes:
        show_as: What each bar represents. ``"count"`` (default), ``"rate"``, or ``"duration"``.
        sort: Bar sort order. ``"alphabetic"`` (default), ``"ascending"``, or ``"descending"``.
        orientation: ``"vertical"`` (default) or ``"horizontal"``.
        display_unit: Output unit for datetime-based pools when ``show_as="duration"``.
            ``None`` keeps raw timestep values (required for numeric timestep pools).
    """

    show_as: ShowAs = "count"
    sort: SortOrder = "alphabetic"
    orientation: Orientation = "vertical"
    display_unit: DisplayUnit | None = None


@dataclass
class BarMarkerSettings:
    """Bar marker visual properties.

    Attributes:
        alpha: Opacity (0.0 to 1.0).
        edge_color: Bar border color. ``None`` means no border.
        bar_width: Width fraction of the available slot (0 to 1).
    """

    alpha: float = 0.85
    edge_color: str | None = None
    bar_width: float = 0.8


@dataclass
class BarplotSettings(BaseVizSettings):
    """Full settings for the barplot builder."""

    # pylint: disable=invalid-field-call
    aesthetics: BarAesthetics = field(default_factory=BarAesthetics)
    null_handling: NullHandling = field(default_factory=NullHandling)
    marker: BarMarkerSettings = field(default_factory=BarMarkerSettings)
    # Override BaseVizSettings default: barplots hide the legend by default.
    legend: LegendSettings = field(default_factory=lambda: LegendSettings(show=False))
