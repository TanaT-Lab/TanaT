#!/usr/bin/env python3
"""
Barplot settings.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from tanat_utils import settings_dataclass as dataclass

from ....style.base import BaseVizSettings
from ....style.legend import LegendSettings

DisplayUnit = Literal["days", "hours", "minutes", "seconds"]
ShowAs = Literal["count", "rate", "duration"]
SortOrder = Literal["alphabetic", "ascending", "descending"]
Orientation = Literal["vertical", "horizontal"]


@dataclass
class BarAesthetics:
    """Visual aesthetics for the barplot."""

    show_as: ShowAs = "count"
    sort: SortOrder = "alphabetic"
    orientation: Orientation = "vertical"
    display_unit: DisplayUnit | None = None
    # display_unit: only meaningful when show_as=duration + datetime pool.
    # None -> raw ms for datetime, raw value for numeric timestep.


@dataclass
class BarMarkerSettings:
    """Bar marker visual properties."""

    alpha: float = 0.85
    edge_color: str | None = None
    bar_width: float = 0.8  # fraction of available slot (0-1)


@dataclass
class BarplotSettings(BaseVizSettings):
    """Full settings for the barplot builder."""

    # pylint: disable=invalid-field-call
    aesthetics: BarAesthetics = field(default_factory=BarAesthetics)
    marker: BarMarkerSettings = field(default_factory=BarMarkerSettings)
    # Override BaseVizSettings default: barplots hide the legend by default.
    legend: LegendSettings = field(default_factory=lambda: LegendSettings(show=False))
