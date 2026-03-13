#!/usr/bin/env python3
"""
BaseVizSettings: common visualization settings shared by all chart types.
"""

from __future__ import annotations

from dataclasses import field

from tanat_utils import settings_dataclass as dataclass

from .axis import XAxisSettings, YAxisSettings
from .legend import LegendSettings


@dataclass
class BaseVizSettings:
    """Base settings shared by all visualization builders."""

    title: str | None = None
    colors: str | dict | list | None = None  # None -> matplotlib default color cycle
    figsize: tuple[float, float] = (10.0, 5.0)
    grid: bool = False
    # pylint: disable=invalid-field-call
    x_axis: XAxisSettings = field(default_factory=XAxisSettings)
    y_axis: YAxisSettings = field(default_factory=YAxisSettings)
    legend: LegendSettings = field(default_factory=LegendSettings)
