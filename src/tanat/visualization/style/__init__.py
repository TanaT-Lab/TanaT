#!/usr/bin/env python3
"""Style package."""

from .axis import XAxisSettings, YAxisSettings
from .base import BaseVizSettings
from .legend import LegendSettings

__all__ = [
    "BaseVizSettings",
    "LegendSettings",
    "XAxisSettings",
    "YAxisSettings",
]
