#!/usr/bin/env python3
"""Register ZenodoAccessor subtypes."""

from .mimic4 import Mimic4ZenodoAccessor
from .mooc_events import MOOCEventsZenodoAccessor
from .mvad import MvadZenodoAccessor
from .sentinel_health import SentinelHealthZenodoAccessor

__all__ = [
    "Mimic4ZenodoAccessor",
    "MOOCEventsZenodoAccessor",
    "MvadZenodoAccessor",
    "SentinelHealthZenodoAccessor",
]
