#!/usr/bin/env python3
"""
Style settings: legend configuration.
"""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass


@dataclass
class LegendSettings:
    """Legend display settings."""

    show: bool = True
    location: str = "best"
    title: str | None = None
