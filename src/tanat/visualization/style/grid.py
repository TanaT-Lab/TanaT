#!/usr/bin/env python3
"""
Style settings: grid configuration.
"""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass


@dataclass
class GridSettings:
    """Grid display settings."""

    show: bool = False
    color: str = "lightgrey"
    linewidth: float = 0.8
    axis: str = "both"  # "x", "y", or "both"
