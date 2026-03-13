#!/usr/bin/env python3
"""
Style settings: axis configuration.
"""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass


@dataclass
class XAxisSettings:
    """Horizontal axis display settings."""

    show: bool = True
    label: str | None = None
    tick_rotation: int = 45
    limit_min: float | None = None
    limit_max: float | None = None
    autofmt_xdate: bool = False


@dataclass
class YAxisSettings:
    """Vertical axis display settings."""

    show: bool = True
    label: str | None = None
    tick_rotation: int = 0
    limit_min: float | None = None
    limit_max: float | None = None
