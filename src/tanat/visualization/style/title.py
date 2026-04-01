#!/usr/bin/env python3
"""
Style settings: title configuration.
"""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass


@dataclass
class TitleSettings:
    """Title display settings."""

    text: str | None = None
    fontsize: int | None = None  # None -> matplotlib default
    fontweight: str = "normal"  # "normal", "bold", "light", …
    pad: float = 6.0  # spacing between title and top of axes/figure (points)
