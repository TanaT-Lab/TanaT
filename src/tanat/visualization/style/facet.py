#!/usr/bin/env python3
"""
Style settings: facet (small-multiples) grid configuration.
"""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass


@dataclass
class FacetSettings:
    """Facet grid display settings.

    ``by=None`` means faceting is disabled (default).  Set via the builder's
    ``.facet()`` chainable method. Never instantiate directly.
    """

    by: str | None = None
    is_static: bool = False
    cols: int = 3
    share_x: bool = True
    share_y: bool = True
    figsize_per_facet: tuple[float, float] = (5.0, 4.0)
    title_template: str = "{by} = {value}"
