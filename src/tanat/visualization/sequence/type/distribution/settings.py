#!/usr/bin/env python3
"""
DistributionVizBuilder settings.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from tanat_utils import settings_dataclass as dataclass

from ....style.base import BaseVizSettings
from ....style.legend import LegendSettings
from ...base.literals import DisplayUnit, TimeMode
from ...base.settings import NullHandling

# Supported aggregation modes for the distribution chart.
DistributionMode = Literal["count", "proportion", "percentage"]


@dataclass
class DistributionAesthetics:
    """Visual aesthetics for the distribution chart.

    Attributes:
        mode: What each stacked area represents.

            * ``"count"``: raw number of IDs occupying each state per bin.
            * ``"proportion"``: fraction of IDs in each state (sums to 1 per bin).
            * ``"percentage"``: same as proportion expressed as 0-100 (default).

        bin_size: Width of each time bin.

            * **Datetime pools**: a Polars duration string such as ``"1d"``,
              ``"12h"``, ``"1w"``, ``"1mo"``.
            * **Timestep pools**: a numeric step value (``int`` or ``float``).

        stacked: When ``True`` (default) render a stacked area chart.
            When ``False`` render overlapping transparent fills, one per label.
        time_mode: ``"absolute"`` (default) uses real timestamps on the x-axis.
            ``"relative"`` aligns all sequences to their per-ID T0 reference date;
            the x-axis shows a numeric offset from T0.  Requires :meth:`set_t0` to
            have been called (or uses the lazy default of ``position=0``).
        display_unit: Target unit for the x-axis when ``time_mode="relative"`` and
            the pool is datetime-based.  One of ``"days"`` (default), ``"hours"``,
            ``"minutes"``, or ``"seconds"``.  Ignored for timestep pools (a
            :class:`UserWarning` is emitted if explicitly set).
    """

    mode: DistributionMode = "percentage"
    bin_size: str | int | float = "1d"
    stacked: bool = True
    time_mode: TimeMode = "absolute"
    display_unit: DisplayUnit | None = None


@dataclass
class DistributionMarkerSettings:
    """Marker visual properties for the distribution chart.

    Attributes:
        alpha: Opacity of the filled areas (0-1).
        line_width: Width of the area boundary line.
    """

    alpha: float = 0.7
    line_width: float = 0.8


@dataclass
class DistributionSettings(BaseVizSettings):
    """Complete settings bundle for :class:`DistributionVizBuilder`.

    Attributes:
        aesthetics: High-level visual choices (mode, bin_size, stacked, time_mode).
        marker: Low-level fill/line properties.
        legend: Legend visibility and placement. Shown by default.
    """

    # pylint: disable=invalid-field-call
    aesthetics: DistributionAesthetics = field(default_factory=DistributionAesthetics)
    null_handling: NullHandling = field(default_factory=NullHandling)
    marker: DistributionMarkerSettings = field(
        default_factory=DistributionMarkerSettings
    )
    legend: LegendSettings = field(default_factory=lambda: LegendSettings(show=True))
