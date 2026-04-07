#!/usr/bin/env python3
"""
Shared base settings for sequence visualization builders.
"""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass

from .literals import NaLabel, NaTimeIndex


@dataclass
class NullHandling:
    """How the pipeline handles null values before rendering.

    This is a **data-preparation** concern, not a visual one: the chosen
    strategies decide which rows survive into the chart, but they do not
    change any visual property.

    Attributes:
        na_time_index: Strategy for null values in time index columns
            (``__TIME__``, ``__START__``, ``__END__``).
            ``"drop"`` (default) removes affected rows with a
            :class:`UserWarning`; ``"raise"`` raises :exc:`ValueError`
            immediately.
        na_label: Strategy for null values in the entity feature label.
            ``"drop"`` (default) removes rows with a :class:`UserWarning`;
            ``"raise"`` raises :exc:`ValueError` immediately;
            ``"category"`` replaces nulls with the string ``"N/A"``,
            making missing values an explicit visible category.
    """

    na_time_index: NaTimeIndex = "drop"
    na_label: NaLabel = "drop"
