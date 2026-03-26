#!/usr/bin/env python3
"""Interval sequence settings."""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass

from ...base.settings import SequenceSettings


@dataclass
class IntervalSequenceSettings(SequenceSettings):
    """Settings for interval sequences (start + end timestamp columns).

    Unlike state sequences, intervals are **not** required to be
    contiguous: gaps between intervals are allowed, and two intervals
    may overlap in time.
    """

    id_column: str
    start_column: str
    end_column: str

    def get_time_columns(self) -> list[str]:
        """Returns time index columns for Interval sequences [start, end]."""
        return [self.start_column, self.end_column]
