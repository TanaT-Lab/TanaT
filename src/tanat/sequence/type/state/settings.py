#!/usr/bin/env python3
"""State sequence settings."""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass

from ...base.settings import SequenceSettings


@dataclass
class StateSequenceSettings(SequenceSettings):
    """Settings for state sequences (start + end timestamp columns).

    States are **contiguous and non-overlapping**: the end of one state
    is always the start of the next, with no gaps in between.
    """

    start_column: str
    end_column: str

    def get_time_columns(self) -> list[str]:
        """Returns time index columns for State sequences [start, end]."""
        return [self.start_column, self.end_column]
