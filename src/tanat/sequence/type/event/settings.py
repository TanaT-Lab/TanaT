#!/usr/bin/env python3
"""Event sequence settings."""

from __future__ import annotations

from tanat_utils import settings_dataclass as dataclass

from ...base.settings import SequenceSettings


@dataclass
class EventSequenceSettings(SequenceSettings):
    """Settings for event sequences (single timestamp column)."""

    id_column: str
    time_column: str

    def get_time_columns(self) -> list[str]:
        """Returns time index columns for Event sequences [time]."""
        return [self.time_column]
