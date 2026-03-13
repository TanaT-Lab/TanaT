#!/usr/bin/env python3
"""
Visualization exceptions.
"""

from __future__ import annotations

from tanat.exceptions import TanaTException


class UnsupportedShowAsError(TanaTException, ValueError):
    """Raised when show_as='duration' is used with a pool type that has no duration."""

    def __init__(self, show_as: str, pool_type: str) -> None:
        super().__init__(
            f"show_as={show_as!r} is not supported for {pool_type}. "
            "'duration' requires an IntervalSequencePool or StateSequencePool."
        )


class IncompatibleDisplayUnitError(TanaTException, ValueError):
    """Raised when display_unit is incompatible with the pool's temporal index."""

    def __init__(self, display_unit: str | None, is_datetime: bool) -> None:
        if is_datetime and display_unit is None:
            msg = (
                "display_unit is required when show_as='duration' on a datetime sequence. "
                "Choose one of: 'days', 'hours', 'minutes', 'seconds'."
            )
        else:
            msg = (
                f"display_unit={display_unit!r} cannot be used on a numeric timestep sequence. "
                "Set display_unit=None to aggregate in raw timestep values."
            )
        super().__init__(msg)
