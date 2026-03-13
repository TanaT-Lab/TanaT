#!/usr/bin/env python3
"""
Shared visualization exceptions for sequence builders.
"""

from __future__ import annotations

from ....exceptions import TanaTException


class IncompatibleDisplayUnitError(TanaTException, ValueError):
    """Raised when display_unit is incompatible with the pool's temporal index.

    Two incompatible situations:

    - A datetime pool requires an explicit ``display_unit`` to convert raw
      millisecond durations to a human-readable unit.
    - A numeric timestep pool does not support ``display_unit``; durations
      are already in raw timestep values.
    """

    def __init__(self, display_unit: str | None, *, is_datetime: bool) -> None:
        if is_datetime and display_unit is None:
            msg = (
                "display_unit is required for a datetime sequence. "
                "Choose one of: 'days', 'hours', 'minutes', 'seconds'."
            )
        else:
            msg = (
                f"display_unit={display_unit!r} cannot be used on a numeric "
                "timestep sequence. Set display_unit=None to use raw timestep values."
            )
        super().__init__(msg)
