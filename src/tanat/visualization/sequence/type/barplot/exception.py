#!/usr/bin/env python3
"""
Barplot visualization exceptions.
"""

from __future__ import annotations

from .....exceptions import TanaTException


class UnsupportedShowAsError(TanaTException, ValueError):
    """Raised when show_as='duration' is used with a pool type that has no duration."""

    def __init__(self, show_as: str, pool_type: str) -> None:
        super().__init__(
            f"show_as={show_as!r} is not supported for {pool_type}. "
            "'duration' requires an IntervalSequencePool or StateSequencePool."
        )
