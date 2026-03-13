#!/usr/bin/env python3
"""
Spanplot visualization exceptions.
"""

from __future__ import annotations

from .....exceptions import TanaTException


class UnsupportedSequenceTypeError(TanaTException, ValueError):
    """Raised when the input sequence type has no duration concept."""

    def __init__(self, found_type: str) -> None:
        super().__init__(
            f"SpanplotVizBuilder requires a state or interval sequence, "
            f"got {found_type!r}. Duration is undefined for {found_type!r} sequences."
        )
