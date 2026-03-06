#!/usr/bin/env python3
"""State entity implementation."""

from __future__ import annotations

from ...base.entity import Entity


class StateEntity(Entity, register_name="state"):
    """Entity representing one state row (start/end).

    States are **contiguous and non-overlapping**: the end of one state
    is always the start of the next, with no gaps in between.
    """
