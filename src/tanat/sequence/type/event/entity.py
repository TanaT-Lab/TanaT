#!/usr/bin/env python3
"""Event entity implementation."""

from __future__ import annotations

from ...base.entity import Entity


class EventEntity(Entity, register_name="event"):
    """Entity representing one event row (single timestamp)."""
