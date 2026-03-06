#!/usr/bin/env python3
"""Interval entity implementation."""

from __future__ import annotations

from ...base.entity import Entity


class IntervalEntity(Entity, register_name="interval"):
    """Entity representing one interval row (start/end timestamps).

    Unlike :class:`StateEntity`, intervals are **not** required to be
    contiguous: gaps between intervals are allowed, and two intervals
    may overlap in time.
    """
