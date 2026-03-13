#!/usr/bin/env python3
"""
Main exceptions module.

All custom exceptions raised by the library inherit from :class:`TanaTException`.
"""

from __future__ import annotations


class TanaTException(Exception):
    """Root exception for all TanaT errors."""
