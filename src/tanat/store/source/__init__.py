#!/usr/bin/env python3
"""Data source package."""

from .base import AbstractSource

# Internal import subtypes to trigger registration via __init_subclass__.
from . import type as _register_subtypes  # pylint: disable=unused-import

__all__ = ["AbstractSource"]
