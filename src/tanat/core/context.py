#!/usr/bin/env python3
"""
Workspace context.
"""

from pathlib import Path

from .workspace import Workspace

_active_ws = None  # pylint: disable=invalid-name
DEFAULT_PATH = Path.home() / ".tanat"


def get_workspace():
    """Get the active workspace instance, creating it if it doesn't exist."""
    global _active_ws  # pylint: disable=global-statement
    if _active_ws is None:
        _active_ws = Workspace(DEFAULT_PATH)
    return _active_ws


def set_active_ws_instance(instance):
    """Set the active workspace instance."""
    global _active_ws  # pylint: disable=global-statement
    _active_ws = instance
