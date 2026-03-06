#!/usr/bin/env python3
"""
Path resolution utilities
"""

from pathlib import Path

from .context import get_workspace


def resolve_path(path_or_name: str | Path) -> Path:
    """
    Resolve a path or workspace name to an absolute Path object.
    """
    if not isinstance(path_or_name, (str, Path)):
        raise TypeError(f"Expected a string or Path, got {type(path_or_name).__name__}")

    if isinstance(path_or_name, Path) or "/" in path_or_name or "\\" in path_or_name:
        return Path(path_or_name).expanduser().resolve()

    return get_workspace().get_store_path(path_or_name)
