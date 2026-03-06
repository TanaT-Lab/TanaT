#!/usr/bin/env python3
"""
User data access utils.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .zenodo import ZenodoAccessor

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def access(data_type: str, cache_dir: Path | None = None, force: bool = False) -> Any:
    """
    Access a dataset from Zenodo and return a ready-to-use object.

    Depending on the type, this may return:
    - a DataFrame (for file-based datasets: CSV, Parquet, etc.)
    - a database connection (for SQL-based datasets)

    Data is cached locally after the first download unless `force=True` is set.

    Args:
        data_type (str): Name of the dataset registered in ZenodoAccessor.
        cache_dir (str, optional): Directory used for caching. Defaults to system temp directory.
        force (bool): If True, forces re-download even if data is already cached.

    Returns:
        Any: A usable object for interacting with the dataset. The concrete type
        depends on the accessor implementation (e.g. a ``Path`` for SQL-based,
        datasets such as ``"mimic4"``, or a ``pandas.DataFrame`` for CSV-based)
        ones such as ``"mvad"``.

    Raises:
        ValueError: If ``data_type`` is not registered in the accessor.

    Examples:
        >>> # Access a MVAD CSV dataset as a DataFrame
        >>> df = access("mvad")

        >>> # Access the mimic4 SQLite database: returns Path to the .db file
        >>> db_path: Path = access("mimic4")
        >>> DB = f"sqlite:///{db_path}"  # SQLAlchemy-compatible URL
    """
    accessor = ZenodoAccessor.init(data_type, cache_dir=cache_dir)
    return accessor.get(force=force)
