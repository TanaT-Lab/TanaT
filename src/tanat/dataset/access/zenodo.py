#!/usr/bin/env python3
"""
Zenodo dataset accessor.
"""

from __future__ import annotations

import json
import logging
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm
from tanat_utils import Registrable

if TYPE_CHECKING:
    from typing import Any

LOGGER = logging.getLogger(__name__)


class ZenodoAccessor(ABC, Registrable):
    """
    Zenodo dataset accessor.
    """

    _REGISTER = {}
    _TYPE_SUBMODULE = "./type"

    _DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "zenodo_datasets"

    def __init__(
        self,
        record_id: int | str,
        filename: str,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Initialize ZenodoAccessor.

        Args:
            record_id: Zenodo record ID.
            filename: Name of the file to download.
            cache_dir: Cache directory. Defaults to the system temp directory.
        """
        self._record_id = record_id
        self._filename = filename
        self._url = (
            f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
        )
        self._api_url = f"https://zenodo.org/api/records/{record_id}/files/{filename}"

        self._cache_dir = cache_dir
        self._expected_size = None

    @property
    def local_path(self) -> Path:
        """Local path to the cached file."""
        return self.cache_dir / f"{self._filename}"

    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        cache_dir = self._cache_dir
        if cache_dir is None:
            self._cache_dir = self._DEFAULT_CACHE_DIR

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    @property
    def expected_size(self) -> int:
        """Get expected file size from Zenodo API."""
        if self._expected_size is not None:
            return self._expected_size

        with urllib.request.urlopen(self._api_url) as response:
            metadata = json.loads(response.read().decode("utf-8"))
            return metadata.get("size", 0)

    def _is_file_valid(self) -> bool:
        """Check if cached file exists and has correct size."""
        if not self.local_path.exists():
            return False

        expected_size = self.expected_size
        if expected_size == 0:
            return True  # Skip validation if size unknown

        actual_size = self.local_path.stat().st_size
        return actual_size == expected_size

    def download(self, force: bool = False) -> Path:
        """
        Download file from Zenodo if not cached or invalid.

        Args:
            force: Force download even if file exists.

        Returns:
            Path to the downloaded file.
        """
        if not force and self._is_file_valid():
            LOGGER.info("Using cached file: %s", self.local_path)
            return self.local_path

        LOGGER.info("Downloading %s...", self._filename)

        with urllib.request.urlopen(self._url) as response:
            total_size = self.expected_size
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=self._filename
            ) as pbar:
                with open(self.local_path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify download
        if not self._is_file_valid():
            self.local_path.unlink()
            raise ValueError(
                f"Downloaded file size mismatch for {self._filename}. Please try again."
            )

        LOGGER.info("Downloaded to %s", self.local_path)
        return self.local_path

    def get(self, force: bool = False) -> Any:
        """
        Download and give access to data from Zenodo dataset.

        Args:
            force: Force download even if file exists.

        Returns:
            Any: The object returned by :meth:`_access_impl`.
            depends on the subclass (e.g. ``Path``, ``pandas.DataFrame``).
        """
        self.download(force)
        return self._access_impl()

    @abstractmethod
    def _access_impl(self) -> Any:
        """
        Implementation of the data access logic.

        Returns:
            Any: The concrete type is defined by each subclass.
        """

    @classmethod
    def init(cls, accessor_type: str, cache_dir: Path | None = None) -> ZenodoAccessor:
        """
        Initialize a Zenodo accessor class dynamically.

        Args:
            accessor_type: Registered name of the accessor to create.
            cache_dir: Cache directory. Defaults to the system temp directory.

        Returns:
            ZenodoAccessor: Instance of the requested accessor subclass.

        Raises:
            ValueError: If ``accessor_type`` is not registered.
        """
        return cls.get_registered(accessor_type)(cache_dir=cache_dir)
