#!/usr/bin/env python3
"""Abstract base class for all ingestion sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl
from tanat_utils import Registrable


class AbstractSource(ABC, Registrable):
    """
    Unified interface for all ingestion sources.

    Subclasses implement :meth:`read` and return a :class:`polars.LazyFrame`.
    Registered by name so the resolver and builder can look them up by string.

    Registration names: ``"csv"``, ``"parquet"``, ``"dataframe"``, ``"sql"``.
    """

    _REGISTER = {}
    _TYPE_SUBMODULE = "type"

    @abstractmethod
    def read(self) -> pl.LazyFrame:
        """Read the source and return a lazy frame."""

    @abstractmethod
    def schema(self) -> pl.Schema:
        """
        Return the column schema without reading the full dataset.

        Implementations should be as cheap as possible - reading metadata
        only (file headers, SQL ``LIMIT 0`` probe, in-memory schema, …).
        """
