#!/usr/bin/env python3
"""
Centralised column-name constants for the Sequence Store layout.

Every internal column name used across the store, pool, sequence and
entity layers is defined here.  Import from this module instead of
hard-coding strings.
"""

from typing import Final


class StoreSchema:
    """Internal column names used by the store layer."""

    # --- Sequence index columns ---
    SEQ_ID: Final[str] = "_seq_id"
    OFFSET: Final[str] = "offset"
    LENGTH: Final[str] = "length"

    # --- Time index columns ---
    T_START: Final[str] = "_t_start"
    T_END: Final[str] = "_t_end"
    T_EVENT: Final[str] = "_t_event"

    # --- Transient columns (never persisted) ---
    ROW_IDX: Final[str] = "__row_idx__"

    # --- File layout ---

    class Files:
        """Physical file names for a sequence store on disk."""

        CORE = "core.json"
        METADATA = "metadata.json"
        SEQUENCE_INDEX = "sequence_index.arrow"
        TIME_INDEX = "time_index.arrow"
        ENTITY_FEATURES = "entity_features.arrow"
        STATIC_FEATURES = "static_features.arrow"

    # --- Column-set helpers ---

    @classmethod
    def time_index_columns(cls) -> list[str]:
        """All possible time index column names, in canonical order (start → end → event)."""
        return [cls.T_START, cls.T_END, cls.T_EVENT]

    @classmethod
    def internal_columns(cls) -> frozenset[str]:
        """All internal (non-feature) column names."""
        return frozenset({cls.SEQ_ID, cls.ROW_IDX, *cls.time_index_columns()})
