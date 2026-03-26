#!/usr/bin/env python3
"""
Abstract class for sequence settings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
import logging
from typing import Literal
import warnings

from pydantic import field_validator
from tanat_utils import settings_dataclass as dataclass

from ...store.sequence.schema import StoreSchema as SCH

LOGGER = logging.getLogger(__name__)


@dataclass
class SequenceSettings(ABC):
    """
    Abstract class for sequence settings.
    """

    id_column: str
    # pylint: disable=invalid-field-call
    entity_features: list[str]
    static_features: list[str] = field(default_factory=list)

    @field_validator("entity_features", mode="before")
    @classmethod
    def normalize_entity_features(cls, v):
        """Normalize to a sorted, deduplicated list; require at least one."""
        if isinstance(v, str):
            v = [v]
        # Remove duplicates while preserving order, then sort.
        result = sorted(dict.fromkeys(v))
        if not result:
            raise ValueError("entity_features must contain at least one feature name.")
        return result

    @field_validator("static_features", mode="before")
    @classmethod
    def normalize_static_features(cls, v):
        """Normalize to a sorted, deduplicated list."""
        if isinstance(v, str):
            v = [v]
        return sorted(dict.fromkeys(v))

    @abstractmethod
    def get_time_columns(self) -> list[str]:
        """
        Returns a list of time index columns configured for this sequence type.
        """

    def get_column_rename_map(self, is_static: bool = False) -> dict[str, str]:
        """
        Returns a mapping from store internal column names
        (``StoreSchema``) to user-facing column names.

        Inferred from ``get_time_columns()``:
        - 1 column  → ``T_EVENT``
        - 2 columns → ``T_START``, ``T_END`` (in order)
        """
        rename_map = {SCH.SEQ_ID: self.id_column}
        if is_static:
            return rename_map

        t_cols = self.get_time_columns()
        if len(t_cols) == 1:
            rename_map[SCH.T_EVENT] = t_cols[0]
        elif len(t_cols) == 2:
            rename_map[SCH.T_START] = t_cols[0]
            rename_map[SCH.T_END] = t_cols[1]
        return rename_map

    def available_features(self, is_static: bool = False) -> list[str]:
        """
        Returns all feature names for the given scope.

        Args:
            is_static: If ``True``, return static features;
                otherwise entity features.

        Returns:
            List of feature names (may be empty).
        """
        features = self.static_features if is_static else self.entity_features
        return list(features)

    def validate_features(
        self,
        features: list[str] | str,
        is_static: bool = False,
        on_missing: Literal["raise", "warn", "ignore"] = "raise",
    ) -> list[str]:
        """
        Validates explicit feature names against the current settings.

        Args:
            features: Feature name(s) to validate.
            is_static: Whether to check static or entity features.
            on_missing: Strategy when a requested feature does not exist.
                ``"raise"``  – raise a ``KeyError`` immediately (default).
                ``"warn"``   – log a warning and skip.
                ``"ignore"`` – silently skip.

        Returns:
            List of validated feature names that exist in the configuration.

        Raises:
            KeyError: If ``on_missing="raise"`` and a feature is not found.
        """
        available = self.available_features(is_static)

        if isinstance(features, str):
            features = [features]
        if not isinstance(features, list):
            raise TypeError("'features' must be a list of strings or a single string")

        feature_type = "static" if is_static else "entity"
        valid = []
        for feature in features:
            if feature in available:
                valid.append(feature)
            elif on_missing == "raise":
                raise KeyError(
                    f"Feature '{feature}' not found in {feature_type} features. "
                    f"Available: {available}"
                )
            elif on_missing == "warn":
                warnings.warn(
                    f"'{feature}' not found in {feature_type} features, skipping.",
                    UserWarning,
                    stacklevel=3,
                )
            # on_missing == "ignore" → silently skip

        return valid

    def is_compatible_with(self, other: SequenceSettings) -> tuple[bool, list[str]]:
        """
        Check if these settings are compatible with another SequenceSettings instance.

        Compatibility rules:
        - id_column must be identical
        - time index columns must be identical
        - entity_features must be identical or a subset (no extra features)
        - static_features must be identical or a subset (no extra features)

        Args:
            other: SequenceSettings instance to compare with.

        Returns:
            Tuple of (is_compatible, list_of_errors)
        """
        errors = []

        # Check id_column
        if self.id_column != other.id_column:
            errors.append(
                f"id_column mismatch: '{self.id_column}' != '{other.id_column}'"
            )

        # Check time index columns
        self_ti = set(self.get_time_columns())
        other_ti = set(other.get_time_columns())
        if self_ti != other_ti:
            errors.append(f"time_index mismatch: {self_ti} != {other_ti}")

        # Check features (must be subset or equal - no extra features allowed)
        for feature_type, self_feat, other_feat in [
            ("Entity features", self.entity_features, other.entity_features),
            ("Static features", self.static_features, other.static_features),
        ]:
            extra = set(self_feat) - set(other_feat)
            if extra:
                errors.append(f"Extra {feature_type} not in store: {extra}")

        return len(errors) == 0, errors

    def validate_compatibility(self, other: SequenceSettings) -> None:
        """
        Validate compatibility with another SequenceSettings instance.

        Args:
            other: SequenceSettings instance to validate against.

        Raises:
            ValueError: If settings are incompatible.
        """
        is_compatible, errors = self.is_compatible_with(other)
        if not is_compatible:
            error_msg = "Settings are incompatible with store settings:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ValueError(error_msg)

        return other
