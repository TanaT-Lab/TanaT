#!/usr/bin/env python3
"""
Settings for TrajectoryPool views.

Mirrors the ``SequenceSettings`` pattern: the Store holds **all**
columns on disk; the view exposes only the features listed here.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import field

from pydantic import field_validator
from tanat_utils import settings_dataclass as dataclass

from ..store.trajectory.schema import TrajectorySchema as TSCH

LOGGER = logging.getLogger(__name__)


@dataclass
class TrajectorySettings:
    """
    View-layer settings for a :class:`TrajectoryPool`.

    Attributes:
        id_column: Name of the trajectory-ID column (always ``_traj_id``).
        static_features: Feature names visible in ``static_data()``.
            ``None`` means *no static features exposed* (the default
            until ``add_static_features`` is called).
    """

    id_column: str = "_traj_id"
    # pylint: disable=invalid-field-call
    static_features: list[str] = field(default_factory=list)

    @field_validator("static_features", mode="before")
    @classmethod
    def normalize_static_features(cls, v):
        """Normalize to a sorted, deduplicated list."""
        if isinstance(v, str):
            v = [v]
        return sorted(dict.fromkeys(v))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def available_features(self) -> list[str]:
        """Returns all visible static feature names (may be empty)."""
        return list(self.static_features)

    def get_column_rename_map(self) -> dict[str, str]:
        """
        Returns the mapping from store-internal column names to
        user-facing names.

        Currently only the trajectory-ID column is renamed:
        ``_traj_id`` → :attr:`id_column`.
        """
        return {TSCH.TRAJ_ID: self.id_column}

    def validate_features(
        self,
        features: list[str] | str,
        *,
        on_missing: str = "raise",
    ) -> list[str]:
        """
        Validates explicit feature names against the current settings.

        Args:
            features: Feature name(s) to validate.
            on_missing: ``"raise"`` (default), ``"warn"`` or ``"ignore"``.

        Returns:
            List of validated feature names.

        Raises:
            KeyError: If ``on_missing="raise"`` and a feature is missing.
        """
        available = self.available_features()

        if isinstance(features, str):
            features = [features]
        if not isinstance(features, list):
            raise TypeError("'features' must be a list of strings or a single string")

        valid: list[str] = []
        for feature in features:
            if feature in available:
                valid.append(feature)
            elif on_missing == "raise":
                raise KeyError(
                    f"Feature '{feature}' not found in static features. "
                    f"Available: {available}"
                )
            elif on_missing == "warn":
                warnings.warn(
                    f"'{feature}' not found in static features, skipping.",
                    UserWarning,
                    stacklevel=3,
                )

        return valid
