#!/usr/bin/env python3
"""
Trajectory Metadata definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from .feature import (
    FeatureInfo,
    CategoricalInfo,
    build_feature_metadata,
    print_features,
)
from .sequence import TemporalIndexInfo

if TYPE_CHECKING:
    from ..store.sequence.store import SequenceStore


@dataclass(frozen=True)
class TrajectoryMetadata:
    """
    Rich Trajectory metadata with semantic profiling.

    Attributes:
        traj_id: Polars DataType of the trajectory ID column.
        temporal: Aggregated temporal info across all linked stores.
            A :class:`TrajectoryStore` always has at least one linked store,
            so this field is never ``None``.
        static_features: List of :class:`FeatureInfo` for each
            static feature, or ``None`` if no static features exist.
    """

    traj_id: pl.DataType
    temporal: TemporalIndexInfo
    static_features: list[FeatureInfo] | None

    def __str__(self) -> str:
        s = "\n"

        id_str = str(self.traj_id).replace("DataType.", "")
        s += f"  Trajectory ID: {id_str}\n"
        s += f"  Temporal Index: {self.temporal}\n\n"

        if self.static_features:
            s += print_features("Static Features", self.static_features)

        return s

    # ------------------------------------------------------------------
    # Inference helpers (one per field)
    # ------------------------------------------------------------------

    @classmethod
    def infer_temporal(cls, seq_stores: dict[str, SequenceStore]) -> TemporalIndexInfo:
        """
        Aggregates the temporal range across all linked stores by taking
        the global min/max.

        Raises:
            ValueError: If *seq_stores* is empty (should never happen for a
                valid :class:`TrajectoryStore`).
        """
        stores = list(seq_stores.values())
        if not stores:
            raise ValueError(
                "Cannot infer temporal metadata: no linked sequence stores found. "
                "A TrajectoryStore must have at least one store."
            )

        infos = [TemporalIndexInfo.from_lazyframe(s.temporal()) for s in stores]
        ref = infos[0]
        all_mins = [i.min for i in infos if i.min is not None]
        all_maxes = [i.max for i in infos if i.max is not None]

        return TemporalIndexInfo(
            dtype=ref.dtype,
            is_datetime=ref.is_datetime,
            min=min(all_mins) if all_mins else None,
            max=max(all_maxes) if all_maxes else None,
            unit=ref.unit,
            time_zone=ref.time_zone,
        )

    @classmethod
    def infer_static(cls, lf: pl.LazyFrame | None) -> list[FeatureInfo] | None:
        """Returns the list of :class:`FeatureInfo` for *lf*, or ``None`` if *lf* is ``None``."""
        return build_feature_metadata(lf) if lf is not None else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_categorical_feature(self, name: str) -> bool:
        """Returns ``True`` if *name* is a ``Categorical`` or ``Enum`` static feature.

        Args:
            name: Feature name to check.

        Raises:
            KeyError: If the feature name is not found.

        Examples::

            >>> traj.metadata.is_categorical_feature("group")
            True
        """
        by_name = {f.name: f for f in (self.static_features or [])}
        if name not in by_name:
            raise KeyError(
                f"Feature '{name}' not found in static features. "
                f"Available: {sorted(by_name)}"
            )
        return isinstance(by_name[name], CategoricalInfo)

    def to_json_dict(self) -> dict:
        """Converts metadata to a JSON-serializable dictionary."""
        return {
            "traj_id": str(self.traj_id),
            "temporal": self.temporal.to_json_dict(),
            "static_features": (
                [f.to_json_dict() for f in self.static_features]
                if self.static_features
                else None
            ),
        }
