#!/usr/bin/env python3
"""
Trajectory Metadata definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
from tanat_utils.pretty_format import format_feature_section

from .feature import (
    FeatureInfo,
    CategoricalInfo,
    build_feature_metadata,
)
from .sequence import TimeIndexInfo

if TYPE_CHECKING:
    from ..store.sequence.store import SequenceStore


@dataclass(frozen=True)
class TrajectoryMetadata:
    """
    Rich Trajectory metadata with semantic profiling.

    Attributes:
        traj_id: Polars DataType of the trajectory ID column.
        time_index: Aggregated time index info across all linked stores.
            A :class:`TrajectoryStore` always has at least one linked store,
            so this field is never ``None``.
        static_features: List of :class:`FeatureInfo` for each
            static feature (alphabetical order), or ``None`` if none exist.
    """

    traj_id: pl.DataType
    time_index: TimeIndexInfo
    static_features: list[FeatureInfo] | None

    def __str__(self) -> str:
        s = "\n"

        id_str = str(self.traj_id).replace("DataType.", "")
        s += f"  Trajectory ID: {id_str}\n"
        s += f"  Time Index: {self.time_index}\n\n"

        if self.static_features:
            sf_section = format_feature_section(
                "Static Features",
                [(f.name, f.summary) for f in self.static_features],
            )
            if sf_section:
                s += sf_section

        return s

    # ------------------------------------------------------------------
    # Inference helpers (one per field)
    # ------------------------------------------------------------------

    @classmethod
    def infer_time_index(cls, seq_stores: dict[str, SequenceStore]) -> TimeIndexInfo:
        """
        Aggregates the time index range across all linked stores by taking
        the global min/max.

        Raises:
            ValueError: If *seq_stores* is empty (should never happen for a
                valid :class:`TrajectoryStore`).
        """
        stores = list(seq_stores.values())
        if not stores:
            raise ValueError(
                "Cannot infer time index metadata: no linked sequence stores found. "
                "A TrajectoryStore must have at least one store."
            )

        infos = [TimeIndexInfo.from_lazyframe(s.time_index()) for s in stores]
        ref = infos[0]
        all_mins = [i.min for i in infos if i.min is not None]
        all_maxes = [i.max for i in infos if i.max is not None]

        return TimeIndexInfo(
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

    def scope(
        self,
        static_features: list[str] | None = None,
    ) -> TrajectoryMetadata:
        """Return a new metadata restricted to the given static feature subset.

        Args:
            static_features: Feature names to keep.  ``None`` keeps all.
                An empty list produces ``static_features=None``.

        Returns:
            A filtered copy, or ``self`` when nothing was actually removed.
            Original feature order is preserved.
        """
        sf = self.static_features
        if (
            static_features is not None
            and sf is not None
            and len(static_features) < len(sf)
        ):
            visible = set(static_features)
            sf = [f for f in sf if f.name in visible] or None

        # Fast-path: nothing was actually filtered → return self unchanged.
        if sf is self.static_features:
            return self

        return TrajectoryMetadata(
            traj_id=self.traj_id,
            time_index=self.time_index,
            static_features=sf,
        )

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
            "time_index": self.time_index.to_json_dict(),
            "static_features": (
                [f.to_json_dict() for f in self.static_features]
                if self.static_features
                else None
            ),
        }
