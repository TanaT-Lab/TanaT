#!/usr/bin/env python3
"""
Sequence Metadata definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from .feature import (
    FeatureInfo,
    CategoricalInfo,
    NumericalInfo,
    TemporalInfo,
    build_feature_metadata,
)
from tanat_utils.pretty_format import format_feature_section


@dataclass(frozen=True)
class TimeIndexInfo:
    """Metadata for the sequence time index (time columns)."""

    dtype: str
    is_datetime: bool
    min: Any | None
    max: Any | None
    unit: str | None = None  # ms, us, ns for datetime
    time_zone: str | None = None

    @classmethod
    def from_lazyframe(cls, lf: pl.LazyFrame) -> TimeIndexInfo:
        """
        Factory: builds a ``TimeIndexInfo`` by inspecting **all**
        columns of the time index LazyFrame.

        Validates that:
        - Every column is a supported type (``pl.Datetime``, ``pl.Date``,
          or a numeric type, integer or float for discrete timesteps).
        - All columns share the **same** base type (no mix of
          ``Datetime`` start with ``Date`` end, for example).

        Also computes the **global min/max** across all time index columns.

        Raises:
            TypeError: If a column has an unsupported type or if
                column types are inconsistent.
        """
        schema = lf.collect_schema()
        col_names = schema.names()
        dtypes = list(schema.dtypes())

        # --- Validate each column individually ---
        for col_name, dtype in schema.items():
            if not (
                isinstance(dtype, (pl.Datetime, pl.Date))
                or dtype.is_integer()
                or dtype.is_float()
            ):
                raise TypeError(
                    f"Unsupported time index type for column '{col_name}': {dtype}. "
                    "Expected pl.Datetime, pl.Date, or a numeric type (discrete timestep)."
                )

        # --- Validate cross-column consistency ---
        if len(dtypes) > 1 and not all(d == dtypes[0] for d in dtypes):
            col_detail = ", ".join(f"{n}: {d}" for n, d in schema.items())
            raise TypeError(
                f"Inconsistent time index types: {col_detail}. "
                "All time index columns must share the same type."
            )

        # --- Compute global min / max ---
        stats = (
            lf.select(
                pl.min_horizontal(*col_names).min().alias("global_min"),
                pl.max_horizontal(*col_names).max().alias("global_max"),
            )
            .collect()
            .row(0)
        )
        global_min, global_max = stats

        dtype = dtypes[0]
        if isinstance(dtype, (pl.Datetime, pl.Date)):
            return cls(
                dtype=str(dtype),
                is_datetime=True,
                min=global_min,
                max=global_max,
                unit=getattr(dtype, "time_unit", None),
                time_zone=getattr(dtype, "time_zone", None),
            )
        return cls(
            dtype=str(dtype),
            is_datetime=False,
            min=global_min,
            max=global_max,
        )

    def __str__(self) -> str:
        s = f"{self.dtype}"
        if not self.is_datetime:
            s += " (Timestep)"
        s += f" [{self.min} → {self.max}]"
        return s

    def is_schema_compatible(self, other: TimeIndexInfo) -> bool:
        """
        Returns ``True`` if *other* has the same **schema** as this instance.

        Only the structural fields are compared (``dtype``, ``is_datetime``,
        ``unit``, ``time_zone``).  Range fields (``min``, ``max``) are
        intentionally ignored. Different stores can cover different time
        periods and still be joinable.
        """
        if self.is_datetime != other.is_datetime:
            return False
        if self.is_datetime:
            return self.unit == other.unit and self.time_zone == other.time_zone
        return self.dtype == other.dtype

    def to_json_dict(self) -> dict:
        """Converts this instance to a JSON-serializable dictionary."""
        d = self.__dict__.copy()
        d.pop("dtype")
        # min/max can be datetime, date, etc. → convert to str for JSON
        for key in ("min", "max"):
            if d[key] is not None:
                d[key] = str(d[key])
        return d


@dataclass(frozen=True)
class SequenceMetadata:
    """
    Rich Sequence metadata with semantic profiling.

    Feature lists (``entity_features``, ``static_features``) are
    guaranteed to be in alphabetical order.
    """

    seq_id: pl.DataType
    time_index: TimeIndexInfo
    entity_features: list[FeatureInfo]
    static_features: list[FeatureInfo] | None

    def scope(
        self,
        entity_features: list[str] | None = None,
        static_features: list[str] | None = None,
    ) -> SequenceMetadata:
        """Return a new metadata restricted to the given feature subsets.

        Args:
            entity_features: Feature names to keep.  ``None`` keeps all.
            static_features: Feature names to keep.  ``None`` keeps all.
                An empty list produces ``static_features=None``.

        Returns:
            A filtered copy, or ``self`` when nothing was actually removed.
            Original feature order is preserved.
        """
        ef = self.entity_features
        if entity_features is not None and len(entity_features) < len(ef):
            visible = set(entity_features)
            ef = [f for f in ef if f.name in visible]

        sf = self.static_features
        if (
            static_features is not None
            and sf is not None
            and len(static_features) < len(sf)
        ):
            visible = set(static_features)
            sf = [f for f in sf if f.name in visible] or None

        # Fast-path: nothing was actually filtered → return self unchanged.
        if ef is self.entity_features and sf is self.static_features:
            return self

        return SequenceMetadata(
            seq_id=self.seq_id,
            time_index=self.time_index,
            entity_features=ef,
            static_features=sf,
        )

    def __str__(self) -> str:
        s = "\n"

        # ID
        # Clean up id type representation
        id_str = str(self.seq_id).replace("DataType.", "")
        s += f"  Sequence ID: {id_str}\n"

        # Temporal
        s += f"  Time Index: {self.time_index}\n\n"

        ef_section = format_feature_section(
            "Entity Features",
            [(f.name, f.summary) for f in self.entity_features],
        )
        if ef_section:
            s += ef_section

        if self.static_features:
            sf_section = format_feature_section(
                "Static Features",
                [(f.name, f.summary) for f in self.static_features],
            )
            if sf_section:
                s += "\n"
                s += sf_section

        return s

    # ------------------------------------------------------------------
    # Inference helpers (one per field)
    # ------------------------------------------------------------------

    @classmethod
    def infer_time_index(cls, time_index: pl.LazyFrame) -> TimeIndexInfo:
        """Returns a :class:`TimeIndexInfo` built from the time index LazyFrame."""
        return TimeIndexInfo.from_lazyframe(time_index)

    @classmethod
    def infer_entity_features(cls, entity_lf: pl.LazyFrame | None) -> list[FeatureInfo]:
        """Returns the list of :class:`FeatureInfo` for *entity_lf*, or ``[]`` if ``None``."""
        return build_feature_metadata(entity_lf) if entity_lf is not None else []

    @classmethod
    def infer_static_features(
        cls, static_lf: pl.LazyFrame | None
    ) -> list[FeatureInfo] | None:
        """Returns the list of :class:`FeatureInfo` for *static_lf*, or ``None`` if ``None``."""
        return build_feature_metadata(static_lf) if static_lf is not None else None

    def _get_features(self, is_static: bool) -> list[FeatureInfo]:
        """Returns the feature list for the given scope."""
        if is_static:
            return self.static_features or []
        return self.entity_features

    def assert_id_compatible_with(
        self,
        other: SequenceMetadata,
        alias: str,
        *,
        context: str = "ID dtypes must match.",
    ) -> None:
        """Raises :class:`TypeError` if *other* has a different ID column dtype.

        Args:
            other: Metadata of the sequence or pool being compared.
            alias: Label used in the error message to identify *other*.
            context: Sentence appended to the error message.

        Raises:
            TypeError: If the ID dtypes differ.
        """
        if self.seq_id != other.seq_id:
            raise TypeError(
                f"'{alias}' has an incompatible ID dtype: "
                f"expected {self.seq_id}, got {other.seq_id}. "
                f"{context}"
            )

    def assert_time_index_compatible_with(
        self,
        other: SequenceMetadata,
        alias: str,
        *,
        context: str = "Temporal schemas must match.",
    ) -> None:
        """Raises :class:`TypeError` if *other* has an incompatible time index schema.

        Two time index schemas are compatible when they share the same Datetime
        unit and time_zone (or identical numeric dtype for timestep sequences).
        Range information (min/max) is **not** checked.

        Args:
            other: Metadata of the sequence or pool being compared.
            alias: Label used in the error message to identify *other*.
            context: Sentence appended to the error message.

        Raises:
            TypeError: If the time index schemas are incompatible.
        """
        if not self.time_index.is_schema_compatible(other.time_index):
            raise TypeError(
                f"'{alias}' has an incompatible time index schema: "
                f"expected {self.time_index.dtype}, got {other.time_index.dtype}. "
                f"Unit and time_zone must match. {context}"
            )

    def assert_features_compatible_with(
        self,
        other: SequenceMetadata,
        alias: str,
        *,
        context: str = "Features must be compatible for merging.",
    ) -> list[str]:
        """
        Check that *other* exposes at least all entity features declared in
        *self*, with matching dtypes.

        Args:
            other: Metadata of the sequence or pool being compared.
            alias: Label used in error messages to identify *other*.
            context: Sentence appended to each error message.

        Returns:
            List of feature names present in *other* but absent in *self*
            (extras).

        Raises:
            ValueError: If *other* is missing a feature present in *self*.
            TypeError: If a shared feature has an incompatible dtype.
        """
        self_feats = {f.name: f.dtype for f in self.entity_features}
        other_feats = {f.name: f.dtype for f in other.entity_features}

        missing = [name for name in self_feats if name not in other_feats]
        if missing:
            raise ValueError(
                f"'{alias}' is missing entity features: {sorted(missing)}. "
                f"Use cast_features() or add_entity_features() on '{alias}' first. "
                f"{context}"
            )

        mismatched = [
            (name, self_feats[name], other_feats[name])
            for name in self_feats
            if other_feats[name] != self_feats[name]
        ]
        if mismatched:
            details = ", ".join(
                f"{name!r}: expected {exp}, got {got}" for name, exp, got in mismatched
            )
            raise TypeError(
                f"'{alias}' has features with incompatible dtypes: {details}. "
                f"Use cast_features() to align dtypes before merging. "
                f"{context}"
            )

        return [name for name in other_feats if name not in self_feats]

    @property
    def is_datetime(self) -> bool:
        """Returns ``True`` if the time index is a Datetime type."""
        return self.time_index.is_datetime

    def feature_info(self, name: str, is_static: bool = False) -> FeatureInfo | None:
        """Return the :class:`~tanat.metadata.feature.FeatureInfo` for *name*, or ``None``.

        Args:
            name: Feature name to look up.
            is_static: ``True`` for static features, ``False`` for entity features.

        Returns:
            The matching :class:`FeatureInfo` instance, or ``None`` if not found.
        """
        for f in self._get_features(is_static):
            if f.name == name:
                return f
        return None

    def is_numeric_feature(self, name: str, is_static: bool = False) -> bool:
        """Returns ``True`` if *name* is a numeric (integer or float) feature.

        Args:
            name: Feature name to check.
            is_static: ``True`` for static features, ``False`` for entity features.

        Raises:
            KeyError: If the feature name is not found.

        Examples::

            >>> pool.metadata.is_numeric_feature("duration_hrs")
            True
        """
        info = self.feature_info(name, is_static=is_static)
        if info is None:
            raise KeyError(
                f"Feature '{name}' not found in "
                f"{'static' if is_static else 'entity'} features."
            )
        return isinstance(info, NumericalInfo)

    def is_duration_feature(self, name: str, is_static: bool = False) -> bool:
        """Returns ``True`` if *name* is a ``pl.Duration`` feature.

        Args:
            name: Feature name to check.
            is_static: ``True`` for static features, ``False`` for entity features.

        Raises:
            KeyError: If the feature name is not found.

        Examples::

            >>> pool.metadata.is_duration_feature("los")
            True
        """
        info = self.feature_info(name, is_static=is_static)
        if info is None:
            raise KeyError(
                f"Feature '{name}' not found in "
                f"{'static' if is_static else 'entity'} features."
            )
        return isinstance(info, TemporalInfo) and info.is_duration

    def is_datetime_feature(self, name: str, is_static: bool = False) -> bool:
        """Returns ``True`` if *name* is a ``pl.Datetime`` or ``pl.Date`` feature (not a Duration).

        Args:
            name: Feature name to check.
            is_static: ``True`` for static features, ``False`` for entity features.

        Raises:
            KeyError: If the feature name is not found.

        Examples::

            >>> pool.metadata.is_datetime_feature("discharge_time")
            True
        """
        info = self.feature_info(name, is_static=is_static)
        if info is None:
            raise KeyError(
                f"Feature '{name}' not found in "
                f"{'static' if is_static else 'entity'} features."
            )
        return isinstance(info, TemporalInfo) and not info.is_duration

    def is_categorical_feature(self, name: str, is_static: bool = False) -> bool:
        """Returns ``True`` if *name* is a ``Categorical`` or ``Enum`` feature.

        Args:
            name: Feature name to check.
            is_static: ``True`` for static features, ``False`` for entity features.

        Raises:
            KeyError: If the feature name is not found.

        Examples::

            >>> pool.metadata.is_categorical_feature("status")
            True
        """
        info = self.feature_info(name, is_static=is_static)
        if info is None:
            raise KeyError(
                f"Feature '{name}' not found in "
                f"{'static' if is_static else 'entity'} features."
            )
        return isinstance(info, CategoricalInfo)

    def to_json_dict(self) -> dict:
        """Converts metadata to a JSON-serializable dictionary."""
        return {
            "seq_id": str(self.seq_id),
            "time_index": self.time_index.to_json_dict(),
            "entity_features": [f.to_json_dict() for f in self.entity_features],
            "static_features": (
                [f.to_json_dict() for f in self.static_features]
                if self.static_features
                else None
            ),
        }
