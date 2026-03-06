#!/usr/bin/env python3
"""
Feature metadata definitions and helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class FeatureInfo(ABC):
    """Base class for feature metadata."""

    name: str
    dtype: str

    def to_json_dict(self) -> dict:
        """Converts this instance to a JSON-serializable dictionary."""
        return self.__dict__

    @classmethod
    @abstractmethod
    def get_aggregations(
        cls, col_name: str, dtype: pl.DataType | None = None
    ) -> list[pl.Expr]:
        """Returns polars expressions needed to compute metadata."""

    @classmethod
    @abstractmethod
    def from_stats(
        cls, col_name: str, dtype: str | pl.DataType, stats: dict
    ) -> FeatureInfo:
        """Factory: Builds instance from stats."""


@dataclass(frozen=True)
class NumericalInfo(FeatureInfo):
    """Metadata for numerical features."""

    min: float | int | None
    max: float | int | None

    def __repr__(self) -> str:
        return f"{self.name} ({self.dtype}): [{self.min} - {self.max}]"

    @classmethod
    def get_aggregations(
        cls, col_name: str, dtype: pl.DataType | None = None
    ) -> list[pl.Expr]:
        """Returns min and max aggregation expressions."""
        return [
            pl.col(col_name).min().alias(f"{col_name}_min"),
            pl.col(col_name).max().alias(f"{col_name}_max"),
        ]

    @classmethod
    def from_stats(
        cls, col_name: str, dtype: str | pl.DataType, stats: dict
    ) -> NumericalInfo:
        """Builds a NumericalInfo from precomputed stats."""
        return cls(
            name=col_name,
            dtype=str(dtype),
            min=stats.get(f"{col_name}_min"),
            max=stats.get(f"{col_name}_max"),
        )


@dataclass(frozen=True)
class CategoricalInfo(FeatureInfo):
    """Metadata for categorical/string features.

    .. note::
        ``n_unique`` is computed from the first 10 000 rows sampled by
        :func:`build_feature_metadata`.  On large datasets this may
        undercount the actual number of distinct categories.
    """

    n_unique: int | None
    ordered: bool = False

    def __repr__(self) -> str:
        ord_str = " (Ordered)" if self.ordered else ""
        return f"{self.name} ({self.dtype}){ord_str}: {self.n_unique} distinct values"

    @classmethod
    def get_aggregations(
        cls,
        col_name: str,
        dtype: pl.DataType | None = None,  # pylint: disable=unused-argument
    ) -> list[pl.Expr]:
        """Returns n_unique aggregation expression."""
        return [pl.col(col_name).n_unique().alias(f"{col_name}_nu")]

    @classmethod
    def from_stats(cls, col_name: str, dtype: Any, stats: dict) -> CategoricalInfo:
        """
        Builds a CategoricalInfo from precomputed stats.

        Only ``Enum`` dtype is considered ordered (user-defined order);
        standard ``Categorical`` uses lexical ordering.
        """
        is_ordered = isinstance(dtype, pl.Enum)

        return cls(
            name=col_name,
            dtype=str(dtype),
            n_unique=stats.get(f"{col_name}_nu"),
            ordered=is_ordered,
        )


@dataclass(frozen=True)
class BooleanInfo(FeatureInfo):
    """Metadata for boolean features."""

    true_count: int | None
    false_count: int | None

    def __repr__(self) -> str:
        if self.true_count is not None and self.false_count is not None:
            total = self.true_count + self.false_count
            ratio = (self.true_count / total * 100) if total > 0 else 0
            return f"{self.name} (Bool): {self.true_count} True, {self.false_count} False ({ratio:.1f}% True)"
        return f"{self.name} (Bool): No stats"

    @classmethod
    def get_aggregations(
        cls,
        col_name: str,
        dtype: pl.DataType | None = None,  # pylint: disable=unused-argument
    ) -> list[pl.Expr]:
        # We cast to UInt32 to prevent overflow on very large datasets,
        # though Polars sum() on bool is usually u32
        return [
            pl.col(col_name).cast(pl.UInt32).sum().alias(f"{col_name}_true"),
            (pl.col(col_name).count() - pl.col(col_name).cast(pl.UInt32).sum()).alias(
                f"{col_name}_false"
            ),
        ]

    @classmethod
    def from_stats(cls, col_name: str, dtype: Any, stats: dict) -> BooleanInfo:
        """Builds a BooleanInfo from precomputed stats."""
        return cls(
            name=col_name,
            dtype=str(dtype),
            true_count=stats.get(f"{col_name}_true"),
            false_count=stats.get(f"{col_name}_false"),
        )


@dataclass(frozen=True)
class StringInfo(FeatureInfo):
    """Metadata for string features."""

    min_length: int | None
    max_length: int | None

    def __repr__(self) -> str:
        return f"{self.name} (String): length range [{self.min_length} - {self.max_length}]"

    @classmethod
    def get_aggregations(
        cls,
        col_name: str,
        dtype: pl.DataType | None = None,  # pylint: disable=unused-argument
    ) -> list[pl.Expr]:
        return [
            pl.col(col_name).str.len_chars().min().alias(f"{col_name}_min_len"),
            pl.col(col_name).str.len_chars().max().alias(f"{col_name}_max_len"),
        ]

    @classmethod
    def from_stats(cls, col_name: str, dtype: Any, stats: dict) -> StringInfo:
        return cls(
            name=col_name,
            dtype=str(dtype),
            min_length=stats.get(f"{col_name}_min_len"),
            max_length=stats.get(f"{col_name}_max_len"),
        )


@dataclass(frozen=True)
class TemporalInfo(FeatureInfo):
    """Metadata for temporal features (Date, Time, Datetime, Duration)."""

    min: Any | None
    max: Any | None

    @property
    def is_duration(self) -> bool:
        """Returns ``True`` if this feature has a ``pl.Duration`` type."""
        return self.dtype.startswith("Duration")

    def __repr__(self) -> str:
        return f"{self.name} ({self.dtype}): [{self.min} - {self.max}]"

    def to_json_dict(self) -> dict:
        d = self.__dict__.copy()
        for key in ("min", "max"):
            if d[key] is not None:
                d[key] = str(d[key])
        return d

    @classmethod
    def get_aggregations(
        cls,
        col_name: str,
        dtype: pl.DataType | None = None,  # pylint: disable=unused-argument
    ) -> list[pl.Expr]:
        return [
            pl.col(col_name).min().alias(f"{col_name}_min"),
            pl.col(col_name).max().alias(f"{col_name}_max"),
        ]

    @classmethod
    def from_stats(cls, col_name: str, dtype: Any, stats: dict) -> TemporalInfo:
        return cls(
            name=col_name,
            dtype=str(dtype),
            min=stats.get(f"{col_name}_min"),
            max=stats.get(f"{col_name}_max"),
        )


@dataclass(frozen=True)
class ArrayInfo(FeatureInfo):
    """Metadata for array features (fixed-size Array or variable-size List)."""

    dimension: int | None

    def __repr__(self) -> str:
        if self.dimension:
            return f"{self.name} (Array): {self.dimension}-dim"
        return f"{self.name} (Array): variable length"

    @classmethod
    def get_aggregations(
        cls,
        col_name: str,
        dtype: pl.DataType | None = None,
    ) -> list[pl.Expr]:
        if isinstance(dtype, pl.Array):
            return []
        return [pl.col(col_name).list.len().max().alias(f"{col_name}_dim")]

    @classmethod
    def from_stats(cls, col_name: str, dtype: Any, stats: dict) -> ArrayInfo:
        dim = stats.get(f"{col_name}_dim")
        if dim is None and isinstance(dtype, pl.Array):
            dim = dtype.shape[0]
        return cls(name=col_name, dtype=str(dtype), dimension=dim)


def get_feature_info_class(dtype: pl.DataType) -> type[FeatureInfo]:
    """Returns the appropriate FeatureInfo class for a given Polars DataType."""
    if dtype in pl.INTEGER_DTYPES or dtype in pl.FLOAT_DTYPES:
        return NumericalInfo
    # pl.Categorical is a parameter-free singleton so equality works.
    # pl.Enum is parameterised (Enum(categories=[...])); use isinstance.
    if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
        return CategoricalInfo
    if dtype == pl.Boolean:
        return BooleanInfo
    if dtype == pl.String:
        return StringInfo
    if dtype in pl.TEMPORAL_DTYPES:
        return TemporalInfo
    if isinstance(dtype, (pl.List, pl.Array)):
        return ArrayInfo

    raise TypeError(
        f"Unsupported feature type: {dtype}. "
        "Supported: Numerical, String, Boolean, Categorical, Enum, Temporal (Date/Time/Datetime/Duration) and List/Array."
    )


def build_feature_metadata(lf: pl.LazyFrame) -> list[FeatureInfo]:
    """
    Compute semantic metadata for every column in *lf*.

    A sample of 10 000 rows is used for statistics so it stays fast on
    large datasets.

    Returns:
        List of :class:`FeatureInfo` instances, one per column in schema order.
    """
    schema = lf.collect_schema()
    if not schema:
        return []

    info_classes: dict[str, type] = {}
    aggs: list = []
    for col, dtype in schema.items():
        info_cls = get_feature_info_class(dtype)
        info_classes[col] = info_cls
        aggs.extend(info_cls.get_aggregations(col, dtype=dtype))

    stats = lf.head(10_000).select(aggs).collect().row(0, named=True) if aggs else {}

    return [
        info_classes[col].from_stats(col_name=col, dtype=schema[col], stats=stats)
        for col in schema.names()
    ]
