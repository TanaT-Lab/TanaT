#!/usr/bin/env python3
"""
Base class for sequence objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Literal

import polars as pl
import pandas as pd
from tanat_utils import CachableSettings, Registrable
from tanat_utils.pretty_format import (
    format_header,
    format_section,
    format_kv,
    format_feature_section,
)

from .entity import Entity
from .cast import SequenceCastRecipe
from .view_mixin import SequenceViewMixin
from ...zeroing import T0Setter, _T0, _T0_NEAREST_RANK, T0Value

if TYPE_CHECKING:
    from ...store.sequence.store import SequenceStore
    from .pool import SequencePool

LOGGER = logging.getLogger(__name__)


class Sequence(
    ABC,
    SequenceViewMixin,
    CachableSettings,
    Registrable,
):
    """
    Interface to a single sequence within a Store.

    A Sequence is a **scoped view** on the data for one specific ID.
    It shares the same ``SequenceStore`` as its parent Pool (no copy).

    Typical creation patterns::

        # From a Pool (recommended)
        seq = pool[42]

        # Standalone
        seq = StateSequence(id_value=42, store="my_store")
    """

    _REGISTER = {}

    def __init__(
        self,
        id_value,
        store: SequenceStore,
        settings,
    ) -> None:
        """Base initialiser.  Delegated to by concrete subclasses and
        :meth:`from_parent` after store and feature resolution have been
        performed.

        Args:
            id_value: Unique identifier for this sequence in the store.
            store: Already-resolved :class:`~tanat.store.sequence.store.SequenceStore`.
            settings: Fully-resolved :class:`SequenceSettings`
                (``entity_features`` and ``static_features`` never ``None``).
        """
        self._id_value = id_value
        self._store = store

        CachableSettings.__init__(self, settings=settings)

        self._parent_pool: SequencePool | None = None
        self._own_row_mask: pl.Series | None = None

    @classmethod
    def from_parent(
        cls,
        id_value,
        store: SequenceStore,
        settings,
        *,
        parent_pool: SequencePool,
    ) -> Sequence:
        """Create a pool-managed sequence.  **Not part of the public API.**

        Bypasses store resolution, feature resolution, and cast probe — all
        already performed by the pool.  Every piece of pool context
        (casts, row mask, virtual ID, T0) is read lazily from *parent_pool*
        via the corresponding cached properties.

        Args:
            id_value: Sequence identifier.
            store: Already-resolved :class:`~tanat.store.sequence.store.SequenceStore`.
            settings: Fully-resolved :class:`SequenceSettings`.
            parent_pool: The owning :class:`~tanat.sequence.base.pool.SequencePool`.

        Returns:
            A new :class:`Sequence` instance bound to *parent_pool*.
        """
        new_seq = object.__new__(cls)
        Sequence.__init__(new_seq, id_value, store, settings)
        new_seq._parent_pool = parent_pool
        return new_seq

    # ------------------------------------------------------------------
    # Lazy pool-delegating properties
    # ------------------------------------------------------------------

    @property
    def _casts(self) -> SequenceCastRecipe:
        """Active cast recipe for this sequence.

        * **Pool path:** delegates to the parent pool's recipe.
        * **Standalone path:** always empty (no casts).
        """
        if self._parent_pool is not None:
            return self._parent_pool._casts
        return SequenceCastRecipe()

    @property
    def _virtual_id(self) -> str | None:
        """Virtual context UUID for this sequence.

        * **Pool path:** delegates to the parent pool's virtual context.
        * **Standalone path:** always ``None``.
        """
        if self._parent_pool is not None:
            return self._parent_pool._virtual_id
        return None

    @CachableSettings.cached_property
    def _row_mask(self) -> pl.Series | None:
        """Composed row mask for this sequence.

        Combines the pool-level mask (sliced to this sequence's rows) with
        the sequence's own ``_own_row_mask`` via logical AND.
        Returns ``None`` when neither source is set.
        """
        # Pool contribution: slice the flat pool mask to this sequence's rows.
        pool_slice: pl.Series | None = None
        if self._parent_pool is not None and self._parent_pool._row_mask is not None:
            offset, length = self._parent_pool._store.get_slice(
                self._id_value, id_cast=self._parent_pool._casts.id
            )
            pool_slice = self._parent_pool._row_mask.slice(offset, length)

        own = self._own_row_mask  # None until a future filter() sets it

        if pool_slice is not None and own is not None:
            return pool_slice & own
        return pool_slice if pool_slice is not None else own

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id_value(self):
        """The sequence identifier."""
        return self._id_value

    @CachableSettings.cached_property
    def _t0_setter(self) -> T0Setter:
        """Active T0 setter for this sequence.

        * **Pool path:** returns the parent pool's setter directly.
          Already computed and validated.
        * **Standalone path:** instantiates and runs the default
          ``position=0`` setter on demand.  Cached after the first access.
        """
        if self._parent_pool is not None:
            return self._parent_pool._t0_setter
        setter = T0Setter.default(is_event=self.get_registration_name() == "event")
        setter.compute(self)
        return setter

    @CachableSettings.cached_property
    def _t0_result(self) -> tuple[T0Value | None, int | None]:
        """T0 ``(value, nearest_rank)`` pair for this sequence.

        * **Pool path:** filters the pool's cached :meth:`_get_t0_df` result
          (single vectorised pass already computed by the pool, zero extra I/O).
        * **Standalone path:** computes from scratch via :attr:`_t0_setter`
          and :meth:`_resolve_nearest_rank`.

        Cached by :class:`~tanat_utils.CachableSettings`.
        """
        if self._parent_pool is not None:
            df = self._parent_pool._get_t0_df()
            row = df.filter(pl.col(self.settings.id_column) == self._id_value)
            if row.height > 0:
                return (row[_T0][0], row[_T0_NEAREST_RANK][0])
            return (None, None)

        # Standalone path
        setter = self._t0_setter
        if setter.df is None:
            return (None, None)
        df = setter.df.filter(pl.col(self.settings.id_column) == self._id_value)
        resolved = self._resolve_nearest_rank(df)
        if len(resolved) > 0:
            return (resolved[_T0][0], resolved[_T0_NEAREST_RANK][0])
        return (None, None)

    @property
    def t0(self) -> T0Value | None:
        """T0 value for this sequence (scalar, not a DataFrame).

        ``None`` when no valid T0 row was found (e.g. sequence too short,
        or no row matched the query).
        """
        return self._t0_result[0]

    @property
    def t0_nearest_rank(self) -> int | None:
        """0-based rank of the nearest row at or before T0 within this sequence.

        ``None`` when no valid T0 row was found (e.g. sequence too short,
        T0 before all timestamps, or no row matched the query).
        """
        return self._t0_result[1]

    def __len__(self) -> int:
        """Number of events/states in this sequence (respects row mask)."""
        if self._row_mask is not None:
            return int(self._row_mask.sum())
        return self._store.get_sequence_length(self._id_value, id_cast=self._casts.id)

    def __repr__(self) -> str:
        cls = type(self).__name__
        n_entity = len(self.settings.entity_features)
        return f"{cls}(id={self._id_value}, length={len(self)}, entity_features={n_entity})"

    def __str__(self) -> str:
        cls = type(self).__name__
        meta = self.metadata
        t_cols = self.settings.get_temporal_columns()

        # Temporal range for this specific sequence
        # avoid using metadata propagated from parent pool.
        df = self._temporal_data_lf().select(t_cols).collect()
        if len(t_cols) == 1:
            t_min = df[t_cols[0]].min()
            t_max = df[t_cols[0]].max()
        else:
            t_min = min(df[c].min() for c in t_cols)
            t_max = max(df[c].max() for c in t_cols)

        t0_val = self.t0
        t0_desc = f"{t0_val} (rank {self.t0_nearest_rank})"

        overview = [
            format_kv("Sequence ID", str(self._id_value)),
            format_kv("Length", str(len(self))),
        ]
        temporal = [
            format_kv("Range", f"{t_min} → {t_max}"),
            format_kv("t0", t0_desc),
        ]

        parts = [
            format_header(f"{cls} Summary"),
            "",
            format_section("Overview", overview),
            "",
            format_section("Temporal", temporal),
        ]

        ef_section = format_feature_section(
            "Entity Features",
            [(f.name, f.summary) for f in meta.entity_features],
        )
        if ef_section:
            parts += ["", ef_section]

        sf_section = format_feature_section(
            "Static Features",
            [(f.name, f.summary) for f in (meta.static_features or [])],
        )
        if sf_section:
            parts += ["", sf_section]

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, rank: int) -> Entity:
        """
        Get the Entity at the specified rank within the sequence.

        Args:
            rank: 0-based index. Negative indexing supported.

        Returns:
            An Entity flyweight proxy for that row.
        """
        if not isinstance(rank, int):
            raise TypeError(f"Entity rank must be an integer, got {type(rank)}")
        if rank < 0:
            rank += len(self)
        if not (0 <= rank < len(self)):
            raise IndexError(
                f"Entity rank {rank} out of range for sequence of length {len(self)}"
            )

        # Map logical rank → physical row index when row mask is active
        if self._row_mask is not None:
            physical_rank = int(self._row_mask.arg_true()[rank])
        else:
            physical_rank = rank

        entity_cls = Entity.get_registered(self.get_registration_name())
        return entity_cls(
            id_value=self._id_value,
            rank=physical_rank,
            store=self._store,
            features=self.settings.entity_features,
            cast_recipe=self._casts,
            virtual_id=self._virtual_id,
            parent_metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _apply_masks(
        self,
        lf: pl.LazyFrame,
        *,
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """
        Scopes a LazyFrame to this sequence's ID and applies the row mask.
        """
        lf = lf.filter(pl.col(self._store.seq_id_col) == self._id_value)
        if not is_static and self._row_mask is not None:
            lf = lf.filter(pl.lit(self._row_mask))
        return lf

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def sequence_data(
        self,
        features: list[str] | str | None = None,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Return temporal data for this sequence.

        Args:
            features: Feature name(s) to include (``None`` -> all).
            output_format: ``"pandas"`` (default) or ``"polars"``.

        Returns:
            DataFrame with columns ``[id, temporal…, feature…]`` scoped to
            this sequence ID.

        Examples::

            seq = pool[42]
            df = seq.sequence_data()                    # pandas, all features
            df = seq.sequence_data("heart_rate")        # single feature
            df = seq.sequence_data(output_format="polars")
        """
        df = self._sequence_data_df(features)
        if output_format == "polars":
            return df
        if output_format == "pandas":
            return df.to_pandas()
        raise ValueError(
            f"Invalid output_format {output_format!r}. "
            "Expected one of: 'pandas', 'polars'."
        )

    def static_data(
        self,
        features: list[str] | str | None = None,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pl.DataFrame | pd.DataFrame | None:
        """
        Return static (non-temporal) data for this sequence.

        Args:
            features: Feature name(s) to include (``None`` -> all).
            output_format: ``"pandas"`` (default) or ``"polars"``.

        Returns:
            Single-row DataFrame with columns ``[id, feature…]``.
            ``None`` when no static features are exposed by this pool.

        Examples::

            seq = pool[42]
            row = seq.static_data()               # pandas, all static features
            row = seq.static_data("age", "sex")   # subset
        """
        df = self._static_data_df(features)
        if df is None:
            return None
        if output_format == "polars":
            return df
        if output_format == "pandas":
            return df.to_pandas()
        raise ValueError(
            f"Invalid output_format {output_format!r}. "
            "Expected one of: 'pandas', 'polars'."
        )

    def apply(
        self,
        exprs: pl.Expr | list[pl.Expr],
        is_static: bool = False,
        *,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Evaluates Polars expressions against this sequence's features.

        This is a **read-only** computation scoped to this single
        sequence.  The result is returned, not stored.

        At the Pool level, use ``Pool.apply(by_id=True)`` for
        per-sequence computations across **all** sequences, then
        ``Pool.add_entity_features()`` or ``Pool.add_static_features()``
        to persist.

        Args:
            exprs: One or more Polars expressions producing new columns.
                Each must use ``.alias()`` to name the output.
            is_static: Whether to read static or entity features.


        Returns:
            The computed columns for this sequence only.

        Examples:
            Local normalization::

                seq = pool[42]
                result = seq.apply(
                    (pl.col("value") - pl.col("value").mean()).alias("v_centered")
                )

            Multiple expressions::

                result = seq.apply([
                    (pl.col("value").diff()).alias("v_diff"),
                    (pl.col("value").rolling_mean(3)).alias("v_rm3"),
                ])

        See Also:
            ``Pool.apply``: Apply across all sequences (with optional ``by_id``).
            ``Pool.add_entity_features``: Persist entity features.
            ``Pool.add_static_features``: Persist static features.
        """
        if isinstance(exprs, pl.Expr):
            exprs = [exprs]

        lf = self._get_data_from_store(is_static=is_static)
        if lf is None:
            raise ValueError(
                f"No data found for sequence '{self._id_value}' (is_static={is_static})"
            )

        lf = self._apply_masks(lf, is_static=is_static)
        lf = self._rename_columns(lf, is_static=is_static)
        lf = lf.select(exprs)

        if output_format == "polars":
            return lf.collect()

        if output_format == "pandas":
            return lf.collect().to_pandas()

        raise ValueError(
            f"Invalid output_format {output_format!r}. "
            "Expected one of: 'pandas', 'polars'."
        )

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _exprs_for_describe(cls, settings) -> list[pl.Expr]:
        """
        Return the type-specific Polars expressions for :meth:`describe`.
        """

    def _describe_exprs(self) -> list[pl.Expr]:
        """Type-appropriate describe expressions for this sequence."""
        return type(self)._exprs_for_describe(self.settings)

    @CachableSettings.cached_method()
    def _describe_result(self) -> pl.DataFrame:
        """Compute the describe result as a Polars DataFrame (cached)."""
        return self.apply(self._describe_exprs(), output_format="polars")

    def describe(
        self,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pl.DataFrame | pd.DataFrame:
        """Compute summary statistics for this single sequence.

        Args:
            output_format: ``"pandas"`` *(default)* or ``"polars"``.

        Returns:
            Single-row DataFrame with columns
            ``[length, n_unique_entities, temporal_span, …]``.

        Examples::

            seq = pool[42]
            seq.describe()
            seq.describe(output_format="polars")
        """
        result = self._describe_result()
        if output_format == "polars":
            return result
        if output_format == "pandas":
            return result.to_pandas()
        raise ValueError(
            f"Invalid output_format {output_format!r}. "
            "Expected one of: 'pandas', 'polars'."
        )
