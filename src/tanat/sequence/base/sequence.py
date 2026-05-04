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
from tanat_utils import Cachable, CachableSettings, Registrable
from tanat_utils.pretty_format import (
    format_header,
    format_section,
    format_kv,
    format_feature_section,
)

from .entity import Entity
from .cast import SequenceCastRecipe
from .view_mixin import SequenceViewMixin
from ...core.format import resolve_fmt, to_pandas
from ...core.validation import ensure_criterion
from ...zeroing import T0Setter, _T0, _T0_NEAREST_RANK, T0Value

if TYPE_CHECKING:
    from ...store.sequence.store import SequenceStore
    from .pool import SequencePool
    from ...criterion.base import Criterion

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
        self._own_entity_row_mask: pl.Series | None = None
        self._own_casts: SequenceCastRecipe | None = None
        self._fallback_t0_setter: T0Setter = T0Setter.default(
            is_event=self.get_registration_name() == "event"
        )

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

        Bypasses store resolution, feature resolution, and cast probe: all
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
        * **Detached copy:** returns the snapshotted recipe from :meth:`copy`.
        * **Standalone path:** always empty (no casts).
        """
        if self._parent_pool is not None:
            return self._parent_pool._casts
        if self._own_casts is not None:
            return self._own_casts
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

    @property
    def _entity_row_mask(self) -> pl.Series | None:
        """Active entity row mask for this sequence.

        Combines the parent pool's ``_entity_row_mask`` (if any) with
        this sequence's own ``_own_entity_row_mask`` via logical AND.
        Returns ``None`` when no mask is active.
        """
        pool_mask: pl.Series | None = (
            self._parent_pool._entity_row_mask  # pylint: disable=protected-access
            if self._parent_pool is not None
            else None
        )
        if pool_mask is None:
            return self._own_entity_row_mask
        if self._own_entity_row_mask is None:
            return pool_mask
        return pool_mask & self._own_entity_row_mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id_value(self):
        """The sequence identifier."""
        return self._id_value

    @property
    def _t0_setter(self) -> T0Setter:
        """Effective T0 setter for this sequence.

        Delegates to the parent pool's :attr:`_t0_setter` when managed,
        otherwise returns :attr:`_fallback_t0_setter` (computation deferred
        to :attr:`_t0_result`).
        """
        if self._parent_pool is not None:
            return self._parent_pool._t0_setter
        return self._fallback_t0_setter

    @Cachable.cached_property
    def _t0_result(self) -> tuple[T0Value | None, int | None]:
        """Cached T0 ``(value, nearest_rank)`` pair for this sequence.

        * **Pool path:** filters the pool's cached :meth:`_get_t0_df`.
        * **Standalone path:** computes via :attr:`_t0_setter` and
          :meth:`_resolve_nearest_rank`.
        """
        df = self._get_t0_df()
        if df.is_empty():
            return (None, None)
        return (df[_T0][0], df[_T0_NEAREST_RANK][0])

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
        """Number of entities in this sequence."""
        # Fast path: no row mask -> read length straight from the store index
        # without materialising the full per-rank DataFrame.
        if self._entity_row_mask is None:
            return self._store.get_sequence_length(
                self._id_value, id_caster=self._casts.id_caster()
            )
        return len(self._entity_ranks_df)

    def __iter__(self):
        """Iterate over entities in index order.

        Yields one :class:`~tanat.sequence.base.entity.Entity` per row,
        from rank 0 to ``len(self) - 1``.  Each entity's
        :pyattr:`~Entity.rank` matches its iteration index.

        Example::

            for entity in seq:
                print(entity.rank, entity.temporal_extent, entity.data())
        """
        for row in self._entity_ranks_df.iter_rows(named=True):
            yield self._build_entity(
                row["__phys_seq_rank__"], logical_rank=row["__logical_seq_rank__"]
            )

    def __repr__(self) -> str:
        cls = type(self).__name__
        n_entity = len(self.settings.entity_features)
        return f"{cls}(id={self._id_value}, length={len(self)}, entity_features={n_entity})"

    def __str__(self) -> str:
        cls = type(self).__name__
        meta = self.metadata
        t_cols = self.settings.get_time_columns()

        # Temporal range for this specific sequence
        # avoid using metadata propagated from parent pool.
        df = self._id_time_index_lf().select(t_cols).collect()
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
        ti_section = [
            format_kv("Range", f"{t_min} → {t_max}"),
            format_kv("t0", t0_desc),
        ]

        parts = [
            format_header(f"{cls} Summary"),
            "",
            format_section("Overview", overview),
            "",
            format_section("Time Index", ti_section),
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
        """Return the entity at the given position.

        The returned entity satisfies ``entity.rank == rank``.

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

        phys = self._entity_ranks_df["__phys_seq_rank__"][rank]
        return self._build_entity(phys, logical_rank=rank)

    def _build_entity(
        self, physical_rank: int, *, logical_rank: int | None = None
    ) -> Entity:
        """Build an Entity by physical store rank.

        Args:
            physical_rank: 0-based physical row index in the store.
            logical_rank: 0-based position within the (possibly
                filtered) sequence view.  ``None`` → same as
                *physical_rank* (no mask active).
        """
        entity_cls = Entity.get_registered(self.get_registration_name())
        return entity_cls(
            id_value=self._id_value,
            rank=physical_rank,
            store=self._store,
            features=self.settings.entity_features,
            logical_rank=logical_rank,
            cast_recipe=self._casts,
            virtual_id=self._virtual_id,
            parent_metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _apply_id_mask(
        self,
        lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Scope a LazyFrame to this sequence's ID."""
        return lf.filter(pl.col(self._store.seq_id_col) == self._id_value)

    # -------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def copy(self) -> Sequence:
        """Return a standalone copy of this sequence, detached from any parent pool.

        Returns:
            A new standalone :class:`Sequence` with ``_parent_pool=None``.

        Examples::

            seq = pool[42]
            standalone = seq.copy()         # detaches from pool
            standalone.filter_entities(crit, inplace=True)  # safe
        """
        new_seq = object.__new__(type(self))
        Sequence.__init__(new_seq, self._id_value, self._store, self.settings)
        # Snapshot the combined entity row mask so the detached copy
        # preserves the full filter state without depending on the parent pool.
        # pylint: disable=protected-access
        mask = self._entity_row_mask
        new_seq._own_entity_row_mask = mask.clone() if mask is not None else None
        # Snapshot the cast recipe so the copy keeps type conversions
        # even after being detached from the parent pool.
        new_seq._own_casts = self._casts
        # Snapshot the effective T0 setter so the copy keeps any custom T0
        # inherited from the parent pool (otherwise it would silently fall
        # back to the default fallback setter on first access).
        new_seq._fallback_t0_setter = self._t0_setter
        return new_seq

    # ------------------------------------------------------------------
    # Criterion API
    # ------------------------------------------------------------------

    def match(self, criterion: Criterion) -> bool:
        """Return ``True`` if this sequence satisfies *criterion*.

        Args:
            criterion: A :class:`~tanat.criterion.base.Criterion` instance.

        Raises:
            TypeError: If *criterion* is not a Criterion object.
            CriterionLevelError: If the criterion is incompatible with this sequence.
        """
        ensure_criterion(criterion)
        return criterion.match(self)

    def filter_entities(
        self, criterion: Criterion, *, inplace: bool = False, verbose: bool = True
    ) -> Sequence:
        """Return a view with entities pruned by *criterion*.

        Args:
            criterion: A :class:`~tanat.criterion.base.Criterion` instance
                supporting :attr:`~tanat.criterion.base.CriterionLevel.ENTITY`.
            inplace: If ``True``, modify this sequence in place.
            verbose: If ``True``, print a one-line report.

        Returns:
            Filtered sequence (or *self* when *inplace=True*).

        Raises:
            TypeError: If *criterion* is not a Criterion object.
            CriterionLevelError: If the criterion does not support entity filtering.
        """
        ensure_criterion(criterion)
        return criterion.filter_entities(self, inplace=inplace, verbose=verbose)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def temporal_data(
        self,
        features: list[str] | str | None = None,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Return temporal data for this sequence.

        Each row is one **entity**: the atomic observation of this sequence
        (an event, a state, or a time-step).  Each entity carries the sequence
        ID, its temporal position (one column for events, two for intervals),
        and **entity features**: the per-row measurements that vary along the
        sequence (e.g. heart rate, label, sensor value).

        Args:
            features: Entity feature name(s) to include.
                ``None`` → all entity features.
            fmt: ``"pandas"`` (default) or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            DataFrame with columns ``[id, temporal…, feature…]`` scoped to
            this sequence ID.

        Examples::

            seq = pool[42]
            df = seq.temporal_data()                    # pandas, all features
            df = seq.temporal_data("heart_rate")        # single feature
            df = seq.temporal_data(fmt="polars")
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        df = self._temporal_data_df(features)
        if fmt == "polars":
            return df
        return to_pandas(df, use_arrow=use_arrow)

    def static_data(
        self,
        features: list[str] | str | None = None,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pl.DataFrame | pd.DataFrame | None:
        """
        Return static (non-temporal) data for this sequence.

        Args:
            features: Feature name(s) to include (``None`` -> all).
            fmt: ``"pandas"`` (default) or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            Single-row DataFrame with columns ``[id, feature…]``.
            ``None`` when no static features are exposed by this pool.

        Examples::

            seq = pool[42]
            row = seq.static_data()               # pandas, all static features
            row = seq.static_data("age", "sex")   # subset
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        df = self._static_data_df(features)
        if df is None:
            return None
        if fmt == "polars":
            return df
        return to_pandas(df, use_arrow=use_arrow)

    def apply(
        self,
        exprs: pl.Expr | list[pl.Expr],
        is_static: bool = False,
        *,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
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
            fmt: ``"pandas"`` *(default)* or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

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
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
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

        if fmt == "polars":
            return lf.collect()
        return to_pandas(lf.collect(), use_arrow=use_arrow)

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

    @Cachable.cached_method()
    def _describe_result(self) -> pl.DataFrame:
        """Compute the describe result as a Polars DataFrame (cached)."""
        return self.apply(self._describe_exprs(), fmt="polars")

    def describe(
        self,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pl.DataFrame | pd.DataFrame:
        """Compute summary statistics for this single sequence.

        Args:
            fmt: ``"pandas"`` *(default)* or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            Single-row DataFrame with columns
            ``[length, n_unique_entities, temporal_span, …]``.

        Examples::

            seq = pool[42]
            seq.describe()
            seq.describe(fmt="polars")
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        result = self._describe_result()
        if fmt == "polars":
            return result
        return to_pandas(result, use_arrow=use_arrow)
