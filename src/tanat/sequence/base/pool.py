#!/usr/bin/env python3
"""
Base class for sequence pool objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
from dataclasses import asdict, replace
import logging
from pathlib import Path
import random
import shutil
import uuid
from typing import TYPE_CHECKING, Any, Iterator, Literal
import warnings
import weakref

import numpy as np
import polars as pl
import pandas as pd
from tanat_utils import Cachable, CachableSettings, Registrable
from tanat_utils.pretty_format import (
    format_header,
    format_section,
    format_kv,
    format_feature_section,
)

from ...core.path import resolve_path
from ...core.format import resolve_fmt, to_pandas
from ...core.validation import ensure_criterion
from ...core import registry as _registry
from ...store.base.utils import normalise_to_lazyframe, check_no_reserved_names
from ...store.sequence.builder.base import SequenceStoreBuilder
from ...cast import SequenceCastRecipe
from .sequence import Sequence
from ._utils import merge_optional_frames, resolve_ids_to_add
from .view_mixin import SequenceViewMixin
from ...zeroing import T0Setter, _T0

if TYPE_CHECKING:
    from ...store.sequence.store import SequenceStore
    from .settings import SequenceSettings
    from ...trajectory.pool import TrajectoryPool
    from ...criterion.base import Criterion


LOGGER = logging.getLogger(__name__)


BinSize = str | int | float  # e.g., "1h", "30min", 3600, 0.5


class SequencePool(
    ABC,
    SequenceViewMixin,
    CachableSettings,
    Registrable,
):
    """Base class for sequence pool objects."""

    _REGISTER = {}
    MAX_BINS_LIMIT: int = 2_000  # safety cap when max_bins is not specified

    def __init__(
        self,
        store: SequenceStore,
        settings: SequenceSettings | dict,
        cast_recipe: SequenceCastRecipe | dict | None = None,
    ) -> None:
        """Base initialiser. Delegated to by concrete subclasses after store and
        feature resolution have been performed.

        Args:
            store: Already-resolved :class:`~tanat.store.sequence.store.SequenceStore`.
            settings: Fully-resolved :class:`SequenceSettings` (or equivalent dict).
                ``entity_features`` and ``static_features`` never ``None``.
            cast_recipe: Optional cast recipe (or dict) applied at read time.
                Normalised via :meth:`SequenceCastRecipe.coerce` and probed
                eagerly.

        Raises:
            TypeError: If *cast_recipe* is not a :class:`SequenceCastRecipe`,
                ``dict``, or ``None``.
        """
        self._store = store

        CachableSettings.__init__(self, settings=settings)

        self._virtual_id: str | None = None

        self._id_mask: set | None = None
        self._entity_filter_expr: pl.Expr | None = None
        self._has_soft_drops: bool = False
        self._casts: SequenceCastRecipe = SequenceCastRecipe.coerce(cast_recipe)
        if not self._casts.is_empty():
            self._casts.probe(self)
        self._fallback_t0_setter: T0Setter = T0Setter.default(
            is_event=self.get_registration_name() == "event"
        )
        self._locked: bool = False
        self._parent_pool: TrajectoryPool | None = None

        # -- GC safety --------------------------------------------------------
        # Registered at the very end: if __init__ raises (e.g. during probe()),
        # the object is never fully constructed and the finalizer won't run on
        # a half-built instance.  weakref.finalize runs before sys.modules is
        # cleared (unlike __del__), so pathlib remains available.
        self._gc_state: list = [store, None]  # [store, virtual_id]
        weakref.finalize(self, SequencePool._finalize_cleanup, self._gc_state)
        # ---------------------------------------------------------------------

    @classmethod
    def from_parent(
        cls,
        store_path: Path,
        *,
        parent_pool: TrajectoryPool,
    ) -> SequencePool:
        """Create a managed sub-pool owned by *parent_pool*.

        Builds the pool from *store_path*, aligns settings and casts with the
        parent :class:`~tanat.trajectory.pool.TrajectoryPool`, and marks it
        locked.  T0 is resolved lazily via the :attr:`_t0_setter` property
        which delegates to *parent_pool*.

        Args:
            store_path: Root path of the sequence store to load.
            parent_pool: The owning
                :class:`~tanat.trajectory.pool.TrajectoryPool`.

        Returns:
            A locked :class:`SequencePool` whose T0 delegates to *parent_pool*.
        """
        # pylint: disable=protected-access
        pool = _registry.build_pool("sequence", store_path)
        pool.update_settings(id_column=parent_pool.settings.id_column)
        pool._casts = pool._casts.replace(
            id=parent_pool._casts.id,
            time_index=parent_pool._casts.time_index,
        )
        if parent_pool._id_mask is not None:
            store_ids = set(pool.unique_ids)
            pool._id_mask = parent_pool._id_mask & store_ids
            pool.clear_cache()
        pool._locked = True
        pool._parent_pool = parent_pool
        return pool

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def _make_builder(cls, **kwargs) -> SequenceStoreBuilder:
        """Instantiate the builder registered for this pool type.

        Internal dispatch used by typed :meth:`builder` overrides.
        """
        seq_type = cls.get_registration_name()
        builder_cls = SequenceStoreBuilder.get_registered(seq_type)
        return builder_cls(**kwargs)

    @staticmethod
    def _finalize_cleanup(gc_state: list) -> None:
        """Called by weakref.finalize when the pool is GC'd or at interpreter exit.

        Using weakref.finalize instead of __del__ guarantees that sys.modules
        (and therefore pathlib) is still intact when this runs, so no broad
        exception handling is needed.
        """
        store, virtual_id = gc_state
        if virtual_id is not None and store is not None:
            store.clear_virtual_context(virtual_id)

    def _check_not_locked(self, operation: str) -> None:
        """Raises if this pool is locked by a parent :class:`TrajectoryPool`.

        Structural casts (*cast_id*, *cast_to_datetime*, *cast_to_timestep*)
        and ``subset(inplace=True)`` are coordinated at trajectory-pool level.
        Calling them directly on a managed pool would desynchronise the
        trajectory-level ``_id_mask`` and sub-pool masks.
        """
        if self._locked:
            raise RuntimeError(
                f"'{operation}' is not allowed on a sub-pool managed by a TrajectoryPool.\n"
                f"  • Apply this operation on the parent TrajectoryPool to keep all sub-pools in sync.\n"
                f"  • Or call pool.copy() first to get an independent standalone pool."
            )

    @contextlib.contextmanager
    def _unlocked(self):
        """Context manager that temporarily unlocks this pool for coordinated mutations.

        Intended for use by :class:`~tanat.trajectory.pool.TrajectoryPool` only.
        Re-entrant: the previous lock state is saved and restored in the
        ``finally`` block, even if an exception is raised inside the
        ``with`` body, so nesting two ``_unlocked()`` blocks does not
        accidentally re-lock a pool that started out unlocked.
        """
        previous = self._locked
        self._locked = False
        try:
            yield self
        finally:
            self._locked = previous

    def _cleanup_virtual(self) -> None:
        """
        Remove this Pool's virtual context if it exists.

        Only clears *this* Pool's ``_virtual_id``. Other Pools sharing
        the same Store are unaffected.
        """
        if self._virtual_id is not None:
            self._store.clear_virtual_context(self._virtual_id)
            self._virtual_id = None
            self._gc_state[1] = None

    def _reset_to(self, dest_path: Path) -> None:
        """Reset all dirty state and redirect the pool to *dest_path*.

        Called after every successful :meth:`save`.  Ensures that
        ``is_dirty`` is ``False`` once the data has been written:

        1. Clears the virtual context against the **current** store (must
           happen before any redirect).
        2. Redirects ``self._store`` to the freshly written store at
           *dest_path* (skipped when saving in-place, since the store
           object already points to the right directory).
        3. Resets scopes, soft-drop flag and cast recipe (all are now
           baked into the written files).
        4. Invalidates the settings cache.
        """
        in_place = dest_path == self._store.root_path
        self._cleanup_virtual()  # must run against the old store
        if not in_place:
            self._store = self._resolve_store(dest_path)
            self._gc_state[0] = self._store
        self._id_mask = None
        self._entity_filter_expr = None
        self._has_soft_drops = False
        self._casts = SequenceCastRecipe()  # baked into the written store
        self.clear_cache()

    @classmethod
    def _construct(
        cls,
        *,
        store: SequenceStore,
        settings: SequenceSettings | dict,
        cast_recipe: SequenceCastRecipe,
        virtual_id: str | None,
        id_mask: set | None,
        entity_filter_expr: pl.Expr | None,
        has_soft_drops: bool,
        t0_setter: T0Setter,
        parent_pool: TrajectoryPool | None = None,
    ) -> SequencePool:
        """Construct a pool of type *cls* without going through ``__init__``.

        Bypasses store probing and feature validation: *settings* and
        *cast_recipe* must already be fully resolved.

        Args:
            store: Already-resolved :class:`~tanat.store.sequence.store.SequenceStore`.
            settings: Fully-resolved :class:`SequenceSettings` (no re-validation).
            cast_recipe: Already-probed :class:`SequenceCastRecipe`.
            virtual_id: Forked virtual context UUID (or ``None``).
            id_mask: Set of sequence IDs to expose (or ``None`` for all).
            entity_filter_expr: Entity predicate applied in view space.
            has_soft_drops: Whether soft-dropped sequences exist.
            t0_setter: T0 strategy (own setter; ignored for managed pools).
            parent_pool: Owning
                :class:`~tanat.trajectory.pool.TrajectoryPool` when this pool
                is managed (or ``None`` for standalone pools).
        """
        pool = object.__new__(cls)
        pool._store = store
        CachableSettings.__init__(pool, settings=settings)
        pool._locked = False
        pool._casts = cast_recipe
        pool._fallback_t0_setter = t0_setter
        pool._gc_state = [store, virtual_id]
        weakref.finalize(pool, SequencePool._finalize_cleanup, pool._gc_state)
        pool._virtual_id = virtual_id
        pool._id_mask = id_mask
        pool._entity_filter_expr = entity_filter_expr
        pool._has_soft_drops = has_soft_drops
        pool._parent_pool = parent_pool
        return pool

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @Cachable.cached_property
    def unique_ids(self) -> list:
        """Visible sequence IDs in store order as a plain Python list.

        Respects ``_id_mask``.  Deterministic order (sorted at build time).

        .. warning::

           ``list`` erases rich Polars dtypes (``Categorical`` → ``str``).
           Prefer :attr:`_id_lf` when the result feeds a Polars join.
        """
        return self._id_lf.collect().to_series().to_list()

    def __len__(self) -> int:
        """Number of sequences visible in this view."""
        return len(self.unique_ids)

    def __repr__(self) -> str:
        cls = type(self).__name__
        n_entity = len(self.settings.entity_features)
        n_static = len(self.settings.static_features)
        return (
            f"{cls}(n={len(self)}, entity_features={n_entity}, "
            f"static_features={n_static}, store='{self._store.root_path}')"
        )

    def __str__(self) -> str:
        cls = type(self).__name__
        meta = self.metadata
        t_cols = self.settings.get_time_columns()

        overview = [
            format_kv("Sequences", f"{len(self):,}"),
            format_kv("Store", str(self._store.root_path)),
            format_kv("id_column", self.settings.id_column),
        ]
        ti_section = [
            format_kv("Type", str(meta.time_index)),
            format_kv("Columns", str(t_cols)),
            format_kv("t0", self._t0_display_label()),
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

    @property
    def is_dirty(self) -> bool:
        """``True`` if the pool has state not yet written to disk.

        Covers virtual features, view scopes, type casts,
        and soft feature drops.  A dirty pool needs :meth:`save` to
        materialise its current view.
        """
        return (
            self._virtual_id is not None
            or self._id_mask is not None
            or self._entity_filter_expr is not None
            or not self._casts.is_empty()
            or self._has_soft_drops
        )

    @Cachable.cached_property
    def _target_seq_cls(self) -> type[Sequence]:
        """Determine the Sequence subclass to use for this pool (e.g. event, state, interval)."""
        reg_name = self.get_registration_name()
        seq_cls = Sequence.get_registered(reg_name)
        return seq_cls

    # ------------------------------------------------------------------
    # T0 / Zeroing
    # ------------------------------------------------------------------

    @property
    def _t0_setter(self) -> T0Setter:
        """Effective T0 setter.

        Delegates to the parent :class:`~tanat.trajectory.pool.TrajectoryPool`
        when this pool is managed; returns the pool's own standalone setter
        otherwise.
        """
        if self._parent_pool is not None:
            return self._parent_pool._t0_setter
        return self._fallback_t0_setter

    def _t0_display_label(self) -> str:
        """Return the T0 strategy label for display in :meth:`__str__`.

        Managed pools append ``" (from trajectory)"`` to the parent's summary.
        """
        label = self._t0_setter.strategy_summary
        if self._parent_pool is not None:
            return f"{label} (from trajectory)"
        return label

    def t0_data(
        self,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Return the T0 table for the sequences visible in this pool view.

        Thin public wrapper around :meth:`_get_t0_df` that handles format
        conversion.

        Args:
            fmt: ``"pandas"`` (default) or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            DataFrame with columns ``[id_col, _T0_, _T0_NEAREST_RANK_]``,
            one row per visible sequence.

        Examples::

            pool.set_t0(position=0, anchor="start")
            df = pool.t0_data()
            df_pl = pool.t0_data(fmt="polars")
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        df = self._get_t0_df()
        if fmt == "polars":
            return df
        return to_pandas(df, use_arrow=use_arrow)

    def set_t0(
        self,
        *,
        position: int | None = None,
        direct=None,
        feature: str | None = None,
        query: pl.Expr | None = None,
        anchor: Literal["start", "end", "middle"] | None = None,
        use_first: bool = True,
    ) -> SequencePool:
        """Configure the T0 strategy for this pool.

        Exactly one strategy keyword must be provided.  All others must
        remain ``None``.

        Args:
            position: Row index (0-based; negative indexing supported).
            direct:   Scalar value or ``{seq_id: value}`` dict.
            feature:  Static feature column name.
            query:    Polars boolean expression on any sequence column (time columns or entity features).
            anchor:   Which end of each interval/state row to use as the
                      reference timestamp for the floor lookup:

                      * ``"start"`` *(default)*: use the start timestamp.
                      * ``"end"``: use the end timestamp.
                      * ``"middle"``: use the midpoint ``(start + end) / 2``.

                      Omitting ``anchor=`` on an interval/state pool emits a
                      :exc:`UserWarning` and defaults to ``"start"``.
                      Passing ``anchor=`` on an event pool emits a
                      :exc:`UserWarning` and the value is ignored (single
                      time column, anchor is irrelevant).
            use_first: For the *query* strategy, whether to take the first
                (``True``) or last (``False``) matching row.

        Returns:
            ``self`` for chaining.
        """
        self._check_not_locked("set_t0")

        strategies = {
            "position": position,
            "direct": direct,
            "feature": feature,
            "query": query,
        }
        provided = [k for k, v in strategies.items() if v is not None]
        if len(provided) == 0:
            raise TypeError(
                "set_t0() requires exactly one strategy keyword: "
                "position, direct, feature, or query."
            )
        if len(provided) > 1:
            raise TypeError(
                f"set_t0() accepts exactly one strategy keyword, got: {provided}."
            )

        name = provided[0]
        strategies_kwargs = {
            "position": {"position": position, "anchor": anchor},
            "direct": {"direct": direct},
            "feature": {"feature": feature},
            "query": {"query": query, "anchor": anchor, "use_first": use_first},
        }
        setter = T0Setter.get_registered(name)(**strategies_kwargs[name])
        setter.compute_from_sequence(
            self
        )  # eager: sets setter._df; errors surface here
        self._fallback_t0_setter = setter

        self.clear_cache()
        return self

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @Cachable.cached_method()
    def get_sequences(
        self,
        entity_features: list[str] | None = None,
        static_features: list[str] | None = None,
    ) -> dict[str, Sequence]:
        """
        Return a mapping of sequence IDs to :class:`Sequence` objects.

        Args:
            entity_features: Entity feature subset to expose in each
                sequence.  ``None`` -> use pool-level settings.
            static_features: Static feature subset to expose in each
                sequence.  ``None`` -> use pool-level settings.

        Returns:
            Dict mapping each visible sequence ID to its
            :class:`~tanat.sequence.base.sequence.Sequence` instance.

        Examples::

            seqs = pool.get_sequences()
            print(seqs[42].temporal_data())
        """
        # Validate feature names early, before iterating over all IDs.
        if entity_features is not None:
            self.settings.validate_features(entity_features, is_static=False)
        if static_features is not None:
            self.settings.validate_features(static_features, is_static=True)
        # Build the overridden SequenceSettings if overrides are provided.
        prebuilt: SequenceSettings | None = None
        if entity_features is not None or static_features is not None:
            overrides: dict = {}
            if entity_features is not None:
                overrides["entity_features"] = entity_features
            if static_features is not None:
                overrides["static_features"] = static_features
            prebuilt = replace(self.settings, **overrides)
        return {sid: self._build_sequence(sid, prebuilt) for sid in self.unique_ids}

    def __getitem__(self, id_value) -> Sequence:
        """
        Access an individual Sequence by its ID.

        Args:
            id_value: The sequence ID.

        Returns:
            A Sequence instance scoped to this ID, sharing the same store.

        Raises:
            KeyError: If *id_value* is excluded by the current ``_id_mask``.

        Examples::

            seq = pool[42]
            seq.temporal_data()
        """

        if id_value not in self.unique_ids:
            raise KeyError(f"Invalid Sequence ID: {id_value!r}")

        return self._build_sequence(id_value)

    def __iter__(self) -> Iterator[Sequence]:
        """Iterate over all sequences visible in this pool.

        Yields one :class:`~tanat.sequence.base.sequence.Sequence` per ID
        in :attr:`unique_ids` order.  Respects the current ID mask,
        virtual context, cast recipe, and row mask.

        Examples::

            for seq in pool:
                print(seq.id_value, len(seq))
        """
        for uid in self.unique_ids:
            yield self._build_sequence(uid)

    def _build_sequence(
        self,
        id_value,
        settings=None,
    ) -> Sequence:
        """Internal helper to build a Sequence by ID without any validity check.

        For internal use only. Callers must guarantee that *id_value* is
        present in the current view (e.g. when iterating over
        :attr:`unique_ids` or building :meth:`get_sequences`).

        Args:
            id_value: A sequence ID already known to be in the view.
            settings: :class:`SequenceSettings` to use for this sequence.
                ``None`` → use :attr:`settings` (pool-level defaults).
        """
        return self._target_seq_cls.from_parent(
            id_value,
            self._store,
            settings if settings is not None else self.settings,
            parent_pool=self,
        )

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _resolve_ids(self, ids: list) -> list | None:
        """
        Validates *ids* against the current view and intersects with ``_id_mask``.

        Raises ``ValueError`` if any ID is not present in :attr:`unique_ids`
        (unknown or masked IDs are both rejected).

        Returns the resolved list (possibly filtered by ``_id_mask``).
        """
        if isinstance(ids, str):  # ensure list
            ids = [ids]

        if ids is None and self._id_mask is None:
            return None

        if ids is None:
            return list(self._id_mask)

        known = set(self.unique_ids)
        unknown = [uid for uid in ids if uid not in known]
        if unknown:
            raise ValueError(
                f"Unknown sequence IDs: {unknown}. "
                f"Use `unique_ids` to inspect available IDs."
            )

        if self._id_mask is None:
            return list(ids)
        return [uid for uid in ids if uid in self._id_mask]

    def _apply_id_mask(
        self,
        lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Apply ``_id_mask`` to a LazyFrame."""
        if self._id_mask is not None:
            lf = lf.filter(pl.col(self.settings.id_column).is_in(self._id_mask))
        return lf

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def temporal_data(
        self,
        features: list[str] | str | None = None,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Return temporal data for all sequences visible in this pool.

        Each row is one **entity**: the atomic observation of a sequence
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
            Long-format DataFrame with columns ``[id, temporal…, feature…]``
            covering every visible sequence.

        Examples::

            df = pool.temporal_data()                    # pandas, all features
            df = pool.temporal_data("heart_rate")        # single feature
            df = pool.temporal_data(["a", "b"], fmt="polars")
            # Restrict to a subset of IDs:
            df = pool.subset([1, 2, 3]).temporal_data()
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
    ) -> pd.DataFrame | pl.DataFrame | None:
        """
        Return static (non-temporal) data for all sequences in this pool.

        Args:
            features: Feature name(s) to include (``None`` -> all static
                features).
            fmt: ``"pandas"`` (default) or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            One-row-per-sequence DataFrame with columns ``[id, feature...]``.
            ``None`` when no static features are exposed by this pool.

        Examples::

            df = pool.static_data()               # pandas, all static features
            df = pool.static_data(["age", "sex"]) # subset
            df = pool.subset([1, 2, 3]).static_data()
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        df = self._static_data_df(features)
        if df is None:
            return None
        if fmt == "polars":
            return df
        return to_pandas(df, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _get_virtual_id(self) -> str:
        """Returns the virtual ID for the current session (lazy-created)."""
        if self._virtual_id is None:
            self._virtual_id = str(uuid.uuid4())
            self._gc_state[1] = self._virtual_id
        return self._virtual_id

    def add_entity_features(
        self,
        df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        *,
        overwrite: bool = False,
    ) -> None:
        """Add new entity features to the virtual store.

        The input DataFrame must be positionally aligned with the **full**
        entity row set of the store (i.e. it must have exactly as many rows
        as there are entity rows in the physical store, not just the current
        view).  Use :meth:`save` first to materialise a filtered view before
        calling this method.

        Args:
            df: Feature-only DataFrame (no ID column) positionally aligned
                with the entity rows in the store.  Can be pandas, Polars
                eager, or Polars lazy.
            overwrite: If ``True``, replace existing features with the same
                name in the virtual context.

        Raises:
            RuntimeError: If the pool has an active ``_id_mask`` or
                entity filter expression.  Call ``pool.save()`` first
                and then add features to the resulting unfiltered pool.
            ValueError: If the number of rows in *df* does not match the
                number of entity rows in the store.
        """
        # Guard: positional alignment requires the full unfiltered store.
        if self._id_mask is not None or self._entity_filter_expr is not None:
            raise RuntimeError(
                "Cannot add entity features on a filtered view. "
                "Save the current view first with pool.save(), "
                "then add features on the resulting pool."
            )

        # Normalise first
        lf = normalise_to_lazyframe(df)

        # Drop the ID column if present. It is structural, not a feature.
        # This makes the output of apply(by_id=True) directly usable here.
        id_col = self.settings.id_column
        if id_col in lf.collect_schema().names():
            lf = lf.drop(id_col)

        # Guard: feature names must not collide with temporal or id column names.
        temporal_reserved = frozenset({id_col} | set(self.settings.get_time_columns()))
        check_no_reserved_names(
            lf.collect_schema().names(),
            temporal_reserved,
            context="id / time columns",
        )

        # Early collision detection via settings
        incoming_cols = lf.collect_schema().names()
        known = set(self.settings.available_features(is_static=False))
        collisions = [c for c in incoming_cols if c in known]
        if collisions and not overwrite:
            raise ValueError(
                f"Feature collision: {collisions} already exist in this pool. "
                "Use overwrite=True to replace them."
            )

        vid = self._get_virtual_id()
        new_cols = self._store.add_entity_features(virtual_id=vid, df=lf)

        current = self.settings.available_features(is_static=False)
        updated = current + [c for c in new_cols if c not in current]
        self.update_settings(entity_features=updated)

    def add_static_features(
        self,
        df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        *,
        id_column: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add static features to the virtual store via an ID-keyed join.

        The input DataFrame **must** include the ID column (either under
        ``settings.id_column`` or under the name given by *id_column*).  A
        LEFT JOIN against the full sequence index is performed internally, so
        partial DataFrames (covering only a subset of IDs) are accepted: IDs
        absent from *df* receive ``null`` in the virtual context.


        Args:
            df: DataFrame containing the ID column plus one or more feature
                columns.  Can be pandas, Polars eager, or Polars lazy.
            id_column: Name of the ID column in *df*.  Defaults to
                ``settings.id_column`` when ``None``.  Pass an explicit name
                when the join key in *df* differs from the pool's public ID
                name (e.g. ``id_column="patient_id"``).
            overwrite: If ``True``, replace existing features with the same
                name in the virtual context.

        Raises:
            KeyError: If the resolved ID column is not found in *df*.
        """
        resolved_id_col = (
            id_column if id_column is not None else self.settings.id_column
        )

        # Normalise first
        lf = normalise_to_lazyframe(df)
        df_cols = lf.collect_schema().names()

        if resolved_id_col not in df_cols:
            fallback = self.settings.id_column
            raise KeyError(
                f"ID column {resolved_id_col!r} not found in df "
                f"(columns: {df_cols}). "
                f'Pass id_column="<name>" or rename the column to {fallback!r}.'
            )

        # Early collision detection via settings
        feature_cols = [c for c in lf.collect_schema().names() if c != resolved_id_col]
        known = set(self.settings.available_features(is_static=True))
        collisions = [c for c in feature_cols if c in known]
        if collisions and not overwrite:
            raise ValueError(
                f"Feature collision: {collisions} already exist in this pool. "
                "Use overwrite=True to replace them."
            )

        vid = self._get_virtual_id()
        new_cols = self._store.add_static(vid, lf, id_col=resolved_id_col)

        current = self.settings.available_features(is_static=True)
        updated = current + [c for c in new_cols if c not in current]
        self.update_settings(static_features=updated)

    def apply(
        self,
        exprs: pl.Expr | list[pl.Expr],
        is_static: bool = False,
        *,
        by_id: bool = False,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Evaluate Polars expressions against the current features.

        This is a **read-only** computation: the result is returned, not
        stored.  Use :meth:`add_entity_features` or
        :meth:`add_static_features` to persist the result.

        Each expression must produce a **named** column (``.alias()``).

        When ``by_id=True``, the result always includes the ID column
        (``settings.id_column``).

        Args:
            exprs: One or more Polars expressions producing new columns.
            is_static: Whether to read static or entity features.
            by_id: If ``True``, expressions are evaluated **per sequence**
                (``group_by`` on the sequence ID). Only valid for entity
                features (``is_static=False``).  The ID column is included in
                the result.
            fmt: Format of the returned object. One of:

                - ``"pandas"`` *(default)*: returns a :class:`pandas.DataFrame`.
                - ``"polars"``: returns a :class:`polars.DataFrame`.

        Returns:
            The computed columns as a DataFrame.  When ``by_id=True``, the
            first column is the sequence ID.

        Raises:
            ValueError: If ``by_id=True`` and ``is_static=True``.

        Examples:
            Compute and inspect::

                result = pool.apply(
                    (pl.col("age") * pl.col("score")).alias("age_score"),
                    is_static=True,
                )
                print(result)

            Persist entity features::

                result = pool.apply(
                    (pl.col("value") - pl.col("value").mean()).alias("centered"),
                )
                pool.add_entity_features(result)

            Per-sequence aggregation (result includes ID column)::

                summary = pool.apply(
                    pl.col("value").mean().alias("value_mean"),
                    by_id=True,
                )
                pool.add_static_features(summary)

            Per-sequence normalization (result includes ID column)::

                normed = pool.apply(
                    (pl.col("value") - pl.col("value").mean()).alias("v_normed"),
                    by_id=True,
                )
                pool.add_entity_features(normed)
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        if isinstance(exprs, pl.Expr):
            exprs = [exprs]

        if by_id and is_static:
            raise ValueError(
                "by_id=True is only valid for entity features (is_static=False). "
                "Static features already have one row per id."
            )

        lf = self._get_data_from_store(is_static=is_static)
        if lf is None:
            raise ValueError(f"No data found (is_static={is_static})")
        lf = self._apply_masks(lf, is_static=is_static)
        lf = self._rename_columns(lf, is_static=is_static)

        if by_id:
            id_col = self.settings.id_column
            result_lf = lf.group_by(id_col, maintain_order=True).agg(exprs)
            output_names = [e.meta.output_name() for e in exprs]
            schema = result_lf.collect_schema()
            cols_to_explode = [
                name for name in output_names if schema[name].is_nested()
            ]
            if cols_to_explode:
                result_lf = result_lf.explode(cols_to_explode)
        else:
            result_lf = lf.select(exprs)

        if fmt == "polars":
            return result_lf.collect()
        return to_pandas(result_lf.collect(), use_arrow=use_arrow)

    def to_dummies(
        self,
        features: list[str] | str,
        is_static: bool = False,
        *,
        drop_first: bool = False,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        One-hot encode categorical features.

        Returns a DataFrame with binary columns for each category.
        Only features typed as ``Categorical`` or ``Enum`` are accepted.
        Cast first with ``cast_features`` if needed.

        This is a **consumption** method: the result is returned, not
        stored in the pool. Use it when preparing data for training.

        Args:
            features: Feature name(s) to encode.
            is_static: Whether these are static or entity features.
            drop_first: Drop the first category column to avoid
                multicollinearity (useful for linear models).
            fmt: Format of the returned object. One of:
                - ``"pandas"`` *(default)*: returns a :class:`pandas.DataFrame`.
                - ``"polars"``: returns a :class:`polars.DataFrame`.

        Returns:
            DataFrame with binary columns (one per category per feature).

        Raises:
            TypeError: If any feature is not ``Categorical`` or ``Enum``.

        Examples:
            Basic usage::

                pool.cast_features(schema={"status": pl.Categorical})
                dummies = pool.to_dummies("status")
                # → status_OK, status_ERROR, status_WARNING columns

            With ``drop_first`` for linear models::

                X = pool.to_dummies("group", is_static=True, drop_first=True)
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        if isinstance(features, str):
            features = [features]

        valid = self.settings.validate_features(features, is_static=is_static)
        if not valid:
            raise ValueError(f"None of the requested features exist: {features}")

        non_cat = [
            name
            for name in valid
            if not self.metadata.is_categorical_feature(name, is_static=is_static)
        ]
        if non_cat:
            raise TypeError(
                f"Features {non_cat} are not Categorical or Enum. "
                f"Use cast_features(schema={{name: pl.Categorical}}) first."
            )

        if is_static:
            df = self.static_data(features=valid, fmt="polars")
        else:
            df = self.temporal_data(features=valid, fmt="polars")

        result = df.to_dummies(columns=valid, drop_first=drop_first)

        if fmt == "polars":
            return result
        return to_pandas(result, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------

    def _describe_exprs(self) -> list[pl.Expr]:
        """Type-appropriate describe expressions for this pool."""
        return self._target_seq_cls._exprs_for_describe(self.settings)

    @Cachable.cached_method()
    def _describe_result(self) -> pl.DataFrame:
        """Compute the per-ID describe result as a Polars DataFrame (cached)."""
        result: pl.DataFrame = self.apply(
            self._describe_exprs(), by_id=True, fmt="polars"
        )

        # IDs that appear in the pool index but have no data rows are absent
        # from the group_by result.  Re-attach them with nulls so the output
        # always has one row per pool ID (consistent with unique_ids).
        id_col = self.settings.id_column
        if result.height < len(self.unique_ids):
            all_ids = self._id_lf.collect()
            result = all_ids.join(result, on=id_col, how="left")

        return result

    def describe(
        self,
        by_id: bool = True,
        add_to_static: bool = False,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Compute summary statistics for every sequence in the pool.

        Args:
            by_id: If ``True`` *(default)*, return one row per sequence ID
                with columns ``[id, length, n_unique_entities, …]``.
                If ``False``, return the cross-sequence pandas
                ``.describe()`` (count, mean, std, min, 25%, …).
            add_to_static: If ``True``, write the per-ID result to the
                static-feature store via :meth:`add_static_features`.
                Ignored (with a warning) when ``by_id=False``.
            fmt: ``"pandas"`` *(default)* or ``"polars"``.
                Ignored when ``by_id=False`` (always pandas).
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            - ``by_id=True``: DataFrame with one row per sequence ID.
            - ``by_id=False``: Aggregated statistics (pandas ``describe()``
              output).

        Examples::

            pool.describe()                          # one row per ID, pandas
            pool.describe(fmt="polars")    # same, polars
            pool.describe(by_id=False)               # cross-ID stats
            pool.describe(add_to_static=True)        # persist as static cols
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        result = self._describe_result()

        if add_to_static:
            if not by_id:
                warnings.warn(
                    "add_to_static=True is ignored when by_id=False "
                    "(no per-ID result to persist).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self.add_static_features(result)

        if not by_id:
            return to_pandas(
                result.drop(self.settings.id_column), use_arrow=use_arrow
            ).describe()

        if fmt == "polars":
            return result
        return to_pandas(result, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def copy(self) -> SequencePool:
        """
        Returns a shallow copy of this Pool, sharing the same store but with
        all view state (masks, casts, virtual features) conserved.

        The virtual context is **forked** into a new UUID so that the copy
        owns its own independent context.  Garbage-collecting either instance
        will not destroy the other's virtual features.

        The T0 strategy (``_t0_setter``) is propagated to the copy.  The T0
        result cache is **not** copied; it is recomputed on the first call to
        :meth:`t0_data` on the copy.
        """
        # pylint: disable=protected-access
        # _construct bypasses XXXSequencePool.__init__ (validation, resolution, ...).
        return self.__class__._construct(
            store=self._store,
            settings=self.settings,
            cast_recipe=self._casts,
            virtual_id=self._store.fork_virtual_context(self._virtual_id),
            id_mask=set(self._id_mask) if self._id_mask is not None else None,
            entity_filter_expr=self._entity_filter_expr,
            has_soft_drops=self._has_soft_drops,
            t0_setter=self._t0_setter,
            parent_pool=self._parent_pool,
        )

    def subset(self, ids, *, inplace=False) -> SequencePool:
        """
        Returns a new Pool containing only the specified sequence IDs.

        Args:
            ids: A list of sequence IDs to include in the subset.
            inplace: If ``True``, modify this Pool's view instead of returning a new one.

        Returns:
            A new SequencePool instance with the subset of IDs, or self if inplace=True.

        Raises:
            ValueError: If any ID in *ids* is not present in :attr:`unique_ids`.
        """
        if isinstance(ids, str):  # ensure list
            ids = [ids]

        if inplace:
            self._check_not_locked("subset(inplace=True)")

        valid_ids = self._resolve_ids(ids)
        if valid_ids is None:
            raise ValueError("No valid IDs found for subsetting.")

        if inplace:
            self._id_mask = set(valid_ids)
            self.clear_cache()
            return self

        new_pool = self.copy()
        # pylint: disable=protected-access
        new_pool._id_mask = set(valid_ids)
        new_pool.clear_cache()
        return new_pool

    # ------------------------------------------------------------------
    # Criterion API
    # ------------------------------------------------------------------

    def which(self, criterion: Criterion, *, verbose: bool = True) -> set:
        """Return the set of IDs in this pool satisfying *criterion*.

        Args:
            criterion: A :class:`~tanat.criterion.base.Criterion` instance.
            verbose: If ``True``, print a one-line report.

        Returns:
            Set of matching IDs.

        Raises:
            TypeError: If *criterion* is not a Criterion object.
        """
        ensure_criterion(criterion)
        return criterion.which_ids(self, verbose=verbose)

    def filter_entities(
        self, criterion: Criterion, *, inplace: bool = False, verbose: bool = True
    ) -> SequencePool:
        """Return a view with entities pruned by *criterion*.

        Args:
            criterion: A :class:`~tanat.criterion.base.Criterion` instance
                supporting :attr:`~tanat.criterion.base.CriterionLevel.ENTITY`.
            inplace: If ``True``, modify this pool in place.
            verbose: If ``True``, print a one-line report.

        Returns:
            Filtered pool (or *self* when *inplace=True*).

        Raises:
            TypeError: If *criterion* is not a Criterion object.
            CriterionLevelError: If the criterion does not support entity filtering.
        """
        self._check_not_locked("filter_entities")
        ensure_criterion(criterion)
        return criterion.filter_entities(self, inplace=inplace, verbose=verbose)

    def train_test_split(
        self,
        *,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int | None = None,
        shuffle: bool = True,
    ) -> tuple[SequencePool, SequencePool]:
        """Split the pool into train and test subsets.

        Mirrors the interface of
        :func:`sklearn.model_selection.train_test_split`.

        Args:
            test_size: Proportion (``float`` in ``(0, 1)``) or absolute count
                (``int``) of samples for the test subset.  Defaults to
                ``0.25`` when both *test_size* and *train_size* are ``None``.
            train_size: Proportion (``float`` in ``(0, 1)``) or absolute
                count (``int``) of samples for the train subset.  Defaults
                to the complement of *test_size*.
            random_state: Seed for the random number generator.  Pass an
                integer for reproducibility.
            shuffle: Whether to shuffle IDs before splitting.  When
                ``False``, the first IDs go to train and the last to test.

        Returns:
            ``(train_pool, test_pool)``: two new non-overlapping pool views.

        Raises:
            ValueError: If the pool is empty, sizes are non-positive, or
                ``n_train + n_test`` exceeds the pool size.
        """
        ids = list(self.unique_ids)
        n = len(ids)
        if n == 0:
            raise ValueError("Cannot split an empty pool.")

        if test_size is None and train_size is None:
            test_size = 0.25

        def _resolve(size: float | int) -> int:
            """Convert a proportion or absolute count to an integer sample count."""
            if isinstance(size, float):
                if not 0.0 < size < 1.0:
                    raise ValueError(f"Float size must be in (0, 1), got {size!r}.")
                return max(1, round(size * n))
            count = int(size)
            if count <= 0:
                raise ValueError(f"Integer size must be > 0, got {size!r}.")
            return count

        if test_size is not None and train_size is not None:
            n_test = _resolve(test_size)
            n_train = _resolve(train_size)
            if n_train + n_test > n:
                raise ValueError(
                    f"n_train + n_test ({n_train + n_test}) exceeds pool size ({n})."
                )
        elif test_size is not None:
            n_test = _resolve(test_size)
            n_train = n - n_test
        else:
            n_train = _resolve(train_size)
            n_test = n - n_train

        if shuffle:
            rng = random.Random(random_state)
            rng.shuffle(ids)

        train_ids = ids[:n_train]
        test_ids = ids[n_train : n_train + n_test]

        return self.subset(train_ids), self.subset(test_ids)

    def cast_features(
        self, schema: dict[str, pl.DataType | type], is_static: bool = False
    ) -> None:
        """
        Casts feature columns to new types, scoped to **this Pool only**.


        To make a cast permanent on disk, save the Pool
        first (``pool.save()``) and reload.
        Persisting changes might affect other views sharing the same store,
        so use with caution.

        Args:
            schema: Dictionary mapping feature names to target Polars DataTypes.
            is_static: Whether these are static features (True) or entity features (False).

        Raises:
            TypeError: If *schema* is not a dict.
            KeyError: If a feature name does not exist.
        """
        if not isinstance(schema, dict):
            raise TypeError(f"'schema' must be a dict, got {type(schema).__name__}")
        if not schema:
            return

        # Validate all feature names exist (raises on missing)
        valid_names = self.settings.validate_features(
            list(schema.keys()), is_static=is_static
        )
        valid_schema = {col: schema[col] for col in valid_names}

        # Build new recipes (existing + new step) and probe the full chain.
        new_recipe = self._casts.append(
            **({"static": valid_schema} if is_static else {"entity": valid_schema})
        )
        new_recipe.features.probe(self, is_static=is_static)
        self._casts = new_recipe
        self.clear_cache()

    def cast_id(self, dtype: pl.DataType) -> None:
        """
        Casts the ID column to a new type.

        Args:
            dtype: The target Polars DataType.
        """
        self._check_not_locked("cast_id")
        new_recipe = self._casts.append(id=dtype)
        new_recipe.structural.probe(self)
        self._casts = new_recipe
        self.clear_cache()

    def cast_to_datetime(self, unit: str = "us", time_zone: str | None = None):
        """
        Cast time columns to Datetime.

        Args:
            unit: The datetime resolution ("ms", "us", "ns").
                Default is "us" (microsecond), the Python standard.
            time_zone: Optional timezone string (e.g. "UTC", "Europe/Paris").
        """
        self._check_not_locked("cast_to_datetime")
        if unit not in ("ms", "us", "ns"):
            raise ValueError(
                f"Invalid time unit: {unit}. Must be one of 'ms', 'us', 'ns'."
            )
        target_dtype = pl.Datetime(unit, time_zone)
        new_recipe = self._casts.append(time_index=target_dtype)
        new_recipe.structural.probe(self)
        self._casts = new_recipe
        self.clear_cache()

    def cast_to_timestep(self, dtype: pl.DataType = pl.Int64):
        """
        Cast time columns to numeric-based timesteps.

        Args:
            dtype: The target numeric type (e.g., pl.UInt32, pl.Int64,
                pl.Float64).  Default is pl.Int64 for safety.

        Raises:
            TypeError: If *dtype* is not a numeric type.
            TypeError: If the underlying data is already in Datetime format.
                    (Conversion from Datetime to Timestep is not allowed).
        """
        self._check_not_locked("cast_to_timestep")
        if not dtype.is_numeric():
            raise TypeError(f"Target dtype must be a numeric type, got {dtype}")
        if self.metadata.time_index.is_datetime:
            raise TypeError("Conversion from Datetime to Timestep is not supported.")
        new_recipe = self._casts.append(time_index=dtype)
        new_recipe.structural.probe(self)
        self._casts = new_recipe
        self.clear_cache()

    def drop_features(
        self,
        features: list[str],
        is_static: bool = False,
        *,
        permanently: bool = False,
    ) -> None:
        """
        Removes features from the current view.

        By default, this is a **soft drop**: features are removed from the
        Pool settings so they no longer appear in ``temporal_data()``,
        ``static_data()`` or ``metadata``, but the underlying files are
        left untouched.

        With ``permanently=True`` the columns are also physically deleted
        from disk (physical store and/or virtual store).

        Args:
            features: Feature names to drop.
            is_static: ``True`` for static features, ``False`` for entity features.
            permanently: If ``True``, also delete the columns from disk.
                This is irreversible for physical features.
        """
        # Validate that requested features exist (raises on missing)
        valid_features = self.settings.validate_features(features, is_static=is_static)

        key = "static_features" if is_static else "entity_features"
        current = self.settings.available_features(is_static=is_static)
        new_list = [f for f in current if f not in valid_features]
        self.update_settings(**{key: new_list})

        if not permanently:
            self._has_soft_drops = True

        if permanently:
            self._store.drop_features(
                features=valid_features,
                is_static=is_static,
                virtual_id=self._virtual_id,
            )

    # ------------------------------------------------------------------
    # Survival target
    # ------------------------------------------------------------------

    def survival_target(
        self,
        endpoint_time: str,
        occurred: str | None = None,
        censure_time: str | None = None,
        fmt: Literal["sksurv", "polars", "pandas"] = "sksurv",
    ) -> tuple[np.ndarray | pl.DataFrame | pd.DataFrame, list]:
        """Build a survival target (occurred, time) from static features stored in the pool.

        For each patient, assembles a binary indicator (did the endpoint occur?) and the
        corresponding duration (time from T0 to the endpoint, or to the last observation
        for censored patients). Patients with unresolvable or non-positive durations are
        excluded and reported via a warning.

        Durations are computed as the difference between the absolute value stored in the
        static column and the per-patient T0.

        If ``censure_time`` is ``None``, the last recorded time in the sequence is used
        as the censoring reference.

        Args:
            endpoint_time: Name of the static column containing the absolute time at
                which the endpoint occurred (e.g. age at death, event datetime). T0 is
                subtracted internally to produce the duration. Null = endpoint not
                observed (censored), when occurred is None. Expected dtype: same as the
                pool time axis (numeric for timestep pools, datetime for datetime pools).
            occurred: Name of the static column with a binary endpoint indicator.
                True (or 1) = endpoint observed, False (or 0) or null = censored.
                Expected dtype: bool or numeric (int or float); cast to bool internally.
                If None, inferred as endpoint_time.is_not_null().
            censure_time: Name of the static column containing the absolute time of
                the last observation for censored patients (e.g. age at last visit,
                last visit datetime). T0 is subtracted internally to produce the
                duration. Expected dtype: same as the pool time axis. If None, derived
                automatically as ``max(get_temporal_columns()[-1])`` per patient
                (i.e. max of time_column for events, max of end_column for
                state/interval pools).
            fmt: Format of the returned target y. "sksurv" returns a
                structured np.ndarray with fields (occurred: bool, time: float)
                compatible with scikit-survival. "polars" and "pandas" return a
                DataFrame with columns ["id", "occurred", "time"].

        Returns:
            A tuple (y, valid_ids). y is the survival target in the requested
            format. valid_ids is the list of patient identifiers retained after
            filtering invalid rows.

        Raises:
            KeyError: If a column name is not found in the pool's static features.
            RuntimeError: If no static features are available in this pool.

        Examples:
            >>> y, valid_ids = pool.survival_target(
            ...     endpoint_time="death_age_occur",
            ... )
            >>> y, valid_ids = pool.survival_target(
            ...     endpoint_time="death_age_occur",
            ...     occurred="death",
            ... )
        """
        fmt = resolve_fmt(fmt, allowed=("sksurv", "polars", "pandas"), default="sksurv")
        valid_df = self._build_survival_frame(endpoint_time, occurred, censure_time)
        valid_ids = valid_df[self.settings.id_column].to_list()

        if fmt == "polars":
            return valid_df, valid_ids
        if fmt == "pandas":
            return valid_df.to_pandas(), valid_ids

        # Sksurv
        time_series = valid_df["time"]
        if str(time_series.dtype).startswith("Duration"):
            time_np = time_series.dt.total_seconds().to_numpy().astype(float)
        else:
            time_np = time_series.to_numpy(allow_copy=True).astype(float)
        dtype = np.dtype([("occurred", bool), ("time", float)])
        y = np.empty(len(valid_df), dtype=dtype)
        y["occurred"] = valid_df["occurred"].to_numpy()
        y["time"] = time_np
        return y, valid_ids

    def _build_survival_frame(
        self,
        endpoint_time: str,
        occurred: str | None,
        censure_time: str | None,
    ) -> pl.DataFrame:
        """Build the (id, occurred, time) DataFrame, filtering invalid rows.

        Called by :meth:`survival_target`. Applies the full Polars lazy pipeline:
        static join, T0 subtraction, default censor computation, exclusion filter,
        and exclusion warning.

        Args:
            endpoint_time: Static column with absolute endpoint time.
            occurred: Static column with binary endpoint indicator, or ``None``
                to infer from ``endpoint_time.is_not_null()``.
            censure_time: Static column with absolute censoring time, or ``None``
                to derive from ``max(get_temporal_columns()[-1])`` per patient.

        Returns:
            Polars DataFrame with columns ``[id_col, "occurred", "time"]``,
            containing only valid (non-excluded) rows.
        """
        id_col = self.settings.id_column

        # 1. Validate static columns (raises KeyError on unknown column)
        cols_to_read = [endpoint_time]
        if occurred is not None:
            cols_to_read.append(occurred)
        if censure_time is not None:
            cols_to_read.append(censure_time)
        self.settings.validate_features(cols_to_read, is_static=True)

        # 2. Read static columns via existing pipeline
        static_lf = self._static_data_lf(features=cols_to_read)
        if static_lf is None:
            raise RuntimeError(
                "No static features available in this pool. "
                "Add static features before calling survival_target()."
            )

        # 3. Join T0 (lazy default position=0 triggered if not explicitly set).
        # Use a left join so patients without a resolvable T0 are kept and
        # surfaced through the standard exclusion + warning pipeline below
        # (their `time` column will be null and they will be reported).
        t0_df = self._get_t0_df()
        lf = static_lf.join(t0_df.lazy().select([id_col, _T0]), on=id_col, how="left")

        # 4. Default censoring: max of last temporal column per patient
        censor_src = censure_time if censure_time is not None else "__censor__"
        if censure_time is None:
            censor_col = self.settings.get_time_columns()[-1]
            default_censor_lf = (
                self._temporal_data_lf()
                .group_by(id_col)
                .agg(pl.col(censor_col).max().alias("__censor__"))
            )
            lf = lf.join(default_censor_lf, on=id_col, how="left")

        # 5. Occurred expression: infer from endpoint_time or cast explicit column.
        # Null in an explicit occurred column is treated as False (censored), per spec.
        if occurred is None:
            lf = lf.with_columns(
                pl.col(endpoint_time).is_not_null().alias("__occurred__")
            )
        else:
            lf = lf.with_columns(
                pl.col(occurred).cast(pl.Boolean).fill_null(False).alias("__occurred__")
            )

        # 6. Compute time = absolute_value - T0
        lf = lf.with_columns(
            pl.when(pl.col("__occurred__"))
            .then(pl.col(endpoint_time) - pl.col(_T0))
            .otherwise(pl.col(censor_src) - pl.col(_T0))
            .alias("time")
        )

        # 7. Collect with sentinel columns needed for exclusion logic
        full_df = lf.select(
            [id_col, pl.col("__occurred__").alias("occurred"), "time", endpoint_time]
        ).collect()

        # 8. Build exclusion mask (handles both Duration and numeric time types)
        time_series = full_df["time"]
        if str(time_series.dtype).startswith("Duration"):
            time_nonpositive = time_series.dt.total_microseconds() <= 0
        else:
            time_nonpositive = time_series <= 0

        excluded_mask = (
            (full_df["occurred"] & full_df[endpoint_time].is_null())
            | time_series.is_null()
            | time_nonpositive
        )

        # 9. Warn on excluded patients
        excluded_df = full_df.filter(excluded_mask)
        if len(excluded_df) > 0:
            excluded_ids = excluded_df[id_col].to_list()
            LOGGER.warning(
                "survival_target: %d patient(s) excluded (unresolvable or "
                "non-positive duration): %s",
                len(excluded_ids),
                excluded_ids,
            )

        # 10. Return valid rows with only the output columns
        return full_df.filter(~excluded_mask).select([id_col, "occurred", "time"])

    # ------------------------------------------------------------------
    # Discretize helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bin_expr(
        col: str, t_min, bin_size_native: int | float, is_datetime: bool
    ) -> pl.Expr:
        """Converts a time column to an integer bin index (relative to *t_min*)."""
        if is_datetime:
            return (
                (pl.col(col) - pl.lit(t_min)).dt.total_microseconds() // bin_size_native
            ).cast(pl.Int64)
        return ((pl.col(col) - t_min) / bin_size_native).floor().cast(pl.Int64)

    @abstractmethod
    def _assign_bins(
        self,
        lf: pl.LazyFrame,
        t_min,
        bin_size_native: int | float,
        is_datetime: bool,
        bin_col: str = "__bin__",
    ) -> pl.LazyFrame:
        """Projects each row onto one or more bin indices.

        Subclasses must implement this to reflect their temporal structure:
        - **Event**: one bin per row (single timestamp).
        - **Interval / State**: explode across ``[bin_start, bin_end]`` inclusive.
        """

    def _validate_discretize_inputs(
        self,
        feature: list[str] | str,
        overlap_rule: str,
    ) -> tuple[list[str], list[str], str]:
        """Validate and normalise *feature* and *overlap_rule*.

        Returns:
            ``(valid_features, time_cols, id_col)``

        Note:
            Categorical dtype enforcement for ``ohe=True`` is delegated to
            :meth:`one_hot_encode`, which raises an explicit :exc:`TypeError`
            with remediation hints.
        """
        if isinstance(feature, str):
            feature = [feature]
        valid_features = self.settings.validate_features(feature, is_static=False)

        if not callable(getattr(pl.Expr, overlap_rule, None)):
            raise ValueError(
                f"overlap_rule {overlap_rule!r} is not a valid polars.Expr "
                "aggregation method. "
                "Examples: 'first', 'last', 'mean', 'max', 'min', 'sum', 'median'."
            )

        return (
            valid_features,
            self.settings.get_time_columns(),
            self.settings.id_column,
        )

    def _build_entity_frame(
        self,
        features: list[str],
        time_cols: list[str],
        id_col: str,
        ohe: bool,
    ) -> tuple[pl.DataFrame, list[str]]:
        """Build and materialise the entity frame for bin projection.

        Shared by :meth:`binned_data` / :meth:`to_tensor` and
        :meth:`~tanat.trajectory.pool.TrajectoryPool.binned_data` /
        :meth:`~tanat.trajectory.pool.TrajectoryPool.to_tensor` (called on each
        alias pool).  *features* must already be validated by the caller.

        Args:
            features: Already-validated feature names to include.
            time_cols: Time column names (excluded from the returned
                *feat_cols* on the OHE path).
            id_col: ID column name (also excluded from *feat_cols* on OHE path).
            ohe: If ``True``, one-hot encode *features* before returning.

        Returns:
            ``(frame, feat_cols)`` - collected DataFrame and the list of
            feature columns to project onto the grid.
        """
        if ohe:
            frame = self.to_dummies(features, fmt="polars")
            temporal_set = set(time_cols)
            feat_cols = [
                c for c in frame.columns if c != id_col and c not in temporal_set
            ]
        else:
            lf = self._get_data_from_store(is_static=False)
            lf = self._apply_masks(lf, is_static=False)
            lf = self._select_columns(lf, features, is_static=False)
            lf = self._rename_columns(lf, is_static=False)
            frame = lf.collect()
            feat_cols = features
        return frame, feat_cols

    def _resolve_bin_params(
        self,
        frame: pl.DataFrame,
        time_cols: list[str],
        bin_size: BinSize,
        max_bins: int | None,
    ) -> tuple[Any, bool, int | float]:
        """Validate temporal data and resolve binning parameters.

        Performs a single in-memory scan to check for nulls and compute
        ``t_min`` / ``t_max``.  Then detects the temporal type, converts
        *bin_size* to its native unit, and optionally checks the safety cap.

        Returns:
            ``(t_min, is_datetime, bin_size_native)``
        """
        # Null check + stats
        stats_row = frame.select(
            [pl.col(c).null_count().alias(f"null_{c}") for c in time_cols]
            + [pl.col(c).min().alias(f"min_{c}") for c in time_cols]
            + [pl.col(c).max().alias(f"max_{c}") for c in time_cols]
        ).row(0, named=True)

        bad_cols = [c for c in time_cols if stats_row[f"null_{c}"] > 0]
        if bad_cols:
            raise ValueError(
                f"Time columns contain null values: {bad_cols}. "
                "Fill or drop them before discretizing."
            )

        t_min = min(stats_row[f"min_{c}"] for c in time_cols)
        t_max = max(stats_row[f"max_{c}"] for c in time_cols)
        if t_min is None or t_max is None:
            raise ValueError("Cannot discretize: pool is empty.")

        is_datetime = self.metadata.time_index.is_datetime

        if is_datetime:
            if not isinstance(bin_size, str):
                raise TypeError(
                    "bin_size must be a duration string (e.g. '1h', '1d') "
                    "for datetime sequences."
                )
            bin_size_native = int(pd.Timedelta(bin_size).total_seconds() * 1_000_000)
        else:
            if not isinstance(bin_size, (int, float)):
                raise TypeError(
                    "bin_size must be numeric for non-datetime (timestep) sequences."
                )
            bin_size_native = bin_size

        if max_bins is None:
            span = (
                (t_max - t_min).total_seconds() * 1_000_000
                if is_datetime
                else float(t_max - t_min)
            )
            estimated_bins = int(span // bin_size_native) + 1
            if estimated_bins > self.MAX_BINS_LIMIT:
                raise ValueError(
                    f"Discretization would produce ~{estimated_bins:,} bins, "
                    f"which exceeds the safety limit of {self.MAX_BINS_LIMIT:,}. "
                    "Use a larger bin_size or set max_bins explicitly to override."
                )
        # When max_bins is provided explicitly the safety cap is intentionally
        # not enforced: the caller has knowingly opted in to a large grid.

        return t_min, is_datetime, bin_size_native

    def _build_binned_data(
        self,
        lf_binned: pl.LazyFrame,
        valid_features: list[str],
        id_col: str,
        overlap_rule: str,
        max_bins: int | None,
        fill_value: Any,
        bin_col: str = "__bin__",
    ) -> tuple[pl.DataFrame, int]:
        """Resolve bin conflicts, truncate/pad to *max_bins*, build the id×bin table.

        Returns:
            ``(df_binned, n_bins)``: the complete ``(N × M)`` long-format
            DataFrame and the effective bin count.
        """
        agg_exprs = [
            getattr(pl.col(f), overlap_rule)().alias(f) for f in valid_features
        ]
        df = (
            lf_binned.group_by([id_col, bin_col], maintain_order=False)
            .agg(agg_exprs)
            .sort([id_col, bin_col])
            .collect()
        )

        # All IDs captured before truncation (preserves pool order and
        # keeps sequences that have no data within the bin range).
        ids_df = self._id_lf.collect()

        if max_bins is None:
            max_b = df[bin_col].max()
            n_bins = int(max_b) + 1 if max_b is not None else 1
        else:
            n_bins = max_bins
            df = df.filter(pl.col(bin_col) < n_bins)
        bins_df = pl.Series(bin_col, list(range(n_bins)), dtype=pl.Int64).to_frame()
        df = ids_df.join(bins_df, how="cross").join(
            df, on=[id_col, bin_col], how="left"
        )

        if fill_value is not None:
            df = df.with_columns(
                [pl.col(f).fill_null(pl.lit(fill_value)) for f in valid_features]
            )

        return df, n_bins

    def _to_binned_axis(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        feat_cols: list[str],
        t_min: Any,
        bin_size_native: int | float,
        is_datetime: bool,
        max_bins: int | None,
        *,
        fill_value: Any = None,
        overlap_rule: str = "first",
        bin_col: str = "__bin__",
    ) -> pl.DataFrame:
        """Bin a pre-built frame against a pre-computed shared temporal axis.

        Bypasses :meth:`_resolve_bin_params` and frame construction:
        ``t_min``, ``bin_size_native``, ``is_datetime`` and ``max_bins`` are
        supplied by the caller, and *frame* is already masked / renamed.

        This is the low-level hook consumed by :meth:`binned_data` /
        :meth:`to_tensor` (which resolve the axis first and build the frame)
        and by
        :meth:`~tanat.trajectory.pool.TrajectoryPool.binned_data` /
        :meth:`~tanat.trajectory.pool.TrajectoryPool.to_tensor` (which compute
        a single shared axis across all stores before calling this method on
        each alias pool).

        Args:
            frame: Already-masked, already-renamed entity LazyFrame (or
                eager DataFrame for OHE path).  Must contain the id column,
                time columns and *feat_cols*.
            feat_cols: Feature columns to project (post-OHE names if
                ``ohe=True`` was applied by the caller).
            t_min: Origin of the time axis.
            bin_size_native: Bin width in the time column's native unit.
            is_datetime: ``True`` if the time column is a Datetime type.
            max_bins: Maximum bins; ``None`` → inferred from data.
            fill_value: Value for empty bins (default ``None`` keeps nulls).
            overlap_rule: Polars aggregation name for conflict resolution.
            bin_col: Name of the bin-index column in the output.

        Returns:
            A :class:`polars.DataFrame` in long format:
            ``[id_col, bin_col, feat…]`` with ``N × max_bins`` rows.
        """
        if not callable(getattr(pl.Expr, overlap_rule, None)):
            raise ValueError(
                f"overlap_rule {overlap_rule!r} is not a valid polars.Expr "
                "aggregation method. "
                "Examples: 'first', 'last', 'mean', 'max', 'min', 'sum', 'median'."
            )
        id_col = self.settings.id_column
        lf_binned = self._assign_bins(
            frame if isinstance(frame, pl.LazyFrame) else frame.lazy(),
            t_min,
            bin_size_native,
            is_datetime,
            bin_col=bin_col,
        )
        df, _ = self._build_binned_data(
            lf_binned,
            feat_cols,
            id_col,
            overlap_rule,
            max_bins,
            fill_value,
            bin_col=bin_col,
        )
        return df

    def _binned_core(
        self,
        features: list[str] | str,
        bin_size: BinSize,
        max_bins: int | None,
        fill_value: Any,
        overlap_rule: str,
        ohe: bool,
        bin_col: str,
    ) -> tuple[pl.DataFrame, list[str], int, str]:
        """Shared computation core for :meth:`binned_data` and :meth:`to_tensor`.

        Returns:
            ``(df_polars_long, feat_cols_out, n_bins, id_col)``
        """
        valid_features, time_cols, id_col = self._validate_discretize_inputs(
            features, overlap_rule
        )

        frame, feat_cols = self._build_entity_frame(
            valid_features, time_cols, id_col, ohe
        )

        t_min, is_datetime, bin_size_native = self._resolve_bin_params(
            frame, time_cols, bin_size, max_bins
        )

        lf_binned = self._assign_bins(
            frame.lazy(),
            t_min,
            bin_size_native,
            is_datetime,
            bin_col=bin_col,
        )
        df, n_bins = self._build_binned_data(
            lf_binned,
            feat_cols,
            id_col,
            overlap_rule,
            max_bins,
            fill_value,
            bin_col=bin_col,
        )
        feat_cols_out = [c for c in df.columns if c not in {id_col, bin_col}]
        return df, feat_cols_out, n_bins, id_col

    def binned_data(
        self,
        features: list[str] | str,
        bin_size: BinSize,
        max_bins: int | None = None,
        fill_value: Any = None,
        overlap_rule: str = "first",
        ohe: bool = False,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
        bin_col: str = "__bin__",
    ) -> pd.DataFrame | pl.DataFrame:
        """Project sequences onto a binned temporal table (long-format dataframe).

        Each sequence is aligned to a shared time axis divided into fixed-size
        bins. When multiple values compete for the same bin, ``overlap_rule``
        resolves the ambiguity. Empty bins are filled with ``fill_value``.

        For an ML-ready 3-D tensor with feature labels and ID order, see
        :meth:`to_tensor`.

        Args:
            features: Feature(s) to project onto the grid.
            bin_size: Width of each bin (duration string for datetime, numeric
                otherwise).
            max_bins: Maximum number of bins. ``None`` infers from the data
                span, capped by :attr:`MAX_BINS_LIMIT`. An explicit value
                **bypasses** the safety cap — the caller opts in knowingly.
            fill_value: Value used to fill empty bins. ``None`` keeps nulls.
            overlap_rule: Polars aggregation name for in-bin conflict
                resolution (``"first"``, ``"last"``, ``"mean"``, ``"max"``,
                ``"sum"``, ``"median"``, ...).
            ohe: One-hot encode features before binning. Requires
                ``Categorical`` or ``Enum`` dtypes.
            fmt: ``"pandas"`` (default) or ``"polars"``.
            use_arrow: Pandas conversion uses Arrow when ``True``.
            bin_col: Output column name for the bin index.

        Returns:
            DataFrame with columns ``[id_col, bin_col, *feature_cols]`` and
            ``len(unique_ids) * n_bins`` rows.
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        df, _, _, _ = self._binned_core(
            features, bin_size, max_bins, fill_value, overlap_rule, ohe, bin_col
        )
        if fmt == "polars":
            return df
        return to_pandas(df, use_arrow=use_arrow)

    def to_tensor(
        self,
        features: list[str] | str,
        bin_size: BinSize,
        max_bins: int | None = None,
        fill_value: Any = None,
        overlap_rule: str = "first",
        ohe: bool = False,
        bin_col: str = "__bin__",
    ) -> tuple[np.ndarray, list, list[str]]:
        """Project sequences onto a 3-D temporal grid (ML-ready tensor).

        Returns a dense ``(N, M, K)`` ndarray together with the list of K-axis
        feature labels. Sequence order on the N axis follows :attr:`unique_ids`.

        For a long-format dataframe variant (joins, plotting, exploration),
        see :meth:`binned_data`.

        Args:
            features: Feature(s) to project.
            bin_size: Bin width (duration string for datetime, numeric
                otherwise).
            max_bins: Maximum bins. ``None`` infers from the data span, capped
                by :attr:`MAX_BINS_LIMIT`. An explicit value **bypasses** the
                safety cap — the caller opts in knowingly.
            fill_value: Value for empty bins. ``None`` keeps NaN.
            overlap_rule: In-bin aggregation. See :meth:`binned_data`.
            ohe: One-hot encode before binning. Post-OHE column names are
                reflected in the returned ``feature_names``.
            bin_col: Internal bin column name (forwarded to the underlying
                pipeline for consistency; not present in the output).

        Returns:
            A 3-tuple ``(arr, ids, feature_names)`` where:

            * ``arr`` has shape ``(N, M, K)`` = ``(len(unique_ids), n_bins,
              len(feature_names))``.
            * ``ids`` is the sequence of entity IDs matching the N-axis order
              (identical to :attr:`unique_ids`).
            * ``feature_names`` lists the K-axis labels in column order
              (post-OHE names when ``ohe=True``).

        Examples::

            arr, ids, names = pool.to_tensor(["dose", "route"], "1d", ohe=True)
            # arr.shape == (N, M, K)
            # names == ["dose", "route_oral", "route_iv"]
        """
        df, feat_cols_out, n_bins, _ = self._binned_core(
            features, bin_size, max_bins, fill_value, overlap_rule, ohe, bin_col
        )
        ids = self.unique_ids
        arr = df.select(feat_cols_out).to_numpy()  # (N*M, K)
        return (
            arr.reshape(len(ids), n_bins, len(feat_cols_out)),
            ids,
            feat_cols_out,
        )

    # ------------------------------------------------------------------
    # Extend
    # ------------------------------------------------------------------

    def extend(
        self,
        other: SequencePool | Sequence,
        destination: str | Path | None = None,
        *,
        on_duplicate: Literal["raise", "skip"] = "raise",
        overwrite: bool = False,
    ) -> SequencePool:
        """Merge *other* into this pool and write the result to disk.

        Mirrors the semantics of :meth:`save`.

        **Same-store fast path**: if ``self`` and ``other`` point to the same
        physical store *and* neither carries virtual content
        (``_virtual_id is None`` on both sides), no read I/O is performed.
        A new pool backed by the same store with the union of both ID masks is
        built immediately.  If *destination* is provided the merged pool is
        then materialised to disk via :meth:`save`; otherwise it is returned
        as an in-memory view with zero I/O.

        **Different stores (or virtual content present)**: a full merge-and-write
        is performed and *destination* is required.  A named destination writes
        the merged data to a new store.  To rewrite in-place, pass
        ``destination=self._store.root_path`` explicitly together with
        ``overwrite=True``.

        Args:
            other: Data to merge.  Accepted types:

                - :class:`SequencePool`: must be the same concrete subclass
                  with an identical entity feature schema.
                - :class:`~tanat.sequence.base.sequence.Sequence`: single
                  sequence object; schema is checked against this pool.

            destination: ``None`` → in-memory view (same-store fast path only;
                no I/O); ``str`` / ``Path`` → materialise the merged data to
                disk.  *destination* is required when merging from different
                stores.
            on_duplicate: Behaviour when *other* contains an ID already present
                in this pool:

                - ``"raise"`` *(default)*: raise ``ValueError`` listing the
                  conflicting IDs.
                - ``"skip"``: silently ignore duplicates.

            overwrite: Allows overwriting an existing *destination* when it
                already exists on disk.

        Returns:
            A new :class:`SequencePool` instance.

        Raises:
            TypeError: If *other* is not a :class:`SequencePool` or
                :class:`~tanat.sequence.base.sequence.Sequence`.
            ValueError: If *other* is a :class:`SequencePool` of a different
                concrete type.
            ValueError: If *other* is missing entity features declared in this
                pool's settings.
            ValueError: If ``on_duplicate="raise"`` and duplicate IDs are found.
            ValueError: If ``destination=None`` and stores differ (same-store
                fast path only supports an in-memory view without I/O).
            FileExistsError: If *destination* exists and ``overwrite=False``.

        See Also:
            :meth:`save`
        """
        # ── Type check ────────────────────────────────────────────────────────────
        if not isinstance(other, (SequencePool, Sequence)):
            raise TypeError(
                f"'other' must be a SequencePool or Sequence, "
                f"got {type(other).__name__!r}."
            )
        if other.get_registration_name() != self.get_registration_name():
            raise TypeError(
                f"Type mismatch: this pool holds {self.get_registration_name()!r} sequences "
                f"but 'other' is {other.get_registration_name()!r}. "
                "Both must share the same sequence type."
            )

        # Schema checks  ───────────────────────────────────────────
        # 1. ID dtype.
        self.metadata.assert_id_compatible_with(
            other.metadata,
            alias="other",
            context="Cannot extend pools with different ID dtypes.",
        )
        # 2. Temporal schema.
        self.metadata.assert_time_index_compatible_with(
            other.metadata,
            alias="other",
            context="Cannot extend pools with different temporal schemas.",
        )
        # 3. Feature-level: presence + dtype compatibility.
        extra_feats = self.metadata.assert_features_compatible_with(
            other.metadata,
            alias="other",
            context="Cannot extend pools with incompatible feature schemas.",
        )
        if extra_feats:
            warnings.warn(
                f"Ignoring extra entity features in 'other' (not in self): {sorted(extra_feats)}",
                UserWarning,
                stacklevel=2,
            )

        # ── Collect IDs from both sides ───────────────────────────────────────────
        self_ids = set(self.unique_ids)
        other_ids_list = (
            [other._id_value] if isinstance(other, Sequence) else list(other.unique_ids)
        )
        other_ids_to_add = resolve_ids_to_add(self_ids, other_ids_list, on_duplicate)

        if not other_ids_to_add:
            warnings.warn(
                "Nothing to extend: all IDs from 'other' are already present. "
                "Returning a copy of self unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return self.copy()

        # ── Same-store fast path ──────────────────────────────────────────────────
        same_store = self._store.root_path == other._store.root_path
        no_virtual = self._virtual_id is None and other._virtual_id is None
        if same_store and no_virtual:
            merged = list(self_ids) + other_ids_to_add
            new_pool = self.copy()
            new_pool._id_mask = set(merged)
            new_pool.clear_cache()
            if destination is not None:
                new_pool.save(destination, overwrite=overwrite)
            return new_pool

        # ── Destination required for cross-store merge ────────────────────────────
        if destination is None:
            raise ValueError(
                "extend() requires a destination when merging sequences from different "
                "stores. Use extend(other, destination='path/to/new_store') to write "
                "to a new store, or pass destination=self._store.root_path with "
                "overwrite=True to rewrite the current store in-place."
            )

        # ── Prepare + write ───────────────────────────────────────────────────────
        merged_entity, merged_static = self._prepare_extend_frames(
            other, other_ids_to_add
        )

        # ── Write ─────────────────────────────────────────────────────────────────
        builder = self.__class__._make_builder()
        dest_path = resolve_path(destination)
        in_place = dest_path == self._store.root_path
        if in_place:
            self._check_not_locked("extend(destination=self._store.root_path)")

        # Always write to a sibling tmp first:
        #   - For in-place: source is read before dest is cleared (lazy frames stay valid).
        #   - For all cases: dest is never left in a partial state on failure.
        tmp = dest_path.parent / f".tmp_{uuid.uuid4().hex}"
        try:
            builder.build_from_frames(tmp, merged_entity, merged_static)
            # _prepare_dest raises FileExistsError or clears the destination.
            dest_path = self._prepare_dest(destination, overwrite)
            shutil.move(str(tmp), str(dest_path))
        finally:
            # tmp has been moved on success - exists() is False, this is a no-op.
            # tmp still exists on failure - clean it up.
            if tmp.exists():
                shutil.rmtree(tmp)

        if in_place:
            self._store._invalidate_all_caches()
        LOGGER.info("Extended store written to %s.", dest_path)

        settings_dict = asdict(self.settings)
        # expose all static from the merged store
        # may contain extra features due to the merge self x other.
        settings_dict["static_features"] = None
        return self.__class__(dest_path, **settings_dict)

    def _prepare_extend_frames(
        self,
        other: SequencePool | Sequence,
        other_ids_to_add: list,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
        """Build merged entity and static LazyFrames for a cross-store extend.

        Reads both sides, applies view masks, projects to *self*'s entity
        feature set, and concatenates.  Returned frames are lazy (no I/O
        until collected by the builder).

        Returns:
            ``(merged_entity, merged_static)`` - second element is ``None``
            when neither side has static features.
        """
        # pylint: disable=protected-access
        entity_lf_self = self._get_data_from_store(is_static=False)
        entity_lf_self = self._apply_masks(entity_lf_self, is_static=False)
        entity_lf_self = self._select_columns(
            entity_lf_self, self.settings.entity_features, is_static=False
        )

        entity_lf_other = other._get_data_from_store(is_static=False)
        entity_lf_other = other._apply_masks(entity_lf_other, is_static=False)
        entity_lf_other = entity_lf_other.filter(
            pl.col(other._store.seq_id_col).is_in(other_ids_to_add)
        )
        entity_lf_other = other._select_columns(
            entity_lf_other, self.settings.entity_features, is_static=False
        )

        static_lf_self = self._get_data_from_store(is_static=True)
        if static_lf_self is not None:
            static_lf_self = self._apply_masks(static_lf_self, is_static=True)

        static_lf_other = other._get_data_from_store(is_static=True)
        if static_lf_other is not None:
            static_lf_other = other._apply_masks(static_lf_other, is_static=True)
            static_lf_other = static_lf_other.filter(
                pl.col(other._store.seq_id_col).is_in(other_ids_to_add)
            )

        return (
            pl.concat([entity_lf_self, entity_lf_other]),
            merge_optional_frames(static_lf_self, static_lf_other),
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        destination: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist the current pool state (virtual features + view masks).

        Without *destination* the store is rewritten in-place.
        With *destination* the pool is rebuilt into that path and then
        **redirects to it** - the original files are left untouched.

        In both cases the pool is left in a clean state after a successful
        save: virtual context, masks, soft-drops and cast recipe are all
        reset, and :attr:`is_dirty` becomes ``False``.

        When a mask is active and *destination* is ``None``, the store is
        overwritten with a subset of the data. ``overwrite=True`` is required
        to confirm.

        Args:
            destination: ``None`` → in-place; path → rebuild into new path
                and redirect the pool there.
            overwrite: Required when saving a filtered view in-place.
                Also allows overwriting an existing *destination*.

        Returns:
            The :class:`~pathlib.Path` of the written store.

        Raises:
            FileExistsError: If *destination* exists and *overwrite* is ``False``.
            RuntimeError: If a mask is active in-place without *overwrite*.
        """
        in_place = destination is None or (
            destination is not None
            and Path(resolve_path(destination)).resolve()
            == self._store.root_path.resolve()
        )

        # --- Nothing to do ---
        if not self.is_dirty:
            if in_place:
                warnings.warn(
                    "Nothing to save. Rewrite skipped.", UserWarning, stacklevel=2
                )
                return self._store.root_path
            # Fast-path: plain copy, type_name written explicitly by the pool
            dest_path = self._prepare_dest(destination, overwrite)
            self._store.copy_to(
                dest_path, self.get_registration_name(), exist_ok=overwrite
            )
            LOGGER.info("Store saved to %s.", dest_path)
            return dest_path

        # --- Guard: any in-place save is destructive - always require confirmation ---
        if in_place and not overwrite:
            raise RuntimeError(
                "Saving in-place rewrites the store with the current view. "
                "Use save(overwrite=True) to confirm, "
                "or save(destination) to create a copy and keep the original."
            )

        # --- Prepare frames (virtual merged, casts applied, masks filtered) ---
        entity_lf = self._get_data_from_store(is_static=False)
        entity_lf = self._apply_masks(entity_lf, is_static=False)
        static_lf = self._get_data_from_store(is_static=True)
        if static_lf is not None:
            static_lf = self._apply_masks(static_lf, is_static=True)
            static_lf = self._select_columns(
                static_lf, self.settings.static_features, is_static=True
            )

        # Select features
        entity_lf = self._select_columns(
            entity_lf, self.settings.entity_features, is_static=False
        )

        builder = self.__class__._make_builder()

        if in_place:
            # Write to a sibling temp directory then atomically replace the store.
            tmp = self._store.root_path.parent / f".tmp_{uuid.uuid4().hex}"
            try:
                builder.build_from_frames(tmp, entity_lf, static_lf, presorted=True)
                shutil.rmtree(self._store.root_path)
                shutil.move(str(tmp), str(self._store.root_path))
            except Exception:
                if tmp.exists():
                    shutil.rmtree(tmp)
                raise
            self._store._invalidate_all_caches()
            LOGGER.info("Store saved in-place.")
            dest_path = self._store.root_path
        else:
            dest_path = self._prepare_dest(destination, overwrite)
            builder.build_from_frames(dest_path, entity_lf, static_lf, presorted=True)
            LOGGER.info("Store saved to %s.", dest_path)

        self._reset_to(dest_path)
        return dest_path

    # ------------------------------------------------------------------
    # Type conversions (private helpers)
    # ------------------------------------------------------------------

    def _reinterpret_as(
        self,
        target_pool_cls: type[SequencePool],
        settings: SequenceSettings | dict,
        virtual_id: str | None = None,
    ) -> SequencePool:
        """Swap pool class and settings in memory, reusing this pool's store.

        Zero-I/O. Propagates *virtual_id*, masks, and cast recipes to the new pool.

        Args:
            target_pool_cls: Concrete subclass to instantiate.
            settings: Settings dict or instance for the new pool.
            virtual_id: Virtual context UUID to attach to the new pool.

        Returns:
            A new pool of *target_pool_cls* pointing to the same store.
        """
        # The temporal schema is fully materialized on disk after a type conversion (e.g.
        # _t_event → _t_start/_t_end).  A temporal cast no longer meaningful.
        cast_for_new = (
            self._casts.replace(time_index=[])
            if self._casts.time_index
            else self._casts
        )
        # pylint: disable=protected-access
        # _construct bypasses target_pool_cls.__init__ (validation, resolution, ...).
        return target_pool_cls._construct(
            store=self._store,
            settings=settings,
            cast_recipe=cast_for_new,
            virtual_id=virtual_id,
            id_mask=self._id_mask,
            entity_filter_expr=self._entity_filter_expr,
            has_soft_drops=self._has_soft_drops,
            t0_setter=self._t0_setter,
        )

    def _prepare_dest(self, destination: str | Path, overwrite: bool) -> Path:
        """Resolve *destination* to an absolute ``Path`` and make room for output.

        Raises :exc:`FileExistsError` when *destination* already exists and
        *overwrite* is ``False``.  Removes the existing directory tree otherwise.
        """
        dest_path = resolve_path(destination)
        if dest_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {dest_path}. "
                    "Use overwrite=True to replace it."
                )
            shutil.rmtree(dest_path)
        return dest_path

    def _persist_as(
        self,
        target_cls: type[SequencePool],
        settings: dict,
        virtual_id: str | None,
        destination: str | Path,
        overwrite: bool = False,
    ) -> SequencePool:
        """Write a prepared conversion to a persistent store and return a fresh pool.

        Creates an ephemeral pool of *target_cls* (via :meth:`_reinterpret_as`)
        then calls its :meth:`save` to delegate all I/O to the builder.  This
        ensures the correct ``type_name`` is baked into ``core.json`` and that
        the full write pipeline (sort gate, metadata inference) runs exactly
        once.

        Args:
            target_cls: Concrete subclass to produce.
            settings: Constructor kwargs for *target_cls*.
            virtual_id: Forked virtual context holding the converted temporal.
            destination: Filesystem path for the new store.
            overwrite: Replace *destination* if it already exists.

        Returns:
            A new pool of *target_cls* loaded from *destination*.
        """
        ephemeral = self._reinterpret_as(target_cls, settings, virtual_id)
        dest_path = ephemeral.save(destination=destination, overwrite=overwrite)
        # Bypass __init__: the written store is clean (all features physical, no
        # virtual context, no masks).  _construct avoids redundant validation and
        # probe.  T0 strategy is propagated; everything else is at fresh defaults.
        return target_cls._construct(
            store=self._resolve_store(dest_path),
            settings=settings,
            cast_recipe=SequenceCastRecipe(),
            virtual_id=None,
            id_mask=None,
            entity_filter_expr=None,
            has_soft_drops=False,
            t0_setter=self._t0_setter,
        )

    def _as_event_from_period(
        self,
        anchor: Literal["start", "end", "middle"],
        time_column: str,
        destination: str | Path | None = None,
        overwrite: bool = False,
    ) -> SequencePool:
        """Shared period→event conversion used by State and Interval pools.

        Forks the virtual context projecting ``_t_event`` from *anchor*
        (``_t_start``, ``_t_end``, or their midpoint), then branches on *destination*.

        Raises:
            ValueError: If *anchor* is not ``'start'``, ``'end'``, or ``'middle'``.
        """
        if anchor not in ("start", "end", "middle"):
            raise ValueError(
                f"anchor must be 'start', 'end', or 'middle', got {anchor!r}"
            )
        new_uuid = self._store._fork_period_to_event(
            self._virtual_id,
            anchor,
            time_index_caster=self._casts.time_index_caster(),
            time_index_dtype=self._casts.time_index_dtype,
        )
        new_settings = {
            "id_column": self.settings.id_column,
            "time_column": time_column,
            "entity_features": list(self.settings.entity_features),
            "static_features": list(self.settings.static_features),
        }
        event_cls = SequencePool.get_registered("event")
        if destination is None:
            return self._reinterpret_as(event_cls, new_settings, new_uuid)
        return self._persist_as(
            event_cls, new_settings, new_uuid, destination, overwrite
        )
