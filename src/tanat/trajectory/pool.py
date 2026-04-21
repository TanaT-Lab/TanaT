#!/usr/bin/env python3
"""TrajectoryPool: aggregation of SequencePool views."""

from __future__ import annotations

import logging
import random
import shutil
import uuid
import warnings
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Literal
import weakref

import numpy as np
import polars as pl
import pandas as pd
from tanat_utils import Cachable, CachableSettings
from tanat_utils.pretty_format import (
    format_header,
    format_section,
    format_kv,
    format_bullet,
    format_feature_section,
)

from ..store.trajectory.builder import TrajectoryStoreBuilder
from ..store.trajectory.schema import TrajectorySchema as TSCH
from ..core.path import resolve_path
from ..core.format import resolve_fmt, to_pandas
from ..sequence.base.pool import BinSize, SequencePool
from ..sequence.base._utils import merge_optional_frames, resolve_ids_to_add
from ..core import registry as _registry
from ..store.base.utils import normalise_to_lazyframe
from ..zeroing import T0Setter, T0Value, _T0, _T0_NEAREST_RANK
from .cast import TrajectoryCastRecipe
from .settings import TrajectorySettings
from .trajectory import Trajectory
from .view_mixin import TrajectoryViewMixin

if TYPE_CHECKING:
    from ..store.trajectory.store import TrajectoryStore

LOGGER = logging.getLogger(__name__)


class TrajectoryPool(TrajectoryViewMixin, CachableSettings):
    """
    Aggregates :class:`SequencePool` views into trajectories.

    Accepts a store name, path, or :class:`TrajectoryStore` instance,
    following the same convention as :class:`SequencePool`.

    Usage::

        store_path = (
            TrajectoryPool.builder()
            .add("medical", medical_pool)
            .add("lab", lab_pool)
            .build("./my_trajectories")
        )
        pool = TrajectoryPool(store="./my_trajectories")
    """

    SETTINGS_CLASS = TrajectorySettings

    def __init__(
        self,
        store: str | Path | TrajectoryStore,
        *,
        id_column: str = "id",
        static_features: list[str] | None = None,
        cast_recipe: TrajectoryCastRecipe | dict | None = None,
    ) -> None:
        """Create a trajectory pool backed by *store*.

        Args:
            store: Store path, name, or :class:`TrajectoryStore` instance.
            id_column: User-facing name for the trajectory ID column.
            static_features: Static feature names to expose.
                ``None`` → all available.  ``[]`` → none.
            cast_recipe: Optional cast recipe (or dict) applied at read time.
                Only ``id`` and ``static`` fields are meaningful at this level.
                Normalised via :meth:`TrajectoryCastRecipe.coerce` and probed
                eagerly.
        Raises:
            TypeError: If *cast_recipe* is not a :class:`TrajectoryCastRecipe`,
                ``dict``, or ``None``.
        """
        self._store = self._resolve_store(store)
        sf = self._resolve_features(self._store, static_features)

        self._alias_mask: set[str] | None = None
        self._id_mask: set | None = None
        self._virtual_id: str | None = None
        self._pools: dict | None = None
        self._has_soft_drops: bool = False
        self._t0_setter: T0Setter = T0Setter.default()

        CachableSettings.__init__(
            self, settings=TrajectorySettings(id_column=id_column, static_features=sf)
        )

        self._casts: TrajectoryCastRecipe = TrajectoryCastRecipe.coerce(cast_recipe)
        if not self._casts.is_empty():
            self._casts.probe(self._store)

        # -- GC safety --------------------------------------------------------
        # Registered at the very end: if __init__ raises (e.g. during probe()),
        # the object is never fully constructed and the finalizer won't run on
        # a half-built instance.  weakref.finalize runs before sys.modules is
        # cleared (unlike __del__), so pathlib remains available.
        self._gc_state: list = [self._store, None]  # [store, virtual_id]
        weakref.finalize(self, TrajectoryPool._finalize_cleanup, self._gc_state)
        # ---------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def builder(cls) -> TrajectoryStoreBuilder:
        """Return a fluent builder for constructing a trajectory store."""
        return TrajectoryStoreBuilder()

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_virtual_id(self) -> str:
        """Returns the virtual ID for the current session (lazy-created)."""
        if self._virtual_id is None:
            self._virtual_id = str(uuid.uuid4())
            self._gc_state[1] = self._virtual_id
        return self._virtual_id

    def _cleanup_virtual(self) -> None:
        """Removes the virtual context from disk."""
        if self._virtual_id is not None:
            self._store.clear_virtual_context(self._virtual_id)
            self._virtual_id = None
            self._gc_state[1] = None

    def _reset_to(self, dest_path: Path) -> None:
        """Reset all dirty state and redirect the pool to *dest_path*.

        Called after every successful :meth:`save`.  Mirrors the contract of
        :meth:`SequencePool._reset_to` with one trajectory-specific nuance:

        For both **in-place** and **destination** saves, all casts are baked
        into the written stores (trajectory files + sequence stores), so
        ``_casts`` is fully reset in both cases.

        Steps:

        1. Clears the virtual context against the **current** store.
        2. Reloads ``self._store`` from *dest_path* (picks up rewritten ``core.json``).
        3. Fully resets ``_casts``, masks and soft-drop flag.
        4. Drops ``_pools`` to force a rebuild against the updated store.
        5. Invalidates the settings cache.
        """
        self._cleanup_virtual()  # resets _gc_state[1] to None
        self._store = self._resolve_store(dest_path)  # reload (links may have changed)
        self._gc_state[0] = self._store
        self._casts = TrajectoryCastRecipe()  # all baked into written stores
        self._id_mask = None
        self._has_soft_drops = False
        self._pools = None  # force rebuild against new/updated store
        self._t0_setter = T0Setter.default(is_event=False)
        self.clear_cache()

    @classmethod
    def _construct(
        cls,
        *,
        store: TrajectoryStore,
        settings: TrajectorySettings,
        cast_recipe: TrajectoryCastRecipe,
        virtual_id: str | None,
        id_mask: set | None,
        alias_mask: set[str] | None,
        has_soft_drops: bool,
        pools: dict | None,
        t0_setter: T0Setter | None = None,
    ) -> TrajectoryPool:
        """Construct a :class:`TrajectoryPool` without going through ``__init__``.

        Bypasses all store probing and feature validation: *settings* and
        *cast_recipe* must already be fully resolved.

        Called by :meth:`copy`.

        Args:
            store: Shared :class:`TrajectoryStore` backing the new pool.
            settings: Already-resolved :class:`TrajectorySettings` (no re-validation).
            cast_recipe: Already-probed :class:`TrajectoryCastRecipe`.
            virtual_id: Forked virtual context UUID (or ``None``).
            id_mask: Set of trajectory IDs to expose (or ``None`` for all).
            alias_mask: Set of sequence-store aliases to expose (or ``None`` for all).
            has_soft_drops: Whether soft-dropped trajectories exist.
            pools: Pre-built sequence pool registry (or ``None`` to rebuild lazily).
            t0_setter: T0 strategy to propagate.  ``None`` creates a fresh default.
        """
        pool = object.__new__(cls)
        pool._store = store
        CachableSettings.__init__(pool, settings=settings)
        pool._casts = cast_recipe
        pool._gc_state = [store, virtual_id]
        weakref.finalize(pool, TrajectoryPool._finalize_cleanup, pool._gc_state)
        pool._virtual_id = virtual_id
        pool._id_mask = id_mask
        pool._alias_mask = alias_mask
        pool._has_soft_drops = has_soft_drops
        pool._pools = pools
        pool._t0_setter = (
            t0_setter if t0_setter is not None else T0Setter.default(is_event=False)
        )
        return pool

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------

    @property
    def sequence_pools(self) -> MappingProxyType:
        """
        Visible :class:`SequencePool` instances, keyed by alias.

        Returns a **read-only** mapping filtered by the current alias mask.
        Direct item assignment (e.g. ``tpool.sequence_pools[alias] = …``)
        raises :exc:`TypeError` - use :meth:`subset` or
        :meth:`drop_sequence_pools` to change the visible pools.
        """
        if self._pools is None:
            self._pools = self._build_pools()
        if self._alias_mask is None:
            return MappingProxyType(self._pools)
        return MappingProxyType(
            {a: self._pools[a] for a in self._alias_mask if a in self._pools}
        )

    def _build_pools(self) -> dict[str, SequencePool]:
        """Create pool objects for all store aliases via :meth:`SequencePool.from_parent`."""
        result = {}
        for alias in self._store.store_aliases:  # ALL aliases, unfiltered
            store_path = self._store.sequence_stores[alias].root_path
            # pylint: disable=protected-access
            pool = SequencePool.from_parent(store_path, parent_pool=self)
            result[alias] = pool
        return result

    def _sync_pool_casts(self) -> None:
        """Propagate trajectory-level id/time index casts to existing pool objects.

        Called **in-place** after :meth:`cast_id`, :meth:`cast_to_datetime`,
        or :meth:`cast_to_timestep` so that pool objects are updated without
        being discarded.  This preserves any entity casts previously applied
        via :meth:`~SequencePool.cast_features` on those pools.

        If ``_pools`` is not yet initialised the casts will be applied on
        first access via :meth:`_build_pools`.
        """
        if self._pools is None:
            return
        for pool in self._pools.values():
            # pylint: disable=protected-access
            # Propagate id/time index casts - already validated at trajectory level.
            pool._casts = pool._casts.replace(
                id=self._casts.id, time_index=self._casts.time_index
            )
            pool.clear_cache()

    @Cachable.cached_method()
    def get_trajectories(
        self,
        static_features: list[str] | None = None,
        aliases: list[str] | None = None,
    ) -> dict[str, Trajectory]:
        """
        All visible :class:`Trajectory` instances, keyed by ID.

        Materialises every trajectory reachable through the current view
        (respecting ``_id_mask`` and ``_alias_mask``).  Useful for
        iteration-heavy workflows where the same trajectory is accessed
        multiple times.

        Args:
            static_features: Static features to expose in each
                :class:`Trajectory`.  ``None`` → use the pool-level
                setting.  ``[]`` → no static features.
            aliases: Sequence-store aliases to expose in each
                :class:`Trajectory`.  ``None`` → use the pool-level
                alias mask.  Must be a subset of the pool's visible
                aliases.

        Raises:
            KeyError: If any alias in *aliases* is not visible in
                the current pool view.
        """
        # Resolve effective alias mask for the returned trajectories
        effective_mask: set[str] | None
        if aliases is not None:
            visible = set(self._store_aliases)
            unknown = set(aliases) - visible
            if unknown:
                raise KeyError(
                    f"Unknown aliases: {sorted(unknown)}. "
                    f"Available: {sorted(visible)}"
                )
            effective_mask = set(aliases)
        else:
            effective_mask = None  # _build_trajectory falls back to self._alias_mask

        # Validate static_features early, before iterating over all IDs.
        # Build settings if static_features is overridden.
        prebuilt: TrajectorySettings | None = None
        if static_features is not None:
            self.settings.validate_features(static_features)
            prebuilt = replace(self.settings, static_features=static_features)
        return {
            tid: self._build_trajectory(tid, prebuilt, alias_mask=effective_mask)
            for tid in self.unique_ids
        }

    def drop_sequence_pools(self, *aliases: str) -> None:
        """
        Hides one or more store aliases from this view.

        The underlying :class:`TrajectoryStore` is **not** modified.
        Only the pool's visible aliases (and derived properties like
        :attr:`trajectory_index` and :attr:`unique_ids`) are affected.

        Args:
            aliases: One or more alias names to hide.

        Raises:
            RuntimeError: If the pool is not built yet.
            KeyError: If an alias does not exist in the store.
        """
        all_store_aliases = set(self._store.store_aliases)
        for alias in aliases:
            if alias not in all_store_aliases:
                raise KeyError(
                    f"Alias '{alias}' not found in store. "
                    f"Available: {sorted(all_store_aliases)}"
                )
        if self._alias_mask is None:
            self._alias_mask = set(all_store_aliases)
        self._alias_mask -= set(aliases)
        # Remove dropped aliases from _pools directly
        if self._pools is not None:
            for alias in aliases:
                self._pools.pop(alias, None)
        self.clear_cache()

    # ------------------------------------------------------------------
    # Properties (delegated to store, filtered by view)
    # ------------------------------------------------------------------

    @Cachable.cached_property
    def unique_ids(self) -> list:
        """Visible trajectory IDs as a plain Python list.

        Respects ``_id_mask``.

        .. warning::

           ``list`` erases rich Polars dtypes.
           Prefer :attr:`_id_lf` when the result feeds a Polars join.
        """
        return self._id_lf.collect().to_series().to_list()

    @Cachable.cached_property
    def _store_aliases(self) -> list[str]:
        """Aliases visible through the current view mask."""
        all_aliases = self._store.store_aliases
        if self._alias_mask is None:
            return all_aliases
        return [a for a in all_aliases if a in self._alias_mask]

    def __len__(self) -> int:
        """Number of trajectories visible in this view."""
        return len(self.unique_ids)

    def __repr__(self) -> str:
        aliases = self._store_aliases
        n_static = len(self.settings.static_features)
        return (
            f"TrajectoryPool(n={len(self)}, sequences={aliases}, "
            f"static_features={n_static}, store='{self._store.root_path}')"
        )

    def __str__(self) -> str:
        meta = self.metadata
        pools = self.sequence_pools

        overview = [
            format_kv("Trajectories", f"{len(self):,}"),
            format_kv("Store", str(self._store.root_path)),
            format_kv("id_column", self.settings.id_column),
        ]
        ti_section = [
            format_kv("Type", str(meta.time_index)),
            format_kv("t0", self._t0_setter.strategy_summary),
        ]
        seq_bullets = [
            format_bullet(alias, repr(pool)) for alias, pool in pools.items()
        ]

        parts = [
            format_header("TrajectoryPool Summary"),
            "",
            format_section("Overview", overview),
            "",
            format_section("Time Index", ti_section),
            "",
            format_section(f"Sequences ({len(pools)})", seq_bullets),
        ]

        sf_section = format_feature_section(
            "Static Features",
            [(f.name, f.summary) for f in (meta.static_features or [])],
        )
        if sf_section:
            parts += ["", sf_section]

        return "\n".join(parts)

    @property
    def is_dirty(self) -> bool:
        """``True`` if the pool (or any linked sequence pool) has unsaved state.

        Trajectory-level: virtual features, ID mask, type casts, soft drops.
        Sub-pool level: delegates to each pool's
        :attr:`~tanat.sequence.base.pool.SequencePool.is_dirty`.

        A dirty pool needs :meth:`save` with a destination to materialise
        all pending changes (sub-pool changes cannot be saved in-place).
        """
        if (
            self._virtual_id is not None
            or self._id_mask is not None
            or not self._casts.is_empty()
            or self._has_soft_drops
        ):
            return True
        return any(p.is_dirty for p in self.sequence_pools.values())

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def _build_trajectory(
        self,
        traj_id,
        settings=None,
        *,
        alias_mask: set[str] | None = None,
    ) -> Trajectory:
        """Build a :class:`Trajectory` for *traj_id* without any validity check.

        For internal use only - callers must guarantee that *traj_id* is
        present in the current view (e.g. when iterating over
        :attr:`unique_ids` or building :attr:`trajectories`).

        Args:
            traj_id: A trajectory ID already known to be in the view.
            settings: Pre-built :class:`TrajectorySettings` to use.
                ``None`` → use :attr:`settings` as-is (no copy, no override).
            alias_mask: Override the pool-level alias mask.
                ``None`` → use :attr:`_alias_mask`.
        """
        s = settings if settings is not None else self.settings
        mask = alias_mask if alias_mask is not None else self._alias_mask
        return Trajectory.from_parent(
            id_value=traj_id,
            store=self._store,
            settings=s,
            parent_pool=self,
            alias_mask=mask,
        )

    def __getitem__(self, traj_id) -> Trajectory:
        if traj_id not in set(self.unique_ids):
            preview = self.unique_ids[:5]
            suffix = "..." if len(self) > 5 else ""
            raise KeyError(
                f"Trajectory '{traj_id}' not found. " f"Available: {preview}{suffix}"
            )
        return self._build_trajectory(traj_id)

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterate over all visible :class:`Trajectory` objects in :attr:`unique_ids` order."""
        for tid in self.unique_ids:
            yield self._build_trajectory(tid)

    def items(self) -> Iterator[tuple]:
        """Yield ``(id, trajectory)`` pairs for all visible trajectories."""
        for tid in self.unique_ids:
            yield tid, self._build_trajectory(tid)

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _resolve_ids(self, ids: list) -> list | None:
        """
        Validates *ids* against the current view and intersects with ``_id_mask``.

        Raises ``KeyError`` if any ID is not present in :attr:`unique_ids`
        (unknown or masked IDs are both rejected).

        Returns ``None`` when no filtering is needed (both are ``None``).
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
            preview = unknown[:5]
            suffix = "..." if len(unknown) > 5 else ""
            raise KeyError(
                f"Unknown trajectory IDs: {preview}{suffix}. "
                f"Use `unique_ids` to inspect available IDs."
            )

        if self._id_mask is None:
            return list(ids)
        return [uid for uid in ids if uid in self._id_mask]

    def _apply_masks(
        self,
        lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Applies ``_id_mask`` to a LazyFrame."""
        if self._id_mask is not None:
            lf = lf.filter(pl.col(self._store.traj_id_col).is_in(self._id_mask))
        return lf

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @Cachable.cached_method()
    def _static_data_raw(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame | None:
        """Cached data layer - always returns a Polars DataFrame or None.

        Used internally by :meth:`static_data`.  Cache invalidated by
        :meth:`clear_cache` (e.g. after ``cast_features``, ``drop_features``).
        """
        visible_features = self._resolve_valid_features(features)
        if not visible_features:
            return None
        lf = self._get_static_data_from_store()
        if lf is None:
            return None
        lf = self._apply_masks(lf)
        lf = self._select_columns(lf, visible_features)
        lf = self._rename_columns(lf)
        return lf.collect()

    def static_data(
        self,
        features: list[str] | str | None = None,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pl.DataFrame | pd.DataFrame | None:
        """Return trajectory-level static data for visible trajectories.

        Args:
            features: Static feature name(s) to include.
                ``None`` -> all visible static features.
            fmt: ``"pandas"`` *(default)* or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            One-row-per-trajectory DataFrame with columns ``[id, feature...]``.
            ``None`` when no static features are exposed by this pool view.

        To restrict to a subset of IDs, use ``pool.subset(ids).static_data()``.
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        df = self._static_data_raw(features)
        if df is None:
            return None
        if fmt == "polars":
            return df
        return to_pandas(df, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_static_features(
        self,
        df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        *,
        id_column: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add static features to the trajectory pool via an ID-keyed join.

        The input DataFrame **must** include the trajectory ID column (either
        under ``settings.id_column`` or under the name given by *id_column*).
        A LEFT JOIN against the full trajectory index is performed internally,
        so partial DataFrames (covering only a subset of trajectory IDs) are
        valid: absent IDs receive ``null`` in the virtual context.

        Because alignment is handled by the join rather than by row position,
        this method works on views with pending changes (cast, virtual features, masks).
        Only the IDs visible in the view are exposed when reading back with
        :meth:`static_data`.

        Args:
            df: DataFrame containing the ID column plus one or more feature
                columns.  Can be pandas, Polars eager, or Polars lazy.
            id_column: Name of the ID column in *df*.  Defaults to
                ``settings.id_column`` when ``None``.  Pass an explicit name
                when the join key in *df* differs from the pool's public ID
                name (e.g. ``id_column="traj_id"``).
            overwrite: If ``True``, replaces features that already exist in
                the virtual context.

        Raises:
            KeyError: If the resolved ID column is not found in *df*.
        """
        resolved_id_col = (
            id_column if id_column is not None else self.settings.id_column
        )

        # Normalise first - schema inspection is then uniform via collect_schema().
        lf = normalise_to_lazyframe(df)
        df_cols = lf.collect_schema().names()

        if resolved_id_col not in df_cols:
            fallback = self.settings.id_column
            raise KeyError(
                f"ID column {resolved_id_col!r} not found in df "
                f"(columns: {df_cols}). "
                f'Pass id_column="<name>" or rename the column to {fallback!r}.'
            )

        # Early collision detection via settings - fail fast before any I/O.
        feature_cols = [c for c in lf.collect_schema().names() if c != resolved_id_col]
        known = set(self.settings.available_features())
        collisions = [c for c in feature_cols if c in known]
        if collisions and not overwrite:
            raise ValueError(
                f"Feature collision: {collisions} already exist in this pool. "
                "Use overwrite=True to replace them."
            )

        vid = self._get_virtual_id()
        new_cols = self._store.add_static(vid, lf, id_col=resolved_id_col)

        # Update settings so the view sees the new features
        current = self.settings.available_features()
        updated = current + [c for c in new_cols if c not in current]
        self.update_settings(static_features=updated)

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------

    @Cachable.cached_method()
    def _describe_result(self, separator: str = "_") -> pl.DataFrame:
        """Cached polars result (one row per trajectory) for :meth:`describe`.

        For each visible sequence pool, calls ``pool.describe()`` and
        prefixes its metric columns with ``{alias}{separator}``.  Results
        are joined horizontally on the trajectory ID.  A ``n_sequences``
        column (number of visible stores) is prepended.  Cached: invalidated
        automatically when settings change.

        Args:
            separator: Separator between alias and metric name (default ``_``).

        Returns:
            Polars DataFrame with columns
            ``[id, n_sequences, {alias}{sep}length, …]``.
        """
        id_col = self.settings.id_column
        pools = self.sequence_pools
        n_aliases = len(pools)

        frames: list[pl.DataFrame] = []
        for alias, seq_pool in pools.items():
            per_id: pl.DataFrame = seq_pool.describe(by_id=True, fmt="polars")
            # Rename metrics with alias prefix; keep the id column unchanged.
            metric_cols = [
                c for c in per_id.columns if c != seq_pool.settings.id_column
            ]
            renamed = {c: f"{alias}{separator}{c}" for c in metric_cols}
            # Ensure id column name matches the trajectory-level id_column.
            if seq_pool.settings.id_column != id_col:
                renamed[seq_pool.settings.id_column] = id_col
            per_id = per_id.rename(renamed)
            frames.append(per_id)

        if not frames:
            raise ValueError("No sequence pools available for describe().")

        # Horizontal join on the ID column
        result = frames[0]
        for frame in frames[1:]:
            result = result.join(frame, on=id_col, how="full", coalesce=True)

        # Prepend n_sequences
        return result.with_columns(pl.lit(n_aliases).alias("n_sequences")).select(
            [id_col, "n_sequences"]
            + [c for c in result.columns if c not in (id_col, "n_sequences")]
        )

    def describe(
        self,
        by_id: bool = True,
        add_to_static: bool = False,
        separator: str = "_",
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Compute summary statistics across all sequences and all trajectories.

        Args:
            by_id: If ``True`` *(default)*, return one row per trajectory.
                If ``False``, return cross-trajectory pandas ``.describe()``.
            add_to_static: If ``True``, persist the per-ID result via
                :meth:`add_static_features`.  Ignored (with a warning) when
                ``by_id=False``.
            separator: Separator between alias and metric name (default ``_``).
            fmt: ``"pandas"`` *(default)* or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            DataFrame with columns
            ``[id, n_sequences, {alias}{sep}length, …]``.

        Examples::

            traj_pool.describe()
            traj_pool.describe(separator=".")
            traj_pool.describe(by_id=False)
            traj_pool.describe(add_to_static=True)
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        result = self._describe_result(separator)

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
            numeric = result.drop(self.settings.id_column)
            return to_pandas(numeric, use_arrow=use_arrow).describe()

        if fmt == "polars":
            return result
        return to_pandas(result, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # T0 / Zeroing
    # ------------------------------------------------------------------

    def set_t0(
        self,
        *,
        position: int | None = None,
        direct: T0Value | dict[Any, T0Value] | None = None,
        feature: str | None = None,
        query: pl.Expr | None = None,
        anchor: Literal["start", "end", "middle"] | None = None,
        use_first: bool = True,
        on: str | None = None,
    ) -> TrajectoryPool:
        """Configure the T0 strategy for this trajectory pool.

        Builds a :class:`~tanat.zeroing.base.T0Setter` via the registry and
        delegates to ``setter.compute_from_trajectory(self, on=on)``.  The
        setter stores the resulting ``[id_col, _T0_]`` DataFrame; per-alias
        nearest ranks are computed lazily in :meth:`_get_traj_t0_df`.

        Args:
            position: Row index (0-based; negative indexing supported).
            direct:   Scalar value or ``{traj_id: value}`` dict.
            feature:  Trajectory-level static feature column name.
            query:    Polars boolean expression evaluated on the reference
                      sub-pool's columns.
            anchor:   ``"start"`` / ``"end"`` / ``"middle"`` for interval/state pools.
            use_first: For the *query* strategy only.
            on:       Alias of the sub-pool used to compute T0.
                      Required for ``position`` and ``query`` strategies.
                      Ignored (with warning) for ``direct`` and ``feature``.

        Returns:
            ``self`` for chaining.

        Raises:
            TypeError: If ``on`` is missing for ``position``/``query``.
            KeyError:  If ``on`` refers to an alias not visible in this pool.
        """
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

        # Validate/warn about `on=` usage.
        if name in ("position", "query") and on is None:
            raise TypeError(
                f"set_t0() with strategy '{name}' requires the 'on=' parameter "
                "to designate the reference sub-pool."
            )
        if name in ("direct", "feature") and on is not None:
            warnings.warn(
                f"'on=' is ignored for the '{name}' strategy and will be silently "
                "discarded.",
                UserWarning,
                stacklevel=2,
            )
            on = None

        strategy_kwargs: dict[str, dict] = {
            "position": {"position": position, "anchor": anchor},
            "direct": {"direct": direct},
            "feature": {"feature": feature},
            "query": {"query": query, "anchor": anchor, "use_first": use_first},
        }
        setter = T0Setter.get_registered(name)(**strategy_kwargs[name])
        setter.compute_from_trajectory(self, on=on)

        self._t0_setter = setter
        # Clear sub-pool caches so they pick up the new setter on next access.
        if self._pools is not None:
            for pool in self._pools.values():
                pool.clear_cache()
        self.clear_cache()
        return self

    @Cachable.cached_method()
    def _get_traj_t0_df(self) -> pl.DataFrame:
        """Cached T0 DataFrame with columns ``[id_col, _T0_, <alias>_T0_NEAREST_RANK_, ...]``.

        Triggers lazy computation if ``set_t0()`` was never called, then
        joins per-alias nearest ranks.
        """
        setter = self._t0_setter
        if setter.df is None:
            # Lazy trigger: no explicit set_t0() yet.  Default setter
            # (position=0, anchor=start) resolves on=None to the first
            # visible alias.
            setter.compute_from_trajectory(self)

        id_col = self.settings.id_column
        t0_lf = setter.df.lazy()
        if self._id_mask is not None:
            t0_lf = t0_lf.filter(pl.col(id_col).is_in(self._id_mask))

        result_lf = t0_lf.select([id_col, _T0])
        for alias, pool in self.sequence_pools.items():
            # pylint: disable=protected-access
            rank_lf = pool._nearest_rank_lf(t0_lf).rename(
                {_T0_NEAREST_RANK: f"{alias}{_T0_NEAREST_RANK}"}
            )
            result_lf = result_lf.join(rank_lf, on=id_col, how="left")

        # Apply same null-handling as _resolve_nearest_rank:
        # _T0_ null → rank null; _T0_ set but no floor → rank 0.
        rank_cols = [f"{a}{_T0_NEAREST_RANK}" for a in self.sequence_pools]
        result_lf = result_lf.with_columns(
            pl.when(pl.col(_T0).is_null())
            .then(pl.lit(None, dtype=pl.UInt32))
            .otherwise(pl.col(rc).fill_null(pl.lit(0, dtype=pl.UInt32)).cast(pl.UInt32))
            .alias(rc)
            for rc in rank_cols
        )
        return result_lf.collect()

    @Cachable.cached_method()
    def _get_traj_t0_lookup(self) -> dict:
        """O(1)-per-trajectory lookup built once from :meth:`_get_traj_t0_df`.

        Returns ``{id_value: (t0, {alias: nearest_rank})}`` where each
        trajectory's alias dict contains only the aliases where that
        trajectory has data, computed in a single trajectory-index read
        rather than one per trajectory.

        Cache invalidated by :meth:`clear_cache`.
        """
        df = self._get_traj_t0_df()
        id_col = self.settings.id_column
        # Columns are named f"{alias}{_T0_NEAREST_RANK}"
        rank_cols = [c for c in df.columns if c.endswith(_T0_NEAREST_RANK)]

        # Build presence map {alias: frozenset_of_ids} from the trajectory
        # index: one .collect() for all aliases instead of one per trajectory.
        traj_id_col = self._store.traj_id_col
        traj_idx = self._store.trajectory_index.collect()
        id_caster = self._casts.id_caster()
        if id_caster is not None:
            traj_idx = traj_idx.with_columns(id_caster(pl.col(traj_id_col)))

        presence: dict[str, frozenset] = {}
        for rc in rank_cols:
            alias = rc[: -len(_T0_NEAREST_RANK)]
            if alias in traj_idx.columns:
                presence[alias] = frozenset(
                    traj_idx.filter(pl.col(alias))[traj_id_col].to_list()
                )
            else:
                presence[alias] = frozenset()

        # Vectorised extraction: all columns in one pass.
        ids = df[id_col].to_list()
        t0s = df[_T0].to_list()
        alias_ranks: dict[str, list] = {
            rc[: -len(_T0_NEAREST_RANK)]: df[rc].to_list() for rc in rank_cols
        }

        return {
            tid: (
                t0,
                {
                    alias: alias_ranks[alias][i]
                    for alias in alias_ranks
                    if tid in presence.get(alias, frozenset())
                },
            )
            for i, (tid, t0) in enumerate(zip(ids, t0s))
        }

    def t0_data(
        self,
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pl.DataFrame | pd.DataFrame:
        """Return the T0 table for all visible trajectories.

        Columns: ``[id_col, _T0_, <alias1>_T0_NEAREST_RANK_, ...]``.
        Each alias gets its own nearest-rank column because the floor lookup
        depends on the alias-specific temporal index.

        Args:
            fmt: ``"pandas"`` (default) or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            One row per visible trajectory ID.
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        df = self._get_traj_t0_df()
        if fmt == "polars":
            return df
        return to_pandas(df, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # Copy / Subset
    # ------------------------------------------------------------------

    def _propagate_id_mask_to_pools(self, effective: set | None) -> None:
        """Propagate a new trajectory-level ID mask to already-initialised sub-pools.

        Each sub-pool's visible IDs are restricted to ``effective ∩ store_ids``
        where *store_ids* is the full unfiltered ID set for that sub-pool's
        store.  If *effective* is ``None`` (no restriction), all sub-pool
        masks are cleared.

        When ``_pools`` is ``None`` (not yet built), the mask will be applied
        automatically on first access via :meth:`_build_pools` →
        :meth:`SequencePool.from_parent`.

        Args:
            effective: New trajectory-level ID mask, or ``None`` to clear.
        """
        if self._pools is None:
            return
        for pool in self._pools.values():
            # pylint: disable=protected-access
            if effective is None:
                pool._id_mask = None
            else:
                # Temporarily lift the sub-pool's mask to enumerate all IDs
                # available in its store (same approach as from_parent()).
                pool._id_mask = None
                pool.clear_cache()
                store_ids = set(pool.unique_ids)
                pool._id_mask = effective & store_ids
            pool.clear_cache()

    def _copy_with_mask(self, id_mask: set | None) -> TrajectoryPool:
        """Build a pool copy with *id_mask* applied in a single pass.

        Handles two cases transparently:

        * **Sub-pools already initialised**: each pool is shallow-copied
          (preserving entity casts from
          :meth:`~tanat.sequence.base.pool.SequencePool.cast_features`) and
          re-parented; :meth:`_propagate_id_mask_to_pools` then applies the
          intersection ``id_mask ∩ store_ids`` exactly once.

        * **Sub-pools not yet built**: ``_pools`` is left ``None``; on first
          access :meth:`_build_pools` calls
          :meth:`~tanat.sequence.base.pool.SequencePool.from_parent` which
          reads ``new_pool._id_mask`` and applies the same intersection
          automatically.

        Args:
            id_mask: Trajectory IDs to expose in the new pool, or ``None``
                for no restriction.
        """
        # pylint: disable=protected-access
        copied_pools = None
        if self._pools is not None:
            copied_pools = {}
            for alias, seqpool in self._pools.items():
                p = seqpool.copy()
                p._locked = True  # re-lock: copy() resets _locked to False
                copied_pools[alias] = p

        new_pool = TrajectoryPool._construct(
            store=self._store,
            settings=self.settings,
            cast_recipe=self._casts,
            virtual_id=self._store.fork_virtual_context(self._virtual_id),
            id_mask=id_mask,
            alias_mask=set(self._alias_mask) if self._alias_mask is not None else None,
            has_soft_drops=self._has_soft_drops,
            pools=copied_pools,
            t0_setter=self._t0_setter,
        )
        if copied_pools is not None:
            for p in copied_pools.values():
                p._parent_pool = new_pool
            new_pool._propagate_id_mask_to_pools(id_mask)
        return new_pool

    def copy(self) -> TrajectoryPool:
        """Return a shallow copy sharing the same store, with all view state preserved.

        The new pool references the same :class:`TrajectoryStore` and the same
        virtual context (``_virtual_id``) so virtual features are immediately
        visible.

        Returns:
            A new :class:`TrajectoryPool` with identical settings, casts,
            masks and virtual context.

        Note:
            Chaining with :meth:`save` produces a fully independent pool at a
            new path **without** mutating the original instance::

                pool2 = TrajectoryPool(store=pool.copy().save("other_path"))

            Use this when you need both the original and a snapshot at a new
            destination.  ``pool.save("other_path")`` alone would redirect
            *pool* itself to ``"other_path"``.

        See Also:
            :meth:`save`
        """
        return self._copy_with_mask(
            set(self._id_mask) if self._id_mask is not None else None
        )

    def subset(self, ids, *, inplace: bool = False) -> TrajectoryPool:
        """Return a view restricted to the given trajectory IDs.

        All IDs must be present in the current :attr:`unique_ids` (i.e. they
        must pass the existing mask, if any).  The new view inherits the full
        pool state (casts, virtual features, alias mask).

        Args:
            ids: Trajectory ID(s) to keep.  A single value is accepted and
                treated as a one-element list.
            inplace: If ``True``, modify this pool in-place rather than
                returning a new instance.

        Returns:
            A :class:`TrajectoryPool` restricted to *ids* (or ``self`` when
            *inplace=True*).

        Raises:
            KeyError: If any ID is not present in :attr:`unique_ids`.
        """
        effective = set(self._resolve_ids(ids))

        if inplace:
            self._id_mask = effective
            self._propagate_id_mask_to_pools(effective)
            self.clear_cache()
            return self

        return self._copy_with_mask(effective)

    def train_test_split(
        self,
        *,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int | None = None,
        shuffle: bool = True,
    ) -> tuple[TrajectoryPool, TrajectoryPool]:
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
            ``(train_pool, test_pool)`` - two new non-overlapping pool views.

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

    def drop_static_features(
        self,
        features: list[str] | str,
        *,
        permanently: bool = False,
    ) -> None:
        """
        Removes static features from the view (and optionally from disk).

        By default this is a **soft drop**: features are removed from
        the settings so they no longer appear in ``static_data()``,
        but the underlying data is left untouched.

        With ``permanently=True`` the columns are also deleted from
        disk / virtual context (irreversible).

        Args:
            features: Feature name(s) to drop.
            permanently: If ``True``, also remove from disk/virtual.
        """
        if isinstance(features, str):
            features = [features]
        valid = self.settings.validate_features(features)
        current = self.settings.available_features()
        new_list = [f for f in current if f not in valid]
        self.update_settings(static_features=new_list)

        if not permanently:
            self._has_soft_drops = True

        if permanently:
            self._store.drop_static(
                valid,
                virtual_id=self._virtual_id,
            )

    def cast_static_features(
        self,
        schema: dict[str, pl.DataType | type],
    ) -> None:
        """
        Casts trajectory-level static-feature columns to new types.

        Only static features can be cast at trajectory level: entity
        features live inside the linked sequence stores and must be
        cast there.

        Args:
            schema: Dictionary mapping feature names to target Polars
                DataTypes (e.g. ``{"group": pl.Categorical}``).

        Raises:
            TypeError: If *schema* is not a dict.
            KeyError: If a feature name does not exist in the current view.
        """
        if not isinstance(schema, dict):
            raise TypeError(f"'schema' must be a dict, got {type(schema).__name__}")
        if not schema:
            return

        valid_names = self.settings.validate_features(list(schema.keys()))
        valid_schema = {col: schema[col] for col in valid_names}

        # Build new recipes and probe the full chain.
        new_recipe = self._casts.append(static=valid_schema)
        new_recipe.probe(self._store)
        self._casts = new_recipe
        self.clear_cache()

    def cast_id(self, dtype: pl.DataType) -> None:
        """
        Casts the trajectory ID column to a new type.

        The cast is propagated automatically to all linked sequence pools
        (accessible via :attr:`sequence_pools`) so that entity data and
        static data surface IDs in the same type at every level.

        Args:
            dtype: Target Polars DataType (e.g. ``pl.String``, ``pl.UInt32``).

        Raises:
            TypeError: If the cast is incompatible with the stored ID values.
        """
        new_recipe = self._casts.append(id=dtype)
        new_recipe.probe(self._store)
        self._casts = new_recipe
        self._sync_pool_casts()
        self.clear_cache()

    def cast_to_datetime(self, unit: str = "us", time_zone: str | None = None) -> None:
        """
        Casts time columns to Datetime across all linked sequence pools.

        All sequence stores are guaranteed to share the same temporal
        schema (enforced at build time), so a single probe against the
        trajectory store is sufficient - exactly like :meth:`cast_id`.
        The cast is stored in the trajectory-level recipe and
        re-propagated to every pool on next :attr:`sequence_pools` access.

        Args:
            unit: Datetime resolution (``"ms"``, ``"us"``, ``"ns"``).
                Default is ``"us"`` (microsecond).
            time_zone: Optional timezone string (e.g. ``"UTC"``, ``"Europe/Paris"``).

        Raises:
            ValueError: If *unit* is not one of the accepted values.
            TypeError: If the cast is incompatible with the temporal data.
        """
        if unit not in ("ms", "us", "ns"):
            raise ValueError(
                f"Invalid time unit: {unit!r}. Must be one of 'ms', 'us', 'ns'."
            )
        target_dtype = pl.Datetime(unit, time_zone)
        new_recipe = self._casts.append(time_index=target_dtype)
        new_recipe.probe(self._store)  # one probe - all stores homogeneous
        self._casts = new_recipe
        self._sync_pool_casts()
        self.clear_cache()

    def cast_to_timestep(self, dtype: pl.DataType = pl.Int64) -> None:
        """
        Casts time columns to numeric-based timesteps across all linked
        sequence pools.

        All sequence stores are guaranteed to share the same temporal
        schema (enforced at build time), so a single probe against the
        trajectory store is sufficient - exactly like :meth:`cast_id`.
        The cast is stored in the trajectory-level recipe and
        re-propagated to every pool on next :attr:`sequence_pools` access.

        Args:
            dtype: Target numeric type (e.g. ``pl.UInt32``, ``pl.Int64``,
                ``pl.Float64``).  Default is ``pl.Int64``.

        Raises:
            TypeError: If *dtype* is not a numeric type, or if the
                temporal data is already in Datetime format.
        """
        if not dtype.is_numeric():
            raise TypeError(f"Target dtype must be a numeric type, got {dtype}")
        if (
            self.metadata.time_index is not None
            and self.metadata.time_index.is_datetime
        ):
            raise TypeError("Conversion from Datetime to Timestep is not supported..")
        new_recipe = self._casts.append(time_index=dtype)
        new_recipe.probe(self._store)  # one probe - all stores homogeneous
        self._casts = new_recipe
        self._sync_pool_casts()
        self.clear_cache()

    # ------------------------------------------------------------------
    # Extend
    # ------------------------------------------------------------------

    def extend(
        self,
        other: TrajectoryPool | Trajectory,
        destination: str | Path | None = None,
        *,
        on_duplicate: Literal["raise", "skip"] = "raise",
        overwrite: bool = False,
    ) -> TrajectoryPool:
        """Merge *other* into this trajectory pool and write the result to disk.

        Mirrors the semantics of :meth:`save`.

        **Same-store fast path** - if both trajectory pools share
        ``_store.root_path`` and neither carries virtual content
        (``_virtual_id is None`` on both sides), no I/O is performed.  A new
        pool backed by the same store with the union of ID masks is returned
        immediately.  If *destination* is provided the merged pool is
        materialised via :meth:`save`; otherwise it is returned as an
        in-memory view with zero I/O.

        **Different stores (or virtual content present)** - *destination* is
        required.  For each alias in this pool,
        :meth:`~tanat.sequence.base.pool.SequencePool.extend` is called on the
        corresponding sub-pools; the results are assembled into a new
        trajectory store via the builder.  Pass
        ``destination=self._store.root_path`` with ``overwrite=True`` to
        rewrite in-place.

        Args:
            other: Trajectory pool or single :class:`Trajectory` to merge.
            destination: ``None`` → in-memory view (same-store fast path only;
                no I/O); ``str`` / ``Path`` → materialise the merged data to
                disk.  *destination* is required when merging from different
                stores.
            on_duplicate: Behaviour when *other* contains a trajectory ID already
                present in this pool:

                - ``"raise"`` *(default)*: raise ``ValueError``.
                - ``"skip"``: silently ignore duplicates.

            overwrite: Allows overwriting an existing *destination* when it
                already exists on disk.

        Returns:
            Always a new :class:`TrajectoryPool` - never ``self``.

        Raises:
            TypeError: If *other* is not a :class:`TrajectoryPool` or
                :class:`Trajectory`.
            TypeError: If a sub-pool has an incompatible ID dtype or temporal
                schema.
            ValueError: If a sub-pool in *other* is missing features present
                in the corresponding sub-pool of *self*.
            ValueError: If ``on_duplicate="raise"`` and duplicate IDs are found.
            ValueError: If ``destination=None`` and stores differ.
            FileExistsError: If *destination* exists and ``overwrite=False``.

        Note:
            Aliases present in *other* but absent from ``self`` are silently
            ignored (logged at ``WARNING``).  Aliases present in ``self`` but
            absent from *other* are carried over unchanged.

        See Also:
            :meth:`save`, :meth:`~tanat.sequence.base.pool.SequencePool.extend`
        """
        # pylint: disable=protected-access

        # ── Type check ────────────────────────────────────────────────────────────
        if not isinstance(other, (TrajectoryPool, Trajectory)):
            raise TypeError(
                f"'other' must be a TrajectoryPool or Trajectory, "
                f"got {type(other).__name__!r}."
            )

        # ── Per-alias schema checks ───────────────────────────────────────────────
        other_aliases = set(other._store_aliases)
        for alias in self._store_aliases:
            if alias not in other_aliases:
                continue  # alias absent from other - will be carried over
            sub_self = self.sequence_pools[alias]
            sub_other_meta = (
                other._build_sequence(alias).metadata
                if isinstance(other, Trajectory)
                else other.sequence_pools[alias].metadata
            )
            sub_self.metadata.assert_id_compatible_with(
                sub_other_meta,
                alias=alias,
                context=f"Cannot extend TrajectoryPool: alias '{alias}' has different ID dtypes.",
            )
            sub_self.metadata.assert_time_index_compatible_with(
                sub_other_meta,
                alias=alias,
                context=f"Cannot extend TrajectoryPool: alias '{alias}' has different time index schemas.",
            )
            extra_feats = sub_self.metadata.assert_features_compatible_with(
                sub_other_meta,
                alias=alias,
                context=f"Cannot extend TrajectoryPool: alias '{alias}' has incompatible features.",
            )
            if extra_feats:
                LOGGER.warning(
                    "Alias '%s': ignoring extra features in 'other' not present in self: %s",
                    alias,
                    sorted(extra_feats),
                )

        # ── Collect IDs from both sides ───────────────────────────────────────────
        self_ids = set(self.unique_ids)
        other_ids_list = (
            [other._id_value]
            if isinstance(other, Trajectory)
            else list(other.unique_ids)
        )
        other_ids_to_add = resolve_ids_to_add(
            self_ids, other_ids_list, on_duplicate, entity_label="trajectory IDs"
        )

        if not other_ids_to_add:
            LOGGER.warning(
                "Nothing to extend: all trajectory IDs from 'other' are already present. "
                "Returning a copy of self unchanged."
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
                "extend() requires a destination when merging from different stores. "
                "Use extend(other, destination='path/to/new_store') to write to a new "
                "store, or pass destination=self._store.root_path with overwrite=True "
                "to rewrite the current store in-place."
            )

        extra_aliases = other_aliases - set(self._store_aliases)
        if extra_aliases:
            LOGGER.warning(
                "Ignoring extra aliases in 'other' not present in self: %s",
                sorted(extra_aliases),
            )

        # ── Resolve write path ────────────────────────────────────────────────────
        dest_path, write_path, in_place = self._resolve_write_path(
            destination, overwrite
        )

        # ── Filter other to only the IDs we want to add ───────────────────────────
        other_for_extend: TrajectoryPool | Trajectory = (
            other
            if isinstance(other, Trajectory)
            else (
                other.subset(other_ids_to_add)
                if set(other_ids_to_add) != set(other_ids_list)
                else other
            )
        )

        # ── Materialise sub-pools + trajectory frames ─────────────────────────────
        try:
            links = self._materialise_sub_pools(
                write_path, other_for_extend, other_aliases, on_duplicate
            )
            merged_traj_idx, merged_static = self._merge_traj_frames(
                other, other_ids_to_add
            )
            TrajectoryStoreBuilder().build_from_frames(
                write_path, merged_traj_idx, merged_static, links, exist_ok=True
            )

            # ── In-place: atomic replace ──────────────────────────────────────────
            if in_place:
                shutil.rmtree(dest_path)
                shutil.move(str(write_path), str(dest_path))
                LOGGER.info("Trajectory pool extended in-place at %s.", dest_path)
            else:
                dest_path = write_path
                LOGGER.info("Extended trajectory store written to %s.", dest_path)

        finally:
            # Clean up the temp dir only if it still exists (i.e. the move
            # above never happened - meaning an exception was raised).
            if in_place and write_path.exists():
                shutil.rmtree(write_path)

        return TrajectoryPool(
            store=dest_path,
            id_column=self.settings.id_column,
            static_features=None,
        )

    def _resolve_write_path(
        self, destination: str | Path, overwrite: bool
    ) -> tuple[Path, Path, bool]:
        """Resolve the destination and working paths for a cross-store extend.

        Returns:
            ``(dest_path, write_path, in_place)`` - *write_path* is a sibling
            tmp directory when writing in-place, or *dest_path* otherwise.
            The working directory is created before returning.
        """
        dest_path = resolve_path(destination)
        in_place = dest_path == self._store.root_path
        if in_place:
            write_path = self._store.root_path.parent / f".tmp_{uuid.uuid4().hex}"
        else:
            if dest_path.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Destination already exists: {dest_path}. "
                        "Use overwrite=True to replace it."
                    )
                shutil.rmtree(dest_path)
            write_path = dest_path
        write_path.mkdir(parents=True, exist_ok=True)
        return dest_path, write_path, in_place

    def _materialise_sub_pools(
        self,
        write_path: Path,
        other_for_extend: TrajectoryPool | Trajectory,
        other_aliases: set[str],
        on_duplicate: str,
    ) -> dict[str, str]:
        """Write each alias sub-pool to *write_path* and return the links dict."""
        # pylint: disable=protected-access
        links: dict[str, str] = {}
        stores_dir = write_path / TSCH.Files.DIR_STORES
        for alias in self._store_aliases:
            sub_self = self.sequence_pools[alias]
            dest_sub = stores_dir / alias
            if alias in other_aliases:
                sub_other = (
                    other_for_extend._build_sequence(alias)
                    if isinstance(other_for_extend, Trajectory)
                    else other_for_extend.sequence_pools[alias]
                )
                merged_sub = sub_self.extend(
                    sub_other,
                    destination=dest_sub,
                    on_duplicate=on_duplicate,
                    overwrite=True,
                )
                links[alias] = TrajectoryStoreBuilder._to_relative(
                    write_path, merged_sub._store.root_path
                )
            else:
                # alias absent from other: materialise self's sub-pool as-is
                sub_self.copy().save(destination=dest_sub, overwrite=True)
                links[alias] = TrajectoryStoreBuilder._to_relative(write_path, dest_sub)
        return links

    def _merge_traj_frames(
        self,
        other: TrajectoryPool | Trajectory,
        other_ids_to_add: list,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
        """Build merged trajectory index and static LazyFrames.

        Returns:
            ``(merged_traj_idx, merged_static)`` - second element is ``None``
            when neither side has static features.
        """
        # pylint: disable=protected-access
        traj_idx_self, static_lf_self = self._store.get_frames_for_save(
            self._id_mask,
            self._casts if not self._casts.is_empty() else None,
            self._virtual_id,
        )
        traj_idx_other, static_lf_other = other._store.get_frames_for_save(
            set(other_ids_to_add),
            other._casts if not other._casts.is_empty() else None,
            other._virtual_id,
        )
        merged_traj_idx = pl.concat(
            [traj_idx_self, traj_idx_other], how="diagonal_relaxed"
        )
        bool_cols = [
            c for c in merged_traj_idx.collect_schema().names() if c != TSCH.TRAJ_ID
        ]
        if bool_cols:
            merged_traj_idx = merged_traj_idx.with_columns(
                [pl.col(c).fill_null(False) for c in bool_cols]
            )
        return merged_traj_idx, merge_optional_frames(static_lf_self, static_lf_other)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bin_size_native(bin_size: BinSize, is_datetime: bool) -> int | float:
        """Convert *bin_size* to the time column's native unit.

        Raises :exc:`TypeError` immediately (no I/O) when the type does not
        match the temporal kind.

        Returns:
            Duration in microseconds (``int``) for Datetime columns, or the
            original numeric value for timestep columns.
        """
        if is_datetime:
            if not isinstance(bin_size, str):
                raise TypeError(
                    "bin_size must be a duration string (e.g. '1h', '1d') "
                    "for Datetime trajectory stores."
                )
            return int(pd.Timedelta(bin_size).total_seconds() * 1_000_000)
        if not isinstance(bin_size, (int, float)):
            raise TypeError(
                "bin_size must be numeric for non-datetime (timestep) stores."
            )
        return bin_size

    def _collect_alias_frames(
        self,
        features: dict[str, list[str] | str],
        ohe: bool,
    ) -> tuple[dict[str, tuple[pl.DataFrame, list[str]]], list, list, str]:
        """Build and cache entity frames for all aliases in one pass.

        Returns frames collected once (reused for stats and binning), the
        per-alias temporal min/max lists, and the common ID column name.

        Returns:
            ``(alias_cache, all_t_mins, all_t_maxs, alias_id_col)``
        """
        all_t_mins: list = []
        all_t_maxs: list = []
        alias_cache: dict[str, tuple[pl.DataFrame, list[str]]] = {}
        alias_id_col: str | None = None

        for alias, feats in features.items():
            pool = self.sequence_pools[alias]  # pylint: disable=protected-access
            feats_list = [feats] if isinstance(feats, str) else feats
            time_cols = (
                pool.settings.get_time_columns()
            )  # pylint: disable=protected-access
            id_col_pool = pool.settings.id_column
            valid = pool.settings.validate_features(
                feats_list, is_static=False
            )  # pylint: disable=protected-access
            frame, feat_cols_alias = (
                pool._build_entity_frame(  # pylint: disable=protected-access
                    valid, time_cols, id_col_pool, ohe
                )
            )
            if alias_id_col is None:
                alias_id_col = id_col_pool

            stats = frame.select(
                [pl.col(c).min().alias(f"min_{c}") for c in time_cols]
                + [pl.col(c).max().alias(f"max_{c}") for c in time_cols]
            ).row(0, named=True)
            all_t_mins.append(
                min(v for c in time_cols if (v := stats[f"min_{c}"]) is not None)
            )
            all_t_maxs.append(
                max(v for c in time_cols if (v := stats[f"max_{c}"]) is not None)
            )
            alias_cache[alias] = (frame, feat_cols_alias)

        assert alias_id_col is not None  # guaranteed: features is non-empty
        return alias_cache, all_t_mins, all_t_maxs, alias_id_col

    def _build_alias_grids(
        self,
        features: dict[str, list[str] | str],
        alias_cache: dict[str, tuple[pl.DataFrame, list[str]]],
        alias_id_col: str,
        t_min: Any,
        bin_size_native: int | float,
        is_datetime: bool,
        max_bins: int,
        *,
        fill_value: Any,
        overlap_rule: str,
        bin_col: str,
    ) -> dict[str, pl.DataFrame]:
        """Run ``_to_grid_with_axis`` on each alias and prefix feature columns.

        Returns:
            Mapping of alias → long-format grid DataFrame with prefixed columns.
        """
        alias_grids: dict[str, pl.DataFrame] = {}
        for alias in features:
            pool = self.sequence_pools[alias]  # pylint: disable=protected-access
            frame, feat_cols_alias = alias_cache[alias]
            df_alias = pool._to_grid_with_axis(  # pylint: disable=protected-access
                frame,
                feat_cols_alias,
                t_min,
                bin_size_native,
                is_datetime,
                max_bins,
                fill_value=fill_value,
                overlap_rule=overlap_rule,
                bin_col=bin_col,
            )
            feat_cols_prefix = [
                c for c in df_alias.columns if c not in {alias_id_col, bin_col}
            ]
            rename_map = {c: f"{alias}_{c}" for c in feat_cols_prefix}
            if rename_map:
                df_alias = df_alias.rename(rename_map)
            alias_grids[alias] = df_alias
        return alias_grids

    @staticmethod
    def _join_and_fill_grids(
        alias_grids: dict[str, pl.DataFrame],
        alias_id_col: str,
        traj_ids: list,
        max_bins: int,
        fill_value: Any,
        bin_col: str,
    ) -> pl.DataFrame:
        """Cross-join id × bin, left-join every alias grid, optionally fill nulls.

        Returns:
            Long-format DataFrame with all alias feature columns aligned on the
            shared ``(alias_id_col, bin_col)`` reference frame.
        """
        first_grid = next(iter(alias_grids.values()))
        ids_df = pl.Series(
            alias_id_col, traj_ids, dtype=first_grid[alias_id_col].dtype
        ).to_frame()
        bins_df = pl.Series(bin_col, list(range(max_bins)), dtype=pl.Int64).to_frame()
        result = ids_df.join(bins_df, how="cross")
        for df_alias in alias_grids.values():
            result = result.join(df_alias, on=[alias_id_col, bin_col], how="left")
        if fill_value is not None:
            feat_cols_all = [
                c for c in result.columns if c not in {alias_id_col, bin_col}
            ]
            result = result.with_columns(
                [pl.col(c).fill_null(pl.lit(fill_value)) for c in feat_cols_all]
            )
        return result

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def to_grid(
        self,
        features: dict[str, list[str] | str],
        bin_size: BinSize,
        max_bins: int | None = None,
        fill_value: Any = None,
        overlap_rule: str = "first",
        ohe: bool = False,
        fmt: Literal["pandas", "polars", "numpy"] = "pandas",
        use_arrow: bool = True,
        bin_col: str = "__bin__",
    ) -> pd.DataFrame | pl.DataFrame | np.ndarray:
        """Project all trajectory stores onto a single shared temporal grid.

        Unlike :meth:`~tanat.sequence.base.pool.SequencePool.to_grid`, which
        computes one independent axis per store, this method derives a single
        global temporal axis from the **union** of all temporal indices across
        the requested stores, so that bin *k* represents the same time window
        regardless of the alias.

        Internally, each alias's :class:`~tanat.sequence.base.pool.SequencePool`
        is called via :meth:`~tanat.sequence.base.pool.SequencePool._to_grid_with_axis`
        with the shared ``(t_min, bin_size_native, is_datetime, max_bins)``
        tuple.  Results are horizontally joined on ``(id_col, bin_col)`` and
        the trajectory ID mask is applied to guarantee that only visible
        trajectories appear in the output.

        Args:
            features: Mapping of ``alias → feature name(s)`` that defines
                both **which stores** to query and **which features** to
                project from each store.  Every key must be a visible alias
                in the current pool view.

                Example::

                    {
                        "vitals": ["heart_rate", "spo2"],
                        "labs": "creatinine",
                    }

            bin_size: Width of each bin.  The expected type depends on the
                time column type:

                - **Datetime sequences** (``pl.Datetime`` / ``pl.Date``):
                  a duration string parsed by :class:`pandas.Timedelta`;
                  any pandas-compatible format is accepted, e.g. ``"1h"``,
                  ``"30min"``, ``"12h"``, ``"1d"``, ``"90s"``,
                  ``"2h30min"``, ``"1W"``.
                - **Timestep sequences** (numeric column): an ``int`` or
                  ``float`` in the same unit as the time column.
                  E.g. if the column holds integer timesteps, ``bin_size=2``
                  produces bins of size 2 timesteps.

                A **single** bin size is applied uniformly across all aliases.
            max_bins: Maximum number of bins.  When ``None``, inferred from
                the global temporal span ``(t_max - t_min) / bin_size``
                (capped by :attr:`~tanat.sequence.base.pool.SequencePool.MAX_BINS_LIMIT`).
            fill_value: Value used to fill empty bins, including bins that
                fall outside a store's own temporal span (default ``None``
                keeps nulls).
            overlap_rule: :class:`polars.Expr` aggregation name for bin
                conflict resolution (e.g. ``"first"``, ``"mean"``).
            ohe: If ``True``, one-hot encode the specified features before
                binning.  All features must be ``Categorical`` or ``Enum``.
            fmt: Format of the returned object:

                - ``"pandas"`` *(default)* / ``"polars"``: **long** format
                  with ``N × M`` rows.  Columns are
                  ``[id_col, bin_col, alias1_feat1, alias1_feat2, …]``,
                  where each ``alias_feat`` column holds the binned values
                  of feature ``feat`` from store ``alias`` (prefixed by
                  the alias name).
                - ``"numpy"``: 3-D :class:`numpy.ndarray` of shape
                  ``(N, M, K)`` where *N* = trajectories (ordered by
                  :attr:`unique_ids`), *M* = bins, *K* = total feature
                  columns across all aliases.

            bin_col: Name of the bin-index column used internally and
                surfaced in the long-format intermediate representation
                (default ``"__bin__"``).

        Returns:
            Grid-aligned multi-modal trajectory data in the requested format.

        Raises:
            KeyError: If any key in *features* is not a visible alias.
            TypeError: If *bin_size* has the wrong type for the temporal kind
                (string for Datetime, numeric for timestep).
            ValueError: If *bin_size* would produce too many bins and
                *max_bins* is not set.
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars", "numpy"), default="pandas")
        # ------------------------------------------------------------------ #
        # Step 1 - Validate aliases
        # ------------------------------------------------------------------ #
        visible = set(self._store_aliases)
        unknown = set(features.keys()) - visible
        if unknown:
            raise KeyError(
                f"Unknown aliases: {sorted(unknown)}. " f"Available: {sorted(visible)}"
            )

        # ------------------------------------------------------------------ #
        # Step 2 - Resolve temporal type + convert bin_size (fail fast, no I/O)
        # ------------------------------------------------------------------ #
        is_datetime: bool = self.metadata.time_index.is_datetime
        bin_size_native = self._resolve_bin_size_native(bin_size, is_datetime)

        # ------------------------------------------------------------------ #
        # Steps 3 & 5 - Build frames + temporal stats in a single pass
        # ------------------------------------------------------------------ #
        alias_cache, all_t_mins, all_t_maxs, alias_id_col = self._collect_alias_frames(
            features, ohe
        )
        t_min = min(all_t_mins)
        t_max = max(all_t_maxs)

        # ------------------------------------------------------------------ #
        # Step 4 - Resolve max_bins from global span
        # ------------------------------------------------------------------ #
        if max_bins is None:
            span = (
                int((t_max - t_min).total_seconds() * 1_000_000)
                if is_datetime
                else float(t_max - t_min)
            )
            estimated_bins = int(span // bin_size_native) + 1
            if estimated_bins > SequencePool.MAX_BINS_LIMIT:
                raise ValueError(
                    f"Discretization would produce ~{estimated_bins:,} bins, "
                    f"which exceeds the safety limit of "
                    f"{SequencePool.MAX_BINS_LIMIT:,}. "
                    "Use a larger bin_size or set max_bins explicitly to override."
                )
            max_bins = estimated_bins

        # ------------------------------------------------------------------ #
        # Steps 6-9 - Per-alias grids → align → join → fill
        # ------------------------------------------------------------------ #
        alias_grids = self._build_alias_grids(
            features,
            alias_cache,
            alias_id_col,
            t_min,
            bin_size_native,
            is_datetime,
            max_bins,
            fill_value=fill_value,
            overlap_rule=overlap_rule,
            bin_col=bin_col,
        )
        traj_ids = self.unique_ids
        result = self._join_and_fill_grids(
            alias_grids, alias_id_col, traj_ids, max_bins, fill_value, bin_col
        )

        # ------------------------------------------------------------------ #
        # Step 10 - Dispatch fmt
        # ------------------------------------------------------------------ #
        traj_id_col = self.settings.id_column
        if alias_id_col != traj_id_col:
            result = result.rename({alias_id_col: traj_id_col})

        feat_cols_result = [
            c for c in result.columns if c not in {traj_id_col, bin_col}
        ]

        if fmt == "numpy":
            arr = result.select(feat_cols_result).to_numpy()  # (N*M, K)
            return arr.reshape(len(traj_ids), max_bins, len(feat_cols_result))

        if fmt == "polars":
            return result
        return to_pandas(result, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        destination: str | Path | None = None,
        *,
        overwrite: bool = False,
        deep: bool = False,
    ) -> Path:
        """
        Persists the current pool state to disk.

        **Trajectory-level** (``trajectory_index.arrow``,
        ``static_features.arrow``) is always written, with ID and static
        casts baked in.

        **Sequence pools** - persisted according to their state:

        * Modified pools (virtual features or casts) are **always** saved to
          ``destination/stores/<alias>/``, regardless of *deep*.
        * Unmodified pools: copied when ``deep=True``; referenced by
          absolute path when ``deep=False``.

        For **in-place** saves the linked sequence stores are never
        touched (they may be shared with other pool instances).

        Without *destination* the trajectory-level files are rewritten
        **in-place**.  With *destination* a copy is created; the original
        is untouched.

        Args:
            destination: Where to save.  Can be:
                - ``None`` → in-place,
                - a workspace store name (no ``/`` or ``\\``),
                - or a filesystem ``Path`` / path string.
                Passing a path that resolves to the current store root is
                equivalent to ``None`` (treated as in-place).
            overwrite: Required when saving in-place (trajectory-level files
                will be overwritten).  Also allows overwriting an existing
                *destination*.  Each dirty sub-pool is saved in-place at its
                current location with ``overwrite=True`` automatically.
            deep: If ``True``, **all** sequence stores (including unmodified
                ones) are copied to ``destination/stores/<alias>/``.
                When ``False`` (default), only modified stores are
                materialised there; the rest are kept as absolute links.
                Ignored when saving in-place.

        Returns:
            The :class:`~pathlib.Path` of the written store - the in-place
            root when *destination* is ``None``, otherwise the resolved
            destination path.  Useful for chaining::

                pool2 = TrajectoryPool(store=pool.save("my_trajectories"))

        Raises:
            RuntimeError: If saving in-place without ``overwrite=True``.
            FileExistsError: If *destination* already exists and
                *overwrite* is ``False``.

        Note:
            This method **mutates** ``self``: after the call, the pool is
            redirected to *destination* (its store, masks and virtual context
            are all reset to the written state).  To keep the original
            instance unchanged while creating an independent copy elsewhere,
            use :meth:`copy` first::

                pool2 = TrajectoryPool(store=pool.copy().save("other_path"))
                # pool is still pointing to its original store

        See Also:
            :meth:`copy`

        Warning:
            Saving in-place with dirty sequence pools will overwrite those
            stores at their current location (``stores/<alias>/`` within the
            trajectory root).  If the stores are shared with other pool
            instances those instances will also reflect the changes.

        Note:
            With ``deep=True`` all links are relative - suitable for
            archiving or transfer.  With ``deep=False``, absolute links
            to unchanged stores are not portable across machines.
        """
        # Resolve upfront - treat same-path as in-place.
        dest_path = (
            resolve_path(destination)
            if destination is not None
            else self._store.root_path
        )
        in_place = dest_path == self._store.root_path

        if not self.is_dirty:
            if in_place:
                warnings.warn("Nothing to save.", UserWarning, stacklevel=2)
                return self._store.root_path
            # Fast-path: plain copy of the trajectory store (no frames to rebuild).
            if dest_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {dest_path}. "
                    "Use overwrite=True to replace it."
                )
            self._store.copy_to(dest_path, exist_ok=overwrite)
            LOGGER.info("Store saved to %s.", dest_path)
            return dest_path

        # Guard: any in-place save is destructive - always require confirmation.
        if in_place and not overwrite:
            raise RuntimeError(
                "Saving in-place rewrites the trajectory store with the current view. "
                "Use save(overwrite=True) to confirm, "
                "or save(destination) to create a copy and keep the original."
            )

        # Prepare destination directory (destination only).
        if not in_place:
            if dest_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {dest_path}. "
                    "Use overwrite=True to replace it."
                )
            if dest_path.exists() and overwrite:
                shutil.rmtree(dest_path)

        # Persist sequence pools and compute store links.
        store_links = TrajectoryStoreBuilder._resolve_pool_links(
            dest_path, self.sequence_pools, deep=deep, overwrite=overwrite
        )

        # Prepare trajectory-level frames (filtering + casting delegated to store).
        traj_idx, static_lf = self._store.get_frames_for_save(
            self._id_mask,
            self._casts if not self._casts.is_empty() else None,
            self._virtual_id,
            features=self.settings.static_features or None,
        )

        builder = TrajectoryStoreBuilder()
        builder.build_from_frames(
            dest_path, traj_idx, static_lf, store_links, exist_ok=True
        )

        self._reset_to(dest_path)
        LOGGER.info("Trajectory store saved to %s.", dest_path)
        return dest_path
