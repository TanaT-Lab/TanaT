#!/usr/bin/env python3
"""
SequenceViewMixin: shared view-layer logic for Pool and Sequence.

Both ``SequencePool`` and ``Sequence`` are **scoped views** on a
``SequenceStore``.  This mixin factors out the logic they share:
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from tanat_utils import Cachable

from ...metadata.sequence import SequenceMetadata
from ...store.sequence.store import SequenceStore
from ...store.sequence.schema import StoreSchema as SCH
from ...zeroing import _T0, _T0_NEAREST_RANK
from ._utils import merge_optional_frames, resolve_store

if TYPE_CHECKING:
    from .pool import SequencePool
    from .sequence import Sequence


class SequenceFrameAssembler:
    """Assembles view-schema LazyFrames from the store for Sequence/SequencePool view"""

    def __init__(self, view: Sequence | SequencePool) -> None:
        self._view = view

    def temporal(
        self,
        features: list[str] | str | None = None,
        *,
        with_store_index: bool = False,
    ) -> pl.LazyFrame:
        """Return temporal data in view schema with scopes and casts applied."""
        view = self._view
        # pylint: disable=protected-access
        valid_features = view._resolve_valid_features(features, is_static=False)
        lf = self._fetch(is_static=False, with_store_index=True)
        lf = self._rename(lf, is_static=False)
        lf = view._casts.structural.apply(
            lf,
            id_col=view.settings.id_column,
            time_cols=view.settings.get_time_columns(),
        )
        lf = view._casts.features.apply(lf, is_static=False)
        lf = view._apply_scopes(lf, is_static=False)
        return lf.select(self._projection(valid_features, with_store_index))

    def temporal_for_store(
        self,
        features: list[str] | str | None = None,
    ) -> pl.LazyFrame:
        """Return temporal data in store schema after view scopes and casts."""
        view = self._view
        # pylint: disable=protected-access
        valid_features = view._resolve_valid_features(features, is_static=False)
        lf = self._fetch(is_static=False, with_store_index=True)
        lf = self._rename(lf, is_static=False)
        lf = view._casts.structural.apply(
            lf,
            id_col=view.settings.id_column,
            time_cols=view.settings.get_time_columns(),
        )
        lf = view._casts.features.apply(lf, is_static=False)
        lf = view._apply_scopes(lf, is_static=False)
        lf = self._to_store(lf, is_static=False)
        structural_cols = view._store.structural_columns(
            is_static=False, virtual_id=view._virtual_id
        )
        return lf.select(structural_cols + valid_features)

    def id_time_index(self, *, with_store_index: bool = False) -> pl.LazyFrame:
        """Return id + time-index columns with view scopes applied."""
        view = self._view
        # pylint: disable=protected-access
        if view.has_entity_filter_expr:
            return self.temporal(features=[], with_store_index=with_store_index)

        lf = view._store.get_id_time_index(
            virtual_id=view._virtual_id,
            with_store_index=True,
        )
        lf = self._rename(lf, is_static=False)
        lf = view._casts.structural.apply(
            lf,
            id_col=view.settings.id_column,
            time_cols=view.settings.get_time_columns(),
        )
        lf = view._apply_scopes(lf, is_static=False)
        if not with_store_index:
            lf = lf.drop(SCH.STORE_INDEX)
        return lf

    def static(
        self,
        features: list[str] | str | None = None,
    ) -> pl.LazyFrame | None:
        """Return static data in view schema with scopes and casts applied."""
        view = self._view
        # pylint: disable=protected-access
        valid_features = view._resolve_valid_features(features, is_static=True)
        if not valid_features:
            return None
        lf = self._fetch(is_static=True)
        if lf is None:
            return None
        lf = self._rename(lf, is_static=True)
        lf = view._casts.structural.apply(lf, id_col=view.settings.id_column)
        lf = view._casts.features.apply(lf, is_static=True)
        lf = view._apply_scopes(lf, is_static=True)
        id_col = view.settings.id_column
        return lf.select([id_col] + valid_features)

    def static_for_store(
        self,
        features: list[str] | str | None = None,
    ) -> pl.LazyFrame | None:
        """Return static data in store schema after view scopes and casts."""
        view = self._view
        # pylint: disable=protected-access
        valid_features = view._resolve_valid_features(features, is_static=True)
        if not valid_features:
            return None
        lf = self._fetch(is_static=True)
        if lf is None:
            return None
        lf = self._rename(lf, is_static=True)
        lf = view._casts.structural.apply(lf, id_col=view.settings.id_column)
        lf = view._casts.features.apply(lf, is_static=True)
        lf = view._apply_scopes(lf, is_static=True)
        lf = self._to_store(lf, is_static=True)
        structural_cols = view._store.structural_columns(
            is_static=True, virtual_id=view._virtual_id
        )
        return lf.select(structural_cols + valid_features)

    def ids(self) -> pl.LazyFrame:
        """Return visible IDs with the view ID dtype and schema."""
        view = self._view
        # pylint: disable=protected-access
        lf = view._store.get_id_lf()
        lf = self._rename(lf)
        lf = view._casts.structural.apply(lf, id_col=view.settings.id_column)
        return view._apply_id_mask(lf)

    def select(
        self,
        lf: pl.LazyFrame,
        feature_names: list[str],
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """Select store structural columns plus *feature_names*."""
        view = self._view
        # pylint: disable=protected-access
        structural = view._store.structural_columns(
            is_static, virtual_id=view._virtual_id
        )
        return lf.select(structural + feature_names)

    def _fetch(
        self,
        *,
        is_static: bool = False,
        with_store_index: bool = False,
    ) -> pl.LazyFrame | None:
        """Fetch raw data from the store."""
        view = self._view
        # pylint: disable=protected-access
        if is_static:
            return view._store.get_static_data(virtual_id=view._virtual_id)
        return view._store.get_temporal_data(
            virtual_id=view._virtual_id,
            with_store_index=with_store_index,
        )

    def _rename(
        self,
        lf: pl.LazyFrame,
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """Rename store-internal columns to user-facing view names."""
        full_map = self._view.settings.get_column_rename_map(is_static=is_static)
        return lf.rename(full_map, strict=False)

    def _to_store(
        self,
        lf: pl.LazyFrame,
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """Rename view columns back to internal store names."""
        full_map = self._view.settings.get_column_rename_map(is_static=is_static)
        inverse = {dst: src for src, dst in full_map.items()}
        return lf.rename(inverse, strict=False)

    def merged_for_extend(
        self,
        other: SequencePool,
        other_ids_to_add: list,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
        """Build merged entity and static LazyFrames for a cross-store extend.

        Reads both sides (``self._view`` and *other*), applies view scopes,
        projects to ``self._view``'s entity feature set, and concatenates.
        Returned frames are lazy (no I/O until collected by the builder).

        Args:
            other: The pool to merge from.
            other_ids_to_add: IDs from *other* to include (duplicates already
                resolved by the caller).

        Returns:
            ``(merged_entity, merged_static)`` — second element is ``None``
            when neither side has static features.
        """
        view = self._view
        # pylint: disable=protected-access

        # --- self side ---
        entity_lf_self = self._fetch(is_static=False)
        entity_lf_self = self._rename(entity_lf_self, is_static=False)
        entity_lf_self = view._apply_scopes(entity_lf_self, is_static=False)
        entity_lf_self = self._to_store(entity_lf_self, is_static=False)
        entity_lf_self = self.select(
            entity_lf_self, view.settings.entity_features, is_static=False
        )

        # --- other side ---
        other_frames = SequenceFrameAssembler(other)
        entity_lf_other = other_frames._fetch(is_static=False)
        entity_lf_other = other_frames._rename(entity_lf_other, is_static=False)
        entity_lf_other = other._apply_scopes(entity_lf_other, is_static=False)
        entity_lf_other = entity_lf_other.filter(
            pl.col(other.settings.id_column).is_in(other_ids_to_add)
        )
        entity_lf_other = other_frames._to_store(entity_lf_other, is_static=False)
        entity_lf_other = other_frames.select(
            entity_lf_other, view.settings.entity_features, is_static=False
        )

        # --- static self ---
        static_lf_self = self._fetch(is_static=True)
        if static_lf_self is not None:
            static_lf_self = self._rename(static_lf_self, is_static=True)
            static_lf_self = view._apply_scopes(static_lf_self, is_static=True)
            static_lf_self = self._to_store(static_lf_self, is_static=True)

        # --- static other ---
        static_lf_other = other_frames._fetch(is_static=True)
        if static_lf_other is not None:
            static_lf_other = other_frames._rename(static_lf_other, is_static=True)
            static_lf_other = other._apply_scopes(static_lf_other, is_static=True)
            static_lf_other = static_lf_other.filter(
                pl.col(other.settings.id_column).is_in(other_ids_to_add)
            )
            static_lf_other = other_frames._to_store(static_lf_other, is_static=True)

        return (
            pl.concat([entity_lf_self, entity_lf_other]),
            merge_optional_frames(static_lf_self, static_lf_other),
        )

    def _projection(
        self,
        features: list[str],
        with_store_index: bool,
    ) -> list[str]:
        """Return temporal view-schema projection columns."""
        view = self._view
        columns = (
            [view.settings.id_column] + view.settings.get_time_columns() + features
        )
        if with_store_index:
            columns.append(SCH.STORE_INDEX)
        return columns


class SequenceViewMixin:
    """
    Mixin providing the view-layer helpers shared by
    ``SequencePool`` and ``Sequence``.

    Note: ``SequencePool`` and ``Sequence`` inherit from CachableSettings, not this mixin.
    """

    # ------------------------------------------------------------------
    # Store / feature resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_store(store: str | Path | SequenceStore) -> SequenceStore:
        """Resolve a store argument to a :class:`SequenceStore` instance."""
        return resolve_store(store)

    @staticmethod
    def _resolve_features(
        store: SequenceStore,
        entity_features: list[str] | None,
        static_features: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        """Resolve ``None`` feature lists from the store; keep explicit lists as-is.

        - ``None``    → take all available from the store.
        - ``[]``      → expose no features (explicit empty selection).
        - ``[...]``   → validate against the store, then use as-is.

        Raises:
            KeyError: If any feature name in a non-``None`` list is not
                available in the store.
        """
        if entity_features is not None:
            unknown = set(entity_features) - set(store.entity_features())
            if unknown:
                raise KeyError(
                    f"Unknown entity features: {sorted(unknown)}. "
                    f"Available: {store.entity_features()}"
                )
        if static_features is not None:
            unknown = set(static_features) - set(store.static_features())
            if unknown:
                raise KeyError(
                    f"Unknown static features: {sorted(unknown)}. "
                    f"Available: {store.static_features()}"
                )
        ef = entity_features if entity_features is not None else store.entity_features()
        sf = static_features if static_features is not None else store.static_features()
        return ef, sf

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _frames(self) -> SequenceFrameAssembler:
        """The frame assembler bound to this view."""
        return SequenceFrameAssembler(self)

    @Cachable.cached_property
    def _id_lf(self) -> pl.LazyFrame:
        """Lazy frame of visible IDs with the correct dtype.

        Renamed to ``settings.id_column``.  Filter depends on the concrete type:

        - :class:`Sequence`: single-ID filter on ``_id_value``.
        - :class:`SequencePool`: ID-mask filter when ``_id_mask`` is set.

        Unlike :attr:`unique_ids`, preserves rich dtypes (e.g. ``Categorical``).
        Cached via ``CachableSettings``; invalidated by ``clear_cache()``.
        """
        return self._frames.ids()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @Cachable.cached_property
    def metadata(self) -> SequenceMetadata:
        """
        Returns rich metadata fully reflecting this view's cast recipes,
        masks, and feature selection.

        When created from a parent :class:`SequencePool`, the pool's
        metadata is returned directly or scoped by the view's settings
        (when built with a feature subset).

        For a standalone view (no parent), metadata is inferred directly
        from the assembled, cast and masked LazyFrames.  The seq_id dtype
        is derived from the cast recipe (or the store schema when no cast
        is active).

        Automatically cached via ``CachableSettings``: the cache is
        invalidated whenever settings change (e.g. after ``cast_features``
        or ``drop_features``).
        """
        # Propagated from parent Pool, scoped to this view's feature selection.
        # hasattr(self, "_id_value") avoid scoping trajectory metadata for Sequence views.
        if (
            hasattr(self, "_id_value")
            and getattr(self, "_parent_pool", None) is not None
        ):
            return self._parent_pool.metadata.scope(
                entity_features=self.settings.entity_features,
                static_features=self.settings.static_features,
            )

        seq_id_dtype = self._casts.id_dtype or self._store.seq_id_dtype
        id_col = self.settings.id_column

        time_index = self._frames.id_time_index().select(
            self.settings.get_time_columns()
        )
        entity_lf = self._frames.temporal().select(self.settings.entity_features)

        static_lf = self._frames.static()
        static_infos = (
            SequenceMetadata.infer_static_features(static_lf.drop(id_col))
            if static_lf is not None
            else None
        )

        return SequenceMetadata(
            seq_id=seq_id_dtype,
            time_index=SequenceMetadata.infer_time_index(time_index),
            entity_features=SequenceMetadata.infer_entity_features(entity_lf),
            static_features=static_infos,
        )

    # ------------------------------------------------------------------
    # Feature resolution
    # ------------------------------------------------------------------

    def _resolve_valid_features(
        self,
        features: list[str] | str | None,
        is_static: bool = False,
    ) -> list[str]:
        """
        Validates explicit feature names or returns all available ones.

        Args:
            features: User-requested names (``None`` → all visible).
            is_static: Static or entity scope.

        Returns:
            List of validated feature names.
        """
        if features is not None:
            return self.settings.validate_features(features, is_static=is_static)
        return self.settings.available_features(is_static=is_static)

    # ------------------------------------------------------------------
    # Physical rank access
    # ------------------------------------------------------------------

    def _apply_entity_filter_expr(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the entity filter expression to a view LazyFrame when present."""
        expr = self._entity_filter_expr
        if expr is not None:
            return lf.filter(expr)
        return lf

    @property
    def has_entity_filter_expr(self) -> bool:
        """Whether this view has an expression-based entity filter."""
        return self._entity_filter_expr is not None

    def _apply_scopes(
        self,
        lf: pl.LazyFrame,
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """Apply view scopes in the stable order: ID mask then entity expression."""
        lf = self._apply_id_mask(lf)
        if not is_static:
            lf = self._apply_entity_filter_expr(lf)
        return lf

    @Cachable.cached_property
    def _store_index_df(self) -> pl.DataFrame:
        """Ordered store indices for all visible entities.

        Columns: ``[id_col, SCH.STORE_INDEX]``.

        Rows are in sequence order: position ``i`` corresponds to rank ``i``
        within its sequence, so ``df[SCH.STORE_INDEX][rank]`` gives the
        absolute physical row index without needing an explicit rank column.

        Three resolution paths, in order of cost:

        1. **Pool-managed sequence**: filter parent pool's cached DataFrame (zero I/O).
        2. **Entity filter active**: compute physical indices on feature-aware data.
        3. **No entity filter**: logical order matches physical order.
        """
        # Pool-managed sequence: delegate to parent, then slice.
        if (
            hasattr(self, "_id_value")
            and getattr(self, "_parent_pool", None) is not None
        ):
            id_col = self.settings.id_column
            # pylint: disable=protected-access
            return self._parent_pool._store_index_df.filter(
                pl.col(id_col) == self._id_value
            )

        id_col = self.settings.id_column

        if self.has_entity_filter_expr:
            lf = self._frames._fetch(is_static=False, with_store_index=True)
            lf = self._frames._rename(lf)
            lf = self._casts.structural.apply(
                lf, id_col=id_col, time_cols=self.settings.get_time_columns()
            )
            lf = self._casts.features.apply(lf, is_static=False)
            lf = self._apply_scopes(lf)
            return lf.select([id_col, SCH.STORE_INDEX]).collect()

        lf = self._store.get_id_lf(explode=True, with_store_index=True)
        lf = self._frames._rename(lf)
        lf = self._casts.structural.apply(lf, id_col=id_col)
        lf = self._apply_scopes(lf)
        return lf.select([id_col, SCH.STORE_INDEX]).collect()

    def _compute_raw_t0_df(self) -> pl.DataFrame:
        """Raw T0 DataFrame ``[id_col, _T0_]`` for this view, no nearest rank.

        Safe to call from within the entity-criteria pipeline (e.g.
        :class:`~tanat.criterion.type.rank.RankCriterion`) because it never
        triggers temporal frame assembly / :meth:`_apply_scopes`.

        Three resolution paths, mirroring :attr:`_store_index_df`:

        1. **Pool-managed sequence**: filter parent pool's cached raw result.
        2. **Managed pool** (owned by a :class:`~tanat.trajectory.pool.TrajectoryPool`):
           compute via trajectory.
        3. **Standalone pool or sequence**: compute from own data.

        ID scoping applied after computation:

        - :class:`Sequence`: filter to ``_id_value``.
        - :class:`SequencePool`: filter by ``_id_mask`` when set.
        """
        id_col = self.settings.id_column

        # Pool-managed sequence: filter parent pool's already-cached raw result.
        if (
            hasattr(self, "_id_value")
            and getattr(self, "_parent_pool", None) is not None
        ):
            full = self._parent_pool._compute_raw_t0_df()
            return full.filter(pl.col(id_col) == self._id_value)

        # Compute T0 if not yet done.
        setter = self._t0_setter
        df = setter.df
        if df is None:
            if getattr(self, "_parent_pool", None) is not None:
                # Managed pool owned by a TrajectoryPool.
                setter.compute_from_trajectory(self._parent_pool)
            else:
                setter.compute_from_sequence(self)
            df = setter.df

        if df is None:
            return pl.DataFrame()

        # Scope to visible IDs.
        if hasattr(self, "_id_value"):
            df = df.filter(pl.col(id_col) == self._id_value)
        elif getattr(self, "_id_mask", None) is not None:
            df = df.filter(pl.col(id_col).is_in(self._id_mask))

        return df

    @Cachable.cached_method()
    def _get_t0_df(self) -> pl.DataFrame:
        """T0 DataFrame ``[id_col, _T0_, _T0_NEAREST_RANK_]`` for this view.

        Thin wrapper: :meth:`_compute_raw_t0_df` + :meth:`_resolve_nearest_rank`.
        """
        # Pool-managed sequence: delegate to parent pool's full cached result.
        id_col = self.settings.id_column
        if (
            hasattr(self, "_id_value")
            and getattr(self, "_parent_pool", None) is not None
        ):
            full = self._parent_pool._get_t0_df()
            return full.filter(pl.col(id_col) == self._id_value)

        return self._resolve_nearest_rank(self._compute_raw_t0_df())

    # ------------------------------------------------------------------
    # Collected data (cached)
    # ------------------------------------------------------------------

    @Cachable.cached_method()
    def _temporal_data_df(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame:
        """Collect and cache temporal data as a Polars DataFrame.

        Wraps ``_frames.temporal`` with a final ``.collect()`` and
        caches the result.  Use ``_frames.temporal`` when further
        lazy operations are needed (e.g. in the visualization layer).
        """
        return self._frames.temporal(features).collect()

    @Cachable.cached_method()
    def _static_data_df(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame | None:
        """Collect and cache static data as a Polars DataFrame.

        Wraps ``_frames.static`` with a final ``.collect()`` and
        caches the result.  Returns ``None`` when no static features are
        visible.
        """
        lf = self._frames.static(features)
        return lf.collect() if lf is not None else None

    # ------------------------------------------------------------------
    # T0 / Zeroing
    # ------------------------------------------------------------------

    def _nearest_rank_lf(self, t0_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Lazy floor lookup: ``[id_col, _T0_NEAREST_RANK_]`` from ``[id_col, _T0_]``.

        Returns a LazyFrame; the caller decides when to collect.

        The comparison always uses the first time column (``start``) — see
        :meth:`_resolve_nearest_rank` for the rationale.

        Args:
            t0_lf: Two-column LazyFrame ``[id_col, _T0_]``.

        Returns:
            Two-column LazyFrame ``[id_col, _T0_NEAREST_RANK_]`` (raw max agg,
            no null-handling applied).  Call :meth:`_resolve_nearest_rank` for
            the full three-case null-handling logic.
        """
        id_col = self.settings.id_column
        t_col = self.settings.get_time_columns()[0]
        lf = (
            self._frames.id_time_index()
            .select([id_col, t_col])
            .with_columns(pl.int_range(pl.len()).over(id_col).alias("__rn__"))
        )
        return (
            lf.join(t0_lf.select([id_col, _T0]), on=id_col)
            .filter(pl.col(t_col) <= pl.col(_T0))
            .group_by(id_col)
            .agg(pl.col("__rn__").max().alias(_T0_NEAREST_RANK))
        )

    def _resolve_nearest_rank(
        self,
        t0_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Floor lookup: last row where ``start[i] ≤ _T0`` per sequence.

        The comparison **always** uses the first time column (``start``),
        regardless of the anchor used to compute ``_T0``.

        Rationale: the anchor controls *which edge of a row* is used to
        compute ``_T0`` (start / end / middle).  The nearest-rank, however,
        must be the row that *contains or precedes* ``_T0`` on the natural
        row-ordering axis (left boundary = start).  Using any other column
        for the ``≤`` comparison gives wrong results:

        * ``end ≤ T0``: excludes the containing row when T0 falls inside an
          interval (``start < T0 < end``), which is one row too early.
        * ``middle ≤ T0``: same issue for overlapping intervals.

        For non-overlapping intervals, ``start[i] ≤ T0`` always identifies
        the containing row, regardless of how ``_T0`` was anchored:

        * ``anchor='start'`` → ``_T0 = start[k]`` → max i where ``start[i] ≤ start[k]`` = k ✓
        * ``anchor='end'``   → ``_T0 = end[k]``   → max i where ``start[i] ≤ end[k]``   = k ✓
        * ``anchor='middle'`` → ``_T0 = mid[k]``  → max i where ``start[i] ≤ mid[k]``   = k ✓

        Args:
            t0_df: Two-column DataFrame ``[id_col, _T0_]``.

        Returns:
            Three-column DataFrame ``[id_col, _T0_, _T0_NEAREST_RANK_]``.
        """
        id_col = self.settings.id_column
        rank_lf = self._nearest_rank_lf(t0_df.lazy())
        return (
            t0_df.lazy()
            .join(rank_lf, on=id_col, how="left")
            .with_columns(
                # Three cases:
                #   _T0_ null                        → null  (T0 unknown)
                #   _T0_ set, floor found            → floor rank (last row ≤ T0)
                #   _T0_ set, all data after T0      → 0     (first row is nearest)
                pl.when(pl.col(_T0).is_null())
                .then(pl.lit(None, dtype=pl.UInt32))
                .otherwise(
                    pl.col(_T0_NEAREST_RANK)
                    .fill_null(pl.lit(0, dtype=pl.UInt32))
                    .cast(pl.UInt32)
                )
                .alias(_T0_NEAREST_RANK)
            )
            .collect()
        )
