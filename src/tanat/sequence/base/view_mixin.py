#!/usr/bin/env python3
"""
SequenceViewMixin: shared view-layer logic for Pool and Sequence.

Both ``SequencePool`` and ``Sequence`` are **scoped views** on a
``SequenceStore``.  This mixin factors out the logic they share:
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from tanat_utils import Cachable

from ...metadata.sequence import SequenceMetadata
from ...store.sequence.store import SequenceStore
from ...zeroing import _T0, _T0_NEAREST_RANK
from ._utils import resolve_store


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

    @Cachable.cached_property
    def _id_lf(self) -> pl.LazyFrame:
        """Lazy frame of visible IDs with the correct dtype.

        Renamed to ``settings.id_column``.  Filter depends on the concrete type:

        - :class:`Sequence`: single-ID filter on ``_id_value``.
        - :class:`SequencePool`: ID-mask filter when ``_id_mask`` is set.

        Unlike :attr:`unique_ids`, preserves rich dtypes (e.g. ``Categorical``).
        Cached via ``CachableSettings``; invalidated by ``clear_cache()``.
        """
        lf = self._store.get_id_lf(id_caster=self._casts.id_caster())
        lf = self._apply_id_mask(lf)
        return lf.rename({self._store.seq_id_col: self.settings.id_column})

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

        time_index = self._id_time_index_lf().select(self.settings.get_time_columns())
        entity_lf = self._temporal_data_lf().select(self.settings.entity_features)

        static_lf = self._static_data_lf()
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

    def _apply_entity_row_mask(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the entity row mask to a store LazyFrame.

        The mask is a positional boolean Series aligned on the full physical store.

        Args:
            lf: LazyFrame to apply the mask.

        Returns:
            Filtered :class:`~polars.LazyFrame`.
        """
        mask = getattr(self, "_entity_row_mask", None)
        if mask is not None:
            return lf.filter(pl.lit(mask))
        return lf

    def _apply_masks(self, lf: pl.LazyFrame, is_static: bool = False) -> pl.LazyFrame:
        """Apply entity row mask then ID mask to a store LazyFrame.

        Order is mandatory:

        1. ``_entity_row_mask`` first: positional boolean :class:`~polars.Series`
           aligned on the **full physical store**.  Must run before any row-count
           change (including ID scoping).
        2. ``_id_mask`` second: value-based filter, safe on any row count.

        Args:
            lf: LazyFrame to apply the mask.
            is_static: whether lf come from static data.

        Returns:
            Filtered :class:`~polars.LazyFrame`.
        """
        if not is_static:
            lf = self._apply_entity_row_mask(lf)
        return self._apply_id_mask(lf)

    @Cachable.cached_property
    def _entity_ranks_df(self) -> pl.DataFrame:
        """Per-sequence rank indices for all visible entities.

        Columns: ``[id_col, "__phys_seq_rank__", "__logical_seq_rank__"]``.

        * ``"__phys_seq_rank__"``    : 0-based rank within the sequence in the store,
          computed **before any view-level mask**.  Stable identifier used by
          :class:`~tanat.sequence.base.entity.Entity` to locate its store row.
        * ``"__logical_seq_rank__"`` : 0-based rank within the sequence in **this
          view**, re-indexed after all masks are applied.

        Three resolution paths, in order of cost:

        1. **Pool-managed sequence**: filter parent pool's cached DataFrame (zero I/O).
        2. **No entity mask** (fast path): ``__logical_seq_rank__`` is a copy of
           ``__phys_seq_rank__`` (no rows were removed).
        3. **Entity mask active** (slow path): ``__logical_seq_rank__`` is
           re-computed after both masks.
        """
        # Pool-managed sequence: delegate to parent, then slice.
        if (
            hasattr(self, "_id_value")
            and getattr(self, "_parent_pool", None) is not None
        ):
            id_col = self.settings.id_column
            # pylint: disable=protected-access
            return self._parent_pool._entity_ranks_df.filter(
                pl.col(id_col) == self._id_value
            )

        id_col = self.settings.id_column

        # __phys_seq_rank__ is stamped by the store on the full N_STORE rows,
        # so _apply_entity_row_mask (positional on N_STORE) must come first.
        lf = self._store.get_id_lf(
            id_caster=self._casts.id_caster(),
            explode=True,
            with_seq_rank=True,
        )
        lf = self._apply_masks(lf)
        lf = lf.rename({self._store.seq_id_col: id_col})

        # Fast path: no entity mask, __logical_seq_rank__ == __phys_seq_rank__.
        if self._entity_row_mask is None:
            return lf.with_columns(
                pl.col("__phys_seq_rank__").alias("__logical_seq_rank__")
            ).collect()

        # Slow path: compute logical rank after mask.
        return lf.with_columns(
            pl.int_range(pl.len())
            .over(id_col)
            .cast(pl.UInt32)
            .alias("__logical_seq_rank__")
        ).collect()

    def _compute_raw_t0_df(self) -> pl.DataFrame:
        """Raw T0 DataFrame ``[id_col, _T0_]`` for this view, no nearest rank.

        Safe to call from within the entity-criteria pipeline (e.g.
        :class:`~tanat.criterion.type.rank.RankCriterion`) because it never
        triggers :meth:`_temporal_data_lf` / :meth:`_apply_masks`.

        Three resolution paths, mirroring :attr:`_entity_ranks_df`:

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
    # Lazy data access (no collect, for internal consumers)
    # ------------------------------------------------------------------

    def _temporal_data_lf(
        self,
        features: list[str] | str | None = None,
        with_store_index: bool = False,
    ) -> pl.LazyFrame:
        """Return temporal data as a :class:`~polars.LazyFrame` without collecting.

        Pipeline (in order):
        1. Fetch all columns from store (optionally with ``__store_idx__``).
        2. Apply masks (entity row mask first, then ID mask).
        3. Rename store columns to user-facing names.
        4. Select structural (id + time) + requested feature columns.

        Args:
            features: Feature names to include (``None`` → all visible).
            with_store_index: When ``True``, prepends ``__store_idx__`` (the
                absolute physical row position in the store) to the result.
                Useful for consumers that need to map view-space rows back to
                store-space positions (e.g. :class:`~tanat.criterion.type.rank.RankCriterion`).
        """
        valid_features = self._resolve_valid_features(features, is_static=False)
        lf = self._get_data_from_store(
            is_static=False, with_store_index=with_store_index
        )
        lf = self._apply_masks(lf, is_static=False)
        lf = self._rename_columns(lf, is_static=False)
        # Select structural (id + time) + requested feature columns.
        id_col = self.settings.id_column
        time_cols = self.settings.get_time_columns()
        select_cols = [id_col] + time_cols + valid_features
        if with_store_index:
            select_cols = select_cols + ["__store_idx__"]
        return lf.select(select_cols)

    def _id_time_index_lf(self, with_store_index: bool = False) -> pl.LazyFrame:
        """Return ``id + time index`` columns as a :class:`~polars.LazyFrame`, masks applied.

        Cheaper than :meth:`_temporal_data_lf` when entity features are not needed.
        Works regardless of the temporal type (datetime, integer timestep, etc.).

        Args:
            with_store_index: When ``True``, prepends ``__store_idx__`` (the
                absolute physical row position in the store) to the result.
        """
        lf = self._store.get_id_time_index(
            virtual_id=self._virtual_id,
            id_caster=self._casts.id_caster(),
            time_index_caster=self._casts.time_index_caster(),
            with_store_index=with_store_index,
        )
        lf = self._apply_masks(lf, is_static=False)
        return self._rename_columns(lf, is_static=False)

    def _static_data_lf(
        self,
        features: list[str] | str | None = None,
    ) -> pl.LazyFrame | None:
        """Return static data as a :class:`~polars.LazyFrame` without collecting.

        Returns ``None`` when no static features are visible.
        """
        valid_features = self._resolve_valid_features(features, is_static=True)
        if not valid_features:
            return None
        lf = self._get_data_from_store(is_static=True)
        if lf is None:
            return None
        lf = self._apply_id_mask(lf)
        lf = self._rename_columns(lf, is_static=True)
        id_col = self.settings.id_column
        return lf.select([id_col] + valid_features)

    # ------------------------------------------------------------------
    # Collected data (cached)
    # ------------------------------------------------------------------

    @Cachable.cached_method()
    def _temporal_data_df(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame:
        """Collect and cache temporal data as a Polars DataFrame.

        Wraps :meth:`_temporal_data_lf` with a final ``.collect()`` and
        caches the result.  Use :meth:`_temporal_data_lf` when further
        lazy operations are needed (e.g. in the visualization layer).
        """
        return self._temporal_data_lf(features).collect()

    @Cachable.cached_method()
    def _static_data_df(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame | None:
        """Collect and cache static data as a Polars DataFrame.

        Wraps :meth:`_static_data_lf` with a final ``.collect()`` and
        caches the result.  Returns ``None`` when no static features are
        visible.
        """
        lf = self._static_data_lf(features)
        return lf.collect() if lf is not None else None

    # ------------------------------------------------------------------
    # Column renaming
    # ------------------------------------------------------------------

    def _rename_columns(
        self,
        lf: pl.LazyFrame,
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """
        Renames store internal columns to user-facing names.

        The mapping is provided by ``settings.get_column_rename_map()``.
        """
        full_map = self.settings.get_column_rename_map(is_static=is_static)
        return lf.rename(full_map)

    # ------------------------------------------------------------------
    # Column selection
    # ------------------------------------------------------------------

    def _select_columns(
        self,
        lf: pl.LazyFrame,
        feature_names: list[str],
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """
        Selects structural columns (ID + temporal) plus the validated
        feature columns from a store LazyFrame.

        This is the **view-level** column filter.  The store always
        returns all columns; the view picks what it needs.
        """
        structural = self._store.structural_columns(
            is_static, virtual_id=self._virtual_id
        )
        return lf.select(structural + feature_names)

    def _get_data_from_store(
        self,
        is_static: bool = False,
        with_store_index: bool = False,
    ) -> pl.LazyFrame | None:
        """
        Fetches all features from the store, merges virtual layer, and applies
        view-level cast recipes.

        Args:
            is_static: When ``True``, returns static features; otherwise temporal.
            with_store_index: When ``True``, prepends ``__store_idx__`` (absolute
                physical row position in the store) to the returned LazyFrame.
                Only meaningful for temporal data (ignored when *is_static* is
                ``True``).

        Returns ``None`` when ``is_static=True`` and no static features exist.
        """
        if is_static:
            return self._store.get_static_data(
                virtual_id=self._virtual_id,
                id_caster=self._casts.id_caster(),
                feature_exprs=self._casts.feature_exprs(is_static=True),
            )
        return self._store.get_temporal_data(
            virtual_id=self._virtual_id,
            id_caster=self._casts.id_caster(),
            time_index_caster=self._casts.time_index_caster(),
            feature_exprs=self._casts.feature_exprs(is_static=False),
            with_store_index=with_store_index,
        )

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
        temporal_lf = (
            self._id_time_index_lf()
            .select([id_col, t_col])
            .with_columns(pl.int_range(pl.len()).over(id_col).alias("__rn__"))
        )
        return (
            temporal_lf.join(t0_lf.select([id_col, _T0]), on=id_col)
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
