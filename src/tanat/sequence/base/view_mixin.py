#!/usr/bin/env python3
"""
SequenceViewMixin: shared view-layer logic for Pool and Sequence.

Both ``SequencePool`` and ``Sequence`` are **scoped views** on a
``SequenceStore``.  This mixin factors out the logic they share:
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from tanat_utils import CachableSettings

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

    @CachableSettings.cached_property
    def _id_lf(self) -> pl.LazyFrame:
        """Lazy frame of visible IDs with the correct dtype.

        Renamed to ``settings.id_column``.  Filter depends on the concrete type:

        - :class:`Sequence`: single-ID filter on ``_id_value``.
        - :class:`SequencePool`: ID-mask filter when ``_id_mask`` is set.

        Unlike :attr:`unique_ids`, preserves rich dtypes (e.g. ``Categorical``).
        Cached via ``CachableSettings``; invalidated by ``clear_cache()``.
        """
        lf = self._store.get_id_lf(id_caster=self._casts.id_caster()).rename(
            {self._store.seq_id_col: self.settings.id_column}
        )
        if hasattr(self, "_id_value"):
            # Sequence path: filter to a single ID
            return lf.filter(pl.col(self.settings.id_column) == self._id_value)
        if self._id_mask is not None:
            # Pool path: filter by ID mask
            return lf.filter(pl.col(self.settings.id_column).is_in(list(self._id_mask)))
        return lf

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @CachableSettings.cached_property
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
        if getattr(self, "_parent_pool", None) is not None:
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
    # Lazy data access (no collect, for internal consumers)
    # ------------------------------------------------------------------

    def _temporal_data_lf(
        self,
        features: list[str] | str | None = None,
    ) -> pl.LazyFrame:
        """Return temporal data as a :class:`~polars.LazyFrame` without collecting.

        Applies masks, column selection, and renaming identically to
        :meth:`temporal_data`, but skips the final ``.collect()`` call.
        Use this when chaining further lazy operations.
        """
        valid_features = self._resolve_valid_features(features, is_static=False)
        lf = self._get_data_from_store(is_static=False)
        lf = self._apply_masks(lf, is_static=False)
        lf = self._select_columns(lf, valid_features, is_static=False)
        return self._rename_columns(lf, is_static=False)

    def _id_time_index_lf(self) -> pl.LazyFrame:
        """Return ``id + time index`` columns as a :class:`~polars.LazyFrame`, masks applied.

        Cheaper than :meth:`_temporal_data_lf` when entity features are not needed.
        Works regardless of the temporal type (datetime, integer timestep, etc.).
        """
        lf = self._store.get_id_time_index(
            virtual_id=self._virtual_id,
            id_caster=self._casts.id_caster(),
            time_index_caster=self._casts.time_index_caster(),
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
        lf = self._apply_masks(lf, is_static=True)
        lf = self._select_columns(lf, valid_features, is_static=True)
        return self._rename_columns(lf, is_static=True)

    # ------------------------------------------------------------------
    # Collected data (cached)
    # ------------------------------------------------------------------

    @CachableSettings.cached_method()
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

    @CachableSettings.cached_method()
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
    ) -> pl.LazyFrame | None:
        """
        Fetches all features from the store, merges virtual layer, and applies
        view-level cast recipes.

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
        )

    # ------------------------------------------------------------------
    # T0 / Zeroing
    # ------------------------------------------------------------------

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
        t_col = self.settings.get_time_columns()[0]

        temporal_lf = (
            self._id_time_index_lf()
            .select([id_col, t_col])
            .with_columns(
                pl.int_range(pl.len()).over(id_col).alias("__rn__"),
            )
        )
        rank_lf = (
            temporal_lf.join(t0_df.lazy().select([id_col, _T0]), on=id_col)
            .filter(pl.col(t_col) <= pl.col(_T0))
            .group_by(id_col)
            .agg(pl.col("__rn__").max().alias(_T0_NEAREST_RANK))
        )
        return (
            t0_df.lazy()
            .join(rank_lf, on=id_col, how="left")
            .with_columns(pl.col(_T0_NEAREST_RANK).cast(pl.Int32))
            .collect()
        )
