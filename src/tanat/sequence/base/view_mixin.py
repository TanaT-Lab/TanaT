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

from ...core.path import resolve_path
from ...metadata.sequence import SequenceMetadata
from ...store.sequence.store import SequenceStore


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
        if isinstance(store, SequenceStore):
            return store
        if isinstance(store, (str, Path)):
            return SequenceStore(root_path=resolve_path(store))
        raise TypeError(
            f"'store' must be a store name, Path, or SequenceStore instance, "
            f"got {type(store)}"
        )

    @staticmethod
    def _resolve_features(
        store: SequenceStore,
        entity_features: list[str] | None,
        static_features: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        """Resolve ``None`` feature lists from the store; keep explicit lists as-is.

        - ``None``    → take all available from the store.
        - ``[]``      → expose no features (explicit empty selection).
        - ``[...]``   → use the provided list as-is.
        """
        ef = entity_features if entity_features is not None else store.entity_features()
        sf = static_features if static_features is not None else store.static_features()
        return ef, sf

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @CachableSettings.cached_property
    def metadata(self) -> SequenceMetadata:
        """
        Returns rich metadata fully reflecting this view's cast recipes,
        masks, and feature selection.

        When created from a parent :class:`SequencePool`, the pool's
        metadata is returned directly (propagation + consistent pool-level
        stats, no extra inference cost).

        For a standalone view (no parent), metadata is inferred directly
        from the assembled, cast and masked LazyFrames.  The seq_id dtype
        is derived from the cast recipe (or the store schema when no cast
        is active).

        Automatically cached via ``CachableSettings``: the cache is
        invalidated whenever settings change (e.g. after ``cast_features``
        or ``drop_features``).
        """
        # Propagated from parent Pool: consistent pool-level metadata.
        if getattr(self, "_parent_metadata", None) is not None:
            return self._parent_metadata

        # seq_id dtype: from cast recipe if set, else store schema
        seq_id_dtype = self._casts.id
        if seq_id_dtype is None:
            seq_id_dtype = self._store.seq_id_dtype

        seq_lf = self._get_data_from_store(is_static=False)
        seq_lf = self._apply_masks(seq_lf, is_static=False)

        # Temporal and entity slices from the single assembled LF.
        temporal_col_names = (
            self._store.temporal(self._virtual_id).collect_schema().names()
        )
        temporal_lf = seq_lf.select(temporal_col_names)
        entity_lf = seq_lf.select(self.settings.entity_features)

        # Static: casts + masks, then drop ID column directly (rename is useless here).
        static_infos = None
        static_lf = self._get_data_from_store(is_static=True)
        if static_lf is not None:
            static_lf = self._apply_masks(static_lf, is_static=True)
            static_features = self.settings.static_features
            if static_features:
                static_infos = SequenceMetadata.infer_static_features(
                    static_lf.select(static_features)
                )

        return SequenceMetadata(
            seq_id=seq_id_dtype,
            temporal=SequenceMetadata.infer_temporal(temporal_lf),
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

    def _sequence_data_lf(
        self,
        features: list[str] | str | None = None,
    ) -> pl.LazyFrame:
        """Return sequence data as a :class:`~polars.LazyFrame` without collecting.

        Applies masks, column selection and renaming identically to
        :meth:`sequence_data`, but skips the final ``.collect()`` call.
        Intended for internal consumers that chain further lazy operations.
        """
        valid_features = self._resolve_valid_features(features, is_static=False)
        lf = self._get_data_from_store(is_static=False)
        lf = self._apply_masks(lf, is_static=False)
        lf = self._select_columns(lf, valid_features, is_static=False)
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
    def _sequence_data_df(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame:
        """Collect and cache sequence data as a Polars DataFrame.

        Wraps :meth:`_sequence_data_lf` with a final ``.collect()`` and
        caches the result.  Use :meth:`_sequence_data_lf` when further
        lazy operations are needed (e.g. in the visualization layer).
        """
        return self._sequence_data_lf(features).collect()

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
                id_cast=self._casts.id,
                feature_casts=self._casts.static or None,
            )
        return self._store.get_sequence_data(
            virtual_id=self._virtual_id,
            id_cast=self._casts.id,
            temporal_cast=self._casts.temporal,
            feature_casts=self._casts.entity or None,
        )
