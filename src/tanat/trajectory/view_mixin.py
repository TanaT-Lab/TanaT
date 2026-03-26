#!/usr/bin/env python3
"""
TrajectoryViewMixin: shared view-layer logic for TrajectoryPool and Trajectory.

Both ``TrajectoryPool`` and ``Trajectory`` are **scoped views** on a
``TrajectoryStore``.  This mixin factors out the logic they share:

"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pandas as pd
from tanat_utils import CachableSettings

from ..core.path import resolve_path
from ..metadata.trajectory import TrajectoryMetadata
from ..store.trajectory.store import TrajectoryStore


class TrajectoryViewMixin:
    """
    Mixin providing view-layer helpers shared by
    ``TrajectoryPool`` and ``Trajectory``.
    """

    # ------------------------------------------------------------------
    # Store / feature resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_store(store: str | Path | TrajectoryStore) -> TrajectoryStore:
        """Resolve a store argument to a :class:`TrajectoryStore` instance."""
        if isinstance(store, TrajectoryStore):
            return store
        if isinstance(store, (str, Path)):
            return TrajectoryStore(root_path=resolve_path(store))
        raise TypeError(
            f"'store' must be a store name, Path, or TrajectoryStore instance, "
            f"got {type(store)}"
        )

    @staticmethod
    def _resolve_features(
        store: TrajectoryStore,
        static_features: list[str] | None,
    ) -> list[str]:
        """Resolve ``None`` static feature list from the store; keep explicit lists as-is.

        - ``None``  → take all available from the store.
        - ``[]``    → expose no features (explicit empty selection).
        - ``[...]`` → validate against the store, then use as-is.

        Raises:
            KeyError: If any feature name in a non-``None`` list is not
                available in the store.
        """
        if static_features is not None:
            unknown = set(static_features) - set(store.static_features())
            if unknown:
                raise KeyError(
                    f"Unknown static features: {sorted(unknown)}. "
                    f"Available: {store.static_features()}"
                )
        return (
            static_features if static_features is not None else store.static_features()
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @CachableSettings.cached_property
    def _id_lf(self) -> pl.LazyFrame:
        """Lazy frame of visible IDs with the correct dtype.

        Renamed to ``settings.id_column``.  Filter depends on the concrete type:

        - :class:`Trajectory`: single-ID filter on ``_id_value``.
        - :class:`TrajectoryPool`: ID-mask filter when ``_id_mask`` is set.

        Cached via ``CachableSettings``; invalidated by ``clear_cache()``.
        """
        lf = self._store.get_id_lf(id_cast=self._casts.id).rename(
            {self._store.traj_id_col: self.settings.id_column}
        )
        if hasattr(self, "_id_value"):
            # Trajectory path: filter to a single ID
            return lf.filter(pl.col(self.settings.id_column) == self._id_value)
        if self._id_mask is not None:
            # Pool path: filter by ID mask
            return lf.filter(pl.col(self.settings.id_column).is_in(list(self._id_mask)))
        return lf

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @CachableSettings.cached_property
    def metadata(self) -> TrajectoryMetadata:
        """
        Returns trajectory-level metadata, fully reflecting this view's
        cast recipes, masks, and feature selection.

        When created from a parent :class:`TrajectoryPool`, the pool's
        metadata is returned directly or scoped by the view's settings
        (when built with a feature subset).

        For a standalone view, the traj_id dtype is derived from the
        cast recipe (or the store schema when no cast is active) - no
        full plan traversal required.

        Automatically cached via ``CachableSettings``: the cache is
        invalidated whenever settings change.
        """
        # Propagated from parent Pool, scoped to this view's feature selection.
        if getattr(self, "_parent_pool", None) is not None:
            return self._parent_pool.metadata.scope(
                static_features=self.settings.static_features,
            )

        # traj_id dtype: from cast recipe if set, else store schema - no plan traversal.
        traj_id_dtype = self._casts.id
        if traj_id_dtype is None:
            traj_id_dtype = self._store.traj_id_dtype

        # Time index: aggregate across visible sequence stores.
        time_index = TrajectoryMetadata.infer_time_index(
            {alias: self._store.sequence_stores[alias] for alias in self._store_aliases}
        )

        # Static: casts + masks, then drop ID column directly (rename is useless here).
        static_lf = self._get_static_data_from_store()
        if static_lf is not None:
            static_lf = self._apply_masks(static_lf)
            visible_features = self.settings.available_features()
            static_lf = static_lf.select(visible_features) if visible_features else None

        return TrajectoryMetadata(
            traj_id=traj_id_dtype,
            time_index=time_index,
            static_features=TrajectoryMetadata.infer_static(static_lf),
        )

    # ------------------------------------------------------------------
    # Feature resolution
    # ------------------------------------------------------------------

    def _resolve_valid_features(
        self,
        features: list[str] | str | None = None,
    ) -> list[str]:
        """
        Validates explicit feature names or returns all available ones.

        Args:
            features: User-requested names (``None`` → all visible).

        Returns:
            List of validated feature names.
        """
        if features is not None:
            return self.settings.validate_features(features)
        return self.settings.available_features()

    # ------------------------------------------------------------------
    # Column selection
    # ------------------------------------------------------------------

    def _select_columns(
        self,
        lf: pl.LazyFrame,
        feature_names: list[str],
    ) -> pl.LazyFrame:
        """
        Selects the trajectory ID column plus the given feature columns.

        This is the view-level column filter.  The store always returns
        all columns; the view picks what it needs.
        """
        return lf.select([self._store.traj_id_col] + feature_names)

    def _rename_columns(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Renames store-internal columns to user-facing names.

        The mapping is provided by ``settings.get_column_rename_map()``.
        """
        return lf.rename(self.settings.get_column_rename_map())

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _get_static_data_from_store(self) -> pl.LazyFrame | None:
        """
        Fetches trajectory-level static data, applies view-level cast recipes.

        Returns ``None`` when no static features exist.
        """
        return self._store.get_static_data(
            virtual_id=self._virtual_id,
            id_cast=self._casts.id or None,
            feature_casts=self._casts.static or None,
        )

    # ------------------------------------------------------------------
    # Apply (read-only computation)
    # ------------------------------------------------------------------

    def apply(
        self,
        exprs: pl.Expr | list[pl.Expr],
        *,
        lazy: bool = False,
        to_pandas: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame | pd.DataFrame:
        """
        Evaluates Polars expressions against trajectory-level static features.

        This is a **read-only** computation: the result is returned,
        not stored.  Use :meth:`add_static_features` to persist the result.

        Each expression must produce a **named** column (``.alias()``).

        Args:
            exprs: One or more Polars expressions producing new columns.
            lazy: If ``True``, returns a ``pl.LazyFrame`` (no collect).
            to_pandas: If ``True``, returns a ``pandas.DataFrame``.

        Returns:
            The computed columns as a DataFrame (or LazyFrame).

        Raises:
            ValueError: If no static features are available.

        Examples::

            result = pool.apply(
                (pl.col("score") * pl.col("weight")).alias("weighted_score"),
            )
            pool.add_static_features(result)
        """
        if isinstance(exprs, pl.Expr):
            exprs = [exprs]

        lf = self._get_static_data_from_store()
        if lf is None:
            raise ValueError("No static features available.")

        lf = self._apply_masks(lf)
        lf = self._rename_columns(lf)
        result_lf = lf.select(exprs)

        if lazy:
            return result_lf
        if to_pandas:
            return result_lf.collect().to_pandas()
        return result_lf.collect()
