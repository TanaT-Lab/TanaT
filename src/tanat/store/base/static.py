#!/usr/bin/env python3
"""
StaticStoreMixin: shared static-feature logic.

Both SequenceStore and TrajectoryStore follow the same pattern:

* A physical ``static_features.arrow`` on disk (optional).
* A virtual layer under ``tmp/<virtual_id>/static_features.arrow``.
* Merge at read time via horizontal concatenation.
* Split physical / virtual at write / drop time.

This mixin extracts that shared logic.  It is a **pure mixin** that
relies on attributes provided by :class:`BaseStore` (``_root_path``,
``_virtual``, ``main_index``, ``main_id_col``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import polars as pl

from ..sequence.schema import StoreSchema as SCH
from .utils import (
    check_no_reserved_names,
    drop_columns_from_file,
    hconcat_physical_virtual,
    apply_casts,
    normalise_to_lazyframe,
    probe_cast,
)


class StaticStoreMixin:
    """
    Mixin that manages **static features** (physical + virtual).

    Expects the host class to provide:

    * ``self._root_path: Path``
    * ``self._virtual: VirtualStore``
    * ``self.main_index: pl.LazyFrame``
    * ``self.main_id_col: str``
    """

    FILE_STATIC_FEATURES: Final[str] = SCH.Files.STATIC_FEATURES

    # Physical cache slot
    _phys_static_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Static features
    # ------------------------------------------------------------------

    def static_features(self, virtual_id: str | None = None) -> list[str]:
        """List of static-feature column names (physical + virtual when *virtual_id* is set)."""
        if virtual_id is not None:
            lf = self.static(virtual_id)
            return lf.collect_schema().names() if lf is not None else []
        if self._phys_static_names is None:
            path: Path = self._root_path / self.FILE_STATIC_FEATURES
            self._phys_static_names = (
                pl.scan_ipc(path).collect_schema().names() if path.exists() else []
            )
        return self._phys_static_names

    def static(self, virtual_id: str | None = None) -> pl.LazyFrame | None:
        """Static feature columns only (physical + virtual), without the id column.

        Virtual features take precedence: any physical column whose name is
        also present in the virtual context is silently shadowed, so the
        virtual value is always returned.
        """
        path: Path = self._root_path / self.FILE_STATIC_FEATURES
        physical_lf = pl.scan_ipc(path) if path.exists() else None
        virtual_lf = (
            self._virtual.features(virtual_id, is_static=True) if virtual_id else None
        )
        return hconcat_physical_virtual(physical_lf, virtual_lf)

    def get_static_data(
        self,
        virtual_id: str | None = None,
        *,
        id_cast: pl.DataType | None = None,
        feature_casts: dict[str, pl.DataType] | None = None,
    ) -> pl.LazyFrame | None:
        """Static features with id column prepended, and optional cast overlays."""
        features_lf = self.static(virtual_id)
        if features_lf is None:
            return None
        id_lf = self.main_index.select(self.main_id_col)
        if id_cast is not None:
            id_lf = id_lf.with_columns(pl.col(self.main_id_col).cast(id_cast))
        lf = pl.concat([id_lf, features_lf], how="horizontal")
        return apply_casts(lf, feature_casts) if feature_casts else lf

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_static(
        self,
        virtual_id: str,
        df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        *,
        id_col: str,
    ) -> list[str]:
        """Add static feature columns into a virtual context via a LEFT JOIN.

        The input *df* must contain the internal ID column named *id_col*.
        A LEFT JOIN against the main index guarantees that the output has
        exactly one row per ID (nulls for IDs absent from *df*), so no
        height validation is required by the virtual store.

        Args:
            virtual_id: Virtual context identifier.
            df: DataFrame carrying *id_col* plus one or more feature columns.
            id_col: Name of the ID column in *df* (already renamed to the
                internal store ID before this call).

        Returns:
            The list of feature column names written (excludes *id_col*).
        """
        # Normalise to LazyFrame
        lf = normalise_to_lazyframe(df)
        # Guard: feature columns must not collide with the internal ID column name.
        feature_cols = [c for c in lf.collect_schema().names() if c != id_col]
        check_no_reserved_names(
            feature_cols,
            frozenset({self.main_id_col}),
            context="the internal store ID column",
        )
        # Cast id_col to match the internal index dtype to prevent JOIN type mismatch
        main_id_dtype = self.main_index.collect_schema()[self.main_id_col]
        lf = lf.with_columns(pl.col(id_col).cast(main_id_dtype))
        aligned_lf = (
            self.main_index.select(self.main_id_col)
            .join(lf, left_on=self.main_id_col, right_on=id_col, how="left")
            .drop(self.main_id_col)
            .collect()  # materialise before sink_ipc: streaming engine does not
            .lazy()  # preserve JOIN row order without collect().lazy()
        )
        return self._virtual.add_static_features(
            virtual_id=virtual_id,
            df=aligned_lf,
        )

    def drop_static(
        self,
        features: list[str],
        virtual_id: str | None = None,
    ) -> None:
        """Permanently removes static-feature columns (physical and/or virtual)."""
        path = self._root_path / self.FILE_STATIC_FEATURES
        if drop_columns_from_file(path, features):
            self._clear_static_cache()
        if virtual_id:
            self._virtual.drop_features(virtual_id, features, is_static=True)

    def probe_static_cast(
        self, schema: dict[str, pl.DataType], n_rows: int = 10
    ) -> None:
        """
        Validates *schema* against a sample of static-feature rows.

        No-op when the store has no static features.

        Args:
            schema: Mapping of feature name → target dtype.
            n_rows: Sample size (default: 10).
        """
        static = self.static()
        if static is not None:
            probe_cast(static, schema, n_rows)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _clear_static_cache(self) -> None:
        """Invalidates the physical static-feature name cache."""
        self._phys_static_names = None
