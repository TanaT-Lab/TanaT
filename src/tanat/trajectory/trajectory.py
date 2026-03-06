#!/usr/bin/env python3
"""Single trajectory: sequences sharing the same ID across stores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import polars as pl
import pandas as pd

from tanat_utils import CachableSettings

from ..sequence.base.sequence import Sequence
from ..store.trajectory.store import TrajectoryStore
from .cast import TrajectoryCastRecipe
from .settings import TrajectorySettings
from .view_mixin import TrajectoryViewMixin


class Trajectory(TrajectoryViewMixin, CachableSettings):
    """
    Access every :class:`Sequence` that shares a given ID across the
    linked stores.

    Usage::

        traj["medical"]         → Sequence
        "medical" in traj       → bool
        for alias in traj: ...  → iterates aliases
        for alias, seq in traj.items(): ...
    """

    SETTINGS_CLASS = TrajectorySettings

    def __init__(
        self,
        id_value,
        store: str | Path | TrajectoryStore,
        *,
        id_column: str = "id",
        static_features: list[str] | None = None,
        cast_recipe: TrajectoryCastRecipe | dict | None = None,
    ) -> None:
        """Create a trajectory view for *id_value*.

        Args:
            id_value: Trajectory identifier.
            store: Store path, name, or :class:`TrajectoryStore` instance.
            id_column: User-facing name for the trajectory ID column.
            static_features: Static feature names to expose.
                ``None`` → all available.  ``[]`` → none.
            cast_recipe: Optional cast recipe (or dict) applied at read time.
                Normalised via :meth:`TrajectoryCastRecipe.coerce` and probed
                eagerly.

        Raises:
            TypeError: If *cast_recipe* is not a :class:`TrajectoryCastRecipe`,
                ``dict``, or ``None``.
        """
        self._id_value = id_value
        self._store = self._resolve_store(store)
        sf = self._resolve_features(self._store, static_features)
        self._alias_mask: set[str] | None = None
        self._virtual_id: str | None = None
        self._parent_metadata = None
        # Set by _inject() when created from a TrajectoryPool.
        # Delegates _build_sequence to pool[id] so pool-level mutations propagate.
        self._sequence_pools: dict | None = None

        CachableSettings.__init__(
            self, settings=TrajectorySettings(id_column=id_column, static_features=sf)
        )

        self._casts: TrajectoryCastRecipe = TrajectoryCastRecipe.coerce(cast_recipe)
        if not self._casts.is_empty():
            self._casts.probe(self._store)

    def _inject(
        self,
        *,
        cast_recipe: TrajectoryCastRecipe | dict | None = None,
        alias_mask: set[str] | None = None,
        virtual_id: str | None = None,
        parent_metadata=None,
        sequence_pools: dict | None = None,
    ) -> Trajectory:
        """Inject pool-managed context into this trajectory.

        Called by :meth:`TrajectoryPool._build_trajectory` immediately after
        construction - **not part of the public API**.  End users create
        trajectories through a pool (``pool[id]``) or standalone without context.

        Args:
            cast_recipe: Cast recipe propagated from the parent pool.
            alias_mask: Set of visible store aliases.
            virtual_id: Virtual context UUID from the parent pool.
            parent_metadata: Pre-computed metadata from the parent pool.
            sequence_pools: Shared sequence pool registry from the parent pool.

        Returns:
            ``self`` - enables fluent construction:
            ``Trajectory(...)._inject(...)``.
        """
        if cast_recipe is not None:
            self._casts = TrajectoryCastRecipe.coerce(cast_recipe)
        self._alias_mask = alias_mask
        self._virtual_id = virtual_id
        self._parent_metadata = parent_metadata
        self._sequence_pools = sequence_pools
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id_value(self):
        """The trajectory identifier, in the cast type if a cast is active."""
        return self._id_value

    @CachableSettings.cached_property
    def _store_aliases(self) -> list[str]:
        """Aliases where this trajectory has data, filtered by mask."""
        all_aliases = self._store.store_aliases
        if self._alias_mask is not None:
            all_aliases = [a for a in all_aliases if a in self._alias_mask]
        # Keep only aliases where this trajectory is actually present
        traj_idx = self._store.trajectory_index
        if self._casts.id is not None:
            traj_idx = traj_idx.with_columns(
                pl.col(self._store.traj_id_col).cast(self._casts.id)
            )
        row = traj_idx.filter(
            pl.col(self._store.traj_id_col) == self._id_value
        ).collect()
        if row.height == 0:
            return []
        row_dict = row.row(0, named=True)
        return [a for a in all_aliases if row_dict.get(a) is True]

    def __len__(self) -> int:
        return len(self._store_aliases)

    def __contains__(self, alias: str) -> bool:
        return alias in self._store_aliases

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def _build_sequence(self, store_alias: str):
        """Build the :class:`Sequence` for *store_alias* without any validity check.

        For internal use only - callers must guarantee that *store_alias* is
        present in :attr:`_store_aliases` (e.g. when iterating or building
        :attr:`sequences`).

        When the parent pool's ``_sequence_pools`` has been injected (via
        ``_build_trajectory``), delegates to ``pool[id]`` so that all pool-level
        mutations (casts, ``add_entity_features``, ``add_static_features``,
        ``drop_features``) propagate
        automatically.  Falls back to building directly from the store when
        the :class:`Trajectory` is used standalone.
        """
        if self._sequence_pools is not None:
            return self._sequence_pools[store_alias][self._id_value]
        # Standalone fallback: build directly from the linked store.
        seq_store = self._store.sequence_stores[store_alias]
        seq_type = seq_store.get_sequence_type()
        seq_cls = Sequence.get_registered(seq_type)
        # pylint: disable=protected-access
        seq = seq_cls(
            id_value=self._id_value,
            store=seq_store,
            **self.settings,
        )._inject(cast_recipe={"id": self._casts.id, "temporal": self._casts.temporal})
        return seq

    def __getitem__(self, store_alias: str):
        if store_alias not in self._store_aliases:
            raise KeyError(
                f"Trajectory '{self._id_value}' has no data in "
                f"store '{store_alias}'. Available: {self._store_aliases}"
            )
        return self._build_sequence(store_alias)

    def __iter__(self) -> Iterator[str]:
        """Iterate over visible store aliases (keys)."""
        yield from self._store_aliases

    def keys(self) -> Iterator[str]:
        """Yield visible store aliases - mirrors ``dict.keys()``."""
        yield from self._store_aliases

    def values(self) -> Iterator[Sequence]:
        """Yield :class:`Sequence` objects for each visible alias - mirrors ``dict.values()``."""
        for alias in self._store_aliases:
            yield self._build_sequence(alias)

    def items(self) -> Iterator[tuple[str, Sequence]]:
        """Yield ``(alias, sequence)`` pairs - mirrors ``dict.items()``."""
        for alias in self._store_aliases:
            yield alias, self._build_sequence(alias)

    @CachableSettings.cached_property
    def sequences(self) -> dict:
        """
        All visible :class:`Sequence` instances for this trajectory,
        keyed by store alias.

        Materialises every sequence reachable through the current alias
        mask.  Useful when several aliases are accessed repeatedly in
        the same computation.

        Cached: rebuilt automatically when the alias mask or cast recipe
        changes (i.e. after :meth:`clear_cache`).
        """
        return {alias: self._build_sequence(alias) for alias in self._store_aliases}

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _apply_masks(self, lf: pl.LazyFrame, **_) -> pl.LazyFrame:
        """Scopes a LazyFrame to this trajectory's ID."""
        return lf.filter(pl.col(self._store.traj_id_col) == self._id_value)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @CachableSettings.cached_method()
    def _static_data_raw(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame | None:
        """Cached data layer - always returns a Polars DataFrame or None.

        Used internally by :meth:`static_data`.  Cache invalidated by
        :meth:`clear_cache` (e.g. after ``cast_features``, ``drop_features``).
        """
        visible = self._resolve_valid_features(features)
        if not visible:
            return None
        lf = self._get_static_data_from_store()
        if lf is None:
            return None
        lf = self._apply_masks(lf)
        lf = self._select_columns(lf, visible)
        lf = self._rename_columns(lf)
        return lf.collect()

    def static_data(
        self,
        features: list[str] | str | None = None,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pl.DataFrame | pd.DataFrame | None:
        """Returns trajectory-level static data for this trajectory only."""
        df = self._static_data_raw(features)
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
