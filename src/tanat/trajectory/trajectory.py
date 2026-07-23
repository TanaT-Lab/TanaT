#!/usr/bin/env python3
"""Single trajectory: sequences sharing the same ID across stores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

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

from ..sequence.base.sequence import Sequence
from ..store.trajectory.store import TrajectoryStore
from ..core import registry as _registry
from ..core.format import resolve_fmt, to_pandas
from ..core.validation import ensure_criterion
from ..zeroing import T0Setter, T0Value, _T0, _T0_NEAREST_RANK
from ..cast import TrajectoryCastRecipe
from .settings import TrajectorySettings
from .view_mixin import TrajectoryViewMixin

if TYPE_CHECKING:
    from .pool import TrajectoryPool
    from ..criterion.base import Criterion


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
    ) -> None:
        """Create a trajectory view for *id_value*.

        Args:
            id_value: Trajectory identifier.
            store: Store path, name, or :class:`TrajectoryStore` instance.
            id_column: User-facing name for the trajectory ID column.
            static_features: Static feature names to expose.
                ``None`` → all available.  ``[]`` → none.
        """
        self._id_value = id_value
        self._store = self._resolve_store(store)
        sf = self._resolve_features(self._store, static_features)
        self._alias_mask: set[str] | None = None

        CachableSettings.__init__(
            self, settings=TrajectorySettings(id_column=id_column, static_features=sf)
        )

        self._parent_pool: TrajectoryPool | None = None
        self._fallback_t0_setter: T0Setter = T0Setter.default()

    @classmethod
    def from_parent(
        cls,
        id_value,
        store: TrajectoryStore,
        settings: TrajectorySettings,
        *,
        parent_pool: TrajectoryPool,
        alias_mask: set[str] | None = None,
    ) -> Trajectory:
        """Create a pool-managed trajectory.  **Not part of the public API.**

        Bypasses store resolution, feature resolution, and cast probe: all
        already performed by the pool.  Pool context (casts, virtual ID,
        sequence pools, metadata) is read lazily from *parent_pool* via
        the corresponding properties.

        Args:
            id_value: Trajectory identifier.
            store: Already-resolved :class:`TrajectoryStore`.
            settings: Fully-resolved :class:`TrajectorySettings`.
            parent_pool: The owning :class:`TrajectoryPool`.
            alias_mask: Override the pool-level alias mask (used by
                :meth:`TrajectoryPool.get_trajectories` with explicit aliases).

        Returns:
            A new :class:`Trajectory` instance bound to *parent_pool*.
        """
        new_traj = object.__new__(cls)
        new_traj._id_value = id_value
        new_traj._store = store
        new_traj._alias_mask = alias_mask
        CachableSettings.__init__(new_traj, settings=settings)
        new_traj._parent_pool = parent_pool
        new_traj._fallback_t0_setter = T0Setter.default()
        return new_traj

    # ------------------------------------------------------------------
    # Lazy pool-delegating properties
    # ------------------------------------------------------------------

    @property
    def _casts(self) -> TrajectoryCastRecipe:
        """Active cast recipe for this trajectory.

        * **Pool path:** delegates to the parent pool's recipe.
        * **Standalone path:** always empty (no casts).
        """
        if self._parent_pool is not None:
            return self._parent_pool._casts
        return TrajectoryCastRecipe()

    @property
    def _virtual_id(self) -> str | None:
        """Virtual context UUID for this trajectory.

        * **Pool path:** delegates to the parent pool's virtual context.
        * **Standalone path:** always ``None``.
        """
        if self._parent_pool is not None:
            return self._parent_pool._virtual_id
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id_value(self):
        """The trajectory identifier, in the cast type if a cast is active."""
        return self._id_value

    @Cachable.cached_property
    def _store_aliases(self) -> list[str]:
        """Aliases where this trajectory has data, filtered by mask."""
        all_aliases = self._store.store_aliases
        if self._alias_mask is not None:
            all_aliases = [a for a in all_aliases if a in self._alias_mask]
        # Keep only aliases where this trajectory is actually present
        traj_idx = self._store.trajectory_index
        id_caster = self._casts.id_caster()
        if id_caster is not None:
            traj_idx = traj_idx.with_columns(id_caster(pl.col(self._store.traj_id_col)))
        row = traj_idx.filter(
            pl.col(self._store.traj_id_col) == self._id_value
        ).collect()
        if row.height == 0:
            return []
        row_dict = row.row(0, named=True)
        return [a for a in all_aliases if row_dict.get(a) is True]

    def __len__(self) -> int:
        return len(self._store_aliases)

    def __repr__(self) -> str:
        return f"Trajectory(id={self._id_value}, sequences={self._store_aliases})"

    def __str__(self) -> str:
        meta = self.metadata
        aliases = self._store_aliases

        overview = [
            format_kv("Trajectory ID", str(self._id_value)),
            format_kv("Sequences", ", ".join(aliases)),
        ]

        t0_val = self.t0
        t0_ranks = self.t0_nearest_rank
        t0_desc = (
            f"{t0_val} ({', '.join(f'{a}: rank {r}' for a, r in t0_ranks.items())})"
            if t0_val is not None
            else "None"
        )
        ti_section = [
            format_kv("Type", str(meta.time_index)),
            format_kv("t0", t0_desc),
        ]

        seq_bullets = [
            format_bullet(alias, repr(seq)) for alias, seq in self.sequences.items()
        ]

        parts = [
            format_header("Trajectory Summary"),
            "",
            format_section("Overview", overview),
            "",
            format_section("Time Index", ti_section),
            "",
            format_section("Sequences", seq_bullets),
        ]

        sf_section = format_feature_section(
            "Static Features",
            [(f.name, f.summary) for f in (meta.static_features or [])],
        )
        if sf_section:
            parts += ["", sf_section]

        return "\n".join(parts)

    def __contains__(self, alias: str) -> bool:
        return alias in self._store_aliases

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def _build_raw_sequence(self, store_alias: str):
        """Build a plain :class:`Sequence` for *store_alias* without T0 binding."""
        seq_store = self._store.sequence_stores[store_alias]
        seq_type = seq_store.get_sequence_type()
        seq_cls = Sequence.get_registered(seq_type)
        return seq_cls(
            id_value=self._id_value,
            store=seq_store,
            id_column=self.settings.id_column,
        )

    def _build_sequence(self, store_alias: str):
        """Build the :class:`Sequence` for *store_alias*."""
        if self._parent_pool is not None:
            return self._parent_pool.sequence_pools[store_alias][self._id_value]
        seq_store = self._store.sequence_stores[store_alias]
        standalone_pool = _registry.build_pool("sequence", seq_store.root_path)
        standalone_pool.update_settings(id_column=self.settings.id_column)
        # pylint: disable=protected-access
        standalone_pool._fallback_t0_setter = self._fallback_t0_setter
        return standalone_pool[self._id_value]

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

    @Cachable.cached_property
    def sequences(self) -> dict:
        """All visible :class:`Sequence` instances for this trajectory, keyed by store alias.

        Cached per trajectory state: built once and reused across calls.
        Invalidated automatically when underlying settings change.
        """
        return {alias: self._build_sequence(alias) for alias in self._store_aliases}

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _apply_id_mask(self, lf: pl.LazyFrame, **_) -> pl.LazyFrame:
        """Scopes a LazyFrame to this trajectory's ID."""
        return lf.filter(pl.col(self.settings.id_column) == self._id_value)

    # ------------------------------------------------------------------
    # Criterion API
    # ------------------------------------------------------------------

    def match(self, criterion: Criterion) -> bool:
        """Return ``True`` if this trajectory satisfies *criterion*.

        Args:
            criterion: A :class:`~tanat.criterion.base.Criterion` instance.

        Raises:
            TypeError: If *criterion* is not a Criterion object.
            CriterionLevelError: If the criterion is incompatible with trajectories.
        """
        ensure_criterion(criterion)
        return criterion.match(self)

    # ------------------------------------------------------------------
    # T0 / Zeroing
    # ------------------------------------------------------------------

    @property
    def _t0_setter(self):
        """Effective T0 setter for this trajectory.

        Delegates to the parent pool's :attr:`_t0_setter` when managed,
        otherwise returns :attr:`_fallback_t0_setter` (computation deferred
        to :attr:`_t0_result`).
        """
        if self._parent_pool is not None:
            return self._parent_pool._t0_setter
        return self._fallback_t0_setter

    @Cachable.cached_property
    def _t0_result(self) -> tuple:
        """Cached T0 ``(value, {alias: nearest_rank})`` pair for this trajectory.

        * **Pool path:** reads from the parent pool's cached T0 lookup.
        * **Standalone path:** computes lazily via :attr:`_t0_setter` and
          resolves per-alias nearest ranks.
        """
        if self._parent_pool is None:
            setter = self._t0_setter
            if setter.df is None:
                aliases = self._store_aliases
                if aliases:
                    raw_seq = self._build_raw_sequence(aliases[0])
                    setter.compute_from_sequence(raw_seq)
                    setter._on = aliases[0]
            df = setter.df
            if df is None or df.height == 0:
                return (None, {})
            id_col = self.settings.id_column
            row = df.filter(pl.col(id_col) == self._id_value)
            if row.height == 0:
                return (None, {})
            t0_value = row[_T0][0]
            # Resolve per-alias nearest ranks using each visible sequence.
            nearest_ranks: dict[str, int | None] = {}
            for alias in self._store_aliases:
                seq = self._build_sequence(alias)
                # pylint: disable=protected-access
                rank_df = seq._resolve_nearest_rank(row)
                nearest_ranks[alias] = rank_df[_T0_NEAREST_RANK][0]
            return (t0_value, nearest_ranks)

        # pylint: disable=protected-access
        lookup = self._parent_pool._get_traj_t0_lookup()
        if self._id_value not in lookup:
            return (None, {})
        return lookup[self._id_value]

    @property
    def t0(self) -> T0Value | None:
        """T0 value for this trajectory.

        ``None`` when T0 could not be determined (e.g. no matching row).
        """
        return self._t0_result[0]

    @property
    def t0_nearest_rank(self) -> dict[str, int | None]:
        """Per-alias nearest rank at or before T0.

        Returns a dict keyed by visible alias name, e.g.
        ``{"medical": 2, "lab": 5}``.  Value is ``None`` when T0 is
        ``None`` or when no row satisfies ``start <= T0`` in that alias.
        An empty dict is returned for standalone (non-pool) trajectories.
        """
        return self._t0_result[1]

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def static_data(
        self,
        features: list[str] | str | None = None,
        fmt: Literal["pandas", "polars", "dict"] = "pandas",
        use_arrow: bool = True,
    ) -> pl.DataFrame | pd.DataFrame | None:
        """Return trajectory-level static data for this trajectory only.

        Args:
            features: Static feature name(s) to include.
                ``None`` -> all visible static features.
            fmt: ``"pandas"`` *(default)* or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            One-row DataFrame with ``[id, feature...]``;
            a python dictionary with named attribute-value pairs or
            ``None`` when no static features are available in the current view.
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars", "dict"), default="pandas")
        df = self._static_data_df(features)
        if df is None or len(df) == 0:
            return None
        if fmt == "polars":
            return df
        elif fmt == "dict":
            return df.row(0, named=True)
        return to_pandas(df, use_arrow=use_arrow)

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------

    @Cachable.cached_method()
    def _describe_result(self, separator: str = "_") -> pl.DataFrame:
        """Cached polars result for :meth:`describe`.

        Calls ``seq.describe()`` for each visible sequence, prefixes metric
        columns with ``{alias}{separator}``, and prepends a ``n_sequences``
        column.  Cached: invalidated automatically when settings change.

        Args:
            separator: Separator between alias and metric name (default ``_``).

        Returns:
            Single-row polars DataFrame with columns
            ``[n_sequences, {alias}{sep}length, …]``.
        """
        frames: list[pl.DataFrame] = []
        for alias, seq in self.items():
            per_seq: pl.DataFrame = seq.describe(fmt="polars")
            renamed = {c: f"{alias}{separator}{c}" for c in per_seq.columns}
            frames.append(per_seq.rename(renamed))

        if not frames:
            raise ValueError("This trajectory has no visible sequences.")

        result = pl.concat(frames, how="horizontal")
        return result.with_columns(pl.lit(len(frames)).alias("n_sequences")).select(
            ["n_sequences"] + [c for c in result.columns if c != "n_sequences"]
        )

    def describe(
        self,
        separator: str = "_",
        fmt: Literal["pandas", "polars"] = "pandas",
        use_arrow: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Compute summary statistics for this single trajectory.

        Calls ``seq.describe()`` for each visible sequence and prefixes
        metric columns with ``{alias}{separator}``.  The result is a
        single-row DataFrame.

        Args:
            separator: Separator between alias and metric name (default ``_``).
            fmt: ``"pandas"`` *(default)* or ``"polars"``.
            use_arrow: Use Arrow extension arrays for polars -> pandas conversion.

        Returns:
            Single-row DataFrame with columns
            ``[n_sequences, {alias}{sep}length, …]``.

        Examples::

            traj = traj_pool[42]
            traj.describe()
            traj.describe(separator=".", fmt="polars")
        """
        fmt = resolve_fmt(fmt, allowed=("pandas", "polars"), default="pandas")
        result = self._describe_result(separator)
        if fmt == "polars":
            return result
        return to_pandas(result, use_arrow=use_arrow)
