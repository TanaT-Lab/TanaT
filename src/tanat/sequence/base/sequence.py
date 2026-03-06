#!/usr/bin/env python3
"""
Base class for sequence objects.
"""

from __future__ import annotations

from abc import ABC
import logging
from typing import TYPE_CHECKING, Literal

import polars as pl
import pandas as pd
from tanat_utils import CachableSettings, Registrable

from .entity import Entity
from .cast import SequenceCastRecipe
from .view_mixin import SequenceViewMixin

if TYPE_CHECKING:
    from ...store.sequence.store import SequenceStore

LOGGER = logging.getLogger(__name__)


class Sequence(
    ABC,
    SequenceViewMixin,
    CachableSettings,
    Registrable,
):
    """
    Interface to a single sequence within a Store.

    A Sequence is a **scoped view** on the data for one specific ID.
    It shares the same ``SequenceStore`` as its parent Pool (no copy).

    Typical creation patterns::

        # From a Pool (recommended)
        seq = pool[42]

        # Standalone
        seq = StateSequence(id_value=42, store="my_store")
    """

    _REGISTER = {}

    def __init__(
        self,
        id_value,
        store: SequenceStore,
        settings,
        *,
        cast_recipe: SequenceCastRecipe | dict | None = None,
    ) -> None:
        """Base initialiser. delegated to by concrete subclasses after store and
        feature resolution have been performed.

        Args:
            id_value: Unique identifier for this sequence in the store.
            store: Already-resolved :class:`~tanat.store.sequence.store.SequenceStore`.
            settings: Fully-resolved :class:`SequenceSettings`
                (``entity_features`` and ``static_features`` never ``None``).
            cast_recipe: Optional cast recipe (or dict) applied at read time.
                Normalised via :meth:`SequenceCastRecipe.coerce` and probed
                eagerly.

        Raises:
            TypeError: If *cast_recipe* is not a :class:`SequenceCastRecipe`,
                ``dict``, or ``None``.
        """
        self._id_value = id_value
        self._store = store

        CachableSettings.__init__(self, settings=settings)

        self._row_mask: pl.Series | None = None
        self._virtual_id: str | None = None
        self._parent_metadata = None
        self._casts: SequenceCastRecipe = SequenceCastRecipe.coerce(cast_recipe)
        if not self._casts.is_empty():
            self._casts.probe(self._store)

    def _inject(
        self,
        *,
        cast_recipe: SequenceCastRecipe | dict | None = None,
        row_mask: pl.Series | None = None,
        virtual_id: str | None = None,
        parent_metadata=None,
    ) -> Sequence:
        """Inject pool-managed context into this sequence.

        **Not part of the public API**.

        Args:
            cast_recipe: Cast recipe propagated from the parent pool.
            row_mask: Boolean Series aligned with this sequence's rows.
            virtual_id: Virtual context UUID from the parent pool.
            parent_metadata: Pre-computed metadata from the parent pool.

        Returns:
            ``self``
        """
        if cast_recipe is not None:
            self._casts = SequenceCastRecipe.coerce(cast_recipe)
        self._row_mask = row_mask
        self._virtual_id = virtual_id
        self._parent_metadata = parent_metadata
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id_value(self):
        """The sequence identifier."""
        return self._id_value

    def __len__(self) -> int:
        """Number of events/states in this sequence (respects row mask)."""
        if self._row_mask is not None:
            return int(self._row_mask.sum())
        return self._store.get_sequence_length(self._id_value, id_cast=self._casts.id)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, rank: int) -> Entity:
        """
        Get the Entity at the specified rank within the sequence.

        Args:
            rank: 0-based index. Negative indexing supported.

        Returns:
            An Entity flyweight proxy for that row.
        """
        if not isinstance(rank, int):
            raise TypeError(f"Entity rank must be an integer, got {type(rank)}")
        if rank < 0:
            rank += len(self)
        if not (0 <= rank < len(self)):
            raise IndexError(
                f"Entity rank {rank} out of range for sequence of length {len(self)}"
            )

        # Map logical rank → physical row index when row mask is active
        if self._row_mask is not None:
            physical_rank = int(self._row_mask.arg_true()[rank])
        else:
            physical_rank = rank

        reg_name = self.get_registration_name()
        entity_cls = Entity.get_registered(reg_name)
        # pylint: disable=protected-access
        return entity_cls(
            id_value=self._id_value,
            rank=physical_rank,
            store=self._store,
            features=self.settings.entity_features,
        )._inject(
            cast_recipe=self._casts,
            virtual_id=self._virtual_id,
            parent_metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _apply_masks(
        self,
        lf: pl.LazyFrame,
        *,
        is_static: bool = False,
    ) -> pl.LazyFrame:
        """
        Scopes a LazyFrame to this sequence's ID and applies the row mask.
        """
        lf = lf.filter(pl.col(self._store.seq_id_col) == self._id_value)
        if not is_static and self._row_mask is not None:
            lf = lf.filter(pl.lit(self._row_mask))
        return lf

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @CachableSettings.cached_method()
    def _sequence_data_raw(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame:
        """Cached data layer - always returns a Polars DataFrame."""
        valid_features = self._resolve_valid_features(features, is_static=False)
        lf = self._get_data_from_store(is_static=False)
        lf = self._apply_masks(lf, is_static=False)
        lf = self._select_columns(lf, valid_features, is_static=False)
        lf = self._rename_columns(lf, is_static=False)
        return lf.collect()

    def sequence_data(
        self,
        features: list[str] | str | None = None,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Return temporal data for this sequence.

        Args:
            features: Feature name(s) to include (``None`` -> all).
            output_format: ``"pandas"`` (default) or ``"polars"``.

        Returns:
            DataFrame with columns ``[id, temporal…, feature…]`` scoped to
            this sequence ID.

        Examples::

            seq = pool[42]
            df = seq.sequence_data()                    # pandas, all features
            df = seq.sequence_data("heart_rate")        # single feature
            df = seq.sequence_data(output_format="polars")
        """
        df = self._sequence_data_raw(features)
        if output_format == "polars":
            return df
        if output_format == "pandas":
            return df.to_pandas()
        raise ValueError(
            f"Invalid output_format {output_format!r}. "
            "Expected one of: 'pandas', 'polars'."
        )

    @CachableSettings.cached_method()
    def _static_data_raw(
        self,
        features: list[str] | str | None = None,
    ) -> pl.DataFrame | None:
        """Cached data layer - always returns a Polars DataFrame or None."""
        valid_features = self._resolve_valid_features(features, is_static=True)
        if not valid_features:
            return None
        lf = self._get_data_from_store(is_static=True)
        if lf is None:
            return None
        lf = self._apply_masks(lf, is_static=True)
        lf = self._select_columns(lf, valid_features, is_static=True)
        lf = self._rename_columns(lf, is_static=True)
        return lf.collect()

    def static_data(
        self,
        features: list[str] | str | None = None,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pl.DataFrame | pd.DataFrame | None:
        """
        Return static (non-temporal) data for this sequence.

        Args:
            features: Feature name(s) to include (``None`` -> all).
            output_format: ``"pandas"`` (default) or ``"polars"``.

        Returns:
            Single-row DataFrame with columns ``[id, feature…]``.
            ``None`` when no static features are exposed by this pool.

        Examples::

            seq = pool[42]
            row = seq.static_data()               # pandas, all static features
            row = seq.static_data("age", "sex")   # subset
        """
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

    def apply(
        self,
        exprs: pl.Expr | list[pl.Expr],
        is_static: bool = False,
        *,
        output_format: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Evaluates Polars expressions against this sequence's features.

        This is a **read-only** computation scoped to this single
        sequence.  The result is returned, not stored.

        At the Pool level, use ``Pool.apply(by_id=True)`` for
        per-sequence computations across **all** sequences, then
        ``Pool.add_entity_features()`` or ``Pool.add_static_features()``
        to persist.

        Args:
            exprs: One or more Polars expressions producing new columns.
                Each must use ``.alias()`` to name the output.
            is_static: Whether to read static or entity features.


        Returns:
            The computed columns for this sequence only.

        Examples:
            Local normalization::

                seq = pool[42]
                result = seq.apply(
                    (pl.col("value") - pl.col("value").mean()).alias("v_centered")
                )

            Multiple expressions::

                result = seq.apply([
                    (pl.col("value").diff()).alias("v_diff"),
                    (pl.col("value").rolling_mean(3)).alias("v_rm3"),
                ])

        See Also:
            ``Pool.apply``: Apply across all sequences (with optional ``by_id``).
            ``Pool.add_entity_features``: Persist entity features.
            ``Pool.add_static_features``: Persist static features.
        """
        if isinstance(exprs, pl.Expr):
            exprs = [exprs]

        lf = self._get_data_from_store(is_static=is_static)
        if lf is None:
            raise ValueError(
                f"No data found for sequence '{self._id_value}' (is_static={is_static})"
            )

        lf = self._apply_masks(lf, is_static=is_static)
        lf = self._rename_columns(lf, is_static=is_static)
        lf = lf.select(exprs)

        if output_format == "polars":
            return lf.collect()

        if output_format == "pandas":
            return lf.collect().to_pandas()

        raise ValueError(
            f"Invalid output_format {output_format!r}. "
            "Expected one of: 'pandas', 'polars'."
        )
