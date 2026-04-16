#!/usr/bin/env python3
"""
HammingEntityMetric: categorical feature distance by Hamming equality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
from numba.typed import List as NumbaList

from pydantic import field_validator
from tanat_utils import settings_dataclass as dataclass

from .....metadata.feature import CategoricalInfo, FeatureInfo
from ...base import EntityMetric
from .kernels import hamming_dist_simple, hamming_dist_weighted

if TYPE_CHECKING:
    from .....sequence.base.entity import Entity
    from .....sequence.base.pool import SequencePool


@dataclass
class HammingSettings:
    """Settings for :class:`HammingEntityMetric`.

    Args:
        entity_feature: Name of the categorical feature to compare.
            ``None`` - first entity feature from the pool/entity metadata.
        cost: Pairwise cost lookup. Keys are ``(val_a, val_b)`` tuples;
            order does not matter (both ``(A, B)`` and ``(B, A)`` are
            checked). Conflicting entries are rejected at construction.
            Default: ``None`` (every mismatch uses ``mismatch_cost``).
        mismatch_cost: Default cost applied when the pair is not in ``cost``
            and values differ (default: ``1.0``).
    """

    entity_feature: str | None = None
    cost: dict[tuple, float] | None = None
    mismatch_cost: float = 1.0

    @field_validator("cost", mode="before")
    @classmethod
    def validate_cost_symmetry(cls, v):
        """Reject cost dicts with conflicting asymmetric entries."""
        if v is None:
            return v
        for (a, b), val in v.items():
            if (b, a) in v and v[(b, a)] != val:
                raise ValueError(
                    f"Asymmetric cost entries are not supported: "
                    f"({a!r}, {b!r})={val} vs ({b!r}, {a!r})={v[(b, a)]}. "
                    f"Use the same value for both orderings."
                )
        return v


class HammingEntityMetric(EntityMetric, register_name="hamming"):
    """Categorical Hamming distance between two entities.

    Returns ``0.0`` when both entities share the same value for the
    configured feature, and ``mismatch_cost`` (default ``1.0``) when
    they differ.  A custom ``cost`` dict enables partial costs.

    Example::

        hamming = HammingEntityMetric()
        hamming(ent_a, ent_b)                           # 0.0 or 1.0

        hamming = HammingEntityMetric(
            entity_feature="status",
            cost={("A", "B"): 0.5},
            mismatch_cost=0.8,
        )
        hamming(ent_a, ent_b)                           # looks up in cost dict
    """

    SETTINGS_CLASS = HammingSettings
    NUMBA_OPTIM: bool = True
    IS_SYMMETRIC: bool = True  # dist(a, b) == dist(b, a) for all cost configurations

    def __init__(
        self,
        entity_feature: str | None = None,
        cost: dict[tuple, float] | None = None,
        mismatch_cost: float = 1.0,
    ) -> None:
        super().__init__(
            settings=HammingSettings(
                entity_feature=entity_feature,
                cost=cost,
                mismatch_cost=mismatch_cost,
            )
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_entity(self, ent_a: Entity, ent_b: Entity | None = None) -> None:
        """Verify the configured feature exists and is categorical."""
        self._validate_entity_instance(ent_a, ent_b)
        feature = self.settings.entity_feature or ent_a.feature_names[0]
        self._validate_categorical(ent_a.metadata.get(feature), feature)
        if ent_b is not None:
            self._validate_categorical(ent_b.metadata.get(feature), feature)

    def _validate_categorical(self, info: FeatureInfo | None, feature: str) -> None:
        """Assert that *feature* metadata is categorical."""
        if info is None:
            raise KeyError(f"Feature '{feature}' not found in entity features.")
        if not isinstance(info, CategoricalInfo):
            raise TypeError(
                f"HammingEntityMetric requires a Categorical or Enum feature, "
                f"got '{feature}' ({type(info).__name__}). "
                f"Cast it first to pl.Categorical or pl.Enum."
            )

    # ------------------------------------------------------------------
    # Numba batch protocol: private helpers
    # ------------------------------------------------------------------

    def _resolve_feature(self, pool: SequencePool) -> str:
        """Resolve and validate the categorical feature name for *pool*."""
        feature = self.settings.entity_feature or pool.metadata.entity_features[0].name
        self._validate_categorical(pool.metadata.feature_info(feature), feature)
        return feature

    @staticmethod
    def _lazy_grouped_feature(
        pool: SequencePool, feature: str
    ) -> tuple[pl.LazyFrame, str]:
        """Lazy plan that groups a categorical feature by pool id.

        Returns:
            ``(lazy_frame, id_column)``
        """
        id_col = pool.settings.id_column
        lf = pool._temporal_data_lf(features=[feature])
        return lf.group_by(id_col).agg(pl.col(feature)), id_col

    @staticmethod
    def _materialize_raw_sequences(
        df: pl.DataFrame, id_col: str, unique_ids
    ) -> list[list]:
        """Reorder a collected grouped DataFrame into per-id value lists."""
        feature_col = [c for c in df.columns if c != id_col][0]
        mapping: dict = dict(zip(df[id_col].to_list(), df[feature_col].to_list()))
        return [mapping.get(sid, []) for sid in unique_ids]

    @staticmethod
    def _build_vocab(*series: pl.Series) -> dict:
        """Build a stable int-encoding vocabulary from list-typed Series.

        Returns:
            ``{value: int_code}`` dict, sorted for reproducibility.
        """
        combined = pl.concat(list(series))
        vocab_values = combined.explode().unique().sort().to_list()
        return {v: i for i, v in enumerate(vocab_values)}

    @staticmethod
    def _encode(raw_seqs: list[list], vocab: dict) -> tuple:
        """Encode raw value lists into Numba-ready int32 arrays.

        Returns:
            ``(arrays, lengths)``
        """
        arrays: NumbaList = NumbaList()
        for vals in raw_seqs:
            arrays.append(np.array([vocab[v] for v in vals], dtype=np.int32))
        lengths = np.array([len(arr) for arr in arrays], dtype=np.int32)
        return arrays, lengths

    def _build_context(self, vocab: dict) -> tuple:
        """Build the Numba context tuple (cost matrix or empty)."""
        if self.settings.cost is None:
            return ()

        V = len(vocab)
        reverse_vocab: dict = {i: v for v, i in vocab.items()}
        cost_matrix = np.full(
            (V, V), float(self.settings.mismatch_cost), dtype=np.float32
        )
        np.fill_diagonal(cost_matrix, 0.0)
        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                val_a, val_b = reverse_vocab[i], reverse_vocab[j]
                pair_ab = (val_a, val_b)
                pair_ba = (val_b, val_a)
                if pair_ab in self.settings.cost:
                    cost_matrix[i, j] = float(self.settings.cost[pair_ab])
                elif pair_ba in self.settings.cost:
                    cost_matrix[i, j] = float(self.settings.cost[pair_ba])
        return (cost_matrix,)

    # ------------------------------------------------------------------
    # Numba batch protocol: public API
    # ------------------------------------------------------------------

    def prepare_batch_data(self, pool: SequencePool) -> tuple:
        """Extract and encode the categorical feature for Numba batch computation.

        Returns:
            ``(arrays, lengths, context)``
        """
        feature = self._resolve_feature(pool)
        lf, id_col = self._lazy_grouped_feature(pool, feature)
        df = lf.collect()
        raw_seqs = self._materialize_raw_sequences(df, id_col, pool.unique_ids)
        feature_col = [c for c in df.columns if c != id_col][0]
        vocab = self._build_vocab(df[feature_col])
        arrays, lengths = self._encode(raw_seqs, vocab)
        context = self._build_context(vocab)
        return arrays, lengths, context

    def prepare_cross_batch_data(self, pool_rows, pool_cols) -> tuple:
        """Encode two pools with a shared vocabulary for cross-distance.

        Returns:
            ``(arrays_rows, lengths_rows, arrays_cols, lengths_cols, context)``
        """
        feature_r = self._resolve_feature(pool_rows)
        feature_c = self._resolve_feature(pool_cols)

        lf_r, id_col_r = self._lazy_grouped_feature(pool_rows, feature_r)
        lf_c, id_col_c = self._lazy_grouped_feature(pool_cols, feature_c)
        df_r, df_c = pl.collect_all([lf_r, lf_c])

        raw_rows = self._materialize_raw_sequences(df_r, id_col_r, pool_rows.unique_ids)
        raw_cols = self._materialize_raw_sequences(df_c, id_col_c, pool_cols.unique_ids)

        feat_col_r = [c for c in df_r.columns if c != id_col_r][0]
        feat_col_c = [c for c in df_c.columns if c != id_col_c][0]
        vocab = self._build_vocab(df_r[feat_col_r], df_c[feat_col_c])

        arrays_rows, lengths_rows = self._encode(raw_rows, vocab)
        arrays_cols, lengths_cols = self._encode(raw_cols, vocab)
        context = self._build_context(vocab)

        return arrays_rows, lengths_rows, arrays_cols, lengths_cols, context

    @property
    def distance_kernel(self) -> Callable:
        """Numba-compiled entity distance kernel (simple or weighted)."""
        if self.settings.cost is None:
            return hamming_dist_simple
        return hamming_dist_weighted

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, ent_a: Entity, ent_b: Entity) -> float:
        """Compute Hamming distance between two validated entities."""
        feature = self.settings.entity_feature or ent_a.feature_names[0]

        val_a = ent_a[feature]
        val_b = ent_b[feature]

        if val_a == val_b:
            return 0.0

        if self.settings.cost is not None:
            # Check both orderings for symmetry
            pair_ab = (val_a, val_b)
            pair_ba = (val_b, val_a)
            if pair_ab in self.settings.cost:
                return float(self.settings.cost[pair_ab])
            if pair_ba in self.settings.cost:
                return float(self.settings.cost[pair_ba])

        return float(self.settings.mismatch_cost)
