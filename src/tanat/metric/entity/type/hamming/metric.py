#!/usr/bin/env python3
"""
HammingEntityMetric: categorical feature distance by Hamming equality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
from numba.typed import List as NumbaList

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
            ``None`` → first entity feature from the pool/entity metadata.
        cost: Asymmetric cost lookup.  Keys are ``(val_a, val_b)`` tuples.
            Symmetric: ``(A, B)`` and ``(B, A)`` are both checked.
            ``None`` → use ``mismatch_cost`` for every mismatch.
        mismatch_cost: Default cost applied when the pair is not in ``cost``
            and values differ (default: ``1.0``).
    """

    entity_feature: str | None = None
    cost: dict[tuple, float] | None = None
    mismatch_cost: float = 1.0


class HammingEntityMetric(EntityMetric, register_name="hamming"):
    """Categorical Hamming distance between two entities.

    Returns ``0.0`` when both entities share the same value for the
    configured feature, and ``mismatch_cost`` (default ``1.0``) when
    they differ.  A custom ``cost`` dict enables asymmetric / partial costs.

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
        """Type-check then verify the configured feature is categorical.

        Args:
            ent_a: Primary entity.
            ent_b: Optional second entity.

        Raises:
            TypeError: If an argument is not an :class:`Entity`, or the
                feature is not ``Categorical`` / ``Enum``.
            KeyError:  If the configured feature is absent.
        """
        self._validate_entity_instance(ent_a, ent_b)
        feature = self.settings.entity_feature or ent_a.feature_names[0]
        self._validate_categorical(ent_a.metadata.get(feature), feature)
        if ent_b is not None:
            self._validate_categorical(ent_b.metadata.get(feature), feature)

    def _validate_categorical(self, info: FeatureInfo | None, feature: str) -> None:
        """Assert that *feature* metadata is categorical.

        Args:
            info: ``FeatureInfo`` for the feature, or ``None`` if not found.
            feature: Feature name (for error messages).

        Raises:
            KeyError:  Feature not found.
            TypeError: Feature is not ``Categorical`` or ``Enum``.
        """
        if info is None:
            raise KeyError(f"Feature '{feature}' not found in entity features.")
        if not isinstance(info, CategoricalInfo):
            raise TypeError(
                f"HammingEntityMetric requires a Categorical or Enum feature, "
                f"got '{feature}' ({type(info).__name__}). "
                f"Cast it first to pl.Categorical or pl.Enum."
            )

    # ------------------------------------------------------------------
    # Numba batch protocol
    # ------------------------------------------------------------------

    def prepare_batch_data(self, pool: SequencePool) -> tuple:
        """Extract categorical feature from pool, encode to int32 for Numba.

        Implementation steps:
            1. Resolve feature name (settings or first entity feature).
            2. Validate that the feature is categorical.
            3. Build a lazy temporal frame restricted to that feature.
            4. Group by id_col, aggregate feature as list (still lazy).
            5. Collect once; build dict id → values (handles empty sequences).
            6. Iterate pool.unique_ids to get per-sequence value lists.
            7. Build a shared vocabulary: unique values → int32 codes.
            8. Encode + pack per-sequence int32 arrays into a
               ``numba.typed.List``.
            9. Build a lengths array (int32).
            10. Build context tuple:
                - simple mode (cost=None): ``()``
                - weighted mode: ``(cost_matrix,)`` where cost_matrix
                  is a float32 (V × V) array indexed by vocabulary codes,
                  mirroring the Python symmetric cost-dict lookup.

        Args:
            pool: The sequence pool to extract data from.

        Returns:
            ``(arrays, lengths, context)``:
                arrays:  ``numba.typed.List`` of int32 arrays, one per
                         sequence, ordered by ``pool.unique_ids``.
                lengths: int32 array of sequence lengths.
                context: tuple passed as-is to the distance kernel.
        """
        # Step 1: resolve feature name
        feature = self.settings.entity_feature or pool.metadata.entity_features[0].name

        # Step 2: validate categorical
        self._validate_categorical(pool.metadata.feature_info(feature), feature)

        # Step 3: lazy temporal frame (id col + feature col)
        id_col = pool.settings.id_column
        lf = pool._temporal_data_lf(features=[feature])  # lazy!

        # Step 4: group by id, aggregate feature as list (still lazy)
        lf = lf.group_by(id_col).agg(pl.col(feature))

        # Step 5: single collect; build id → values dict
        df = lf.collect()
        id_to_values: dict = dict(zip(df[id_col].to_list(), df[feature].to_list()))

        # Step 6: iterate pool.unique_ids (preserves pool order; handles
        # absent IDs, i.e. empty sequences, naturally)
        raw_sequences: list[list] = [
            id_to_values.get(sid, []) for sid in pool.unique_ids
        ]

        # Step 7: shared vocabulary: stable sort for reproducibility
        all_values: set = set()
        for vals in raw_sequences:
            all_values.update(vals)
        vocab: dict = {v: i for i, v in enumerate(sorted(all_values))}

        # Step 8: encode + pack into NumbaList
        arrays: NumbaList = NumbaList()
        for vals in raw_sequences:
            encoded = np.array([vocab[v] for v in vals], dtype=np.int32)
            arrays.append(encoded)

        # Step 9: lengths array
        lengths = np.array([len(arr) for arr in arrays], dtype=np.int32)

        # Step 10: context tuple
        if self.settings.cost is not None:
            V = len(vocab)
            reverse_vocab: dict = {i: v for v, i in vocab.items()}
            cost_matrix = np.full(
                (V, V), float(self.settings.mismatch_cost), dtype=np.float32
            )
            np.fill_diagonal(cost_matrix, 0.0)
            # Mirror the Python symmetric lookup: for each (i, j) pair,
            # check (val_a, val_b) first then (val_b, val_a).
            for i in range(V):
                for j in range(V):
                    if i == j:
                        continue
                    val_a = reverse_vocab[i]
                    val_b = reverse_vocab[j]
                    pair_ab = (val_a, val_b)
                    pair_ba = (val_b, val_a)
                    if pair_ab in self.settings.cost:
                        cost_matrix[i, j] = float(self.settings.cost[pair_ab])
                    elif pair_ba in self.settings.cost:
                        cost_matrix[i, j] = float(self.settings.cost[pair_ba])
            context: tuple = (cost_matrix,)
        else:
            context = ()

        return arrays, lengths, context

    @property
    def distance_kernel(self) -> Callable:
        """Return the Numba-compiled entity distance kernel.

        Returns :func:`~.kernels.hamming_dist_simple` when ``cost`` is
        ``None``, or :func:`~.kernels.hamming_dist_weighted` when a cost
        dict is configured.
        """
        if self.settings.cost is None:
            return hamming_dist_simple
        return hamming_dist_weighted

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, ent_a: Entity, ent_b: Entity) -> float:
        """Compute Hamming distance (entities are already validated).

        Args:
            ent_a: First entity.
            ent_b: Second entity.

        Returns:
            ``0.0`` if equal, cost-dict lookup or ``mismatch_cost`` otherwise.
        """
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
