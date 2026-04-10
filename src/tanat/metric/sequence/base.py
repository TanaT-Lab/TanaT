#!/usr/bin/env python3
"""
SequenceMetric ABC: base class for all sequence-level distance metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from tanat_utils import SettingsMixin, Registrable, DisplayMixin

from ..matrix import DistanceMatrix
from ..entity.base import EntityMetric
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence


class SequenceMetric(SettingsMixin, Registrable, DisplayMixin, ABC):
    """Abstract base for sequence-level distance metrics.

    Computes a scalar distance between two :class:`~tanat.sequence.base.sequence.Sequence`
    objects and a full pairwise :class:`~tanat.metric.DistanceMatrix` over a pool.
    """

    _REGISTER: dict = {}
    _TYPE_SUBMODULE = "type"

    @SettingsMixin.shadow_dispatch
    def __call__(  # pylint: disable=unused-argument
        self, seq_a: Sequence, seq_b: Sequence, **kwargs
    ) -> float:
        """Compute distance between two sequences.

        Settings-matching ``kwargs`` create a temporary shadow view.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.
            **kwargs: Settings overrides.

        Returns:
            A non-negative scalar distance.
        """
        self._validate_sequences(seq_a, seq_b)
        self.validate_composition(seq_a, seq_b)
        return self._compute(seq_a, seq_b)

    @abstractmethod
    def _compute(self, seq_a: Sequence, seq_b: Sequence) -> float:
        """Core distance computation.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            A non-negative scalar distance.
        """

    # ------------------------------------------------------------------
    # Matrix computation
    # ------------------------------------------------------------------

    @SettingsMixin.shadow_dispatch
    def compute_matrix(  # pylint: disable=unused-argument
        self, pool: SequencePool, **kwargs
    ) -> DistanceMatrix:
        """Compute the full pairwise distance matrix for *pool*.

        Args:
            pool: A :class:`~tanat.sequence.base.pool.SequencePool`.

        Settings-matching ``kwargs`` create a temporary shadow view.

        Returns:
            A symmetric :class:`~tanat.metric.DistanceMatrix` with zeros
            on the diagonal.
        """
        self._validate_pool(pool)
        for sid in pool.unique_ids:
            seq = pool[sid]
            if seq:
                self.validate_composition(seq)
                break
        self._display_header()
        dm = self._compute_matrix_impl(pool)
        self._display_footer(f"{len(pool)} sequences")
        return dm

    def _compute_matrix_impl(self, pool: SequencePool) -> DistanceMatrix:
        """Default O(n²) double-loop implementation.

        Correct for all metrics.  Subclasses may override to use
        batch-optimised kernels (e.g. Numba).

        Args:
            pool: Sequence pool.

        Returns:
            Symmetric :class:`DistanceMatrix`.
        """
        ids = pool.unique_ids
        n = len(ids)
        result = np.zeros((n, n), dtype=np.float32)
        seqs = {sid: pool[sid] for sid in ids}

        with self._create_progress_bar(total=n * (n - 1) // 2, desc="Pairs") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    d = self._compute(seqs[ids[i]], seqs[ids[j]])
                    result[i, j] = result[j, i] = float(d)
                    pbar.update(1)

        return DistanceMatrix(result, ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def entity_metric(self) -> EntityMetric:
        """Resolve ``settings.entity_metric`` (string → instance or pass-through).

        Returns:
            The resolved :class:`~tanat.metric.entity.base.EntityMetric`.

        Raises:
            AttributeError: If the concrete settings class has no
                ``entity_metric`` field.
        """
        metric = self.settings.entity_metric
        if isinstance(metric, str):
            return EntityMetric.get_registered(metric)()
        return metric

    def _validate_pool(self, pool: SequencePool) -> None:
        """Type-check the pool argument.

        Raises:
            TypeError: If *pool* is not a
                :class:`~tanat.sequence.base.pool.SequencePool`.
        """
        if not isinstance(pool, SequencePool):
            raise TypeError(f"pool must be a SequencePool, got {type(pool).__name__}")

    @abstractmethod
    def validate_composition(
        self, seq_a: Sequence, seq_b: Sequence | None = None
    ) -> None:
        """Composition compatibility check between this metric and the given sequence(s).

        Called from :meth:`__call__` with both sequences, and from
        :meth:`compute_matrix` with a single sample sequence (``seq_b=None``).
        Subclasses that compose with an :class:`~tanat.metric.entity.base.EntityMetric`
        probe a sample entity to surface schema errors early.

        Args:
            seq_a: Primary sequence to probe.
            seq_b: Optional second sequence.

        Raises:
            TypeError: If the entity feature has an incompatible dtype.
            KeyError:  If a required feature is absent.
        """

    def _validate_sequences(self, seq_a: Sequence, seq_b: Sequence) -> None:
        """Type-check both sequence arguments.

        Raises:
            TypeError: If either argument is not a
                :class:`~tanat.sequence.base.sequence.Sequence`.
        """
        if not isinstance(seq_a, Sequence):
            raise TypeError(f"seq_a must be a Sequence, got {type(seq_a).__name__}")
        if not isinstance(seq_b, Sequence):
            raise TypeError(f"seq_b must be a Sequence, got {type(seq_b).__name__}")
