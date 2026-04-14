#!/usr/bin/env python3
"""
SequenceMetric ABC: base class for all sequence-level distance metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tanat_utils import SettingsMixin, Registrable, DisplayMixin

from ..matrix import DistanceMatrix
from ..entity.base import EntityMetric
from .._utils import resolve_storage, default_pairwise_matrix, validate_pair
from ...sequence.base.pool import SequencePool
from ...sequence.base.sequence import Sequence
from .._storage import StorageOptions


class SequenceMetric(SettingsMixin, Registrable, DisplayMixin, ABC):
    """Abstract base for sequence-level distance metrics.

    Computes a scalar distance between two :class:`~tanat.sequence.base.sequence.Sequence`
    objects and a full pairwise :class:`~tanat.metric.DistanceMatrix` over a pool.
    """

    _REGISTER: dict = {}
    _TYPE_SUBMODULE = "type"

    #: Set to ``True`` in subclasses that implement disk-backed (memmap) computation.
    #: When ``False``, passing ``store_path`` or an instance-level ``StorageOptions``
    #: raises :class:`NotImplementedError` early with a clear message.
    MEMMAP_SUPPORT: bool = False

    def __init__(
        self, settings=None, storage: StorageOptions | dict | None = None
    ) -> None:
        super().__init__(settings)
        self._storage = resolve_storage(
            type(self).__name__, self.MEMMAP_SUPPORT, storage
        )

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
            Scalar distance.
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
            Scalar distance.
        """

    # ------------------------------------------------------------------
    # Matrix computation
    # ------------------------------------------------------------------

    @SettingsMixin.shadow_dispatch
    def compute_matrix(  # pylint: disable=unused-argument
        self,
        pool: SequencePool,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
        **kwargs,
    ) -> DistanceMatrix:
        """Compute the full pairwise distance matrix for *pool*.

        Storage kwargs are forwarded to
        :class:`~tanat.metric.StorageOptions`.  Other ``kwargs``
        (e.g. ``agg_fun``) create a temporary settings override.

        Args:
            pool:       A :class:`~tanat.sequence.base.pool.SequencePool`.
            store_path: Storage directory (``None`` → in-memory).
            chunk_size: Rows per flush chunk (default 500).
            resume:     Skip already-computed chunks (default ``True``).
            dtype:      Numpy dtype for the matrix (default ``"float32"``).
            **kwargs:   Settings overrides (e.g. ``agg_fun``, ``padding_penalty``).

        Returns:
            A :class:`~tanat.metric.DistanceMatrix`.
        """
        storage = resolve_storage(
            type(self).__name__,
            self.MEMMAP_SUPPORT,
            self._storage,
            store_path=store_path,
            chunk_size=chunk_size,
            resume=resume,
            dtype=dtype,
        )

        self._validate_pool(pool)
        for sid in pool.unique_ids:
            seq = pool[sid]
            if seq:
                self.validate_composition(seq)
                break
        self._display_header()
        dm = self._compute_matrix_impl(pool, storage)
        self._display_footer(f"{len(pool)} sequences")
        return dm

    def _compute_matrix_impl(
        self,
        pool: SequencePool,
        _storage=None,  # pylint: disable=unused-argument
    ) -> DistanceMatrix:
        """Default O(n^2) double-loop implementation.

        Subclasses may override for batch-optimised kernels (e.g. Numba).
        """
        items = {sid: pool[sid] for sid in pool.unique_ids}
        return default_pairwise_matrix(
            items, pool.unique_ids, self._compute, self._create_progress_bar
        )

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
        """Type-check the pool argument."""
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
        """Type-check both sequence arguments."""
        validate_pair(seq_a, seq_b, Sequence, "seq_a", "seq_b")
