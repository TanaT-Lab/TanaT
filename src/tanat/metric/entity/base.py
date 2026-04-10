#!/usr/bin/env python3
"""
EntityMetric ABC: base class for all entity-level distance metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tanat_utils import SettingsMixin, Registrable

from ...sequence.base.entity import Entity


class EntityMetric(SettingsMixin, Registrable, ABC):
    """Abstract base for entity-level distance metrics.

    Computes a scalar distance between two :class:`~tanat.sequence.base.entity.Entity`
    objects.
    """

    _REGISTER: dict = {}
    _TYPE_SUBMODULE = "type"

    #: Subclasses that provide ``prepare_batch_data`` / ``distance_kernel``
    #: set this to ``True`` to opt into the Numba fast path.
    NUMBA_OPTIM: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @SettingsMixin.shadow_dispatch
    def __call__(  # pylint: disable=unused-argument
        self, ent_a: Entity, ent_b: Entity, **kwargs
    ) -> float:
        """Compute distance between two entities.

        Validates both entities, then delegates to :meth:`_compute`.
        Settings-matching ``kwargs`` create a temporary shadow view.

        Args:
            ent_a: First entity.
            ent_b: Second entity.
            **kwargs: Settings overrides (e.g. ``mismatch_cost=0.5``).

        Returns:
            Non-negative scalar distance.
        """
        self.validate_entity(ent_a, ent_b)
        return self._compute(ent_a, ent_b)

    # ------------------------------------------------------------------
    # Abstract methods: subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def validate_entity(self, ent_a: Entity, ent_b: Entity | None = None) -> None:
        """Validate one or two entities against this metric's requirements.

        Called from :meth:`__call__` (both entities) and from
        :meth:`~tanat.metric.sequence.base.SequenceMetric.validate_composition`
        (single sample entity for early schema check).

        Implementations should call :meth:`_validate_entity_instance` first
        for the type check, then add metric-specific checks.

        Args:
            ent_a: Primary entity.
            ent_b: Optional second entity (``None`` → probe single entity only).

        Raises:
            TypeError: Wrong argument type or incompatible feature dtype.
            KeyError:  Required feature absent from the entity.
        """

    @abstractmethod
    def _compute(self, ent_a: Entity, ent_b: Entity) -> float:
        """Core distance computation (entities are already validated).

        Args:
            ent_a: First entity.
            ent_b: Second entity.

        Returns:
            Non-negative scalar distance.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_entity_instance(
        self, ent_a: Entity, ent_b: Entity | None = None
    ) -> None:
        """Type-check that arguments are :class:`Entity` instances.

        Raises:
            TypeError: If either argument is not an :class:`Entity`.
        """
        if not isinstance(ent_a, Entity):
            raise TypeError(f"ent_a must be an Entity, got {type(ent_a).__name__}")
        if ent_b is not None and not isinstance(ent_b, Entity):
            raise TypeError(f"ent_b must be an Entity, got {type(ent_b).__name__}")
