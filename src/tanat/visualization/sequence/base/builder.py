#!/usr/bin/env python3
"""
BaseSequenceVizBuilder: abstract base for all sequence visualization builders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import polars as pl
from tanat_utils import CachableSettings, Registrable

from ....sequence.base.pool import SequencePool
from ....sequence.base.sequence import Sequence
from ...utils.color_manager import ColorManager
from ...utils.result import VisualizationResult

if TYPE_CHECKING:
    from ...style.base import BaseVizSettings


class BaseSequenceVizBuilder(ABC, CachableSettings, Registrable):
    """Abstract base class for sequence visualization builders.

    The :meth:`draw` method orchestrates the full pipeline:
    prepare data → create figure → render → apply styling.
    """

    _REGISTER: dict = {}
    SETTINGS_CLASS: type[BaseVizSettings] | None = (
        None  # must be overridden by subclass
    )

    def __init__(
        self,
        settings: dict | BaseVizSettings | None = None,
        *,
        allow_large: bool = False,
    ) -> None:
        CachableSettings.__init__(self, settings=settings)
        self._facet_by: str | None = None
        self._facet_cols: int = 3
        self._facet_share_y: bool = True
        self.allow_large: bool = allow_large


    # ------------------------------------------------------------------
    # Chainable configuration
    # ------------------------------------------------------------------

    def colors(self, spec: str | dict | list) -> BaseSequenceVizBuilder:
        """Set the color spec. Chainable."""
        self.update_settings(colors=spec)
        return self

    def title(self, text: str) -> BaseSequenceVizBuilder:
        """Set the figure title. Chainable."""
        self.update_settings(title=text)
        return self

    def facet(
        self,
        by: str,
        *,
        cols: int = 3,
        share_y: bool = True,
    ) -> BaseSequenceVizBuilder:
        """Enable faceted view. Chainable."""
        self._facet_by = by
        self._facet_cols = cols
        self._facet_share_y = share_y
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @CachableSettings.cached_method()
    def prepare_data(
        self,
        sequence_or_pool: SequencePool | Sequence,
        *,
        entity_feature: str,
        drop_na: bool = False,
    ) -> pl.DataFrame:
        """Prepare the aggregated Polars DataFrame for rendering.

        Args:
            sequence_or_pool: A :class:`~tanat.sequence.base.pool.SequencePool` or an
                individual :class:`~tanat.sequence.base.sequence.Sequence`
                (e.g. obtained via ``pool[42]``).
            entity_feature: Entity feature column to use as category labels.
            drop_na: Drop rows where the label is null before aggregation.

        Returns:
            Polars DataFrame with columns ``[__LABEL__, __VALUE__]``,
            plus ``__COLOR__`` when a color spec has been set via :meth:`colors`.

        Raises:
            TypeError: If *sequence_or_pool* is not a
                :class:`~tanat.sequence.base.pool.SequencePool` or a
                :class:`~tanat.sequence.base.sequence.Sequence`.
            KeyError: If *entity_feature* is not a declared entity feature.
        """
        if not isinstance(sequence_or_pool, (SequencePool, Sequence)):
            raise TypeError(
                f"Expected a SequencePool or Sequence, got {type(sequence_or_pool).__name__!r}."
            )
        sequence_or_pool.settings.validate_features(entity_feature)

        return self._prepare_data(
            sequence_or_pool, entity_feature=entity_feature, drop_na=drop_na
        )

    def draw(
        self,
        sequence_or_pool: SequencePool | Sequence,
        *,
        entity_feature: str,
        drop_na: bool = False,
    ) -> VisualizationResult:
        """Full pipeline: prepare data, create figure, render, return result.

        Args:
            sequence_or_pool: A :class:`~tanat.sequence.base.pool.SequencePool` or an
                individual :class:`~tanat.sequence.base.sequence.Sequence`
                (e.g. obtained via ``pool[42]``).
            entity_feature: Entity feature column to use as category labels.
            drop_na: Drop rows where the label is null before aggregation.

        Returns:
            A :class:`~tanat.visualization.utils.result.VisualizationResult`.
        """
        data = self.prepare_data(
            sequence_or_pool, entity_feature=entity_feature, drop_na=drop_na
        )

        fig, ax = plt.subplots(figsize=self.settings.figsize)
        self._render(ax, data)
        self._apply_styling(ax)

        return VisualizationResult(fig)

    # ------------------------------------------------------------------
    # Internals to implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare_data(
        self,
        sequence_or_pool: SequencePool | Sequence,
        *,
        entity_feature: str,
        drop_na: bool,
    ) -> pl.DataFrame:
        """Subclass-specific data preparation (aggregation)."""

    @abstractmethod
    def _render(self, ax: Any, data: pl.DataFrame) -> None:
        """Subclass-specific matplotlib rendering."""

    # ------------------------------------------------------------------
    # Shared styling helpers
    # ------------------------------------------------------------------

    def _apply_styling(self, ax: Any) -> None:
        """Apply common axis styling from settings."""
        s = self.settings

        if s.title:
            ax.set_title(s.title)

        x = s.x_axis
        if x.label:
            ax.set_xlabel(x.label)
        if x.tick_rotation:
            ax.tick_params(axis="x", rotation=x.tick_rotation)
        if x.limit_min is not None or x.limit_max is not None:
            ax.set_xlim(x.limit_min, x.limit_max)

        y = s.y_axis
        if y.label:
            ax.set_ylabel(y.label)
        if y.tick_rotation:
            ax.tick_params(axis="y", rotation=y.tick_rotation)
        if y.limit_min is not None or y.limit_max is not None:
            ax.set_ylim(y.limit_min, y.limit_max)

        if s.legend.show:
            ax.legend(
                loc=s.legend.location,
                title=s.legend.title,
            )

    @staticmethod
    def _build_color_map(data: pl.DataFrame, spec: Any) -> dict:
        """Build a ``{label: hex}`` color map from the __LABEL__ column."""
        labels = data["__LABEL__"].unique().to_list()
        return ColorManager.build(labels, spec)
