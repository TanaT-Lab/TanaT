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
    # Beyond ~30 unique label values the legend and axis labels become unreadable.
    # Override per-instance or pass allow_large=True at the factory.
    MAX_CATEGORY: int = 30

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
    # Chainable configuration: figure layout
    # ------------------------------------------------------------------

    def title(self, text: str) -> BaseSequenceVizBuilder:
        """Set the figure title. Chainable.

        Args:
            text: Title string displayed above the chart.
        """
        self.update_settings(title=text)
        return self

    def figsize(self, width: float, height: float) -> BaseSequenceVizBuilder:
        """Set the figure dimensions. Chainable.

        Args:
            width: Figure width in inches.
            height: Figure height in inches.
        """
        self.update_settings(figsize=(width, height))
        return self

    def grid(
        self,
        *,
        show: bool = True,
        color: str | None = None,
        linewidth: float | None = None,
        axis: str | None = None,
    ) -> BaseSequenceVizBuilder:
        """Configure the background grid. Chainable.

        Grid lines are always rendered **behind** markers and bars.

        Args:
            show: Display grid lines when ``True`` (default). Pass ``False`` to
                explicitly hide a grid that was previously enabled.
            color: Line color, any matplotlib color string (default ``"lightgrey"``).
            linewidth: Line width in points (default ``0.8``).
            axis: Which axis to draw lines for: ``"both"`` (default), ``"x"``,
                or ``"y"``.
        """
        patch: dict = {"show": show}
        if color is not None:
            patch["color"] = color
        if linewidth is not None:
            patch["linewidth"] = linewidth
        if axis is not None:
            patch["axis"] = axis
        self.update_settings(grid=patch)
        return self

    # ------------------------------------------------------------------
    # Chainable configuration: colors & legend
    # ------------------------------------------------------------------

    def colors(self, spec: str | dict | list) -> BaseSequenceVizBuilder:
        """Set the color specification. Chainable.

        Args:
            spec: A matplotlib colormap name (``str``), a mapping of
                ``{label: color}`` (``dict``), or an ordered list of colors
                (``list``). ``None`` falls back to the matplotlib default cycle.
        """
        self.update_settings(colors=spec)
        return self

    def legend(
        self,
        *,
        show: bool = True,
        location: str = "best",
        title: str | None = None,
    ) -> BaseSequenceVizBuilder:
        """Configure legend display. Chainable.

        Args:
            show: Display the legend when ``True`` (default).
            location: Matplotlib location string, e.g. ``"upper right"``.
            title: Optional legend title.
        """
        self.update_settings(
            legend={"show": show, "location": location, "title": title}
        )
        return self

    def legend_off(self) -> BaseSequenceVizBuilder:
        """Hide the legend. Convenience shortcut for ``.legend(show=False)``. Chainable."""
        return self.legend(show=False)

    # ------------------------------------------------------------------
    # Chainable configuration: faceting
    # ------------------------------------------------------------------

    def facet(
        self,
        by: str,
        *,
        cols: int = 3,
        share_y: bool = True,
    ) -> BaseSequenceVizBuilder:
        """Enable faceted (small-multiples) view. Chainable.

        Args:
            by: Entity feature name to split by.
            cols: Number of columns in the facet grid (default 3).
            share_y: Share the y-axis scale across facets (default ``True``).
        """
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

        df = self._prepare_data(
            sequence_or_pool,
            entity_feature=entity_feature,
            drop_na=drop_na,
        )

        n_categories = df["__LABEL__"].n_unique()
        if n_categories > self.MAX_CATEGORY and not self.allow_large:
            raise ValueError(
                f"'{entity_feature}' has {n_categories} unique values, which exceeds "
                f"MAX_CATEGORY={self.MAX_CATEGORY}. "
                "Reduce the feature cardinality, increase builder.MAX_CATEGORY, "
                "or pass allow_large=True to bypass this guard."
            )

        return df

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
            sequence_or_pool,
            entity_feature=entity_feature,
            drop_na=drop_na,
        )

        fig, ax = plt.subplots(figsize=self.settings.figsize)
        self._render(ax, data)
        self._apply_styling(ax)

        return VisualizationResult(fig)

    # ------------------------------------------------------------------
    # Internals: patch helpers (used by subclass axis/marker methods)
    # ------------------------------------------------------------------

    def _axis_patch(self, axis: str, **kwargs: Any) -> None:
        """Patch non-``None`` kwargs into an axis settings field.

        Handles the ``rotation`` → ``tick_rotation`` rename transparently so
        subclass axis methods can expose the more intuitive ``rotation`` name.

        Args:
            axis: Settings field name, either ``"x_axis"`` or ``"y_axis"``.
            **kwargs: Axis parameters to update (``None`` values are ignored).
        """
        current = getattr(self.settings, axis)
        patch: dict = vars(current).copy()
        for key, value in kwargs.items():
            if value is not None:
                # rotation maps to tick_rotation in settings
                patch["tick_rotation" if key == "rotation" else key] = value
        self.update_settings(**{axis: patch})

    def _marker_patch(self, **kwargs: Any) -> None:
        """Patch non-``None`` kwargs into the marker settings field.

        Args:
            **kwargs: Marker parameters to update (``None`` values are ignored).
        """
        patch: dict = vars(self.settings.marker).copy()
        for key, value in kwargs.items():
            if value is not None:
                patch[key] = value
        self.update_settings(marker=patch)

    # ------------------------------------------------------------------
    # Abstract interface: subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare_data(
        self,
        sequence_or_pool: SequencePool | Sequence,
        *,
        entity_feature: str,
        drop_na: bool,
    ) -> pl.DataFrame:
        """Subclass-specific data preparation (aggregation, labelling, guards)."""

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

        if s.grid.show:
            ax.set_axisbelow(True)
            ax.grid(
                True,
                color=s.grid.color,
                linewidth=s.grid.linewidth,
                axis=s.grid.axis,
            )

        x = s.x_axis
        if not x.show:
            ax.xaxis.set_visible(False)
        else:
            if x.label:
                ax.set_xlabel(x.label)
            if x.tick_rotation:
                ax.tick_params(axis="x", rotation=x.tick_rotation)
            if x.limit_min is not None or x.limit_max is not None:
                ax.set_xlim(x.limit_min, x.limit_max)
            if x.autofmt_xdate:
                ax.figure.autofmt_xdate()

        y = s.y_axis
        if not y.show:
            ax.yaxis.set_visible(False)
        else:
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
    def _build_color_map(keys: list[str], spec: Any) -> dict[str, str]:
        """Return a ``{label: hex}`` color map for the given keys."""
        return ColorManager.build(sorted(keys), spec)
