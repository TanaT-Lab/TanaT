#!/usr/bin/env python3
"""
BaseSequenceVizBuilder: abstract base for all sequence visualization builders.
"""

from __future__ import annotations

import math
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
    # Beyond ~20 facets the individual panels become too small to read.
    # Override per-instance or pass allow_large=True at the factory.
    MAX_FACET: int = 20

    def __init__(
        self,
        settings: dict | BaseVizSettings | None = None,
        *,
        allow_large: bool = False,
    ) -> None:
        CachableSettings.__init__(self, settings=settings)
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
        is_static: bool = False,
        cols: int = 3,
        share_x: bool = True,
        share_y: bool = True,
        figsize_per_facet: tuple[float, float] = (5.0, 4.0),
        title_template: str = "{by} = {value}",
    ) -> BaseSequenceVizBuilder:
        """Enable faceted (small-multiples) view. Chainable.

        Calling this method updates the settings and **clears the data cache**,
        so any previously cached :meth:`prepare_data` result is discarded.

        Args:
            by: Feature name to split by.
            is_static: ``True`` → static feature;
                ``False`` (default) → entity feature.
            cols: Number of columns in the facet grid (default 3).
            share_x: Share the x-axis scale across facets (default ``True``).
            share_y: Share the y-axis scale across facets (default ``True``).
            figsize_per_facet: Width × height of each cell in inches.
            title_template: Format string for each facet title.  Placeholders:
                ``{by}``, ``{value}``, ``{index}``.
        """
        self.update_settings(
            facet={
                "by": by,
                "is_static": is_static,
                "cols": cols,
                "share_x": share_x,
                "share_y": share_y,
                "figsize_per_facet": figsize_per_facet,
                "title_template": title_template,
            }
        )
        return self

    @property
    def _facet_enabled(self) -> bool:
        """``True`` when a facet has been configured via :meth:`facet`."""
        return self.settings.facet.by is not None

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
            facet_by=self.settings.facet.by if self._facet_enabled else None,
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
        if self._facet_enabled:
            return self._draw_faceted(
                sequence_or_pool,
                entity_feature=entity_feature,
                drop_na=drop_na,
            )

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
    # Faceting internals
    # ------------------------------------------------------------------

    def _inject_facet_column(
        self,
        lf: pl.LazyFrame,
        pool: SequencePool,
        *,
        id_col: str,
    ) -> pl.LazyFrame:
        """Attach ``__FACET__`` to *lf*.

        This is the **only** method that knows about ``is_static``.
        Subclasses call it and never inspect ``settings.facet.is_static``
        directly.

        - *is_static=True*: left-join static data on *id_col*.
        - *is_static=False*: rename the entity feature column.

        Args:
            lf: LazyFrame already containing *id_col* when ``is_static=True``.
            pool: The source pool (needed for the static data join).
            id_col: Name of the ID column in *lf* (``"__ID__"`` for builders
                that called :func:`~.utils.rename_id_column`, or the raw
                ``pool.settings.id_column`` otherwise).
        """
        f = self.settings.facet
        if f.is_static:
            facet_lf = pool._static_data_lf([f.by])
            if facet_lf is None:
                raise ValueError(
                    f"Feature '{f.by}' not found in static data. "
                    "Set is_static=False or add the feature as static data."
                )
            facet_lf = facet_lf.rename({f.by: "__FACET__"})
            return lf.join(
                facet_lf,
                left_on=id_col,
                right_on=pool.settings.id_column,
                how="left",
            )
        # Entity (sequence) feature: just rename the column.
        return lf.rename({f.by: "__FACET__"})

    def _update_per_facet_state(self, df_i: pl.DataFrame) -> pl.DataFrame:
        """Hook called by :meth:`_draw_faceted` before rendering each panel.

        The base implementation is a no-op.  Subclasses that store
        instance-level rendering state (e.g. tick orders, y-tick maps) should
        override this to recompute that state from the facet slice *df_i*.

        Args:
            df_i: DataFrame for a single facet panel (``__FACET__`` already
                dropped, ``__COLOR__`` already re-applied from global map).

        Returns:
            The (possibly modified) *df_i* to pass to :meth:`_render`.
        """
        return df_i

    def _draw_faceted(
        self,
        pool: SequencePool,
        *,
        entity_feature: str,
        drop_na: bool,
    ) -> VisualizationResult:
        """Orchestrate the faceted (small-multiples) render.

        Called by :meth:`draw` when :attr:`_facet_enabled` is ``True``.
        Single-pass: prepares the full dataset once, then partitions by
        ``__FACET__`` for each panel.
        """
        f = self.settings.facet

        # 1 ─ Prepare full dataset (bypasses public cache to avoid pollution)
        full_df = self._prepare_data(
            pool,
            entity_feature=entity_feature,
            drop_na=drop_na,
            facet_by=f.by,
        )
        if "__FACET__" not in full_df.columns:
            raise RuntimeError(
                "_prepare_data did not produce a '__FACET__' column. "
                "Make sure the subclass implementation handles facet_by."
            )
        full_df = full_df.drop_nulls("__FACET__")
        full_df = full_df.with_columns(pl.col("__FACET__").cast(pl.Utf8))

        # 2 ─ Global colour map (ensures stability across panels)
        global_color_map = dict(
            full_df.select(["__LABEL__", "__COLOR__"]).unique().rows()
        )

        # 3 ─ Sorted facet values
        facet_values = sorted(full_df["__FACET__"].unique().to_list())
        n_facets = len(facet_values)

        if n_facets == 0:
            raise ValueError(f"No non-null values found for facet feature '{f.by}'.")
        if n_facets > self.MAX_FACET and not self.allow_large:
            raise ValueError(
                f"Facet feature '{f.by}' has {n_facets} unique values, which exceeds "
                f"MAX_FACET={self.MAX_FACET}. "
                "Reduce the feature cardinality, increase builder.MAX_FACET, "
                "or pass allow_large=True to bypass this guard."
            )

        # 4 ─ Figure layout
        cols = f.cols
        rows = math.ceil(n_facets / cols)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(f.figsize_per_facet[0] * cols, f.figsize_per_facet[1] * rows),
            sharex=f.share_x,
            sharey=f.share_y,
            squeeze=False,
        )
        axes_flat = axes.flatten().tolist()

        # 5 ─ Per-facet render loop
        legend_handles: list | None = None
        legend_labels: list | None = None

        for i, value in enumerate(facet_values):
            ax = axes_flat[i]
            df_i = (
                full_df.filter(pl.col("__FACET__") == value)
                .drop("__FACET__")
                .with_columns(
                    pl.col("__LABEL__").replace(global_color_map).alias("__COLOR__")
                )
            )

            if df_i.is_empty():
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                df_i = self._update_per_facet_state(df_i)
                self._render(ax, df_i)
                self._apply_styling(ax)
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

            # Remove per-subplot legend (shared figure legend added below)
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            ax.set_title(f.title_template.format(by=f.by, value=value, index=i))

        # 6 ─ Hide unused cells
        for i in range(n_facets, len(axes_flat)):
            axes_flat[i].axis("off")

        # 7 ─ Figure-level decorations
        if self.settings.title:
            fig.suptitle(self.settings.title, y=1.01)

        if self.settings.legend.show and legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                title=self.settings.legend.title,
                frameon=True,
            )
            fig.subplots_adjust(right=0.85)

        plt.tight_layout()
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
        facet_by: str | None = None,
    ) -> pl.DataFrame:
        """Subclass-specific data preparation (aggregation, labelling, guards).

        When *facet_by* is not ``None`` the returned DataFrame must contain a
        ``__FACET__`` column.  The base :meth:`_draw_faceted` will then
        partition on that column and render each facet panel separately.
        """

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
