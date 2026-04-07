#!/usr/bin/env python3
"""
BarplotVizBuilder: bar chart visualizer for a SequencePool or individual Sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ...base.builder import BaseSequenceVizBuilder
from ...base.exceptions import IncompatibleDisplayUnitError
from ...base.utils import (
    resolve_label,
    handle_null_labels,
    handle_null_time_index,
    rename_id_column,
    rename_time_index_columns,
)
from .exception import UnsupportedShowAsError
from .data import (
    aggregate_count,
    aggregate_duration,
    aggregate_rate,
    apply_sort,
)
from .settings import BarplotSettings

if TYPE_CHECKING:
    from ....sequence.base.pool import SequencePool
    from ....sequence.base.sequence import Sequence


class BarplotVizBuilder(BaseSequenceVizBuilder, register_name="barplot"):
    """Builds bar charts from a SequencePool or an individual Sequence.

    Typical usage via :class:`~tanat.visualization.sequence.core.SequenceVisualizer`::

        SequenceVisualizer.barplot(show_as="count") \\
            .title("Event counts") \\
            .draw(pool, entity_feature="status") \\
            .show()
    """

    SETTINGS_CLASS = BarplotSettings

    # ------------------------------------------------------------------
    # Chainable configuration: axes
    # ------------------------------------------------------------------

    def x_axis(
        self,
        *,
        show: bool | None = None,
        label: str | None = None,
        rotation: int | None = None,
        limit_min: float | None = None,
        limit_max: float | None = None,
    ) -> BarplotVizBuilder:
        """Configure the x (horizontal) axis. Chainable.

        Role depends on orientation:

        - ``"vertical"`` (default): categories on x, values on y → ``limit_min``/``limit_max``
          are not meaningful here.
        - ``"horizontal"``: values on x, categories on y → use ``limit_min``/``limit_max``
          to constrain the value range.

        Args:
            show: Hide the axis entirely when ``False``.
            label: Axis label text.
            rotation: Tick label rotation in degrees.
            limit_min: Minimum x value (meaningful in horizontal orientation only).
            limit_max: Maximum x value (meaningful in horizontal orientation only).
        """
        self._axis_patch(
            "x_axis",
            show=show,
            label=label,
            rotation=rotation,
            limit_min=limit_min,
            limit_max=limit_max,
        )
        return self

    def y_axis(
        self,
        *,
        show: bool | None = None,
        label: str | None = None,
        rotation: int | None = None,
        limit_min: float | None = None,
        limit_max: float | None = None,
    ) -> BarplotVizBuilder:
        """Configure the y (vertical) axis. Chainable.

        Role depends on orientation:

        - ``"vertical"`` (default): values on y, categories on x → use ``limit_min``/``limit_max``
          to constrain the value range (e.g. ``limit_max=1.0`` for rates).
        - ``"horizontal"``: categories on y, values on x → ``limit_min``/``limit_max``
          are not meaningful here.

        Args:
            show: Hide the axis entirely when ``False``.
            label: Axis label text.
            rotation: Tick label rotation in degrees.
            limit_min: Minimum y value (meaningful in vertical orientation only).
            limit_max: Maximum y value (meaningful in vertical orientation only).
        """
        self._axis_patch(
            "y_axis",
            show=show,
            label=label,
            rotation=rotation,
            limit_min=limit_min,
            limit_max=limit_max,
        )
        return self

    # ------------------------------------------------------------------
    # Chainable configuration: markers
    # ------------------------------------------------------------------

    def marker(
        self,
        *,
        alpha: float | None = None,
        edge_color: str | None = None,
        bar_width: float | None = None,
    ) -> BarplotVizBuilder:
        """Configure bar visual properties. Chainable.

        Args:
            alpha: Opacity (0–1).
            edge_color: Bar border color. ``None`` means no border.
            bar_width: Width fraction of available slot, 0–1 (default 0.8).
        """
        self._marker_patch(alpha=alpha, edge_color=edge_color, bar_width=bar_width)
        return self

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        sequence_or_pool: SequencePool | Sequence,
        *,
        entity_feature: str,
        facet_by: str | None = None,
    ) -> pl.DataFrame:
        """Aggregate sequence data into a barplot-ready DataFrame (see ``data.py``).

        Raises:
            TypeError: If *entity_feature* is not a categorical feature.
            UnsupportedShowAsError: If ``show_as="duration"`` on an event pool.
            IncompatibleDisplayUnitError: If *display_unit* is inconsistent with the
                pool's time representation.
            ValueError: If :attr:`~BaseSequenceVizBuilder.MAX_CATEGORY` is exceeded
                and ``allow_large=False``.
        """
        if not sequence_or_pool.metadata.is_categorical_feature(entity_feature):
            raise TypeError(
                f"'{entity_feature}' is not a categorical feature. "
                "Barplot requires a Categorical or Enum feature."
            )
        feature = entity_feature
        show_as = self.settings.aesthetics.show_as

        if (
            show_as == "duration"
            and sequence_or_pool.get_registration_name() == "event"
        ):
            raise UnsupportedShowAsError("duration", type(sequence_or_pool).__name__)

        if show_as == "duration":
            ti = sequence_or_pool.metadata.time_index
            display_unit = self.settings.aesthetics.display_unit
            if ti.is_datetime and display_unit is None:
                raise IncompatibleDisplayUnitError(None, is_datetime=True)
            if not ti.is_datetime and display_unit is not None:
                raise IncompatibleDisplayUnitError(display_unit, is_datetime=False)

        # Build the feature list; include facet_by for entity facets
        is_static_facet = self.settings.facet.is_static
        features: list[str] = [feature]
        if facet_by and not is_static_facet:
            features.append(facet_by)

        # Fetch sequence data as LazyFrame
        lf = sequence_or_pool._temporal_data_lf(features=features)

        # Rename ID column (needed for static-facet join; harmless otherwise)
        id_col = sequence_or_pool.settings.id_column
        lf = rename_id_column(lf, id_col)

        # Handle null time index for duration mode (uses start/end columns)
        if show_as == "duration":
            time_cols = sequence_or_pool.settings.get_time_columns()
            lf = rename_time_index_columns(lf, time_cols)
            lf = handle_null_time_index(lf, self.settings.null_handling.na_time_index)

        # Resolve label column
        lf = resolve_label(lf, feature)
        label_col = "__LABEL__"

        lf = handle_null_labels(lf, self.settings.null_handling.na_label)

        # Inject __FACET__ column
        if facet_by:
            lf = self._inject_facet_column(lf, sequence_or_pool, id_col="__ID__")

        # Aggregate
        facet_col = "__FACET__" if facet_by else None
        lf = self._aggregate(lf, label_col, show_as, facet_col=facet_col)
        lf = apply_sort(lf, self.settings.aesthetics.sort)

        df = lf.collect()

        # Cast labels to string
        df = df.with_columns(pl.col("__LABEL__").cast(pl.Utf8))

        # Always assign colors (defaults to tab10 when no spec is provided)
        color_map = self._build_color_map(
            df["__LABEL__"].unique().to_list(), self.settings.colors
        )
        df = df.with_columns(pl.col("__LABEL__").replace(color_map).alias("__COLOR__"))

        return df

    def _aggregate(
        self,
        lf: pl.LazyFrame,
        label_col: str,
        show_as: str,
        *,
        facet_col: str | None = None,
    ) -> pl.LazyFrame:
        """Aggregate *lf* according to *show_as* and return a ``__VALUE__`` LazyFrame."""
        if show_as == "count":
            return aggregate_count(lf, label_col, facet_col=facet_col)

        if show_as == "rate":
            return aggregate_rate(lf, label_col, facet_col=facet_col)

        # DURATION (Interval or State pools only)
        return aggregate_duration(
            lf,
            label_col,
            "__START__",
            "__END__",
            display_unit=self.settings.aesthetics.display_unit,
            facet_col=facet_col,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, ax: Any, data: pl.DataFrame) -> None:
        if self.settings.aesthetics.orientation == "horizontal":
            self._render_horizontal(ax, data)
        else:
            self._render_vertical(ax, data)

    def _render_vertical(self, ax: Any, data: pl.DataFrame) -> None:
        marker = self.settings.marker
        labels = data["__LABEL__"].to_list()
        values = data["__VALUE__"].to_list()
        x = np.arange(len(labels))
        colors = data["__COLOR__"].to_list()
        color_kwarg = {"color": colors}

        ax.bar(
            x,
            values,
            width=marker.bar_width,
            alpha=marker.alpha,
            edgecolor=marker.edge_color,
            **color_kwarg,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    def _render_horizontal(self, ax: Any, data: pl.DataFrame) -> None:
        marker = self.settings.marker
        labels = data["__LABEL__"].to_list()
        values = data["__VALUE__"].to_list()
        y = np.arange(len(labels))
        colors = data["__COLOR__"].to_list()
        color_kwarg = {"color": colors}
        ax.barh(
            y,
            values,
            height=marker.bar_width,
            alpha=marker.alpha,
            edgecolor=marker.edge_color,
            **color_kwarg,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
