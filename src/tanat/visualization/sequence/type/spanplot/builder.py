#!/usr/bin/env python3
"""
SpanplotVizBuilder: duration distribution visualizer for a SequencePool or Sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ...base.builder import BaseSequenceVizBuilder
from ...base.utils import resolve_label, drop_null_labels, rename_id_column
from ...base.exceptions import (
    IncompatibleDisplayUnitError,
    UnsupportedSequenceTypeError,
)
from .data import (
    extract_durations,
    rename_temporal_columns,
    sort_ids,
    sort_labels,
)
from .settings import SpanplotSettings

if TYPE_CHECKING:
    from tanat.sequence.base.pool import SequencePool
    from tanat.sequence.base.sequence import Sequence


class SpanplotVizBuilder(BaseSequenceVizBuilder, register_name="spanplot"):
    """Builds duration distribution charts from a SequencePool or an individual Sequence.

    Each point in the resulting chart represents one segment (interval or state
    occurrence). Distributions can be grouped by category label or by sequence ID.

    Only compatible with ``state`` and ``interval`` sequence types: duration is
    undefined for events.

    Typical usage via :class:`~tanat.visualization.sequence.core.SequenceVisualizer`::

        SequenceVisualizer.spanplot(kind="violin", display_unit="hours") \\
            .title("ICU stay durations by status") \\
            .colors("Set2") \\
            .draw(pool, entity_feature="status") \\
            .show()
    """

    SETTINGS_CLASS = SpanplotSettings

    # Sequence types that expose a meaningful duration concept.
    # Override in a subclass to support different type sets.
    _COMPATIBLE_TYPES: frozenset[str] = frozenset({"state", "interval"})

    # Maximum number of distinct IDs rendered in group_by="id" mode.
    # Beyond this the chart becomes unreadable. Override per-instance or pass
    # allow_large=True at the factory.
    MAX_IDS: int = 30

    def __init__(
        self, settings: Any | None = None, *, allow_large: bool = False
    ) -> None:
        super().__init__(settings=settings, allow_large=allow_large)
        self._label_order: list[str] = []
        self._id_order: list[str] = []

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
    ) -> SpanplotVizBuilder:
        """Configure the x (horizontal) axis. Chainable.

        Role depends on orientation:

        - ``"vertical"`` (default): group labels on x, durations on y.
        - ``"horizontal"``: durations on x, group labels on y. Use
          ``limit_min``/``limit_max`` to clip outliers.

        Args:
            show: Hide the axis entirely when ``False``.
            label: Axis label text.
            rotation: Tick label rotation in degrees.
            limit_min: Minimum x value (meaningful in horizontal orientation).
            limit_max: Maximum x value (meaningful in horizontal orientation).
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
    ) -> SpanplotVizBuilder:
        """Configure the y (vertical) axis. Chainable.

        Role depends on orientation:

        - ``"vertical"`` (default): durations on y, group labels on x. Use
          ``limit_min``/``limit_max`` to clip outliers.
        - ``"horizontal"``: group labels on y, durations on x.

        Args:
            show: Hide the axis entirely when ``False``.
            label: Axis label text (e.g. ``"Duration (hours)"``)..
            rotation: Tick label rotation in degrees.
            limit_min: Minimum y value (meaningful in vertical orientation).
            limit_max: Maximum y value (meaningful in vertical orientation).
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
        point_size: float | None = None,
        line_width: float | None = None,
    ) -> SpanplotVizBuilder:
        """Configure marker visual properties. Chainable.

        Args:
            alpha: Opacity (0–1).
            edge_color: Marker/box border color. ``None`` means no border.
            point_size: Scatter point diameter in points (strip kind only).
            line_width: Box or violin outline thickness.
        """
        self._marker_patch(
            alpha=alpha,
            edge_color=edge_color,
            point_size=point_size,
            line_width=line_width,
        )
        return self

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        sequence_or_pool: SequencePool | Sequence,
        *,
        entity_feature: str,
        drop_na: bool,
    ) -> pl.DataFrame:
        """Orchestrate data transformations for the spanplot (see ``data.py``).

        Raises:
            UnsupportedSequenceTypeError: If pool type is not state or interval.
            IncompatibleDisplayUnitError: If display_unit/temporal type mismatch.
        """
        pool_type = sequence_or_pool.get_registration_name()
        if pool_type not in self._COMPATIBLE_TYPES:
            raise UnsupportedSequenceTypeError(
                pool_type, compatible_types=self._COMPATIBLE_TYPES
            )

        # Validate display_unit vs temporal type
        temporal = sequence_or_pool.metadata.temporal
        display_unit = self.settings.aesthetics.display_unit
        if temporal.is_datetime and display_unit is None:
            raise IncompatibleDisplayUnitError(None, is_datetime=True)
        if not temporal.is_datetime and display_unit is not None:
            raise IncompatibleDisplayUnitError(display_unit, is_datetime=False)

        id_col = sequence_or_pool.settings.id_column
        temporal_cols = sequence_or_pool.settings.get_temporal_columns()

        lf = sequence_or_pool._sequence_data_lf(features=[entity_feature])
        lf = rename_id_column(lf, id_col)
        lf = rename_temporal_columns(lf, temporal_cols)
        lf = resolve_label(lf, entity_feature)

        if drop_na:
            lf = drop_null_labels(lf)

        lf = extract_durations(lf, display_unit)

        df = lf.collect()
        df = df.with_columns(pl.col("__LABEL__").cast(pl.Utf8).fill_null("null"))

        # Mode-sensitive size guards
        group_by = self.settings.aesthetics.group_by
        if group_by == "category":
            n_cat = df["__LABEL__"].n_unique()
            if n_cat > self.MAX_CATEGORY and not self.allow_large:
                raise ValueError(
                    f"'{entity_feature}' has {n_cat} unique values, which exceeds "
                    f"MAX_CATEGORY={self.MAX_CATEGORY}. "
                    "Reduce the feature cardinality, increase builder.MAX_CATEGORY, "
                    "or pass allow_large=True to bypass this guard."
                )
        else:  # "id"
            n_ids = df["__ID__"].n_unique()
            if n_ids > self.MAX_IDS and not self.allow_large:
                raise ValueError(
                    f"Input has {n_ids} unique sequence IDs, which exceeds "
                    f"MAX_IDS={self.MAX_IDS} for group_by='id'. "
                    "Reduce the input data, increase builder.MAX_IDS, "
                    "or pass allow_large=True to bypass this guard."
                )

        # Build sorted tick orders (stored for _render / _apply_styling)
        sort = self.settings.aesthetics.sort
        self._label_order = sort_labels(df, sort)
        self._id_order = sort_ids(df, sort)

        # Always assign colors (defaults to tab10 when no spec is provided)
        if group_by == "id":
            id_keys = df["__ID__"].cast(pl.Utf8).unique().to_list()
            color_map = self._build_color_map(id_keys, self.settings.colors)
            df = df.with_columns(
                pl.col("__ID__").cast(pl.Utf8).replace(color_map).alias("__COLOR__")
            )
        else:  # "category": color by label
            label_keys = df["__LABEL__"].unique().to_list()
            color_map = self._build_color_map(label_keys, self.settings.colors)
            df = df.with_columns(
                pl.col("__LABEL__").replace(color_map).alias("__COLOR__")
            )

        return df

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, ax: Any, data: pl.DataFrame) -> None:
        """Dispatch to box / violin / strip renderer."""
        kind = self.settings.aesthetics.kind

        if kind == "box":
            self._render_box(ax, data)
        elif kind == "violin":
            self._render_violin(ax, data)
        else:
            self._render_strip(ax, data)

    def _render_box(self, ax: Any, data: pl.DataFrame) -> None:
        """Draw boxplots — one box per group."""
        marker = self.settings.marker
        group_by = self.settings.aesthetics.group_by
        vert = self.settings.aesthetics.orientation == "vertical"
        keys, group_col, df = self._resolve_groups(data, group_by)

        groups: list[list[float]] = []
        colors: list[str] = []

        for key in keys:
            grp = df.filter(pl.col(group_col) == key)
            groups.append(grp["__DURATION__"].to_list())
            colors.append(grp["__COLOR__"][0])

        positions = list(range(len(keys)))
        bplot = ax.boxplot(
            groups,
            positions=positions,
            patch_artist=True,
            widths=0.5,
            vert=vert,
            medianprops={"linewidth": marker.line_width, "color": "black"},
            whiskerprops={"linewidth": marker.line_width},
            capprops={"linewidth": marker.line_width},
            boxprops={"linewidth": marker.line_width},
            flierprops={"markersize": marker.point_size},
        )
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(marker.alpha)
            if marker.edge_color:
                patch.set_edgecolor(marker.edge_color)

    def _render_violin(self, ax: Any, data: pl.DataFrame) -> None:
        """Draw violin plots — one violin per group."""
        marker = self.settings.marker
        group_by = self.settings.aesthetics.group_by
        vert = self.settings.aesthetics.orientation == "vertical"
        keys, group_col, df = self._resolve_groups(data, group_by)

        groups: list[list[float]] = []
        colors: list[str] = []

        for key in keys:
            grp = df.filter(pl.col(group_col) == key)
            dur = grp["__DURATION__"].to_list()
            # violinplot needs at least 2 distinct values
            if len(dur) < 2:
                dur = dur * 2
            groups.append(dur)
            colors.append(grp["__COLOR__"][0])

        positions = list(range(len(keys)))
        vplot = ax.violinplot(
            groups,
            positions=positions,
            vert=vert,
            showmedians=True,
            showextrema=True,
        )
        for body, color in zip(vplot["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(marker.alpha)
            if marker.edge_color:
                body.set_edgecolor(marker.edge_color)
                body.set_linewidth(marker.line_width)
        # Style median/whisker lines
        for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part_name in vplot:
                vplot[part_name].set_linewidth(marker.line_width)

    def _render_strip(self, ax: Any, data: pl.DataFrame) -> None:
        """Draw strip plots with jitter — one column per group."""
        marker = self.settings.marker
        group_by = self.settings.aesthetics.group_by
        horizontal = self.settings.aesthetics.orientation == "horizontal"
        keys, group_col, df = self._resolve_groups(data, group_by)

        rng = np.random.default_rng(42)  # deterministic jitter

        for i, key in enumerate(keys):
            grp = df.filter(pl.col(group_col) == key)
            dur_values = grp["__DURATION__"].to_list()
            n = len(dur_values)
            jitter = (i + rng.uniform(-0.15, 0.15, n)).tolist()

            color_kwarg: dict[str, Any] = {"color": grp["__COLOR__"][0]}

            # horizontal: durations on x, jitter on y
            scatter_x, scatter_y = (
                (dur_values, jitter) if horizontal else (jitter, dur_values)
            )
            ax.scatter(
                scatter_x,
                scatter_y,
                s=marker.point_size**2,
                alpha=marker.alpha,
                edgecolors=marker.edge_color,
                linewidths=0.5,
                **color_kwarg,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_groups(
        self,
        data: pl.DataFrame,
        group_by: str,
    ) -> tuple[list[str], str, pl.DataFrame]:
        """Return (ordered_keys, group_col_name, df_with_group_col).

        For ``group_by="id"`` the group column is ``__ID_STR__`` (ID cast to
        string) rather than ``__ID__`` to guarantee string comparison works
        regardless of the original ID dtype.
        """
        if group_by == "id":
            df = data.with_columns(pl.col("__ID__").cast(pl.Utf8).alias("__ID_STR__"))
            return self._id_order, "__ID_STR__", df
        # "category": group by label
        return self._label_order, "__LABEL__", data

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def _apply_styling(self, ax: Any) -> None:
        """Apply common styling then set group-axis tick labels from ordered keys."""
        super()._apply_styling(ax)

        group_by = self.settings.aesthetics.group_by
        horizontal = self.settings.aesthetics.orientation == "horizontal"
        tick_labels = self._id_order if group_by == "id" else self._label_order

        if not tick_labels:
            return

        positions = list(range(len(tick_labels)))
        if horizontal:
            ax.set_yticks(positions)
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_xticks(positions)
            rotation = 45 if len(tick_labels) > 10 else 0
            ax.set_xticklabels(
                tick_labels,
                rotation=rotation,
                ha="right" if rotation else "center",
            )
