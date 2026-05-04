#!/usr/bin/env python3
"""
TimelineVizBuilder: timeline visualizer for a SequencePool or individual Sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from ...base.builder import BaseSequenceVizBuilder
from ...base.utils import (
    rename_id_column,
    resolve_label,
    handle_null_time_index,
    handle_null_labels,
    shift_time_to_relative,
    resolve_display_unit,
    UNIT_LABELS,
)
from .data import (
    assign_y_positions,
    build_y_tick_map,
    rename_time_index_columns,
)
from .settings import TimelineSettings

if TYPE_CHECKING:
    from ....sequence.base.pool import SequencePool
    from ....sequence.base.sequence import Sequence


class TimelineVizBuilder(BaseSequenceVizBuilder, register_name="timeline"):
    """Builds timeline charts from a SequencePool or an individual Sequence.

    Typical usage via :class:`~tanat.visualization.sequence.core.SequenceVisualizer`::

        SequenceVisualizer.timeline() \\
            .title("Status over time") \\
            .draw(pool, entity_feature="status") \\
            .show()
    """

    SETTINGS_CLASS = TimelineSettings
    MAX_MARKERS: int = 1000
    # In group_by='id' mode each unique ID gets its own row: beyond ~30 rows the
    # y-axis labels start overlapping at the default figsize=(10, 5). Raise this
    # per-instance (builder.MAX_IDS = 80) or set allow_large=True.
    MAX_IDS: int = 30

    def __init__(
        self, settings: Any | None = None, *, allow_large: bool = False
    ) -> None:
        super().__init__(settings=settings, allow_large=allow_large)
        self._y_tick_map: dict[int, str] = {}

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
        autofmt_xdate: bool | None = None,
    ) -> TimelineVizBuilder:
        """Configure the time (x) axis. Chainable.

        Args:
            show: Hide the axis entirely when ``False``.
            label: Axis label text.
            rotation: Tick label rotation in degrees.
            limit_min: Left bound (zooms into a time window).
            limit_max: Right bound (zooms into a time window).
            autofmt_xdate: Auto-rotate date tick labels (useful for dense datetime scales).
        """
        self._axis_patch(
            "x_axis",
            show=show,
            label=label,
            rotation=rotation,
            limit_min=limit_min,
            limit_max=limit_max,
            autofmt_xdate=autofmt_xdate,
        )
        return self

    def y_axis(
        self,
        *,
        show: bool | None = None,
        label: str | None = None,
    ) -> TimelineVizBuilder:
        """Configure the sequence / category (y) axis. Chainable.

        ``rotation`` and ``limit_min``/``limit_max`` are intentionally not
        exposed: y-positions are integer ranks managed internally, so rotating
        string labels or constraining numeric limits is not meaningful here.

        Args:
            show: Hide the axis entirely when ``False`` (useful when y-axis labels
                are too dense to read).
            label: Axis label text.
        """
        self._axis_patch("y_axis", show=show, label=label)
        return self

    # ------------------------------------------------------------------
    # Chainable configuration: markers
    # ------------------------------------------------------------------

    def marker(
        self,
        *,
        alpha: float | None = None,
        edge_color: str | None = None,
        size: float | None = None,
        bar_height: float | None = None,
        shape: str | None = None,
    ) -> TimelineVizBuilder:
        """Configure marker visual properties. Chainable.

        Args:
            alpha: Opacity (0–1).
            edge_color: Marker border color. ``None`` means no border.
            size: Scatter point size (event sequence only).
            bar_height: Height fraction of each row slot, 0–1 (interval/state pools only).
            shape: Matplotlib marker string, e.g. ``"o"``, ``"s"``, ``"^"`` (event sequence only).
        """
        self._marker_patch(
            alpha=alpha,
            edge_color=edge_color,
            size=size,
            bar_height=bar_height,
            shape=shape,
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
        facet_by: str | None = None,
    ) -> pl.DataFrame:
        """Orchestrate data transformations for the timeline (see ``data.py``).

        Raises:
            TypeError: If *entity_feature* is not a categorical feature.
            ValueError: If safety guards are exceeded and ``allow_large=False``.
        """
        if not sequence_or_pool.metadata.is_categorical_feature(entity_feature):
            raise TypeError(
                f"'{entity_feature}' is not a categorical feature. "
                "Timeline requires a Categorical or Enum feature."
            )

        allow_large = self.allow_large

        id_col = sequence_or_pool.settings.id_column
        time_cols = sequence_or_pool.settings.get_time_columns()

        # Include facet_by in features for entity (non-static) facets
        is_static_facet = self.settings.facet.is_static
        features: list[str] = [entity_feature]
        if facet_by and not is_static_facet:
            features.append(facet_by)

        lf = sequence_or_pool._temporal_data_lf(features=features)
        lf = rename_id_column(lf, id_col)
        lf = rename_time_index_columns(lf, time_cols)
        lf = handle_null_time_index(lf, self.settings.null_handling.na_time_index)
        lf = resolve_label(lf, entity_feature)
        lf = handle_null_labels(lf, self.settings.null_handling.na_label)

        # Relative time: subtract per-ID T0 from temporal columns.
        if self.settings.aesthetics.time_mode == "relative":
            is_dt = sequence_or_pool.metadata.time_index.is_datetime
            resolved_unit = resolve_display_unit(
                self.settings.aesthetics.display_unit,
                is_datetime=is_dt,
            )
            x_unit_label = UNIT_LABELS[resolved_unit] if is_dt else None
            self._default_x_label = (
                f"{x_unit_label} from T0" if x_unit_label else "Time from T0"
            )
            end_col = "__END__" if "__END__" in lf.collect_schema().names() else None
            lf = shift_time_to_relative(
                lf, sequence_or_pool, "__TIME__", end_col, display_unit=resolved_unit
            )

        # Inject __FACET__ column (propagates naturally — no explicit select drops it)
        if facet_by:
            lf = self._inject_facet_column(lf, sequence_or_pool, id_col="__ID__")

        lf = assign_y_positions(lf, mode=self.settings.aesthetics.group_by)

        df = lf.collect()
        df = df.with_columns(pl.col("__LABEL__").cast(pl.Utf8))

        # Safety guard: total markers
        n = len(df)
        if n > self.MAX_MARKERS and not allow_large:
            raise ValueError(
                f"Timeline would render {n:,} markers, which exceeds "
                f"MAX_MARKERS={self.MAX_MARKERS:,}. "
                "Reduce the input data (subset, filter, or split), narrow the time window, "
                "or pass allow_large=True to bypass this guard."
            )

        # Safety guard: id mode row count (y-axis readability)
        if self.settings.aesthetics.group_by == "id" and not allow_large:
            n_ids = df["__ID__"].n_unique()
            if n_ids > self.MAX_IDS:
                raise ValueError(
                    f"group_by='id' would draw {n_ids} y-axis rows, which exceeds "
                    f"MAX_IDS={self.MAX_IDS}. "
                    "Reduce the input data (subset or filter), switch to "
                    "group_by='category', increase builder.MAX_IDS, "
                    "or pass allow_large=True to bypass this guard."
                )

        # Always assign colors (defaults to tab10 when no spec is provided)
        color_map = self._build_color_map(
            df["__LABEL__"].unique().to_list(), self.settings.colors
        )
        df = df.with_columns(pl.col("__LABEL__").replace(color_map).alias("__COLOR__"))

        # Build y-tick map for use in _apply_styling
        self._y_tick_map = build_y_tick_map(df, self.settings.aesthetics.group_by)

        return df

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _update_per_facet_state(self, df_i: pl.DataFrame) -> pl.DataFrame:
        """Recompute y-positions and tick map for the facet slice."""
        mode = self.settings.aesthetics.group_by
        rank_col = "__ID__" if mode == "id" else "__LABEL__"
        df_i = df_i.with_columns(
            pl.col(rank_col).rank("dense").sub(1).cast(pl.Int32).alias("__Y_POSITION__")
        )
        self._y_tick_map = build_y_tick_map(df_i, mode)
        return df_i

    def _render(self, ax: Any, data: pl.DataFrame) -> None:
        """Dispatch to interval or event renderer."""
        if "__END__" in data.columns:
            self._render_intervals(ax, data)
        else:
            self._render_events(ax, data)

    def _render_intervals(self, ax: Any, data: pl.DataFrame) -> None:
        """Draw horizontal bars (``ax.barh``) for interval/state pools."""
        marker = self.settings.marker
        labels = data["__LABEL__"].unique().to_list()

        for label in labels:
            group = data.filter(pl.col("__LABEL__") == label)
            y_positions = group["__Y_POSITION__"].to_list()
            starts = group["__TIME__"].to_list()
            ends = group["__END__"].to_list()
            widths = [e - s for s, e in zip(starts, ends)]

            color_kwarg: dict[str, Any] = {"color": group["__COLOR__"][0]}

            ax.barh(
                y_positions,
                widths,
                left=starts,
                height=marker.bar_height,
                alpha=marker.alpha,
                edgecolor=marker.edge_color,
                label=label,
                **color_kwarg,
            )

    def _render_events(self, ax: Any, data: pl.DataFrame) -> None:
        """Draw scatter points for event pools."""
        marker = self.settings.marker
        labels = data["__LABEL__"].unique().to_list()

        for label in labels:
            group = data.filter(pl.col("__LABEL__") == label)
            x_values = group["__TIME__"].to_list()
            y_values = group["__Y_POSITION__"].to_list()

            color_kwarg: dict[str, Any] = {"color": group["__COLOR__"][0]}

            ax.scatter(
                x_values,
                y_values,
                s=marker.size,
                alpha=marker.alpha,
                edgecolors=marker.edge_color,
                marker=marker.shape,
                label=label,
                **color_kwarg,
            )

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def _set_custom_ticks(self, ax: Any) -> None:
        """Install y-tick labels from the tick map.

        Rotation is applied later by the base via tick_params; we don't pass
        it here so the base remains the single source of truth.
        """
        if self._y_tick_map:
            ax.set_yticks(list(self._y_tick_map.keys()))
            ax.set_yticklabels(list(self._y_tick_map.values()))
