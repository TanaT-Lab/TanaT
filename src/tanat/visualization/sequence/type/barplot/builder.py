#!/usr/bin/env python3
"""
BarplotVizBuilder: bar chart visualizer for a SequencePool or individual Sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ...base.builder import BaseSequenceVizBuilder
from .exception import UnsupportedShowAsError, IncompatibleDisplayUnitError
from .data import (
    aggregate_count,
    aggregate_duration,
    aggregate_rate,
    apply_sort,
    drop_null_labels,
    resolve_label,
)
from .settings import BarplotSettings

if TYPE_CHECKING:
    from tanat.sequence.base.pool import SequencePool
    from tanat.sequence.base.sequence import Sequence


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
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        sequence_or_pool: SequencePool | Sequence,
        *,
        entity_feature: str,
        drop_na: bool,
    ) -> pl.DataFrame:
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
            temporal = sequence_or_pool.metadata.temporal
            display_unit = self.settings.aesthetics.display_unit
            if temporal.is_datetime and display_unit is None:
                raise IncompatibleDisplayUnitError(None, is_datetime=True)
            if not temporal.is_datetime and display_unit is not None:
                raise IncompatibleDisplayUnitError(display_unit, is_datetime=False)

        # Fetch sequence data as LazyFrame
        lf = sequence_or_pool._sequence_data_lf(features=[feature])

        # Resolve label column
        lf, label_col = resolve_label(lf, feature)

        if drop_na:
            lf = drop_null_labels(lf, label_col)

        # Aggregate
        lf = self._aggregate(lf, sequence_or_pool, label_col, show_as)
        lf = apply_sort(lf, self.settings.aesthetics.sort)

        df = lf.collect()

        # Cast labels to string and fill nulls so None becomes "null" and sort is stable
        df = df.with_columns(pl.col("__LABEL__").cast(pl.Utf8).fill_null("null"))

        # Assign colors via expression, only when a spec is provided
        if self.settings.colors is not None:
            color_map = self._build_color_map(df, self.settings.colors)
            df = df.with_columns(
                pl.col("__LABEL__").replace(color_map).alias("__COLOR__")
            )

        return df

    def _aggregate(
        self,
        lf: pl.LazyFrame,
        sequence_or_pool: SequencePool | Sequence,
        label_col: str,
        show_as: str,
    ) -> pl.LazyFrame:
        if show_as == "count":
            return aggregate_count(lf, label_col)

        if show_as == "rate":
            return aggregate_rate(lf, label_col)

        # DURATION (Interval or State pools only)
        temporal_cols = sequence_or_pool.settings.get_temporal_columns()
        start_col, end_col = temporal_cols[0], temporal_cols[1]
        return aggregate_duration(
            lf,
            label_col,
            start_col,
            end_col,
            display_unit=self.settings.aesthetics.display_unit,
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
        colors = data["__COLOR__"].to_list() if "__COLOR__" in data.columns else None
        color_kwarg = {"color": colors} if colors is not None else {}

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
        colors = data["__COLOR__"].to_list() if "__COLOR__" in data.columns else None
        color_kwarg = {"color": colors} if colors is not None else {}
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
