#!/usr/bin/env python3
"""
DistributionVizBuilder: state-occupancy distribution visualizer for a SequencePool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ...base.builder import BaseSequenceVizBuilder
from ...base.utils import (
    drop_null_labels,
    rename_id_column,
    resolve_label,
    shift_time_to_relative,
    resolve_display_unit,
    UNIT_LABELS,
)
from ...base.exceptions import UnsupportedSequenceTypeError
from .data import aggregate_distribution, assign_time_bins, rename_time_index_columns
from .settings import DistributionSettings

if TYPE_CHECKING:
    from ....sequence.base.pool import SequencePool
    from ....sequence.base.sequence import Sequence


class DistributionVizBuilder(BaseSequenceVizBuilder, register_name="distribution"):
    """Builds state-occupancy distribution charts from a StateSequencePool.

    Each time bin on the x-axis counts (or normalises) how many sequences are
    in each state at that point in time, using occupancy-based binning: a
    segment contributes to every bin it overlaps, not just the one containing
    its start.

    Only compatible with ``state`` sequence types.

    Typical usage via :class:`~tanat.visualization.sequence.core.SequenceVisualizer`::

        SequenceVisualizer.distribution(mode="percentage", bin_size="1d") \\
            .title("State distribution over time") \\
            .colors("Set2") \\
            .draw(pool, entity_feature="status") \\
            .show()
    """

    SETTINGS_CLASS = DistributionSettings

    _COMPATIBLE_TYPES: frozenset[str] = frozenset({"state"})

    def __init__(
        self, settings: Any | None = None, *, allow_large: bool = False
    ) -> None:
        super().__init__(settings=settings, allow_large=allow_large)

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
    ) -> DistributionVizBuilder:
        """Configure the time (x) axis. Chainable.

        Args:
            show: Hide the axis entirely when ``False``.
            label: Axis label text.
            rotation: Tick label rotation in degrees.
            limit_min: Left bound (zooms into a time window).
            limit_max: Right bound (zooms into a time window).
            autofmt_xdate: Auto-rotate date tick labels.
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
        rotation: int | None = None,
        limit_min: float | None = None,
        limit_max: float | None = None,
    ) -> DistributionVizBuilder:
        """Configure the value (y) axis. Chainable.

        Args:
            show: Hide the axis entirely when ``False``.
            label: Axis label text.
            rotation: Tick label rotation in degrees.
            limit_min: Minimum y value.
            limit_max: Maximum y value.
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
        line_width: float | None = None,
    ) -> DistributionVizBuilder:
        """Configure fill visual properties. Chainable.

        Args:
            alpha: Opacity of the filled areas (0-1).
            line_width: Width of the area boundary line.
        """
        self._marker_patch(alpha=alpha, line_width=line_width)
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
        facet_by: str | None = None,
    ) -> pl.DataFrame:
        """Orchestrate data transformations for the distribution chart.

        Steps:

        1. Guard: pool type must be ``"state"``.
        2. Guard: *bin_size* type must match the pool temporal type.
        3. Rename ID and time columns, resolve label, optionally drop nulls.
        4. Relative time shift (when ``time_mode="relative"``).
        5. Inject ``__FACET__`` when *facet_by* is set.
        6. Assign time bins (occupancy-based, one collect for global bounds).
        7. Aggregate by ``[__TIME_BIN__, __LABEL__]`` (plus ``__FACET__``)
           and compute ``__VALUE__``.
        8. Collect; cast ``__LABEL__`` to string.
        9. Build per-label color map if ``colors`` is set.

        Args:
            sequence_or_pool: Input pool (must be ``StateSequencePool``).
            entity_feature: Categorical feature column to use as the state label.
            drop_na: Drop rows with a null entity feature value when ``True``.

        Returns:
            Collected DataFrame with columns
            ``__TIME_BIN__``, ``__LABEL__``, ``__VALUE__``.

        Raises:
            UnsupportedSequenceTypeError: If pool type is not ``"state"``.
            TypeError: If *bin_size* type does not match the pool temporal type.
        """
        pool_type = sequence_or_pool.get_registration_name()
        if pool_type not in self._COMPATIBLE_TYPES:
            raise UnsupportedSequenceTypeError(
                pool_type, compatible_types=self._COMPATIBLE_TYPES
            )

        ti = sequence_or_pool.metadata.time_index
        is_datetime = ti.is_datetime
        bin_size = self.settings.aesthetics.bin_size

        # bin_size type guard (raises TypeError if incompatible)
        if is_datetime and not isinstance(bin_size, str):
            raise TypeError(
                f"bin_size must be a Polars duration string for datetime pools "
                f"(e.g. '1d', '12h', '1w'), got {type(bin_size).__name__!r}."
            )
        if not is_datetime and isinstance(bin_size, str):
            raise TypeError(
                f"bin_size must be int or float for numeric timestep pools, "
                f"got a string {bin_size!r}. Use a numeric value instead."
            )

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
        lf = resolve_label(lf, entity_feature)

        if drop_na:
            lf = drop_null_labels(lf)

        # Relative time: subtract per-ID T0 from the start and end columns.
        resolved_unit = None
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
            lf = shift_time_to_relative(
                lf, sequence_or_pool, "__START__", "__END__", display_unit=resolved_unit
            )

        # Inject __FACET__ column (cross-join in assign_time_bins preserves all cols)
        if facet_by:
            lf = self._inject_facet_column(lf, sequence_or_pool, id_col="__ID__")

        # Occupancy-based binning: 1 collect (2 scalars), then cross-join + filter
        lf = assign_time_bins(
            lf, bin_size, is_datetime=is_datetime, display_unit=resolved_unit
        )

        # Aggregate counts / proportion / percentage
        mode = self.settings.aesthetics.mode
        facet_col = "__FACET__" if facet_by else None
        lf = aggregate_distribution(lf, mode, facet_col=facet_col)

        df = lf.collect()
        df = df.with_columns(pl.col("__LABEL__").cast(pl.Utf8).fill_null("null"))

        # Category count guard
        n_cat = df["__LABEL__"].n_unique()
        if n_cat > self.MAX_CATEGORY and not self.allow_large:
            raise ValueError(
                f"'{entity_feature}' has {n_cat} unique values, which exceeds "
                f"MAX_CATEGORY={self.MAX_CATEGORY}. "
                "Reduce the feature cardinality, increase builder.MAX_CATEGORY, "
                "or pass allow_large=True to bypass this guard."
            )

        # Always assign colors (defaults to tab10 when no spec is provided)
        color_map = self._build_color_map(
            df["__LABEL__"].unique().to_list(), self.settings.colors
        )
        df = df.with_columns(pl.col("__LABEL__").replace(color_map).alias("__COLOR__"))

        return df

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, ax: Any, data: pl.DataFrame) -> None:
        """Dispatch to stacked or flat renderer."""
        if self.settings.aesthetics.stacked:
            self._render_stacked(ax, data)
        else:
            self._render_flat(ax, data)

    def _render_stacked(self, ax: Any, data: pl.DataFrame) -> None:
        """Draw a stacked area chart using ``ax.stackplot``."""
        alpha = self.settings.marker.alpha
        line_width = self.settings.marker.line_width

        labels, x_vals, matrix = self._pivot_data(data)
        color_lookup = dict(data.select(["__LABEL__", "__COLOR__"]).unique().rows())
        colors = [color_lookup.get(lbl) for lbl in labels]

        ax.stackplot(
            x_vals,
            matrix,
            labels=labels,
            colors=colors,
            alpha=alpha,
            linewidth=line_width,
        )

    def _render_flat(self, ax: Any, data: pl.DataFrame) -> None:
        """Draw overlapping transparent fills -- one per label."""
        alpha = self.settings.marker.alpha
        line_width = self.settings.marker.line_width

        color_lookup = dict(data.select(["__LABEL__", "__COLOR__"]).unique().rows())
        labels, x_vals, matrix = self._pivot_data(data)

        for i, lbl in enumerate(labels):
            color = color_lookup.get(lbl)
            y = matrix[i]
            ax.fill_between(
                x_vals,
                y,
                alpha=alpha,
                label=lbl,
                color=color,
                linewidth=line_width,
            )
            ax.plot(x_vals, y, color=color, linewidth=line_width)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pivot_data(
        data: pl.DataFrame,
    ) -> tuple[list[str], list, np.ndarray]:
        """Pivot aggregated data into (labels, x_values, matrix) for plotting.

        The returned *matrix* is shaped ``(n_labels, n_bins)`` -- the format
        expected by :func:`matplotlib.axes.Axes.stackplot`.

        Missing label-bin combinations are filled with 0.

        Returns:
            labels: Ordered list of unique label strings.
            x_vals: Ordered list of bin values (datetime or numeric).
            matrix: NumPy array of shape ``(n_labels, n_bins)``.
        """
        wide = (
            data.pivot(
                on="__LABEL__",
                index="__TIME_BIN__",
                values="__VALUE__",
                aggregate_function="first",
            )
            .sort("__TIME_BIN__")
            .fill_null(0.0)
        )

        x_vals = wide["__TIME_BIN__"].to_list()
        labels = sorted(c for c in wide.columns if c != "__TIME_BIN__")
        matrix = np.array([wide[lbl].to_numpy() for lbl in labels])

        return labels, x_vals, matrix
