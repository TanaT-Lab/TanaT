#!/usr/bin/env python3
"""
Clusterer ABC: base class for all clustering algorithms.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from tanat_utils import SettingsMixin, Registrable, DisplayMixin
from tanat_utils.pretty_format import format_header, format_section, format_kv

from ._cluster import Cluster
from ..metric.sequence.base import SequenceMetric
from ..metric.trajectory.base import TrajectoryMetric
from ..sequence.base.pool import SequencePool
from ..trajectory.pool import TrajectoryPool

if TYPE_CHECKING:
    from typing import Self


def _format_setting_value(value: object) -> str:
    """Format a settings field value for __str__ display."""
    if hasattr(value, "SETTINGS_CLASS"):
        return type(value).__name__
    return str(value)


class Clusterer(SettingsMixin, Registrable, DisplayMixin, ABC):
    """Abstract base class for all clustering algorithms."""

    _REGISTER: dict = {}
    _TYPE_SUBMODULE = "type"

    def __init__(self, settings=None) -> None:
        """Initialise with settings built from kwargs."""
        super().__init__(settings)
        self._clusters: list[Cluster] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def clusters(self) -> list[Cluster] | None:
        """Cluster results.  ``None`` before :meth:`fit`."""
        return self._clusters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        pool: SequencePool | TrajectoryPool,
        **kwargs,
    ) -> Self:
        """Fit the clustering model to *pool*.

        Keyword arguments matching settings fields override them for this
        call only (e.g. ``fit(pool, n_clusters=3)``).

        Returns:
            ``self`` for method chaining.
        """
        settings = self._resolve_settings(kwargs)

        self._validate_pool(pool)
        if len(pool) < 2:
            raise ValueError(
                f"Found {len(pool)} id(s) in pool while a minimum of 2 is "
                f"required for clustering."
            )

        self._display_header()

        metric = self._resolve_metric_for_pool(settings.metric, pool)
        labels, item_ids = self._fit_impl(pool, metric, settings)

        self._clusters = self._build_clusters(labels, item_ids)
        self._inject_labels(pool, item_ids, cluster_col=settings.cluster_column)

        self._display_footer(f"{len(item_ids)} items, {len(self._clusters)} clusters")
        return self

    def _resolve_settings(self, overrides: dict):
        """Return settings with *overrides* applied, without mutating *self*."""
        # TODO : reflexion about migrate to SettingsMixin and remove shadow dispatch
        if not overrides or self._settings is None:
            return self._settings
        settings_fields = set(self._settings.__dataclass_fields__.keys())
        valid = {k: v for k, v in overrides.items() if k in settings_fields}
        return dataclasses.replace(self._settings, **valid) if valid else self._settings

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit_impl(
        self,
        pool: SequencePool | TrajectoryPool,
        metric: SequenceMetric | TrajectoryMetric,
        settings,
    ) -> tuple[list, list]:
        """Algorithm-specific clustering logic.

        Returns:
            ``(labels, item_ids)``: cluster label per item and item identifiers.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_metric(
        self, pool: SequencePool | TrajectoryPool
    ) -> SequenceMetric | TrajectoryMetric:
        """Resolve ``settings.metric`` against *pool* type."""
        return self._resolve_metric_for_pool(self.settings.metric, pool)

    @staticmethod
    def _resolve_metric_for_pool(
        metric,
        pool: SequencePool | TrajectoryPool,
    ) -> SequenceMetric | TrajectoryMetric:
        """Resolve *metric* (name or instance) against *pool* type."""
        is_traj = isinstance(pool, TrajectoryPool)

        if isinstance(metric, str):
            if is_traj:
                return TrajectoryMetric.get_registered(metric)()
            return SequenceMetric.get_registered(metric)()

        return metric

    def _validate_pool(self, pool) -> None:
        """Raise :class:`TypeError` if *pool* is not a recognised pool type."""
        if not isinstance(pool, (SequencePool, TrajectoryPool)):
            raise TypeError(
                f"Expected SequencePool or TrajectoryPool, "
                f"got {type(pool).__name__}."
            )

    def _build_clusters(self, labels: list, item_ids: list) -> list[Cluster]:
        """Create :class:`Cluster` objects from label and id arrays."""
        groups: dict[int, list] = defaultdict(list)
        for item_id, label in zip(item_ids, labels):
            groups[int(label)].append(item_id)
        return [
            Cluster(cluster_id=label, items=items)
            for label, items in sorted(groups.items())
        ]

    def _inject_labels(self, pool, item_ids: list, *, cluster_col: str) -> None:
        """Write cluster labels to *pool* as a static feature."""
        id_to_label: dict = {}
        for cluster in self._clusters:
            for item in cluster.items:
                id_to_label[item] = cluster.id

        id_col = pool.settings.id_column
        df = pl.DataFrame(
            {
                id_col: item_ids,
                cluster_col: [id_to_label[item_id] for item_id in item_ids],
            }
        )
        pool.add_static_features(df, id_column=id_col, overwrite=True)

    # ------------------------------------------------------------------
    # Repr / Str
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = len(self._clusters) if self._clusters is not None else "not fitted"
        return f"{type(self).__name__}(clusters={fitted})"

    def __str__(self) -> str:
        cls = type(self).__name__

        # --- Settings section (always shown) ---
        settings_lines = []
        if self._settings is not None:
            for key, value in self._settings.__dict__.items():
                if key.startswith("_"):
                    continue
                display = _format_setting_value(value)
                settings_lines.append(format_kv(key, display))

        parts = [
            format_header(cls),
            "",
            format_section("Settings", settings_lines),
        ]

        # --- Results section (only after fit) ---
        if self._clusters is not None:
            sizes = [c.size for c in self._clusters]
            results_lines = [
                format_kv("Clusters", len(self._clusters)),
                format_kv("Avg size", f"{np.mean(sizes):.1f}"),
                format_kv("Min size", int(np.min(sizes))),
                format_kv("Max size", int(np.max(sizes))),
            ]
            parts += ["", format_section("Results", results_lines)]

            cluster_lines = [
                format_kv(f"#{c.id}", f"{c.size} items") for c in self._clusters[:10]
            ]
            if len(self._clusters) > 10:
                cluster_lines.append(f"... and {len(self._clusters) - 10} more")
            parts += ["", format_section("Clusters", cluster_lines)]
        else:
            parts += ["", format_section("Status", [format_kv("Fitted", "No")])]

        return "\n".join(parts)
