#!/usr/bin/env python3
"""
HierarchicalClusterer: agglomerative hierarchical clustering via sklearn.
"""

from __future__ import annotations

from pydantic import Field
from sklearn.cluster import AgglomerativeClustering

from tanat_utils import settings_dataclass

from ..base import Clusterer
from ...metric.sequence.base import SequenceMetric
from ...metric.trajectory.base import TrajectoryMetric

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@settings_dataclass
class HierarchicalSettings:
    """Settings for :class:`HierarchicalClusterer`.

    Args:
        metric:             Metric name or instance.
        n_clusters:         Target number of clusters (ignored when
                            *distance_threshold* is set).
        distance_threshold: Cut-off distance for dendrogram trimming.
        linkage:            ``"complete"``, ``"average"``, ``"single"``,
                            or ``"ward"``.
        cluster_column:     Static-feature column injected after ``fit()``.
    """

    metric: str | SequenceMetric | TrajectoryMetric = "linearpairwise"
    n_clusters: int = Field(default=2, gt=0)
    distance_threshold: float | None = Field(default=None, ge=0)
    linkage: str = "complete"
    cluster_column: str = "__HCLUSTERS__"


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------


class HierarchicalClusterer(Clusterer, register_name="hierarchical"):
    """Agglomerative hierarchical clustering (sklearn ``AgglomerativeClustering``).

    Consumes a precomputed :class:`~tanat.metric.DistanceMatrix` with
    ``metric="precomputed"``.

    Example::

        clusterer = HierarchicalClusterer(metric="linearpairwise", n_clusters=5)
        clusterer.fit(pool)
        clusterer.clusters    # list[Cluster]
    """

    SETTINGS_CLASS = HierarchicalSettings

    def __init__(
        self,
        metric: str | SequenceMetric | TrajectoryMetric = "linearpairwise",
        n_clusters: int = 2,
        distance_threshold: float | None = None,
        linkage: str = "complete",
        cluster_column: str = "__HCLUSTERS__",
    ) -> None:
        settings = HierarchicalSettings(
            metric=metric,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage=linkage,
            cluster_column=cluster_column,
        )
        Clusterer.__init__(self, settings=settings)

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def _fit_impl(self, pool, metric, settings) -> tuple[list, list]:
        """Compute distance matrix then run AgglomerativeClustering.

        Returns:
            ``(labels, item_ids)``
        """
        self._display_step(1, 2, "Computing distance matrix")
        with self._nested_display():
            dist_matrix = metric.compute_matrix(pool)

        self._display_step(2, 2, f"Clustering ({type(self).__name__})")
        n_clusters = (
            None if settings.distance_threshold is not None else settings.n_clusters
        )

        model = AgglomerativeClustering(
            metric="precomputed",
            n_clusters=n_clusters,
            linkage=settings.linkage,
            distance_threshold=settings.distance_threshold,
        )
        model.fit(dist_matrix.to_numpy())
        return model.labels_.tolist(), dist_matrix.ids
