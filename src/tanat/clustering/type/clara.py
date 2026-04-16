#!/usr/bin/env python3
"""
CLARAClusterer: sampling-based PAM for large datasets.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from tanat_utils import settings_dataclass

from ..base import Clusterer
from ...metric.sequence.base import SequenceMetric
from ...metric.trajectory.base import TrajectoryMetric
from .pam import PAMClusterer, MedoidMixin

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@settings_dataclass
class CLARASettings:
    """Settings for :class:`CLARAClusterer`.

    Args:
        metric:          Metric name or instance.  Default: ``"linearpairwise"``.
        sampling_ratio:  Fraction of the pool used per PAM instance.
                         Must be in ``(0, 1]``.  Default: ``0.1``.
        nb_pam_instances: Number of PAM runs on independent random samples.
                          Default: ``5``.
        n_clusters:      Number of medoids per PAM run.  Default: ``2``.
        max_iter:        Maximum SWAP iterations per PAM run.  Default: ``50``.
        random_state:    Seed for the random number generator.
                         ``None`` → non-reproducible.
        cluster_column:  Static-feature column name injected after ``fit()``.
    """

    metric: str | SequenceMetric | TrajectoryMetric = "linearpairwise"
    sampling_ratio: float = Field(default=0.1, gt=0, le=1.0)
    nb_pam_instances: int = Field(default=5, gt=0)
    n_clusters: int = Field(default=2, gt=0)
    max_iter: int = Field(default=50, gt=0)
    random_state: int | None = None
    cluster_column: str = "__CLARA_CLUSTERS__"


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------


class CLARAClusterer(MedoidMixin, Clusterer, register_name="clara"):
    """CLARA (Clustering Large Applications), sampling-based PAM.

    Runs PAM on *nb_pam_instances* random sub-samples of the pool,
    evaluates each result on the full pool, and keeps the best medoids.

    Example::

        clara = CLARAClusterer(
            metric="linearpairwise",
            n_clusters=5,
            sampling_ratio=0.1,
            nb_pam_instances=5,
            random_state=42,
        )
        clara.fit(pool)
        clara.medoids   # best medoids across all PAM instances
    """

    SETTINGS_CLASS = CLARASettings

    def __init__(
        self,
        metric: str | SequenceMetric | TrajectoryMetric = "linearpairwise",
        sampling_ratio: float = 0.1,
        nb_pam_instances: int = 5,
        n_clusters: int = 2,
        max_iter: int = 50,
        random_state: int | None = None,
        cluster_column: str = "__CLARA_CLUSTERS__",
    ) -> None:
        settings = CLARASettings(
            metric=metric,
            sampling_ratio=sampling_ratio,
            nb_pam_instances=nb_pam_instances,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            cluster_column=cluster_column,
        )
        Clusterer.__init__(self, settings=settings)
        MedoidMixin.__init__(self)

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def _fit_impl(self, pool, metric, settings) -> tuple[list, list]:
        """Sample, run PAM per sample, keep best medoids.

        Returns:
            ``(labels, sorted_ids)``
        """
        nb_instances = settings.nb_pam_instances
        sorted_ids = sorted(pool.unique_ids)
        sample_size = max(
            settings.n_clusters,
            int(settings.sampling_ratio * len(sorted_ids)),
        )

        rng = np.random.default_rng(settings.random_state)

        optimal_medoids = None
        optimal_inertia = float("inf")
        best_labels = None

        for i in range(nb_instances):
            self._display_step(
                i + 1,
                nb_instances,
                f"PAM instance {i + 1}/{nb_instances} (sample: {sample_size})",
            )

            # Sample and run PAM on a sub-pool.
            sampled_ids = rng.choice(sorted_ids, size=sample_size, replace=False)
            sub_pool = pool.subset(sampled_ids.tolist())

            with self._nested_display():
                pam = PAMClusterer(
                    metric=metric,
                    n_clusters=settings.n_clusters,
                    max_iter=settings.max_iter,
                )
                pam.fit(sub_pool)

            # Evaluate medoid quality on the FULL pool: n×k distances (NOT n×n).
            medoid_pool = pool.subset(pam.medoids)
            dists = metric.compute_cross_matrix(pool, medoid_pool)  # (n, k) float32
            inertia = float(np.sum(np.min(dists, axis=1)))

            if inertia < optimal_inertia:
                optimal_medoids = pam.medoids
                optimal_inertia = inertia
                best_labels = np.argmin(dists, axis=1).tolist()

        self._medoids = optimal_medoids
        return best_labels, sorted_ids
