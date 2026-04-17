#!/usr/bin/env python3
"""
PAMClusterer: Partition Around Medoids clustering with Numba-optimized kernels.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from tanat_utils import settings_dataclass

from ..base import Clusterer
from ...metric.sequence.base import SequenceMetric
from ...metric.trajectory.base import TrajectoryMetric
from ._kernels import pam_build_optimized, pam_swap_optimized

# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class MedoidMixin:
    """Mixin for clusterers that expose medoids (representative objects)."""

    def __init__(self) -> None:
        self._medoids: list | None = None

    @property
    def medoids(self) -> list | None:
        """Medoid item IDs.  ``None`` before :meth:`fit`."""
        return self._medoids


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@settings_dataclass
class PAMSettings:
    """Settings for :class:`PAMClusterer`.

    Args:
        metric:         Metric name or instance.  Default: ``"linearpairwise"``.
        n_clusters:     Number of medoids to find.  Must be ``> 0``.
        max_iter:       Maximum number of SWAP iterations.  Must be ``> 0``.
        cluster_column: Static-feature column name injected after ``fit()``.
    """

    metric: str | SequenceMetric | TrajectoryMetric = "linearpairwise"
    n_clusters: int = Field(default=2, gt=0)
    max_iter: int = Field(default=50, gt=0)
    cluster_column: str = "__PAM_CLUSTERS__"


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------


class PAMClusterer(MedoidMixin, Clusterer, register_name="pam"):
    """Partition Around Medoids (PAM) clustering.

    Two-phase algorithm:

    1. **BUILD**: greedy initial medoid selection.
    2. **SWAP**: iterative improvement by swapping medoid/non-medoid pairs.

    Inner loops use Numba-compiled kernels for performance.

    Example::

        pam = PAMClusterer(metric="linearpairwise", n_clusters=3, max_iter=100)
        pam.fit(pool)
        pam.medoids   # list of representative item IDs
        pam.clusters  # list[Cluster]
    """

    SETTINGS_CLASS = PAMSettings

    def __init__(
        self,
        metric: str | SequenceMetric | TrajectoryMetric = "linearpairwise",
        n_clusters: int = 2,
        max_iter: int = 50,
        cluster_column: str = "__PAM_CLUSTERS__",
    ) -> None:
        settings = PAMSettings(
            metric=metric,
            n_clusters=n_clusters,
            max_iter=max_iter,
            cluster_column=cluster_column,
        )
        Clusterer.__init__(self, settings=settings)
        MedoidMixin.__init__(self)

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def _fit_impl(self, pool, metric) -> tuple[list, list]:
        """Compute distance matrix then run PAM.

        Sets :attr:`medoids` as a side-effect.

        Returns:
            ``(labels, item_ids)``
        """
        self._display_step(1, 2, "Computing distance matrix")
        with self._nested_display():
            dist_matrix = metric.compute_matrix(pool)

        self._display_step(2, 2, f"Clustering ({type(self).__name__})")
        np_matrix = dist_matrix.to_numpy().astype(np.float64)
        ids = dist_matrix.ids
        n_clusters = self.settings.n_clusters

        # --- BUILD phase ---
        selected, unselected = pam_build_optimized(np_matrix, n_clusters)

        # --- SWAP phase ---
        n_iter = 0
        swap = pam_swap_optimized(selected, unselected, np_matrix)

        while self.settings.max_iter and n_iter < self.settings.max_iter and swap:
            old_swap = swap

            selected.remove(swap[0])
            selected.append(swap[1])
            unselected.remove(swap[1])
            unselected.append(swap[0])

            n_iter += 1
            swap = pam_swap_optimized(selected, unselected, np_matrix)

            # Loop detection: if the new swap would undo what we just did, stop.
            if swap and old_swap[0] == swap[1] and old_swap[1] == swap[0]:
                break

        # --- Assign labels ---
        medoids_idx = selected
        self._medoids = [ids[ix] for ix in medoids_idx]
        labels = np.argmin(np_matrix[:, medoids_idx], axis=1).tolist()
        return labels, ids
