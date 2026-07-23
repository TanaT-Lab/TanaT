#!/usr/bin/env python3
"""
Tests for PAMClusterer and MedoidMixin.
"""

from __future__ import annotations

from tanat.clustering import Clusterer, PAMClusterer, MedoidMixin


class TestPAMMedoidMixin:
    """MedoidMixin: medoids property is None before fit."""

    def test_medoids_none_before_fit(self) -> None:
        """medoids property is None before fit() is called."""
        pam = PAMClusterer(n_clusters=2)
        assert pam.medoids is None

    def test_pam_has_medoid_mixin(self) -> None:
        """PAMClusterer is an instance of MedoidMixin."""
        assert isinstance(PAMClusterer(), MedoidMixin)


# ---------------------------------------------------------------------------
# Basic fit: sequence pools
# ---------------------------------------------------------------------------


class TestPAMFitSequence:
    """PAMClusterer.fit() on sequence pools."""

    def test_n_clusters_respected(self, cat_pool, seq_metric_name) -> None:
        """fit() produces exactly n_clusters cluster objects."""
        pam = PAMClusterer(metric=seq_metric_name, n_clusters=2)
        pam.fit(cat_pool)
        assert len(pam.clusters) == 2

    def test_medoids_count(self, cat_pool, seq_metric_name) -> None:
        """After fit, medoids list length equals n_clusters."""
        pam = PAMClusterer(metric=seq_metric_name, n_clusters=2)
        pam.fit(cat_pool)
        assert pam.medoids is not None
        assert len(pam.medoids) == 2

    def test_medoids_are_valid_ids(self, cat_pool, seq_metric_name) -> None:
        """Every medoid is a valid pool ID."""
        pam = PAMClusterer(metric=seq_metric_name, n_clusters=2)
        pam.fit(cat_pool)
        valid = set(cat_pool.unique_ids)
        assert set(pam.medoids).issubset(valid)

    def test_each_medoid_in_different_cluster(self, cat_pool, seq_metric_name) -> None:
        """Each medoid belongs to a distinct cluster."""
        pam = PAMClusterer(metric=seq_metric_name, n_clusters=2)
        pam.fit(cat_pool)
        cluster_of = {}
        for cluster in pam.clusters:
            for item in cluster.items:
                cluster_of[item] = cluster.id
        medoid_cluster_ids = {cluster_of[m] for m in pam.medoids}
        assert len(medoid_cluster_ids) == len(pam.medoids)

    def test_all_items_assigned(self, cat_pool, seq_metric_name) -> None:
        """Every pool ID appears in exactly one cluster."""
        pam = PAMClusterer(metric=seq_metric_name, n_clusters=2)
        pam.fit(cat_pool)
        assigned = {item for cluster in pam.clusters for item in cluster.items}
        assert assigned == set(cat_pool.unique_ids)

    def test_static_feature_injected(self, cat_pool, seq_metric_name) -> None:
        """cluster_column is present in the pool's static data after fit."""
        pam = PAMClusterer(
            metric=seq_metric_name,
            n_clusters=2,
            cluster_column="__TEST_PAM__",
        )
        pam.fit(cat_pool)
        static = cat_pool.static_data()
        assert static is not None
        assert "__TEST_PAM__" in static.columns


# ---------------------------------------------------------------------------
# Basic fit: trajectory pools
# ---------------------------------------------------------------------------


class TestPAMFitTrajectory:
    """PAMClusterer.fit() on trajectory pools."""

    def test_n_clusters_respected(self, small_traj_pool, traj_metric) -> None:
        """fit() on a TrajectoryPool produces exactly n_clusters cluster objects."""
        pam = PAMClusterer(metric=traj_metric, n_clusters=2)
        pam.fit(small_traj_pool)
        assert len(pam.clusters) == 2

    def test_medoids_count(self, small_traj_pool, traj_metric) -> None:
        """After fit on a TrajectoryPool, medoids list length equals n_clusters."""
        pam = PAMClusterer(metric=traj_metric, n_clusters=2)
        pam.fit(small_traj_pool)
        assert pam.medoids is not None
        assert len(pam.medoids) == 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestPAMRegistry:
    """Registration and config serialisation."""

    def test_registered_as_pam(self) -> None:
        """'pam' key resolves to PAMClusterer."""
        cls = Clusterer.get_registered("pam")
        assert cls is PAMClusterer

    def test_to_config_type(self) -> None:
        """to_config() emits type='pam'."""
        pam = PAMClusterer(n_clusters=3)
        cfg = pam.to_config()
        assert cfg["type"] == "pam"

    def test_from_config_roundtrip(self) -> None:
        """from_config(to_config()) reconstructs identical settings."""
        pam = PAMClusterer(n_clusters=3, max_iter=10)
        cfg = pam.to_config()
        pam2 = PAMClusterer.from_config(cfg)
        assert pam2.settings.n_clusters == 3
        assert pam2.settings.max_iter == 10
