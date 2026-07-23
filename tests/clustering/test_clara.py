#!/usr/bin/env python3
"""
Tests for CLARAClusterer.
"""

from __future__ import annotations

from tanat.clustering import Clusterer, CLARAClusterer, MedoidMixin


class TestCLARAMedoidMixin:
    """MedoidMixin: medoids property is None before fit."""

    def test_medoids_none_before_fit(self) -> None:
        """medoids property is None before fit() is called."""
        clara = CLARAClusterer(n_clusters=2, random_state=0)
        assert clara.medoids is None

    def test_clara_has_medoid_mixin(self) -> None:
        """CLARAClusterer is an instance of MedoidMixin."""
        assert isinstance(CLARAClusterer(), MedoidMixin)


# ---------------------------------------------------------------------------
# Basic fit: sequence pools
# ---------------------------------------------------------------------------


class TestCLARAFitSequence:
    """CLARAClusterer.fit() on sequence pools."""

    def test_n_clusters_respected(self, cat_pool, seq_metric_name) -> None:
        """fit() produces exactly n_clusters cluster objects."""
        clara = CLARAClusterer(
            metric=seq_metric_name,
            n_clusters=2,
            sampling_ratio=0.8,
            nb_pam_instances=2,
            random_state=42,
        )
        clara.fit(cat_pool)
        assert len(clara.clusters) == 2

    def test_medoids_count(self, cat_pool, seq_metric_name) -> None:
        """After fit, medoids list length equals n_clusters."""
        clara = CLARAClusterer(
            metric=seq_metric_name,
            n_clusters=2,
            sampling_ratio=0.8,
            nb_pam_instances=2,
            random_state=42,
        )
        clara.fit(cat_pool)
        assert clara.medoids is not None
        assert len(clara.medoids) == 2

    def test_all_items_assigned(self, cat_pool, seq_metric_name) -> None:
        """Every pool ID appears in exactly one cluster."""
        clara = CLARAClusterer(
            metric=seq_metric_name,
            n_clusters=2,
            sampling_ratio=0.8,
            nb_pam_instances=2,
            random_state=42,
        )
        clara.fit(cat_pool)
        assigned = {item for cluster in clara.clusters for item in cluster.items}
        assert assigned == set(cat_pool.unique_ids)

    def test_static_feature_injected(self, cat_pool, seq_metric_name) -> None:
        """cluster_column is present in the pool's static data after fit."""
        clara = CLARAClusterer(
            metric=seq_metric_name,
            n_clusters=2,
            sampling_ratio=0.8,
            nb_pam_instances=2,
            random_state=42,
            cluster_column="__TEST_CLARA__",
        )
        clara.fit(cat_pool)
        static = cat_pool.static_data()
        assert static is not None
        assert "__TEST_CLARA__" in static.columns

    def test_reproducibility(self, cat_pool, seq_metric_name) -> None:
        """Same random_state → same medoids."""
        kwargs = dict(
            metric=seq_metric_name,
            n_clusters=2,
            sampling_ratio=0.8,
            nb_pam_instances=2,
            random_state=42,
        )
        clara1 = CLARAClusterer(**kwargs)
        clara1.fit(cat_pool.copy())

        clara2 = CLARAClusterer(**kwargs)
        clara2.fit(cat_pool.copy())

        assert sorted(clara1.medoids) == sorted(clara2.medoids)


# ---------------------------------------------------------------------------
# Basic fit: trajectory pools
# ---------------------------------------------------------------------------


class TestCLARAFitTrajectory:
    """CLARAClusterer.fit() on trajectory pools."""

    def test_n_clusters_respected(self, small_traj_pool, traj_metric) -> None:
        """fit() on a TrajectoryPool produces exactly n_clusters cluster objects."""
        clara = CLARAClusterer(
            metric=traj_metric,
            n_clusters=2,
            sampling_ratio=0.8,
            nb_pam_instances=2,
            random_state=0,
        )
        clara.fit(small_traj_pool)
        assert len(clara.clusters) == 2

    def test_medoids_count(self, small_traj_pool, traj_metric) -> None:
        """After fit on a TrajectoryPool, medoids list length equals n_clusters."""
        clara = CLARAClusterer(
            metric=traj_metric,
            n_clusters=2,
            sampling_ratio=0.8,
            nb_pam_instances=2,
            random_state=0,
        )
        clara.fit(small_traj_pool)
        assert clara.medoids is not None
        assert len(clara.medoids) == 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCLARARegistry:
    """Registration and config serialisation."""

    def test_registered_as_clara(self) -> None:
        """'clara' key resolves to CLARAClusterer."""
        cls = Clusterer.get_registered("clara")
        assert cls is CLARAClusterer

    def test_to_config_type(self) -> None:
        """to_config() emits type='clara'."""
        clara = CLARAClusterer(n_clusters=3)
        cfg = clara.to_config()
        assert cfg["type"] == "clara"

    def test_from_config_roundtrip(self) -> None:
        """from_config(to_config()) reconstructs identical settings."""
        clara = CLARAClusterer(n_clusters=3, nb_pam_instances=3, random_state=7)
        cfg = clara.to_config()
        clara2 = CLARAClusterer.from_config(cfg)
        assert clara2.settings.n_clusters == 3
        assert clara2.settings.nb_pam_instances == 3
        assert clara2.settings.random_state == 7
