#!/usr/bin/env python3
"""
Tests for HierarchicalClusterer.
"""

from __future__ import annotations

from tanat.clustering import Clusterer, HierarchicalClusterer

# ---------------------------------------------------------------------------
# Basic fit: sequence pools
# ---------------------------------------------------------------------------


class TestHierarchicalFitSequence:
    """fit() on sequence pools (cat_pool × seq_metric_name)."""

    def test_n_clusters_respected(self, cat_pool, seq_metric_name) -> None:
        """fit() produces exactly n_clusters cluster objects."""
        c = HierarchicalClusterer(metric=seq_metric_name, n_clusters=2)
        c.fit(cat_pool)
        assert len(c.clusters) == 2

    def test_all_items_assigned(self, cat_pool, seq_metric_name) -> None:
        """Every pool ID appears in exactly one cluster."""
        c = HierarchicalClusterer(metric=seq_metric_name, n_clusters=2)
        c.fit(cat_pool)
        assigned = {item for cluster in c.clusters for item in cluster.items}
        assert assigned == set(cat_pool.unique_ids)

    def test_static_feature_injected(self, cat_pool, seq_metric_name) -> None:
        """cluster_column is present in the pool's static data after fit."""
        c = HierarchicalClusterer(
            metric=seq_metric_name,
            n_clusters=2,
            cluster_column="__TEST_H__",
        )
        c.fit(cat_pool)
        static = cat_pool.static_data()
        assert static is not None
        assert "__TEST_H__" in static.columns

    def test_no_overlapping_clusters(self, cat_pool, seq_metric_name) -> None:
        """No item ID appears in more than one cluster."""
        c = HierarchicalClusterer(metric=seq_metric_name, n_clusters=2)
        c.fit(cat_pool)
        all_items: list = []
        for cluster in c.clusters:
            all_items.extend(cluster.items)
        assert len(all_items) == len(set(all_items))


# ---------------------------------------------------------------------------
# Basic fit: trajectory pools
# ---------------------------------------------------------------------------


class TestHierarchicalFitTrajectory:
    """fit() on trajectory pools (small_traj_pool × traj_metric_name)."""

    def test_n_clusters_respected(self, small_traj_pool, traj_metric) -> None:
        """fit() on a TrajectoryPool produces exactly n_clusters cluster objects."""
        c = HierarchicalClusterer(metric=traj_metric, n_clusters=2)
        c.fit(small_traj_pool)
        assert len(c.clusters) == 2

    def test_all_items_assigned(self, small_traj_pool, traj_metric) -> None:
        """Every pool ID appears in exactly one cluster."""
        c = HierarchicalClusterer(metric=traj_metric, n_clusters=2)
        c.fit(small_traj_pool)
        assigned = {item for cluster in c.clusters for item in cluster.items}
        assert assigned == set(small_traj_pool.unique_ids)

    def test_static_feature_injected(self, small_traj_pool, traj_metric) -> None:
        """cluster_column is present in the pool's static data after fit."""
        c = HierarchicalClusterer(
            metric=traj_metric,
            n_clusters=2,
            cluster_column="__TEST_H_TRAJ__",
        )
        c.fit(small_traj_pool)
        static = small_traj_pool.static_data()
        assert static is not None
        assert "__TEST_H_TRAJ__" in static.columns


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestHierarchicalConfig:
    """to_config / from_config roundtrip."""

    def test_to_config_type(self) -> None:
        """to_config() emits type='hierarchical' and preserves n_clusters."""
        c = HierarchicalClusterer(metric="linearpairwise", n_clusters=5)
        cfg = c.to_config()
        assert cfg["type"] == "hierarchical"
        assert cfg["settings"]["n_clusters"] == 5

    def test_from_config_roundtrip(self) -> None:
        """from_config(to_config()) reconstructs identical settings."""
        c = HierarchicalClusterer(
            metric="linearpairwise", n_clusters=4, linkage="average"
        )
        cfg = c.to_config()
        c2 = HierarchicalClusterer.from_config(cfg)
        assert c2.settings.n_clusters == 4
        assert c2.settings.linkage == "average"

    def test_from_config_via_base(self) -> None:
        """Clusterer.from_config() dispatches to HierarchicalClusterer via the type field."""
        c = HierarchicalClusterer(n_clusters=3)
        cfg = c.to_config()
        c2 = Clusterer.from_config(cfg)
        assert isinstance(c2, HierarchicalClusterer)
