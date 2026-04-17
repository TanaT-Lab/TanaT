#!/usr/bin/env python3
"""
Tests for Clusterer ABC: _validate_pool, _resolve_metric_for_pool, ..
"""

from __future__ import annotations

import pytest

from tanat_utils.registrable.exceptions import UnregisteredTypeError
from tanat.clustering import HierarchicalClusterer
from tanat.metric.sequence.base import SequenceMetric
from tanat.metric.trajectory.base import TrajectoryMetric

# ---------------------------------------------------------------------------
# _validate_pool
# ---------------------------------------------------------------------------


class TestValidatePool:
    """_validate_pool accepts pools, rejects anything else."""

    # pylint: disable=protected-access

    def test_accepts_sequence_pool(self, cat_pool) -> None:
        """Valid SequencePool passes without raising."""
        c = HierarchicalClusterer(n_clusters=2)
        c._validate_pool(cat_pool)  # must not raise

    def test_accepts_trajectory_pool(self, small_traj_pool) -> None:
        """Valid TrajectoryPool passes without raising."""
        c = HierarchicalClusterer(n_clusters=2)
        c._validate_pool(small_traj_pool)

    def test_rejects_list(self) -> None:
        """Plain list raises TypeError with a pool-type hint."""
        c = HierarchicalClusterer()
        with pytest.raises(TypeError, match="SequencePool or TrajectoryPool"):
            c._validate_pool([1, 2, 3])

    def test_rejects_none(self) -> None:
        """None raises TypeError with a pool-type hint."""
        c = HierarchicalClusterer()
        with pytest.raises(TypeError, match="SequencePool or TrajectoryPool"):
            c._validate_pool(None)


# ---------------------------------------------------------------------------
# _resolve_metric_for_pool
# ---------------------------------------------------------------------------


class TestResolveMetric:
    """_resolve_metric_for_pool resolves strings and passes instances through."""

    # pylint: disable=protected-access

    def test_str_with_seq_pool_returns_sequence_metric(self, cat_pool) -> None:
        """String name + SequencePool → SequenceMetric instance."""
        c = HierarchicalClusterer(metric="linearpairwise", n_clusters=2)
        metric = c._resolve_metric_for_pool(c.settings.metric, cat_pool)
        assert isinstance(metric, SequenceMetric)

    def test_str_with_traj_pool_returns_trajectory_metric(
        self, small_traj_pool
    ) -> None:
        """String name + TrajectoryPool → TrajectoryMetric instance."""
        c = HierarchicalClusterer(metric="aggregation", n_clusters=2)
        metric = c._resolve_metric_for_pool(c.settings.metric, small_traj_pool)
        assert isinstance(metric, TrajectoryMetric)

    def test_unknown_str_raises(self, cat_pool) -> None:
        """Unknown metric name raises UnregisteredTypeError."""
        c = HierarchicalClusterer(metric="__nonexistent__", n_clusters=2)
        with pytest.raises(UnregisteredTypeError):
            c._resolve_metric_for_pool(c.settings.metric, cat_pool)

    def test_instance_returned_as_is(self, cat_pool) -> None:
        """SequenceMetric instance is returned by identity (no copy)."""
        lp = SequenceMetric.get_registered("linearpairwise")()
        c = HierarchicalClusterer(metric=lp, n_clusters=2)
        resolved = c._resolve_metric_for_pool(c.settings.metric, cat_pool)
        assert resolved is lp

    def test_trajectory_metric_instance_returned_as_is(self, small_traj_pool) -> None:
        """TrajectoryMetric instance is returned by identity (no copy)."""
        agg = TrajectoryMetric.get_registered("aggregation")()
        c = HierarchicalClusterer(metric=agg, n_clusters=2)
        resolved = c._resolve_metric_for_pool(c.settings.metric, small_traj_pool)
        assert resolved is agg


# ---------------------------------------------------------------------------
# Settings: str stays as str (no eager coercion)
# ---------------------------------------------------------------------------


class TestSettingsMetricType:
    """'metric' field stays a raw string in settings when given as string."""

    def test_str_metric_stays_str(self) -> None:
        """String metric is stored as-is in settings without eager coercion."""
        c = HierarchicalClusterer(metric="linearpairwise", n_clusters=2)
        assert isinstance(c.settings.metric, str)
        assert c.settings.metric == "linearpairwise"

    def test_instance_metric_stays_instance(self) -> None:
        """Metric instance is stored by reference in settings."""
        lp = SequenceMetric.get_registered("linearpairwise")()
        c = HierarchicalClusterer(metric=lp, n_clusters=2)
        assert c.settings.metric is lp


# ---------------------------------------------------------------------------
# Edge cases: empty pool and single-item pool
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty and single-item pools."""

    def test_empty_pool_raises(self, cat_pool) -> None:
        """fit() on an empty pool raises ValueError."""
        empty = cat_pool.subset([cat_pool.unique_ids[0]])
        empty = empty.subset([])  # 0 items
        c = HierarchicalClusterer(n_clusters=2)
        with pytest.raises(ValueError, match="minimum of 2"):
            c.fit(empty)

    def test_single_item_pool_raises(self, cat_pool) -> None:
        """fit() on a pool with 1 item raises ValueError (minimum 2 required)."""
        single = cat_pool.subset([cat_pool.unique_ids[0]])
        c = HierarchicalClusterer(n_clusters=1)
        with pytest.raises(ValueError, match="minimum of 2"):
            c.fit(single)


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Pydantic validators reject invalid settings."""

    def test_n_clusters_zero_raises(self) -> None:
        """n_clusters=0 raises ValidationError."""
        with pytest.raises(Exception):
            HierarchicalClusterer(n_clusters=0)

    def test_n_clusters_negative_raises(self) -> None:
        """n_clusters=-1 raises ValidationError."""
        with pytest.raises(Exception):
            HierarchicalClusterer(n_clusters=-1)
