#!/usr/bin/env python3
"""
Tests: StaticMetric
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion


from tanat.metric.static import StaticMetric
from tanat.metric.entity import HammingEntityMetric
from tanat.metric.sequence import EditSequenceMetric
from tanat.metric.matrix import DistanceMatrix
from tanat.metric.trajectory import AggregationTrajectoryMetric

# ---------------------------------------------------------------------------
# Static Metric comparison between sequences or trajectories
# ---------------------------------------------------------------------------


class TestStatic:
    """Static metric between objects."""

    def test_sequence_default_snapshot(
        self, event_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq1, seq2) matches snapshot."""

        csmet = StaticMetric()
        seq1 = event_pool[event_pool.unique_ids[0]]
        seq2 = event_pool[event_pool.unique_ids[1]]
        assert snapshot == round(csmet(seq1, seq2), 4)

    def test_sequence_function_snapshot(
        self, event_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq1, seq2) matches snapshot."""

        def smet(static1, static2):
            """comparions"""
            return abs(float(static1["age"]) - float(static2["age"]))

        csmet = StaticMetric(cmp_fnct=smet)
        seq1 = event_pool[event_pool.unique_ids[11]]
        seq2 = event_pool[event_pool.unique_ids[12]]
        assert snapshot == round(csmet(seq1, seq2), 4)

    def test_trajectory_default_snapshot(
        self, traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq1, seq2) matches snapshot."""

        csmet = StaticMetric()
        traj1 = traj_pool[traj_pool.unique_ids[11]]
        traj2 = traj_pool[traj_pool.unique_ids[12]]
        assert snapshot == round(csmet(traj1, traj2), 4)

    def test_trajectory_function_snapshot(
        self, traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq1, seq2) matches snapshot."""

        def smet(static1, static2):
            """comparions"""
            return abs(float(static1["age"]) - float(static2["age"]))

        csmet = StaticMetric(cmp_fnct=smet)
        traj1 = traj_pool[traj_pool.unique_ids[11]]
        traj2 = traj_pool[traj_pool.unique_ids[12]]
        assert snapshot == round(csmet(traj1, traj2), 4)

    def test_trajectory_within_agg_default_snapshot(
        self, traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq1, seq2) matches snapshot."""

        traj_pool.sequence_pools["events"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["states"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["intervals"].cast_features({"status": pl.Categorical})

        def smet(static1, static2):
            """comparions"""
            return abs(float(static1["age"]) - float(static2["age"]))

        seq_metric = EditSequenceMetric(
            entity_metric=HammingEntityMetric(entity_feature="status")
        )

        metric = AggregationTrajectoryMetric(
            sequence_metrics={
                "events": seq_metric,
                "intervals": seq_metric,
                "states": seq_metric,
            },
            static_metric=smet,
        )
        traj1 = traj_pool[traj_pool.unique_ids[11]]
        traj2 = traj_pool[traj_pool.unique_ids[12]]
        assert snapshot == round(metric(traj1, traj2), 4)

    def test_trajectory_within_agg_weight_snapshot(
        self, traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq1, seq2) matches snapshot."""

        traj_pool.sequence_pools["events"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["states"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["intervals"].cast_features({"status": pl.Categorical})

        def smet(static1, static2):
            """comparions"""
            return abs(float(static1["age"]) - float(static2["age"]))

        seq_metric = EditSequenceMetric(
            entity_metric=HammingEntityMetric(entity_feature="status")
        )

        metric = AggregationTrajectoryMetric(
            sequence_metrics={
                "events": seq_metric,
                "intervals": seq_metric,
                "states": seq_metric,
            },
            static_metric=smet,
            static_metric_weight=0.5,
        )
        traj1 = traj_pool[traj_pool.unique_ids[11]]
        traj2 = traj_pool[traj_pool.unique_ids[12]]
        assert snapshot == round(metric(traj1, traj2), 4)

    def test_trajectory_within_agg_function_snapshot(
        self, traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq1, seq2) matches snapshot."""

        traj_pool.sequence_pools["events"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["states"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["intervals"].cast_features({"status": pl.Categorical})

        def smet(static1, static2):
            """comparions"""
            return abs(float(static1["age"]) - float(static2["age"]))

        seq_metric = EditSequenceMetric(
            entity_metric=HammingEntityMetric(entity_feature="status")
        )

        csmet = StaticMetric(cmp_fnct=smet)
        metric = AggregationTrajectoryMetric(
            sequence_metrics={
                "events": seq_metric,
                "intervals": seq_metric,
                "states": seq_metric,
            },
            static_metric=csmet,
        )
        traj1 = traj_pool[traj_pool.unique_ids[11]]
        traj2 = traj_pool[traj_pool.unique_ids[12]]
        assert snapshot == round(metric(traj1, traj2), 4)


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestStaticPool:
    """Compute matrix of StaticMetric values from a pool of trajectories or
    sequences."""

    def test_static_matrix_snapshot(
        self, traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """compute_matrix() results on a static metric."""

        subpool = traj_pool.subset([11, 12, 13, 14, 15])

        def smet(static1, static2):
            """comparions"""
            return abs(float(static1["age"]) - float(static2["age"]))

        csmet = StaticMetric(cmp_fnct=smet)

        mat = csmet.compute_matrix(subpool)

        assert snapshot == str(mat)

    def test_agg_static_matrix_snapshot(
        self, traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """compute_matrix() results within aggregation function."""

        traj_pool.sequence_pools["events"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["states"].cast_features({"status": pl.Categorical})
        traj_pool.sequence_pools["intervals"].cast_features({"status": pl.Categorical})

        def smet(static1, static2):
            """comparions"""
            return abs(float(static1["age"]) - float(static2["age"]))

        seq_metric = EditSequenceMetric(
            entity_metric=HammingEntityMetric(entity_feature="status")
        )

        csmet = StaticMetric(cmp_fnct=smet)
        metric = AggregationTrajectoryMetric(
            sequence_metrics={
                "events": seq_metric,
                "intervals": seq_metric,
                "states": seq_metric,
            },
            static_metric=csmet,
        )

        subpool = traj_pool.subset([11, 12, 13, 14, 15])

        mat = metric.compute_matrix(subpool)

        assert snapshot == str(mat)


class TestStaticValidation:
    """Evaluation the validation capability when using a metric
    of static metric in aggregation function."""

    def test_no_static_data(self, event_pool_no_static) -> None:
        """test error when no static data."""

        csmet = StaticMetric()
        seq1 = event_pool_no_static[event_pool_no_static.unique_ids[0]]
        seq2 = event_pool_no_static[event_pool_no_static.unique_ids[1]]

        with pytest.raises(TypeError):
            csmet(seq1, seq2)

    def test_static_fnct_in_agg(self) -> None:
        """Error in metrics."""

        with pytest.raises(TypeError, match="Expected a function"):
            metric = AggregationTrajectoryMetric(static_metric=45)

    def test_static_other_metric_in_agg(self, traj_pool) -> None:
        """Error in metrics."""

        traj1 = traj_pool[traj_pool.unique_ids[11]]
        traj2 = traj_pool[traj_pool.unique_ids[12]]

        seq_metric = EditSequenceMetric(
            entity_metric=HammingEntityMetric(entity_feature="status")
        )

        # another metric is callable ... so it works
        metric = AggregationTrajectoryMetric(
            sequence_metrics={
                "events": seq_metric,
                "intervals": seq_metric,
                "states": seq_metric,
            }
        )

        # until, evaluate a metric
        with pytest.raises(TypeError):
            metric(traj1, traj2)

    def test_static_bad_args_fnct(self, event_pool) -> None:
        """cmp function with bad ."""

        def smet(static1):
            """comparions"""
            return abs(float(static1["age"]))

        csmet = StaticMetric(cmp_fnct=smet)
        seq1 = event_pool[event_pool.unique_ids[0]]
        seq2 = event_pool[event_pool.unique_ids[1]]

        with pytest.raises(TypeError, match="1 positional argument"):
            csmet(seq1, seq2)
