#!/usr/bin/env python3
"""
Tests: AggregationTrajectoryMetric
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.sequence import SequenceMetric, LinearPairwiseSequenceMetric
from tanat.metric.trajectory import (
    TrajectoryMetric,
    AggregationTrajectoryMetric,
)
from tanat.metric.matrix import DistanceMatrix

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Type-checking of pool and trajectory arguments."""

    def test_compute_matrix_rejects_non_traj_pool(self) -> None:
        """compute_matrix raises TypeError for non-TrajectoryPool."""
        agg = AggregationTrajectoryMetric()
        with pytest.raises(TypeError, match="TrajectoryPool"):
            agg.compute_matrix("not a pool")

    def test_call_rejects_non_trajectory(self, traj_pair) -> None:
        """__call__ raises TypeError if an argument is not a Trajectory."""
        agg = AggregationTrajectoryMetric()
        traj_a, _ = traj_pair
        with pytest.raises(TypeError, match="traj_b"):
            agg(traj_a, "not a trajectory")

        # bis repetita
        _, traj_b = traj_pair
        with pytest.raises(TypeError, match="traj_a"):
            agg("not a trajectory", traj_b)


class TestSinglePair:
    """Single-pair distance properties: identity, sign, symmetry, aggregation modes."""

    def test_identity(self, traj_pair) -> None:
        """Distance from a trajectory to itself is zero."""
        agg = AggregationTrajectoryMetric()
        traj_a, _ = traj_pair
        assert agg(traj_a, traj_a) == 0.0

    def test_symmetry(self, traj_pair) -> None:
        """Distance is symmetric: agg(a, b) == agg(b, a)."""
        agg = AggregationTrajectoryMetric()
        traj_a, traj_b = traj_pair
        assert agg(traj_a, traj_b) == pytest.approx(agg(traj_b, traj_a))

    def test_weights_change_result(self, traj_pair) -> None:
        """Assigning non-uniform weights changes the result (when aliases differ in distance)."""
        agg_no_weights = AggregationTrajectoryMetric()
        traj_a, traj_b = traj_pair
        aliases = sorted(set(traj_a) & set(traj_b))

        if len(aliases) < 2:
            pytest.skip("Need at least 2 common aliases to test weight effect")

        # Get per-alias distances via the default metric
        metric = agg_no_weights.settings.default_metric
        per_alias = [metric(traj_a[a], traj_b[a]) for a in aliases]
        if all(d == per_alias[0] for d in per_alias):
            pytest.skip("All alias distances are equal; weights have no effect")

        # Skew weights heavily toward the first alias
        skewed = {aliases[0]: 100.0}
        agg_skewed = AggregationTrajectoryMetric(weights=skewed)
        d_default = agg_no_weights(traj_a, traj_b)
        d_skewed = agg_skewed(traj_a, traj_b)
        assert d_default != pytest.approx(d_skewed)

    def test_agg_fun_sum_vs_mean(self, traj_pair) -> None:
        """agg_fun='sum' and agg_fun='mean' produce different results when distance > 0."""
        agg_mean = AggregationTrajectoryMetric(agg_fun="mean")
        agg_sum = AggregationTrajectoryMetric(agg_fun="sum")
        traj_a, traj_b = traj_pair
        d_mean = agg_mean(traj_a, traj_b)
        d_sum = agg_sum(traj_a, traj_b)

        n_common = len(sorted(set(traj_a) & set(traj_b)))
        if d_mean > 0 and n_common > 1:
            # sum = n_common * mean  (uniform weights)
            assert d_sum == pytest.approx(d_mean * n_common, rel=1e-5)

    def test_no_common_aliases_raises(self, disjoint_alias_traj_pool) -> None:
        """Trajectories with no shared alias raise ValueError.

        In ``disjoint_alias_traj_pool``: ID 1 lives only in ``'events'``,
        ID 11 only in ``'intervals'`` → empty intersection → :exc:`ValueError`.
        """
        agg = AggregationTrajectoryMetric()
        with pytest.raises(ValueError, match="no common sequence aliases"):
            agg(disjoint_alias_traj_pool[1], disjoint_alias_traj_pool[11])

    def test_sum_with_weights_is_weighted_sum(self, traj_pair) -> None:
        """agg_fun='sum' + weights produces a weighted sum (np.dot), not a weighted mean."""
        traj_a, traj_b = traj_pair
        aliases = sorted(set(traj_a) & set(traj_b))

        if len(aliases) < 2:
            pytest.skip("Need at least 2 common aliases")

        w = {a: float(i + 1) for i, a in enumerate(aliases)}
        agg = AggregationTrajectoryMetric(agg_fun="sum", weights=w)

        # Compute expected weighted sum manually
        metric = agg.settings.default_metric
        per_alias = [metric(traj_a[a], traj_b[a]) for a in aliases]
        weights_list = [w.get(a, 1.0) for a in aliases]
        expected = float(np.dot(per_alias, weights_list))

        assert agg(traj_a, traj_b) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    """Full pairwise matrix: structure and numerical properties."""

    def test_returns_distance_matrix(self, small_traj_pool) -> None:
        """compute_matrix() returns a DistanceMatrix instance."""
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(small_traj_pool)
        assert isinstance(dm, DistanceMatrix)

    def test_shape_and_ids_match(self, small_traj_pool) -> None:
        """Returned matrix is (n × n) and ids match pool.unique_ids."""
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(small_traj_pool)
        n = len(small_traj_pool)
        assert dm.shape == (n, n)
        assert dm.ids == small_traj_pool.unique_ids

    def test_diagonal_zero(self, small_traj_pool) -> None:
        """Diagonal entries are zero (distance from a trajectory to itself)."""
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(small_traj_pool)
        np.testing.assert_array_almost_equal(np.diag(dm.to_numpy()), 0.0, decimal=5)

    def test_symmetric(self, small_traj_pool) -> None:
        """Matrix is symmetric: M[i, j] == M[j, i]."""
        agg = AggregationTrajectoryMetric()
        arr = agg.compute_matrix(small_traj_pool).to_numpy()
        np.testing.assert_array_almost_equal(arr, arr.T, decimal=5)

    def test_single_trajectory_pool(self, small_traj_pool) -> None:
        """Pool with 1 trajectory → 1×1 matrix with a zero."""
        sub = small_traj_pool.subset(small_traj_pool.unique_ids[:1])
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(sub)
        assert dm.shape == (1, 1)
        assert dm.to_numpy()[0, 0] == 0.0

    def test_consistency_single_pair_vs_matrix(self, small_traj_pool) -> None:
        """dm[i,j] must equal agg(traj_i, traj_j) for a few pairs."""
        agg = AggregationTrajectoryMetric()
        ids = small_traj_pool.unique_ids[:4]
        sub = small_traj_pool.subset(ids)
        dm = agg.compute_matrix(sub)
        arr = dm.to_numpy()

        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if i == j:
                    continue
                expected = agg(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(
                    expected, abs=1e-5
                ), f"Mismatch at ({i},{j}): matrix={arr[i,j]:.4f}, direct={expected:.4f}"

    def test_matrix_values_snapshot(
        self, small_traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """Full distance matrix matches snapshot (regression guard)."""
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(small_traj_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))

    def test_compute_matrix_nan_on_disjoint_aliases(
        self, disjoint_alias_traj_pool
    ) -> None:
        """compute_matrix inserts nan for pairs with no common alias.

        In ``disjoint_alias_traj_pool``: ID 1 is events-only, ID 11 is
        intervals-only → the cell ``(id1, id11)`` must be ``nan`` (not a
        ``ValueError``).  ID 1 and ID 6 share ``"events"`` → finite.
        """
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(disjoint_alias_traj_pool)
        arr = dm.to_numpy()
        ids = list(dm.ids)

        idx_1 = ids.index(1)
        idx_11 = ids.index(11)
        idx_6 = ids.index(6)

        assert np.isnan(
            arr[idx_1, idx_11]
        ), "Expected nan for ID 1 vs 11 (no common alias)"
        assert np.isnan(
            arr[idx_11, idx_1]
        ), "Expected nan for ID 11 vs 1 (no common alias)"
        assert not np.isnan(
            arr[idx_1, idx_6]
        ), "ID 1 and 6 share 'events': should be finite"

    def test_compute_matrix_nan_on_disjoint_aliases_sum(
        self, disjoint_alias_traj_pool
    ) -> None:
        """agg_fun='sum' also produces nan (not 0.0) for pairs with no common alias."""
        agg = AggregationTrajectoryMetric(agg_fun="sum")
        dm = agg.compute_matrix(disjoint_alias_traj_pool)
        arr = dm.to_numpy()
        ids = list(dm.ids)

        idx_1 = ids.index(1)
        idx_11 = ids.index(11)

        assert np.isnan(arr[idx_1, idx_11]), (
            "Expected nan for ID 1 vs 11 with agg_fun='sum' (no common alias), "
            f"got {arr[idx_1, idx_11]}"
        )


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


class TestRegistryResolution:
    """String → SequenceMetric auto-resolution via Pydantic + Registrable."""

    def test_string_default_metric_resolved(self) -> None:
        """default_metric='linearpairwise' → instance at construction time."""
        agg = AggregationTrajectoryMetric(default_metric="linearpairwise")
        assert isinstance(agg.settings.default_metric, SequenceMetric)

    def test_string_sequence_metrics_resolved(self) -> None:
        """sequence_metrics={'events': 'linearpairwise'} → resolved per alias."""
        agg = AggregationTrajectoryMetric(sequence_metrics={"events": "linearpairwise"})
        assert isinstance(agg.settings.sequence_metrics["events"], SequenceMetric)

    def test_instance_default_metric_passthrough(self) -> None:
        """Passing a SequenceMetric instance directly is preserved."""
        metric = LinearPairwiseSequenceMetric()
        agg = AggregationTrajectoryMetric(default_metric=metric)
        assert agg.settings.default_metric is metric

    def test_get_registered(self) -> None:
        """TrajectoryMetric.get_registered('aggregation') → AggregationTrajectoryMetric."""
        cls = TrajectoryMetric.get_registered("aggregation")
        assert cls is AggregationTrajectoryMetric

    def test_settings_default_metric_is_always_instance(self) -> None:
        """settings.default_metric is always a SequenceMetric, never a string."""
        agg = AggregationTrajectoryMetric()
        assert isinstance(agg.settings.default_metric, SequenceMetric)
        assert not isinstance(agg.settings.default_metric, str)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundtrip:
    """Serialization round-trips and registry dispatch."""

    def test_to_config_type(self) -> None:
        """to_config() sets type='aggregation'."""
        agg = AggregationTrajectoryMetric()
        assert agg.to_config()["type"] == "aggregation"

    def test_from_config_roundtrip(self) -> None:
        """Settings survive a to_config / from_config round-trip."""
        agg = AggregationTrajectoryMetric(agg_fun="sum")
        config = agg.to_config()
        agg2 = AggregationTrajectoryMetric.from_config(config)
        assert agg2.settings.agg_fun == "sum"

    def test_from_config_via_base_registry(self) -> None:
        """TrajectoryMetric.from_config dispatches to AggregationTrajectoryMetric."""
        config = {"type": "aggregation"}
        agg = TrajectoryMetric.from_config(config)
        assert isinstance(agg, AggregationTrajectoryMetric)

    def test_to_config_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Config dict matches snapshot (regression guard for serialization format)."""
        agg = AggregationTrajectoryMetric(agg_fun="sum")
        assert agg.to_config() == snapshot


# ---------------------------------------------------------------------------
# Optimised path (two-step: per-alias matrices then aggregation)
# ---------------------------------------------------------------------------


class TestOptimisedPath:
    """Two-step optimised path consistency and edge cases."""

    def test_consistency_optimised_vs_naive(self, small_traj_pool) -> None:
        """Optimised matrix path matches naive pair-by-pair _compute."""
        agg = AggregationTrajectoryMetric()
        dm_opt = agg.compute_matrix(small_traj_pool)

        ids = small_traj_pool.unique_ids
        n = len(ids)
        naive = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = agg._compute(  # pylint: disable=protected-access
                    small_traj_pool[ids[i]], small_traj_pool[ids[j]]
                )
                naive[i, j] = float(d)

        fast = dm_opt.to_numpy()

        # NaN positions must match
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(naive))
        # Finite values must be close
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], naive[mask], decimal=4)

    def test_single_alias_pool(self, small_traj_pool) -> None:
        """K=1: pool with one alias still produces a valid square matrix."""
        sub = small_traj_pool.copy()
        sub.drop_sequence_pools("events", "states")  # keep only 'intervals'
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(sub)
        assert dm.shape == (len(sub), len(sub))
        np.testing.assert_array_almost_equal(np.diag(dm.to_numpy()), 0.0, decimal=5)

    def test_disjoint_alias_gives_nan(self, disjoint_alias_traj_pool) -> None:
        """Optimised path: nan at (i,j) when traj_i and traj_j share no alias."""
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(disjoint_alias_traj_pool)
        arr = dm.to_numpy()
        ids = list(dm.ids)

        idx_1 = ids.index(1)
        idx_11 = ids.index(11)
        assert np.isnan(arr[idx_1, idx_11])
        assert np.isnan(arr[idx_11, idx_1])

    def test_snapshot_unchanged(
        self, small_traj_pool, snapshot: SnapshotAssertion
    ) -> None:
        """Optimised path produces the same snapshot values as the previous implementation."""
        agg = AggregationTrajectoryMetric()
        dm = agg.compute_matrix(small_traj_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


# ---------------------------------------------------------------------------
# Memmap + chunks + resume
# ---------------------------------------------------------------------------


class TestMemmapChunked:
    """End-to-end memmap/chunked computation and resume logic."""

    def test_memmap_matches_inmemory(self, small_traj_pool, tmp_path) -> None:
        """Memmap result matches in-memory result exactly."""
        agg_mem = AggregationTrajectoryMetric()
        agg_disk = AggregationTrajectoryMetric(store_path=str(tmp_path), chunk_size=3)
        dm_mem = agg_mem.compute_matrix(small_traj_pool)
        dm_disk = agg_disk.compute_matrix(small_traj_pool)
        np.testing.assert_array_almost_equal(
            dm_mem.to_numpy(), dm_disk.to_numpy(), decimal=5
        )

    def test_is_memmap(self, small_traj_pool, tmp_path) -> None:
        """Result from disk path has is_memmap=True."""
        agg = AggregationTrajectoryMetric(store_path=str(tmp_path))
        dm = agg.compute_matrix(small_traj_pool)
        assert dm.is_memmap

    def test_resume_skips_computed_chunks(self, small_traj_pool, tmp_path) -> None:
        """Second call skips completed chunks and produces the same result."""
        agg = AggregationTrajectoryMetric(store_path=str(tmp_path), chunk_size=3)
        dm1 = agg.compute_matrix(small_traj_pool)
        dm2 = agg.compute_matrix(small_traj_pool)  # all done → resume
        np.testing.assert_array_almost_equal(dm1.to_numpy(), dm2.to_numpy(), decimal=5)

    def test_resume_after_partial(self, small_traj_pool, tmp_path) -> None:
        """Decrement completed_chunks → resume recomputes the missing chunk."""
        import json

        agg = AggregationTrajectoryMetric(store_path=str(tmp_path), chunk_size=3)
        dm1 = agg.compute_matrix(small_traj_pool)
        expected = dm1.to_numpy().copy()

        prog_path = tmp_path / "progress.json"
        prog = json.loads(prog_path.read_text())
        if prog["completed_chunks"] > 0:
            prog["completed_chunks"] -= 1
            prog["status"] = "computing"
            prog_path.write_text(json.dumps(prog))

        dm2 = agg.compute_matrix(small_traj_pool)
        np.testing.assert_array_almost_equal(dm2.to_numpy(), expected, decimal=5)

    def test_settings_change_forces_recompute(self, small_traj_pool, tmp_path) -> None:
        """Changing agg_fun wipes the old matrix and recomputes."""
        agg1 = AggregationTrajectoryMetric(agg_fun="mean", store_path=str(tmp_path))
        dm1 = agg1.compute_matrix(small_traj_pool)

        agg2 = AggregationTrajectoryMetric(agg_fun="sum", store_path=str(tmp_path))
        dm2 = agg2.compute_matrix(small_traj_pool)

        if np.any(dm1.to_numpy() > 0):
            assert not np.allclose(dm1.to_numpy(), dm2.to_numpy())

    def test_metadata_and_progress_written(self, small_traj_pool, tmp_path) -> None:
        """metadata.json and progress.json are present and well-formed."""
        import json

        agg = AggregationTrajectoryMetric(store_path=str(tmp_path))
        agg.compute_matrix(small_traj_pool)

        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["shape"] == [len(small_traj_pool), len(small_traj_pool)]
        assert "metric_config" in meta

        prog = json.loads((tmp_path / "progress.json").read_text())
        assert prog["status"] == "complete"
        assert isinstance(prog["completed_chunks"], int)
        assert prog["completed_chunks"] > 0

    def test_single_trajectory_memmap(self, small_traj_pool, tmp_path) -> None:
        """Pool with 1 trajectory + memmap → 1×1 matrix with a zero, is_memmap=True."""
        sub = small_traj_pool.subset(small_traj_pool.unique_ids[:1])
        agg = AggregationTrajectoryMetric(store_path=str(tmp_path), chunk_size=10)
        dm = agg.compute_matrix(sub)
        assert dm.shape == (1, 1)
        assert dm.to_numpy()[0, 0] == 0.0
        assert dm.is_memmap

    def test_chunk_size_larger_than_pool(self, small_traj_pool, tmp_path) -> None:
        """chunk_size > n → single chunk, same result as in-memory."""
        agg_mem = AggregationTrajectoryMetric()
        agg_disk = AggregationTrajectoryMetric(
            store_path=str(tmp_path), chunk_size=9999
        )
        dm_mem = agg_mem.compute_matrix(small_traj_pool)
        dm_disk = agg_disk.compute_matrix(small_traj_pool)
        np.testing.assert_array_almost_equal(
            dm_mem.to_numpy(), dm_disk.to_numpy(), decimal=5
        )

    def test_callsite_store_path_override(self, small_traj_pool, tmp_path) -> None:
        """store_path kwarg on compute_matrix() overrides instance default (no storage)."""
        agg = AggregationTrajectoryMetric()  # no storage
        dm = agg.compute_matrix(small_traj_pool, store_path=str(tmp_path))
        assert dm.is_memmap
        assert (tmp_path / "metadata.json").exists()
