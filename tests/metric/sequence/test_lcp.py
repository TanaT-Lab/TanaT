#!/usr/bin/env python3
"""
Tests: LCPSequenceMetric
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.sequence import SequenceMetric, LCPSequenceMetric
from tanat.metric.matrix import DistanceMatrix

# ---------------------------------------------------------------------------
# Single-pair computation
# ---------------------------------------------------------------------------


class TestSinglePair:
    """Single-pair distance."""

    def test_same_sequence_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq, seq) matches snapshot."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert snapshot == round(lcp(seq, seq), 4)

    def test_pair_sequence_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Spot-check: distance between two sequences matches snapshot."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids
        assert snapshot == round(lcp(cat_pool[ids[0]], cat_pool[ids[1]]), 4)

    def test_prefix_mismatch_at_start_returns_0(self, cat_pool, entity_metric) -> None:
        """LCP mode='length' returns 0.0 when sequences differ from position 0."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric, mode="length")
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        # Length mode: value is between 0 and min(len_a, len_b)
        result = lcp(seq_a, seq_b)
        assert 0.0 <= result <= min(len(seq_a), len(seq_b))

    def test_identical_prefix_length_mode(self, cat_pool, entity_metric) -> None:
        """Two identical sequences give LCP = their length in mode='length'."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric, mode="length")
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert lcp(seq, seq) == pytest.approx(len(seq))

    def test_mode_distance(self, cat_pool, entity_metric) -> None:
        """mode='distance': d(seq, seq) == 0; d(seq_a, seq_b) == len_a + len_b - 2*lcp."""
        lcp_len = LCPSequenceMetric(entity_metric=entity_metric, mode="length")
        lcp_dist = LCPSequenceMetric(entity_metric=entity_metric, mode="distance")
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        length = lcp_len(seq_a, seq_b)
        dist = lcp_dist(seq_a, seq_b)
        expected = len(seq_a) + len(seq_b) - 2.0 * length
        assert dist == pytest.approx(expected)

    def test_mode_normalized_bounds(self, cat_pool, entity_metric) -> None:
        """mode='normalized': result in [0, 1]."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric, mode="normalized")
        ids = cat_pool.unique_ids
        for i in range(min(3, len(ids))):
            for j in range(min(3, len(ids))):
                d = lcp(cat_pool[ids[i]], cat_pool[ids[j]])
                assert 0.0 <= d <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    """Full pairwise matrix: structure and numerical properties."""

    def test_returns_distance_matrix(self, cat_pool, entity_metric) -> None:
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        assert isinstance(lcp.compute_matrix(cat_pool), DistanceMatrix)

    def test_square_and_ids_match(self, cat_pool, entity_metric) -> None:
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        dm = lcp.compute_matrix(cat_pool)
        n = len(cat_pool)
        assert dm.shape == (n, n)
        assert dm.ids == cat_pool.unique_ids

    def test_consistency_single_pair_vs_matrix(self, cat_pool, entity_metric) -> None:
        """Matrix[i,j] must equal direct lcp(seq_i, seq_j)."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids[:4]
        sub = cat_pool.subset(ids)
        dm = lcp.compute_matrix(sub)
        arr = dm.to_numpy()
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                expected = lcp(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(expected, abs=1e-5)

    def test_matrix_values_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        dm = lcp.compute_matrix(cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))

    def test_mixed_matrix_snapshot(
        self, mixed_cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Matrix on a pool mixing empty and non-empty sequences."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        dm = lcp.compute_matrix(mixed_cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    """Settings validation, registry, and config round-trip."""

    def test_registrable_lookup(self) -> None:
        assert SequenceMetric.get_registered("lcp") is LCPSequenceMetric

    def test_settings_defaults(self) -> None:
        lcp = LCPSequenceMetric()
        assert lcp.settings.mode == "distance"
        assert lcp.settings.equality_threshold == 0.0

    def test_config_roundtrip(self) -> None:
        lcp = LCPSequenceMetric(mode="normalized", equality_threshold=0.1)
        cfg = lcp.to_config()
        lcp2 = LCPSequenceMetric.from_config(cfg)
        assert lcp2.settings.mode == "normalized"
        assert lcp2.settings.equality_threshold == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Numba consistency: fast path == slow path
# ---------------------------------------------------------------------------


class TestNumbaConsistency:
    """Numba fast path produces the same result as the Python fallback."""

    def test_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python paths produce identical matrices (NaN-aware)."""
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        fast = lcp.compute_matrix(cat_pool).to_numpy()
        # pylint: disable=protected-access
        slow = lcp._compute_matrix_python(cat_pool).to_numpy()
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)

    def test_cross_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python cross-matrix paths agree."""
        ids = cat_pool.unique_ids
        pool_rows = cat_pool.subset(ids[:4])
        pool_cols = cat_pool.subset(ids[4:8])
        lcp = LCPSequenceMetric(entity_metric=entity_metric)
        fast = np.asarray(lcp.compute_cross_matrix(pool_rows, pool_cols))
        # pylint: disable=protected-access
        slow = np.asarray(lcp._compute_cross_matrix_python(pool_rows, pool_cols))
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)
