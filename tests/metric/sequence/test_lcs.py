#!/usr/bin/env python3
"""
Tests: LCSSequenceMetric
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.sequence import SequenceMetric, LCSSequenceMetric
from tanat.metric.matrix import DistanceMatrix
from tanat.metric.entity import HammingEntityMetric

# ---------------------------------------------------------------------------
# Single-pair computation
# ---------------------------------------------------------------------------


class TestSinglePair:
    """Single-pair distance."""

    def test_same_sequence_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq, seq) matches snapshot."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert snapshot == round(lcs(seq, seq), 4)

    def test_pair_sequence_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Spot-check: distance between two sequences matches snapshot."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids
        assert snapshot == round(lcs(cat_pool[ids[0]], cat_pool[ids[1]]), 4)

    def test_mode_length_returns_int_like(self, cat_pool, entity_metric) -> None:
        """mode='length': result is 0 ≤ LCS ≤ min(n, m)."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric, mode="length")
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        result = lcs(seq_a, seq_b)
        assert 0.0 <= result <= min(len(seq_a), len(seq_b))

    def test_mode_distance_vs_length(self, cat_pool, entity_metric) -> None:
        """mode='distance' == len_a + len_b - 2 * mode='length'."""
        lcs_len = LCSSequenceMetric(entity_metric=entity_metric, mode="length")
        lcs_dist = LCSSequenceMetric(entity_metric=entity_metric, mode="distance")
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        length = lcs_len(seq_a, seq_b)
        dist = lcs_dist(seq_a, seq_b)
        expected = len(seq_a) + len(seq_b) - 2.0 * length
        assert dist == pytest.approx(expected)

    def test_mode_normalized_bounds(self, cat_pool, entity_metric) -> None:
        """mode='normalized': result in [0, 1]."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric, mode="normalized")
        ids = cat_pool.unique_ids
        for i in range(min(3, len(ids))):
            for j in range(min(3, len(ids))):
                d = lcs(cat_pool[ids[i]], cat_pool[ids[j]])
                assert 0.0 <= d <= 1.0 + 1e-9

    def test_equality_threshold_relaxes_matching(self, cat_pool, entity_metric) -> None:
        """A larger equality_threshold can only increase or maintain LCS length."""
        lcs_strict = LCSSequenceMetric(
            entity_metric=entity_metric, mode="length", equality_threshold=0.0
        )
        lcs_relax = LCSSequenceMetric(
            entity_metric=entity_metric, mode="length", equality_threshold=1.0
        )
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        assert lcs_relax(seq_a, seq_b) >= lcs_strict(seq_a, seq_b)

    def test_identical_sequence_length_mode(self, cat_pool, entity_metric) -> None:
        """LCS of a sequence with itself equals its length."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric, mode="length")
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert lcs(seq, seq) == pytest.approx(len(seq))


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    """LCS matrix structure and regression checks."""

    def test_returns_distance_matrix(self, cat_pool, entity_metric) -> None:
        """compute_matrix() returns a DistanceMatrix instance."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        assert isinstance(lcs.compute_matrix(cat_pool), DistanceMatrix)

    def test_square_and_ids_match(self, cat_pool, entity_metric) -> None:
        """Returned matrix is square and reuses pool identifiers."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        dm = lcs.compute_matrix(cat_pool)
        n = len(cat_pool)
        assert dm.shape == (n, n)
        assert dm.ids == cat_pool.unique_ids

    def test_consistency_single_pair_vs_matrix(self, cat_pool, entity_metric) -> None:
        """Matrix entries match direct LCS computations."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids[:4]
        sub = cat_pool.subset(ids)
        dm = lcs.compute_matrix(sub)
        arr = dm.to_numpy()
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                expected = lcs(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(expected, abs=1e-5)

    def test_matrix_values_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Full LCS matrix stays stable against the snapshot."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        dm = lcs.compute_matrix(cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))

    def test_mixed_matrix_snapshot(
        self, mixed_cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Matrix on a pool mixing empty and non-empty sequences."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        dm = lcs.compute_matrix(mixed_cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    """Registry, defaults, and config round-trip checks."""

    def test_registrable_lookup(self) -> None:
        """Registry lookup resolves the LCS metric class."""
        assert SequenceMetric.get_registered("lcs") is LCSSequenceMetric

    def test_settings_defaults(self) -> None:
        """Default settings use distance mode and zero threshold."""
        lcs = LCSSequenceMetric()
        assert lcs.settings.mode == "distance"
        assert lcs.settings.equality_threshold == 0.0

    def test_config_roundtrip(self) -> None:
        """Configuration serialization preserves LCS options."""
        lcs = LCSSequenceMetric(mode="normalized", equality_threshold=0.2)
        cfg = lcs.to_config()
        lcs2 = LCSSequenceMetric.from_config(cfg)
        assert lcs2.settings.mode == "normalized"
        assert lcs2.settings.equality_threshold == pytest.approx(0.2)


#
# ---------------------------------------------------------------------------
# Numba consistency: fast path == slow path
# ---------------------------------------------------------------------------


class TestNumbaConsistency:
    """Numba fast path produces the same result as the Python fallback."""

    def test_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python paths produce identical matrices (NaN-aware)."""
        lcs = LCSSequenceMetric(entity_metric=entity_metric)
        fast = lcs.compute_matrix(cat_pool).to_numpy()
        # pylint: disable=protected-access
        slow = lcs._compute_matrix_python(cat_pool).to_numpy()
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)

    def test_cross_matrix(self, cat_pool) -> None:
        """compute_cross_matrix returns (n × k) consistent with single-pair."""
        hamming = HammingEntityMetric(entity_feature="status")
        ids = cat_pool.unique_ids[:8]
        sub = cat_pool.subset(ids)
        pool_rows = sub.subset(ids[:4])
        pool_cols = sub.subset(ids[4:])

        lcs = LCSSequenceMetric(entity_metric=hamming)
        cross = lcs.compute_cross_matrix(pool_rows, pool_cols)

        assert cross.shape == (4, 4)
        seq_r = pool_rows[ids[0]]
        seq_c = pool_cols[ids[4]]
        assert float(cross[0, 0]) == pytest.approx(lcs(seq_r, seq_c), abs=1e-4)
