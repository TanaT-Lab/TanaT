#!/usr/bin/env python3
"""
Tests: DTWSequenceMetric
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.sequence import SequenceMetric, DTWSequenceMetric
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
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert snapshot == round(dtw(seq, seq), 4)

    def test_pair_sequence_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Spot-check: distance between two sequences matches snapshot."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids
        assert snapshot == round(dtw(cat_pool[ids[0]], cat_pool[ids[1]]), 4)

    def test_empty_returns_nan(self, empty_cat_pool, entity_metric) -> None:
        """Two empty sequences → DTW returns nan (undefined)."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        ids = empty_cat_pool.unique_ids
        seq_a, seq_b = empty_cat_pool[ids[0]], empty_cat_pool[ids[1]]
        result = dtw(seq_a, seq_b)
        assert math.isnan(result)

    def test_window_constraint_effect(self, cat_pool, entity_metric) -> None:
        """DTW with a tight window ≥ DTW without window (more constrained)."""
        dtw_full = DTWSequenceMetric(entity_metric=entity_metric, window=None)
        dtw_tight = DTWSequenceMetric(entity_metric=entity_metric, window=1)
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        # Constrained DTW ≥ unconstrained DTW
        assert dtw_tight(seq_a, seq_b) >= dtw_full(seq_a, seq_b) - 1e-9

    def test_no_window_equals_large_window(self, cat_pool, entity_metric) -> None:
        """DTW with window=None equals DTW with window >= max(len_a, len_b)."""
        dtw_full = DTWSequenceMetric(entity_metric=entity_metric, window=None)
        dtw_wide = DTWSequenceMetric(entity_metric=entity_metric, window=1000)
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        assert dtw_full(seq_a, seq_b) == pytest.approx(dtw_wide(seq_a, seq_b), abs=1e-5)


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    """DTW matrix structure and regression checks."""

    def test_returns_distance_matrix(self, cat_pool, entity_metric) -> None:
        """compute_matrix() returns a DistanceMatrix instance."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        assert isinstance(dtw.compute_matrix(cat_pool), DistanceMatrix)

    def test_square_and_ids_match(self, cat_pool, entity_metric) -> None:
        """Returned matrix is square and reuses pool identifiers."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        dm = dtw.compute_matrix(cat_pool)
        n = len(cat_pool)
        assert dm.shape == (n, n)
        assert dm.ids == cat_pool.unique_ids

    def test_consistency_single_pair_vs_matrix(self, cat_pool, entity_metric) -> None:
        """Matrix entries match direct DTW computations."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids[:4]
        sub = cat_pool.subset(ids)
        dm = dtw.compute_matrix(sub)
        arr = dm.to_numpy()
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                expected = dtw(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(expected, abs=1e-5)

    def test_matrix_values_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Full DTW matrix stays stable against the snapshot."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        dm = dtw.compute_matrix(cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))

    def test_mixed_matrix_snapshot(
        self, mixed_cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Matrix on a pool mixing empty and non-empty sequences."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        dm = dtw.compute_matrix(mixed_cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    """Registry, defaults, and validation around DTW settings."""

    def test_registrable_lookup(self) -> None:
        """Registry lookup resolves the DTW metric class."""
        assert SequenceMetric.get_registered("dtw") is DTWSequenceMetric

    def test_settings_defaults(self) -> None:
        """Default settings keep window unset and normalization disabled."""
        dtw = DTWSequenceMetric()
        assert dtw.settings.window is None
        assert dtw.settings.normalize is False

    def test_invalid_window_rejected(self) -> None:
        """Non-positive window values are rejected."""
        with pytest.raises(Exception):
            DTWSequenceMetric(window=0)
        with pytest.raises(Exception):
            DTWSequenceMetric(window=-1)

    def test_config_roundtrip(self) -> None:
        """Configuration serialization preserves DTW options."""
        dtw = DTWSequenceMetric(window=3, normalize=True)
        cfg = dtw.to_config()
        dtw2 = DTWSequenceMetric.from_config(cfg)
        assert dtw2.settings.window == 3
        assert dtw2.settings.normalize is True


# ---------------------------------------------------------------------------
# Numba consistency: fast path == slow path
# ---------------------------------------------------------------------------


class TestNumbaConsistency:
    """Numba fast path produces the same result as the Python fallback."""

    def test_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python paths produce identical matrices (NaN-aware)."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        fast = dtw.compute_matrix(cat_pool).to_numpy()
        # pylint: disable=protected-access
        slow = dtw._compute_matrix_python(cat_pool).to_numpy()
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)

    def test_cross_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python cross-matrix paths agree."""
        ids = cat_pool.unique_ids
        pool_rows = cat_pool.subset(ids[:4])
        pool_cols = cat_pool.subset(ids[4:8])
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        fast = np.asarray(dtw.compute_cross_matrix(pool_rows, pool_cols))
        # pylint: disable=protected-access
        slow = np.asarray(dtw._compute_cross_matrix_python(pool_rows, pool_cols))
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)
