#!/usr/bin/env python3
"""
Tests: EditSequenceMetric
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.sequence import SequenceMetric, EditSequenceMetric
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
        edit = EditSequenceMetric(entity_metric=entity_metric)
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert snapshot == round(edit(seq, seq), 4)

    def test_pair_sequence_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Spot-check: distance between two sequences matches snapshot."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids
        assert snapshot == round(edit(cat_pool[ids[0]], cat_pool[ids[1]]), 4)

    def test_both_empty_returns_0(self, empty_cat_pool, entity_metric) -> None:
        """Two empty sequences → edit distance = 0."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        ids = empty_cat_pool.unique_ids
        seq_a, seq_b = empty_cat_pool[ids[0]], empty_cat_pool[ids[1]]
        assert edit(seq_a, seq_b) == 0.0

    def test_indel_cost_effect(self, cat_pool, entity_metric) -> None:
        """Higher indel_cost increases or maintains the distance."""
        edit_low = EditSequenceMetric(entity_metric=entity_metric, indel_cost=0.5)
        edit_high = EditSequenceMetric(entity_metric=entity_metric, indel_cost=2.0)
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        assert edit_high(seq_a, seq_b) >= edit_low(seq_a, seq_b)

    def test_normalize_bounds_0_1(self, cat_pool, entity_metric) -> None:
        """Normalized edit distance ∈ [0, 1]."""
        edit = EditSequenceMetric(entity_metric=entity_metric, normalize=True)
        ids = cat_pool.unique_ids
        for i in range(min(3, len(ids))):
            for j in range(min(3, len(ids))):
                d = edit(cat_pool[ids[i]], cat_pool[ids[j]])
                assert 0.0 <= d <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    """Edit-distance matrix structure and regression checks."""

    def test_returns_distance_matrix(self, cat_pool, entity_metric) -> None:
        """compute_matrix() returns a DistanceMatrix instance."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        assert isinstance(edit.compute_matrix(cat_pool), DistanceMatrix)

    def test_square_and_ids_match(self, cat_pool, entity_metric) -> None:
        """Returned matrix is square and reuses pool identifiers."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        dm = edit.compute_matrix(cat_pool)
        n = len(cat_pool)
        assert dm.shape == (n, n)
        assert dm.ids == cat_pool.unique_ids

    def test_consistency_single_pair_vs_matrix(self, cat_pool, entity_metric) -> None:
        """Matrix entries match direct edit-distance computations."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids[:4]
        sub = cat_pool.subset(ids)
        dm = edit.compute_matrix(sub)
        arr = dm.to_numpy()
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                expected = edit(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(expected, abs=1e-5)

    def test_matrix_values_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Full edit-distance matrix stays stable against the snapshot."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        dm = edit.compute_matrix(cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))

    def test_mixed_matrix_snapshot(
        self, mixed_cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Matrix on a pool mixing empty and non-empty sequences."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        dm = edit.compute_matrix(mixed_cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    """Registry, defaults, and validation around edit settings."""

    def test_registrable_lookup(self) -> None:
        """Registry lookup resolves the edit metric class."""
        assert SequenceMetric.get_registered("edit") is EditSequenceMetric

    def test_settings_defaults(self) -> None:
        """Default settings keep unit indel cost and no normalization."""
        edit = EditSequenceMetric()
        assert edit.settings.indel_cost == 1.0
        assert edit.settings.normalize is False

    def test_invalid_indel_cost_rejected(self) -> None:
        """Non-positive indel costs are rejected."""
        with pytest.raises(Exception):
            EditSequenceMetric(indel_cost=0.0)
        with pytest.raises(Exception):
            EditSequenceMetric(indel_cost=-1.0)

    def test_config_roundtrip(self) -> None:
        """Configuration serialization preserves edit options."""
        edit = EditSequenceMetric(indel_cost=0.5, normalize=True)
        cfg = edit.to_config()
        edit2 = EditSequenceMetric.from_config(cfg)
        assert edit2.settings.indel_cost == pytest.approx(0.5)
        assert edit2.settings.normalize is True


# ---------------------------------------------------------------------------
# Numba consistency: fast path == slow path
# ---------------------------------------------------------------------------


class TestNumbaConsistency:
    """Numba fast path produces the same result as the Python fallback."""

    def test_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python paths produce identical matrices (NaN-aware)."""
        edit = EditSequenceMetric(entity_metric=entity_metric)
        fast = edit.compute_matrix(cat_pool).to_numpy()
        # pylint: disable=protected-access
        slow = edit._compute_matrix_python(cat_pool).to_numpy()
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)

    def test_cross_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python cross-matrix paths agree."""
        ids = cat_pool.unique_ids
        pool_rows = cat_pool.subset(ids[:4])
        pool_cols = cat_pool.subset(ids[4:8])
        edit = EditSequenceMetric(entity_metric=entity_metric)
        fast = np.asarray(edit.compute_cross_matrix(pool_rows, pool_cols))
        # pylint: disable=protected-access
        slow = np.asarray(edit._compute_cross_matrix_python(pool_rows, pool_cols))
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)
