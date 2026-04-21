#!/usr/bin/env python3
"""
Tests: Chi2SequenceMetric
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.sequence import SequenceMetric, Chi2SequenceMetric
from tanat.metric.matrix import DistanceMatrix

# ---------------------------------------------------------------------------
# Single-pair computation
# ---------------------------------------------------------------------------


class TestSinglePair:
    """Single-pair distance."""

    def test_same_sequence_snapshot(
        self, cat_pool, snapshot: SnapshotAssertion
    ) -> None:
        """d(seq, seq) matches snapshot."""
        chi2 = Chi2SequenceMetric(entity_feature="status")
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert snapshot == round(chi2(seq, seq), 4)

    def test_pair_sequence_snapshot(
        self, cat_pool, snapshot: SnapshotAssertion
    ) -> None:
        """Spot-check: distance between two sequences matches snapshot."""
        chi2 = Chi2SequenceMetric(entity_feature="status")
        ids = cat_pool.unique_ids
        assert snapshot == round(chi2(cat_pool[ids[0]], cat_pool[ids[1]]), 4)

    def test_same_distribution_zero(self, cat_pool) -> None:
        """Two sequences with identical distributions give distance = 0."""
        chi2 = Chi2SequenceMetric(entity_feature="status")
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert chi2(seq, seq) == pytest.approx(0.0)

    def test_no_entity_metric_needed(self) -> None:
        """Chi2SequenceMetric can be instantiated without specifying an entity metric."""
        chi2 = Chi2SequenceMetric(entity_feature="status")
        assert not hasattr(chi2.settings, "entity_metric")

    def test_entity_feature_subset(self, cat_pool) -> None:
        """Chi2 with entity_feature='status' is valid and computes finite distances."""
        chi2 = Chi2SequenceMetric(entity_feature="status")
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        result = chi2(seq_a, seq_b)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    def test_returns_distance_matrix(self, cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        assert isinstance(chi2.compute_matrix(cat_pool), DistanceMatrix)

    def test_square_and_ids_match(self, cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        dm = chi2.compute_matrix(cat_pool)
        n = len(cat_pool)
        assert dm.shape == (n, n)
        assert dm.ids == cat_pool.unique_ids

    def test_consistency_single_pair_vs_matrix(self, cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        ids = cat_pool.unique_ids[:4]
        sub = cat_pool.subset(ids)
        dm = chi2.compute_matrix(sub)
        arr = dm.to_numpy()
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                expected = chi2(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(expected, abs=1e-5)

    def test_matrix_values_snapshot(
        self, cat_pool, snapshot: SnapshotAssertion
    ) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        dm = chi2.compute_matrix(cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))

    def test_mixed_matrix_snapshot(
        self, mixed_cat_pool, snapshot: SnapshotAssertion
    ) -> None:
        """Matrix on a pool mixing empty and non-empty sequences."""
        chi2 = Chi2SequenceMetric(entity_feature="status")
        dm = chi2.compute_matrix(mixed_cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


class TestComputeCrossMatrix:
    def test_cross_matrix_shape_and_dtype(self, cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        rows = cat_pool.subset(cat_pool.unique_ids[:4])
        cols = cat_pool.subset(cat_pool.unique_ids[4:7])

        result = chi2.compute_cross_matrix(rows, cols)

        assert result.shape == (len(rows), len(cols))
        assert result.dtype == np.float32

    def test_consistency_single_pair_vs_cross_matrix(self, cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        rows = cat_pool.subset(cat_pool.unique_ids[:3])
        cols = cat_pool.subset(cat_pool.unique_ids[3:5])
        result = chi2.compute_cross_matrix(rows, cols)

        for i, id_r in enumerate(rows.unique_ids):
            for j, id_c in enumerate(cols.unique_ids):
                expected = chi2(rows[id_r], cols[id_c])
                assert result[i, j] == pytest.approx(expected, abs=1e-5)

    def test_same_pool_matches_compute_matrix(self, cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        sub = cat_pool.subset(cat_pool.unique_ids[:4])

        cross = chi2.compute_cross_matrix(sub, sub)
        matrix = chi2.compute_matrix(sub).to_numpy()

        np.testing.assert_allclose(cross, matrix, atol=1e-5)

    def test_cross_matrix_with_pool_specific_categories(self, mixed_cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        rows = mixed_cat_pool.subset(["seq_1"])
        cols = mixed_cat_pool.subset(["seq_2"])

        result = chi2.compute_cross_matrix(rows, cols)

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(
            chi2(rows["seq_1"], cols["seq_2"]), abs=1e-5
        )

    def test_cross_matrix_mixed_empty_sequences(self, mixed_cat_pool) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        rows = mixed_cat_pool.subset(["empty_1", "seq_1"])
        cols = mixed_cat_pool.subset(["empty_2", "seq_2"])
        result = chi2.compute_cross_matrix(rows, cols)

        row_empty = rows.unique_ids.index("empty_1")
        row_non_empty = rows.unique_ids.index("seq_1")
        col_empty = cols.unique_ids.index("empty_2")
        col_non_empty = cols.unique_ids.index("seq_2")

        assert result[row_empty, col_empty] == pytest.approx(0.0)
        assert result[row_empty, col_non_empty] == pytest.approx(1.0)
        assert result[row_non_empty, col_empty] == pytest.approx(1.0)
        assert result[row_non_empty, col_non_empty] == pytest.approx(
            chi2(rows["seq_1"], cols["seq_2"]),
            abs=1e-5,
        )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_registrable_lookup(self) -> None:
        assert SequenceMetric.get_registered("chi2") is Chi2SequenceMetric

    def test_settings_defaults(self) -> None:
        chi2 = Chi2SequenceMetric()
        assert chi2.settings.entity_feature is None

    def test_config_roundtrip(self) -> None:
        chi2 = Chi2SequenceMetric(entity_feature="status")
        cfg = chi2.to_config()
        chi2b = Chi2SequenceMetric.from_config(cfg)
        assert chi2b.settings.entity_feature == "status"
