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
