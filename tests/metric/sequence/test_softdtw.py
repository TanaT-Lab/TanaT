#!/usr/bin/env python3
"""
Tests: SoftDTWSequenceMetric
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.sequence import (
    SequenceMetric,
    SoftDTWSequenceMetric,
    DTWSequenceMetric,
)
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
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert snapshot == round(sdtw(seq, seq), 4)

    def test_pair_sequence_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Spot-check: distance between two sequences matches snapshot."""
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids
        assert snapshot == round(sdtw(cat_pool[ids[0]], cat_pool[ids[1]]), 4)

    def test_empty_returns_nan(self, empty_cat_pool, entity_metric) -> None:
        """Two empty sequences → SoftDTW returns nan (undefined)."""
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        ids = empty_cat_pool.unique_ids
        seq_a, seq_b = empty_cat_pool[ids[0]], empty_cat_pool[ids[1]]
        result = sdtw(seq_a, seq_b)
        assert math.isnan(result)

    def test_gamma_small_approaches_dtw(self, cat_pool, entity_metric) -> None:
        """With a very small gamma, SoftDTW ≈ DTW (from above, so ≤ DTW + epsilon)."""
        dtw = DTWSequenceMetric(entity_metric=entity_metric)
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric, gamma=1e-6)
        ids = cat_pool.unique_ids
        seq_a, seq_b = cat_pool[ids[0]], cat_pool[ids[1]]
        dtw_val = dtw(seq_a, seq_b)
        sdtw_val = sdtw(seq_a, seq_b)
        assert sdtw_val == pytest.approx(dtw_val, abs=0.5)

    def test_gamma_validation(self) -> None:
        with pytest.raises(Exception):
            SoftDTWSequenceMetric(gamma=0.0)
        with pytest.raises(Exception):
            SoftDTWSequenceMetric(gamma=-1.0)


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    def test_returns_distance_matrix(self, cat_pool, entity_metric) -> None:
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        assert isinstance(sdtw.compute_matrix(cat_pool), DistanceMatrix)

    def test_square_and_ids_match(self, cat_pool, entity_metric) -> None:
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        dm = sdtw.compute_matrix(cat_pool)
        n = len(cat_pool)
        assert dm.shape == (n, n)
        assert dm.ids == cat_pool.unique_ids

    def test_consistency_single_pair_vs_matrix(self, cat_pool, entity_metric) -> None:
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids[:4]
        sub = cat_pool.subset(ids)
        dm = sdtw.compute_matrix(sub)
        arr = dm.to_numpy()
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                expected = sdtw(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(expected, abs=1e-5)

    def test_matrix_values_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        dm = sdtw.compute_matrix(cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))

    def test_mixed_matrix_snapshot(
        self, mixed_cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Matrix on a pool mixing empty and non-empty sequences."""
        sdtw = SoftDTWSequenceMetric(entity_metric=entity_metric)
        dm = sdtw.compute_matrix(mixed_cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_registrable_lookup(self) -> None:
        assert SequenceMetric.get_registered("softdtw") is SoftDTWSequenceMetric

    def test_settings_defaults(self) -> None:
        sdtw = SoftDTWSequenceMetric()
        assert sdtw.settings.gamma == pytest.approx(1.0)

    def test_config_roundtrip(self) -> None:
        sdtw = SoftDTWSequenceMetric(gamma=0.5)
        cfg = sdtw.to_config()
        sdtw2 = SoftDTWSequenceMetric.from_config(cfg)
        assert sdtw2.settings.gamma == pytest.approx(0.5)
