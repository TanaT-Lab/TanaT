#!/usr/bin/env python3
"""
Tests: LinearPairwiseSequenceMetric
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric.entity import HammingEntityMetric
from tanat.metric.sequence import (
    SequenceMetric,
    LinearPairwiseSequenceMetric,
)
from tanat.metric.matrix import DistanceMatrix
from tanat.metric.sequence.type.linear_pairwise.kernels import (
    compute_pairwise_chunk,
    compute_pairwise_matrix,
    _AGG_NUMBA_KERNELS,
)

# ---------------------------------------------------------------------------
# Single-pair computation
# ---------------------------------------------------------------------------


class TestSinglePair:
    """Single-pair distance properties: identity, sign, symmetry, aggregation modes."""

    def test_same_sequence_zero(self, cat_pool, entity_metric) -> None:
        """Distance from a sequence to itself is 0."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        seq = cat_pool[cat_pool.unique_ids[0]]
        assert lp(seq, seq) == 0.0

    def test_result_is_float_nonnegative(self, cat_pool, entity_metric) -> None:
        """Distance between two sequences is a non-negative float."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids
        result = lp(cat_pool[ids[0]], cat_pool[ids[1]])
        assert isinstance(result, float)
        assert result >= 0.0

    def test_symmetry(self, cat_pool, entity_metric) -> None:
        """d(seq_a, seq_b) == d(seq_b, seq_a)."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids
        seq_a = cat_pool[ids[0]]
        seq_b = cat_pool[ids[1]]
        assert lp(seq_a, seq_b) == pytest.approx(lp(seq_b, seq_a))


# ---------------------------------------------------------------------------
# Compute matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    """Full pairwise matrix: structure, numerical properties, and value regression."""

    def test_returns_distance_matrix(self, cat_pool, entity_metric) -> None:
        """compute_matrix() returns a DistanceMatrix instance."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        dm = lp.compute_matrix(cat_pool)
        assert isinstance(dm, DistanceMatrix)

    def test_square_and_ids_match(self, cat_pool, entity_metric) -> None:
        """Returned matrix is (n × n) and ids match pool.unique_ids."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        dm = lp.compute_matrix(cat_pool)
        n = len(cat_pool)
        assert dm.shape == (n, n)
        assert dm.ids == cat_pool.unique_ids

    def test_diagonal_zero(self, cat_pool, entity_metric) -> None:
        """Diagonal entries are zero (distance from a sequence to itself)."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        dm = lp.compute_matrix(cat_pool)
        np.testing.assert_array_almost_equal(np.diag(dm.to_numpy()), 0.0, decimal=5)

    def test_symmetric(self, cat_pool, entity_metric) -> None:
        """Matrix is symmetric: M[i, j] == M[j, i]."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        arr = lp.compute_matrix(cat_pool).to_numpy()
        np.testing.assert_array_almost_equal(arr, arr.T, decimal=5)

    def test_single_sequence_pool(self, cat_pool, entity_metric) -> None:
        """Pool with one sequence → 1×1 matrix with a zero."""
        one_id = cat_pool.unique_ids[:1]
        sub = cat_pool.subset(one_id)
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        dm = lp.compute_matrix(sub)
        assert dm.shape == (1, 1)
        assert dm.to_numpy()[0, 0] == 0.0

    def test_consistency_single_pair_vs_matrix(self, cat_pool, entity_metric) -> None:
        """Matrix[i,j] must equal direct lp(seq_i, seq_j)."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        ids = cat_pool.unique_ids[:4]
        sub = cat_pool.subset(ids)
        dm = lp.compute_matrix(sub)
        arr = dm.to_numpy()
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                expected = lp(sub[id_i], sub[id_j])
                assert arr[i, j] == pytest.approx(
                    expected, abs=1e-5
                ), f"Mismatch at ({i},{j}): matrix={arr[i,j]:.4f}, direct={expected:.4f}"

    def test_matrix_values_snapshot(
        self, cat_pool, entity_metric, snapshot: SnapshotAssertion
    ) -> None:
        """Full distance matrix values match snapshot (regression guard against logic changes)."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        dm = lp.compute_matrix(cat_pool)
        assert snapshot == dm.to_frame("polars").with_columns(pl.exclude("id").round(4))


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundtrip:
    """Serialization round-trips and registry dispatch via to_config / from_config."""

    def test_to_config_type(self) -> None:
        """to_config() sets the correct type key."""
        lp = LinearPairwiseSequenceMetric()
        config = lp.to_config()
        assert config["type"] == "linearpairwise"

    def test_from_config_roundtrip(self) -> None:
        """Settings survive a to_config / from_config round-trip."""
        lp = LinearPairwiseSequenceMetric(agg_fun="sum", padding_penalty=0.5)
        config = lp.to_config()
        lp2 = LinearPairwiseSequenceMetric.from_config(config)
        assert lp2.settings.agg_fun == "sum"
        assert lp2.settings.padding_penalty == 0.5

    def test_from_config_via_base_registry(self) -> None:
        """SequenceMetric.from_config dispatches to the correct subclass via registry."""
        config = {"type": "linearpairwise"}
        lp = SequenceMetric.from_config(config)
        assert isinstance(lp, LinearPairwiseSequenceMetric)

    def test_to_config_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Full config dict matches snapshot (regression guard for serialization format)."""
        lp = LinearPairwiseSequenceMetric(agg_fun="sum", padding_penalty=0.5)
        assert lp.to_config() == snapshot


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and invalid inputs."""

    def test_invalid_agg_fun_raises(self) -> None:
        """An unsupported agg_fun raises ValueError mentioning 'agg_fun'."""
        lp = LinearPairwiseSequenceMetric(agg_fun="max")  # not supported
        with pytest.raises(ValueError, match="agg_fun"):
            lp._get_agg_fn()  # pylint: disable=protected-access

    def test_empty_vs_nonempty_no_padding_raises(
        self, empty_cat_pool, cat_pool_status_only
    ) -> None:
        """Empty vs non-empty with padding_penalty=None raises ValueError."""
        seq_empty = empty_cat_pool[empty_cat_pool.unique_ids[0]]
        seq_full = cat_pool_status_only[cat_pool_status_only.unique_ids[0]]
        assert len(seq_empty) == 0
        assert len(seq_full) > 0
        lp = LinearPairwiseSequenceMetric(padding_penalty=None)
        with pytest.raises(ValueError, match="padding_penalty"):
            lp(seq_empty, seq_full)

    def test_empty_vs_nonempty_with_padding_ok(
        self, empty_cat_pool, cat_pool_status_only
    ) -> None:
        """Empty vs non-empty with padding_penalty set returns a valid distance.

        With 0 overlap entities, all positions are padded.  mean(N × p) = p.
        """
        seq_empty = empty_cat_pool[empty_cat_pool.unique_ids[0]]
        seq_full = cat_pool_status_only[cat_pool_status_only.unique_ids[0]]
        penalty = 0.75
        lp = LinearPairwiseSequenceMetric(padding_penalty=penalty)
        result = lp(seq_empty, seq_full)
        # mean of len(seq_full) identical penalties == penalty
        assert result == pytest.approx(penalty)

    def test_both_empty_returns_nan(self, empty_cat_pool) -> None:
        """Two empty sequences return NaN (distance is undefined)."""
        ids = empty_cat_pool.unique_ids
        seq_a = empty_cat_pool[ids[0]]
        seq_b = empty_cat_pool[ids[1]]
        lp = LinearPairwiseSequenceMetric()
        result = lp(seq_a, seq_b)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# Shadow dispatch
# ---------------------------------------------------------------------------


class TestShadowDispatch:
    """Temporary kwarg overrides via shadow dispatch."""

    def test_shadow_override_agg_fun(self, cat_pool_status_only) -> None:
        """Kwarg override of agg_fun applies only to the call; stored settings remain unchanged."""
        lp = LinearPairwiseSequenceMetric(
            agg_fun="mean",
        )
        ids = cat_pool_status_only.unique_ids
        seq_a, seq_b = cat_pool_status_only[ids[0]], cat_pool_status_only[ids[1]]
        result_mean = lp(seq_a, seq_b)
        result_sum = lp(seq_a, seq_b, agg_fun="sum")
        assert lp.settings.agg_fun == "mean"  # unchanged
        if result_mean > 0:
            assert result_sum >= result_mean

    def test_shadow_override_padding_penalty(self, cat_pool_status_only) -> None:
        """Kwarg override of padding_penalty applies only to the call."""
        lp = LinearPairwiseSequenceMetric(
            agg_fun="mean",
            padding_penalty=None,
        )
        ids = cat_pool_status_only.unique_ids
        seq_a, seq_b = cat_pool_status_only[ids[0]], cat_pool_status_only[ids[1]]
        result_no_pad = lp(seq_a, seq_b)
        result_pad = lp(seq_a, seq_b, padding_penalty=1.0)
        assert lp.settings.padding_penalty is None  # unchanged
        if len(seq_a) != len(seq_b):
            assert result_pad >= result_no_pad


# ---------------------------------------------------------------------------
# Numba consistency: fast path == slow path
# ---------------------------------------------------------------------------


class TestNumbaConsistency:
    """Numba fast path produces the same result as the Python fallback."""

    def test_matrix_numba_vs_python(self, cat_pool, entity_metric) -> None:
        """Numba and Python paths produce identical matrices (NaN-aware)."""
        lp = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        fast = lp.compute_matrix(cat_pool).to_numpy()  # auto-selects Numba
        # pylint: disable=protected-access
        slow = lp._compute_matrix_python(cat_pool).to_numpy()
        # NaN positions must match
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        # Finite values must be close
        mask = ~np.isnan(fast)
        np.testing.assert_array_almost_equal(fast[mask], slow[mask], decimal=5)

    def test_matrix_numba_vs_python_with_padding(self, cat_pool, entity_metric) -> None:
        """Same comparison with padding_penalty enabled (no NaN expected)."""
        lp = LinearPairwiseSequenceMetric(
            entity_metric=entity_metric, padding_penalty=1.0
        )
        fast = lp.compute_matrix(cat_pool).to_numpy()
        # pylint: disable=protected-access
        slow = lp._compute_matrix_python(cat_pool).to_numpy()
        np.testing.assert_array_almost_equal(fast, slow, decimal=5)

    def test_matrix_numba_vs_python_with_cost(self, cat_pool) -> None:
        """Same comparison with a custom cost dict."""
        em = HammingEntityMetric(
            entity_feature="status",
            cost={("f_0", "f_1"): 0.5},
            mismatch_cost=0.8,
        )
        lp = LinearPairwiseSequenceMetric(entity_metric=em)
        dm_fast = lp.compute_matrix(cat_pool)
        # pylint: disable=protected-access
        dm_slow = lp._compute_matrix_python(cat_pool)
        np.testing.assert_array_almost_equal(
            dm_fast.to_numpy(), dm_slow.to_numpy(), decimal=5
        )

    def test_matrix_numba_sum_vs_python(self, cat_pool) -> None:
        """Numba path with agg_fun='sum' matches Python path."""
        em = HammingEntityMetric(entity_feature="status")
        lp = LinearPairwiseSequenceMetric(entity_metric=em, agg_fun="sum")
        fast = lp.compute_matrix(cat_pool).to_numpy()
        # pylint: disable=protected-access
        slow = lp._compute_matrix_python(cat_pool).to_numpy()
        np.testing.assert_array_almost_equal(fast, slow, decimal=5)


# ---------------------------------------------------------------------------
# Chunked computation (memmap + resume)
# ---------------------------------------------------------------------------


class TestChunkedComputation:
    """End-to-end memmap/chunked computation and resume logic."""

    def test_memmap_matches_inmemory(self, cat_pool, entity_metric, tmp_path) -> None:
        """Memmap result matches in-memory result exactly."""
        lp_mem = LinearPairwiseSequenceMetric(entity_metric=entity_metric)
        lp_disk = LinearPairwiseSequenceMetric(
            entity_metric=entity_metric, store_path=str(tmp_path), chunk_size=3
        )
        dm_mem = lp_mem.compute_matrix(cat_pool)
        dm_disk = lp_disk.compute_matrix(cat_pool)
        np.testing.assert_array_almost_equal(
            dm_mem.to_numpy(), dm_disk.to_numpy(), decimal=5
        )

    def test_resume_skips_computed_chunks(self, cat_pool_status_only, tmp_path) -> None:
        """Second call skips chunks and produces the same result."""
        lp = LinearPairwiseSequenceMetric(store_path=str(tmp_path), chunk_size=3)
        dm1 = lp.compute_matrix(cat_pool_status_only)
        dm2 = lp.compute_matrix(cat_pool_status_only)  # all done → resume
        np.testing.assert_array_equal(dm1.to_numpy(), dm2.to_numpy())

    def test_resume_after_partial(self, cat_pool_status_only, tmp_path) -> None:
        """Remove last chunk from progress.json → resume recomputes it."""
        lp = LinearPairwiseSequenceMetric(store_path=str(tmp_path), chunk_size=3)
        dm1 = lp.compute_matrix(cat_pool_status_only)
        expected = dm1.to_numpy().copy()

        # Decrement completed_chunks to simulate partial computation
        prog_path = tmp_path / "progress.json"
        prog = json.loads(prog_path.read_text())
        if prog["completed_chunks"] > 0:
            prog["completed_chunks"] -= 1
            prog["status"] = "computing"
            prog_path.write_text(json.dumps(prog))

        dm2 = lp.compute_matrix(cat_pool_status_only)
        np.testing.assert_array_almost_equal(dm2.to_numpy(), expected, decimal=5)

    def test_settings_change_forces_recompute(
        self, cat_pool_status_only, tmp_path
    ) -> None:
        """Changing agg_fun wipes the old matrix and recomputes."""
        lp1 = LinearPairwiseSequenceMetric(
            agg_fun="mean",
            store_path=str(tmp_path),
        )
        dm1 = lp1.compute_matrix(cat_pool_status_only)

        lp2 = LinearPairwiseSequenceMetric(
            agg_fun="sum",
            store_path=str(tmp_path),
        )
        dm2 = lp2.compute_matrix(cat_pool_status_only)
        # sum vs mean → results must differ (unless all distances are 0)
        if np.any(dm1.to_numpy() > 0):
            assert not np.allclose(dm1.to_numpy(), dm2.to_numpy())

    def test_is_memmap(self, cat_pool_status_only, tmp_path) -> None:
        """Result from disk path has is_memmap=True."""
        lp = LinearPairwiseSequenceMetric(
            store_path=str(tmp_path),
        )
        dm = lp.compute_matrix(cat_pool_status_only)
        assert dm.is_memmap

    def test_metadata_and_progress_written(
        self, cat_pool_status_only, tmp_path
    ) -> None:
        """metadata.json and progress.json are present and well-formed."""
        lp = LinearPairwiseSequenceMetric(
            store_path=str(tmp_path),
        )
        lp.compute_matrix(cat_pool_status_only)

        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["shape"] == [len(cat_pool_status_only), len(cat_pool_status_only)]
        assert "metric_config" in meta

        prog = json.loads((tmp_path / "progress.json").read_text())
        assert prog["status"] == "complete"
        assert isinstance(prog["completed_chunks"], int)
        assert prog["completed_chunks"] > 0

    def test_single_sequence_memmap(self, cat_pool, entity_metric, tmp_path) -> None:
        """Pool with one sequence + memmap -> 1x1 matrix with a zero."""
        sub = cat_pool.subset(cat_pool.unique_ids[:1])
        lp = LinearPairwiseSequenceMetric(
            entity_metric=entity_metric, store_path=str(tmp_path), chunk_size=10
        )
        dm = lp.compute_matrix(sub)
        assert dm.shape == (1, 1)
        assert dm.to_numpy()[0, 0] == 0.0
        assert dm.is_memmap

    def test_chunk_size_larger_than_pool(self, cat_pool_status_only, tmp_path) -> None:
        """chunk_size > n -> single chunk, equivalent to in-memory."""
        lp_mem = LinearPairwiseSequenceMetric()
        lp_disk = LinearPairwiseSequenceMetric(
            store_path=str(tmp_path), chunk_size=9999
        )
        dm_mem = lp_mem.compute_matrix(cat_pool_status_only)
        dm_disk = lp_disk.compute_matrix(cat_pool_status_only)
        np.testing.assert_array_almost_equal(
            dm_mem.to_numpy(), dm_disk.to_numpy(), decimal=5
        )

    def test_callsite_store_path_override(self, cat_pool_status_only, tmp_path) -> None:
        """store_path kwarg on compute_matrix() overrides instance default."""
        lp = LinearPairwiseSequenceMetric()  # no storage
        dm = lp.compute_matrix(cat_pool_status_only, store_path=str(tmp_path))
        assert dm.is_memmap
        assert (tmp_path / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Chunk kernel consistency
# ---------------------------------------------------------------------------


class TestChunkKernelConsistency:
    """compute_pairwise_chunk over all rows == compute_pairwise_matrix."""

    def test_chunk_vs_full_matrix(self, cat_pool_status_only) -> None:
        """Chunked iteration produces the same matrix as the full kernel."""
        em = HammingEntityMetric(entity_feature="status")
        arrays, lengths, context = em.prepare_batch_data(cat_pool_status_only)
        n = len(arrays)
        agg_kernel = _AGG_NUMBA_KERNELS["mean"]
        padding = np.float32(np.nan)

        full = np.zeros((n, n), dtype=np.float32)
        compute_pairwise_matrix(
            full,
            arrays,
            lengths,
            em.distance_kernel,
            context,
            agg_kernel,
            padding,
        )

        chunked = np.full((n, n), np.nan, dtype=np.float32)
        chunk_size = 3
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            compute_pairwise_chunk(
                chunked,
                start,
                end,
                arrays,
                lengths,
                em.distance_kernel,
                context,
                agg_kernel,
                padding,
            )
            for i in range(start, end):
                chunked[i, i] = 0.0

        # NaN positions match
        np.testing.assert_array_equal(np.isnan(full), np.isnan(chunked))
        # Finite values match
        mask = ~np.isnan(full)
        np.testing.assert_array_almost_equal(full[mask], chunked[mask], decimal=5)
