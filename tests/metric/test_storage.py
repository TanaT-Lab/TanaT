#!/usr/bin/env python3
"""
Tests: StorageOptions, compute_metric_config, open_or_create_matrix, save_matrix_metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from tanat.metric._storage import (
    StorageOptions,
    compute_metric_config,
    open_or_create_matrix,
    save_matrix_metadata,
    save_progress,
)
from tanat.metric.sequence import LinearPairwiseSequenceMetric

# ---------------------------------------------------------------------------
# StorageOptions
# ---------------------------------------------------------------------------


class TestStorageOptions:
    """Construction and defaults."""

    def test_default_values(self) -> None:
        """Defaults: store_path=None, chunk_size=500, resume=True, dtype=float32."""
        opts = StorageOptions(store_path="./out")
        assert opts.store_path == "./out"
        assert opts.chunk_size == 500
        assert opts.resume is True
        assert opts.dtype == "float32"

    def test_store_path_accepts_path_object(self) -> None:
        """store_path accepts a Path in addition to str."""
        opts = StorageOptions(store_path=Path("./out"))
        assert opts.store_path == Path("./out")


# ---------------------------------------------------------------------------
# compute_metric_config
# ---------------------------------------------------------------------------


class TestComputeMetricConfig:
    """Config dict for cache invalidation."""

    def test_excludes_storage(self) -> None:
        """Changing storage does not change the config."""
        lp1 = LinearPairwiseSequenceMetric(store_path="./a")
        lp2 = LinearPairwiseSequenceMetric(store_path="./b")
        assert compute_metric_config(lp1) == compute_metric_config(lp2)

    def test_includes_metric_params(self) -> None:
        """Changing agg_fun changes the config."""
        lp1 = LinearPairwiseSequenceMetric(agg_fun="mean")
        lp2 = LinearPairwiseSequenceMetric(agg_fun="sum")
        assert compute_metric_config(lp1) != compute_metric_config(lp2)

    def test_json_round_trip(self) -> None:
        """Config survives a JSON serialization round-trip."""
        config = compute_metric_config(LinearPairwiseSequenceMetric())
        assert json.loads(json.dumps(config)) == config

    def test_config_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Config structure matches snapshot (regression guard)."""
        lp = LinearPairwiseSequenceMetric(agg_fun="sum", padding_penalty=0.5)
        assert compute_metric_config(lp) == snapshot


# ---------------------------------------------------------------------------
# open_or_create_matrix
# ---------------------------------------------------------------------------


class TestOpenOrCreateMatrix:
    """Memmap creation, resume, and invalidation logic."""

    _IDS = ["seq-0", "seq-1", "seq-2"]
    _CONFIG = {"type": "linearpairwise", "settings": {"agg_fun": "mean"}}

    def test_creates_fresh_matrix(self, tmp_path) -> None:
        """New directory → NaN-filled memmap + metadata.json."""
        storage = StorageOptions(store_path=str(tmp_path), chunk_size=10)
        mm, is_resuming, completed = open_or_create_matrix(
            storage, 3, self._IDS, self._CONFIG
        )
        assert mm.shape == (3, 3)
        assert not is_resuming
        assert completed == 0
        assert np.all(np.isnan(mm))
        assert (tmp_path / "metadata.json").exists()

    def test_reopens_existing(self, tmp_path) -> None:
        """Existing valid matrix with same config → reopen, is_resuming=True."""
        storage = StorageOptions(store_path=str(tmp_path))
        mm1, _, _ = open_or_create_matrix(storage, 3, self._IDS, self._CONFIG)
        mm1[0, 1] = 0.42
        mm1.flush()
        save_progress(storage, completed_chunks=1, status="computing")

        mm2, is_resuming, completed = open_or_create_matrix(
            storage, 3, self._IDS, self._CONFIG
        )
        assert is_resuming
        assert completed == 1
        assert mm2[0, 1] == pytest.approx(0.42)  # preserved

    def test_config_mismatch_wipes(self, tmp_path) -> None:
        """Different metric_config → delete and recreate."""
        storage = StorageOptions(store_path=str(tmp_path))
        mm1, _, _ = open_or_create_matrix(storage, 3, self._IDS, self._CONFIG)
        mm1[0, 1] = 0.42
        mm1.flush()
        save_progress(storage, completed_chunks=1)

        other_config = {**self._CONFIG, "settings": {"agg_fun": "sum"}}
        mm2, is_resuming, completed = open_or_create_matrix(
            storage, 3, self._IDS, other_config
        )
        assert not is_resuming
        assert completed == 0
        assert np.all(np.isnan(mm2))

    def test_resume_false_wipes(self, tmp_path) -> None:
        """resume=False → always delete and recreate."""
        storage = StorageOptions(store_path=str(tmp_path), resume=False)
        mm1, _, _ = open_or_create_matrix(storage, 3, self._IDS, self._CONFIG)
        mm1[0, 1] = 0.42
        mm1.flush()
        save_progress(storage, completed_chunks=1)

        mm2, is_resuming, completed = open_or_create_matrix(
            storage, 3, self._IDS, self._CONFIG
        )
        assert not is_resuming
        assert completed == 0
        assert np.all(np.isnan(mm2))

    def test_ids_mismatch_wipes(self, tmp_path) -> None:
        """Pool ids changed → delete and recreate."""
        storage = StorageOptions(store_path=str(tmp_path))
        open_or_create_matrix(storage, 3, self._IDS, self._CONFIG)
        save_progress(storage, completed_chunks=1)

        new_ids = ["a", "b", "c"]
        _, is_resuming, completed = open_or_create_matrix(
            storage, 3, new_ids, self._CONFIG
        )
        assert not is_resuming
        assert completed == 0

    def test_accepts_path_object(self, tmp_path) -> None:
        """store_path as Path object works (resolve_path handles it)."""
        storage = StorageOptions(store_path=tmp_path / "sub")
        mm, is_resuming, completed = open_or_create_matrix(
            storage, 3, self._IDS, self._CONFIG
        )
        assert mm.shape == (3, 3)
        assert not is_resuming
        assert (tmp_path / "sub" / "metadata.json").exists()

    def test_corrupted_metadata_creates_fresh(self, tmp_path) -> None:
        """Corrupted metadata.json (invalid JSON) -> fresh matrix."""
        storage = StorageOptions(store_path=str(tmp_path))
        (tmp_path / "metadata.json").write_text("{invalid json")

        mm, is_resuming, completed = open_or_create_matrix(
            storage, 3, self._IDS, self._CONFIG
        )
        assert not is_resuming
        assert completed == 0
        assert np.all(np.isnan(mm))


# ---------------------------------------------------------------------------
# save_matrix_metadata
# ---------------------------------------------------------------------------


class TestSaveMatrixMetadata:
    """Metadata persistence (written once)."""

    def test_metadata_snapshot(self, tmp_path, snapshot: SnapshotAssertion) -> None:
        """metadata.json structure matches snapshot."""
        storage = StorageOptions(store_path=str(tmp_path), chunk_size=100)
        ids = ["a", "b"]
        config = {"type": "linearpairwise", "settings": {"agg_fun": "mean"}}
        save_matrix_metadata(storage, ids, config)

        meta = json.loads((tmp_path / "metadata.json").read_text())
        meta.pop("ids", None)
        assert meta == snapshot


# ---------------------------------------------------------------------------
# save_progress
# ---------------------------------------------------------------------------


class TestSaveProgress:
    """Progress persistence (written per chunk, ~200 bytes)."""

    def test_progress_written(self, tmp_path) -> None:
        """progress.json contains completed_chunks and status."""
        storage = StorageOptions(store_path=str(tmp_path))
        save_progress(storage, completed_chunks=2, status="computing")
        prog = json.loads((tmp_path / "progress.json").read_text())
        assert prog["completed_chunks"] == 2
        assert prog["status"] == "computing"

    def test_progress_complete(self, tmp_path) -> None:
        """Final save sets status to complete."""
        storage = StorageOptions(store_path=str(tmp_path))
        save_progress(storage, completed_chunks=3, status="complete")
        prog = json.loads((tmp_path / "progress.json").read_text())
        assert prog["status"] == "complete"
