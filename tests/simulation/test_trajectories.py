#!/usr/bin/env python3
"""
Tests: simulate_trajectories.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tanat.dataset import simulate_trajectories


class TestSimulateTrajectories:
    """Tests for simulate_trajectories()."""

    def test_shared_ids_same_id_set(self) -> None:
        """All aliases share the same ID set when shared_ids=True."""
        data = simulate_trajectories(
            {
                "events": {"type": "event", "n_ids": 50},
                "intervals": {"type": "interval", "n_ids": 50},
            },
            shared_ids=True,
            seed=42,
        )
        ids_events = set(data["events"]["id"])
        ids_intervals = set(data["intervals"]["id"])
        assert ids_events == ids_intervals == set(range(1, 51))

    def test_mismatched_n_ids_raises(self) -> None:
        """Raises ValueError when shared_ids=True and n_ids differ."""
        with pytest.raises(ValueError, match="n_ids"):
            simulate_trajectories(
                {
                    "events": {"type": "event", "n_ids": 50},
                    "intervals": {"type": "interval", "n_ids": 100},
                },
                shared_ids=True,
            )

    def test_result_keys_match_aliases(self) -> None:
        """Returned dict has the same keys as the input sequences dict."""
        data = simulate_trajectories(
            {
                "admissions": {"type": "interval", "n_ids": 20},
                "procedures": {"type": "event", "n_ids": 20},
                "episodes": {"type": "state", "n_ids": 20},
            },
            seed=0,
        )
        assert set(data.keys()) == {"admissions", "procedures", "episodes"}

    def test_unknown_type_raises(self) -> None:
        """Raises ValueError for an unknown sequence type."""
        with pytest.raises(ValueError, match="Unknown sequence type"):
            simulate_trajectories({"x": {"type": "unknown", "n_ids": 10}})

    def test_deterministic_with_seed(self) -> None:
        """Two calls with the same master seed produce identical results."""
        data1 = simulate_trajectories({"ev": {"type": "event", "n_ids": 20}}, seed=7)
        data2 = simulate_trajectories({"ev": {"type": "event", "n_ids": 20}}, seed=7)
        pd.testing.assert_frame_equal(data1["ev"], data2["ev"])
