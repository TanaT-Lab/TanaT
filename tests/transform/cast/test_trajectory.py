#!/usr/bin/env python3
"""Cast on TrajectoryPool."""

from __future__ import annotations

import polars as pl
import pytest

from tanat.trajectory.pool import TrajectoryPool

# ---------------------------------------------------------------------------
# strict=False (lenient)
# ---------------------------------------------------------------------------


class TestCastStaticFeaturesLenient:
    """cast_static_features with strict=False silently converts non-castable values to null."""

    def test_returns_nulls(self, strict_traj: TrajectoryPool) -> None:
        """Non-castable static values become null; no exception is raised."""
        traj = strict_traj.copy()
        traj.cast_static_features({"score_str": pl.Float32}, strict=False)

        static = traj.static_data(fmt="polars")
        assert static["score_str"].dtype == pl.Float32
        assert static["score_str"].is_null().sum() == 1  # only id=2 → "bad"

    def test_valid_values_preserved(self, strict_traj: TrajectoryPool) -> None:
        """Castable static values convert correctly; only the non-castable one becomes null."""
        traj = strict_traj.copy()
        traj.cast_static_features({"score_str": pl.Float32}, strict=False)

        non_null = traj.static_data(fmt="polars")["score_str"].drop_nulls().to_list()
        assert sorted(non_null) == pytest.approx(sorted([1.5, 3.5]), abs=1e-5)


# ---------------------------------------------------------------------------
# strict=True (default)
# ---------------------------------------------------------------------------


class TestCastStaticFeaturesStrictDefault:
    """Default strict=True raises at probe time when values cannot be cast."""

    def test_raises_on_bad_values(self, strict_traj: TrajectoryPool) -> None:
        """probe raises TypeError when any static value is incompatible with the target type."""
        traj = strict_traj.copy()
        with pytest.raises(TypeError):
            traj.cast_static_features(
                {"score_str": pl.Float32}
            )  # "bad" → Float32 raises
